"""Bounded native RMS normalization and original rotary-position arithmetic."""

from contextlib import ExitStack
import ctypes as C
import hashlib
import math
from pathlib import Path
import threading

import numpy as np

from .cuda import Device, Driver, compile_probe
from .resources import GiB, MiB, MemoryBudget

WIDTHS, MAX_ROWS, MAX_POSITION, MAX_FREQUENCIES = (512, 1280, 5120), 514, 131072, 4096
NORM_EPS = 1e-20
NORM_CUBIN_SHA256 = "41b9fbd12ea07f2c1bcb8766f1725b730f570e4969de86ebd41ceaf090156607"
NORM_STAGES = ("square_fp32", "mean_fp32", "mean_eps_fp32", "rstd_fp32", "normalized_fp32", "weighted_fp32", "output_bf16")
FREQUENCY_STAGES = ("inverse_frequency_fp32", "angle_fp32", "frequency_fp32")
ROPE_PROFILES = {"window": (0, 10000), "compressed-yarn": (65536, 160000)}


def mean_threads(rows, width):
    if type(rows) is not int or type(width) is not int or not 1 <= rows <= MAX_ROWS or width not in WIDTHS:
        raise ValueError("Unsupported RMS reduction profile")
    height = min(16, 1 << (rows.bit_length() - 1))
    return min(512, 1 << ((width // 4).bit_length() - 1), 512 // height)


def validate_norm(values, weight):
    if (not isinstance(values, np.ndarray) or values.dtype != np.uint16 or not 2 <= values.ndim <= 4
            or values.shape[-1] not in WIDTHS or not values.flags.c_contiguous
            or not 1 <= math.prod(values.shape[:-1]) <= MAX_ROWS
            or not isinstance(weight, np.ndarray) or weight.dtype != np.uint16
            or weight.shape != (values.shape[-1],) or not weight.flags.c_contiguous):
        raise ValueError("Expected bounded contiguous BF16 RMS inputs and BF16 weight")
    if any(np.any((a & 0x7F80) == 0x7F80) for a in (values, weight)):
        raise ValueError("RMS inputs and weight must be finite")
    return math.prod(values.shape[:-1]), values.shape[-1]


def validate_positions(positions, profile):
    if (profile not in ROPE_PROFILES or not isinstance(positions, np.ndarray) or positions.dtype != np.int32
            or positions.ndim != 1 or not 1 <= positions.size <= MAX_FREQUENCIES or not positions.flags.c_contiguous
            or np.any((positions < 0) | (positions >= MAX_POSITION))):
        raise ValueError("Unknown or unbounded original rotary frequency profile")


def validate_rope(values, start_pos, profile, inverse):
    if (not isinstance(values, np.ndarray) or values.dtype != np.uint16 or values.ndim not in (3, 4)
            or values.shape[-1] not in (64, 512) or values.shape[0] not in (1, 2)
            or not 1 <= values.shape[1] <= 257 or not values.flags.c_contiguous
            or not 1 <= math.prod(values.shape[:-1]) <= MAX_ROWS or type(start_pos) is not int
            or start_pos < 0 or start_pos + values.shape[1] > MAX_POSITION or type(inverse) is not bool
            or profile not in ROPE_PROFILES):
        raise ValueError("Expected bounded contiguous BF16 rotary input and original position/profile")
    if values.ndim == 4 and values.shape[2] not in (1, 16, 32, 64):
        raise ValueError("Unsupported rotary head shape")
    if np.any((values & 0x7F80) == 0x7F80):
        raise ValueError("Rotary input must be finite")
    return values.shape[0], values.shape[1], 1 if values.ndim == 3 else values.shape[2], values.shape[-1]


class NativeNormRoPE:
    """Serialized NumPy/Driver boundary; all GPU call buffers have scoped cleanup."""

    def __init__(self, ordinal=0):
        self.ordinal, self.opened = ordinal, False
        self.lock, self.stack = threading.Lock(), ExitStack()
        self.budget = MemoryBudget(cap=96 * MiB)

    def __enter__(self):
        with self.lock:
            if self.opened:
                raise RuntimeError("Normalization runtime is already open")
            try:
                self.budget.acquire(96 * MiB)
                self.stack.callback(self.budget.release, 96 * MiB)
                self.driver = Driver()
                self.device_info = next(d for d in self.driver.inventory()["devices"] if d["ordinal"] == self.ordinal)
                if self.device_info["compute_capability"] != "8.9":
                    raise ValueError("Normalization/rotary arithmetic is qualified only for SM89")
                self.device = self.stack.enter_context(Device(self.driver, self.ordinal, cap=64 * MiB, free_floor=2 * GiB))
                self.compiler = {}
                binary = compile_probe("89", normalization=True, compiler_info=self.compiler)
                self.cubin_sha256 = hashlib.sha256(binary).hexdigest()
                if self.compiler["nvrtc_version"] != [12, 4] or self.cubin_sha256 != NORM_CUBIN_SHA256:
                    raise ValueError("Unregistered normalization compiler/binary profile; requalification required")
                source = (Path(__file__).parent / "kernels/normalization_probe.cu").read_bytes().replace(b"\r\n", b"\n")
                self.kernel_source_sha256 = hashlib.sha256(source).hexdigest()
                self.module = self.device.load(binary)
                self.opened = True
                return self
            except BaseException:
                self.stack.close()
                raise

    def _ready(self):
        if not self.opened:
            raise RuntimeError("Normalization runtime is not open")
        self.budget.acquire(0)
        self.device.current()

    def _allocator(self, stack):
        def allocate(size):
            pointer = self.device.allocate(size)
            stack.callback(self.device.free, pointer)
            return pointer
        return allocate

    def _upload(self, array, allocate):
        pointer = allocate(array.nbytes)
        self.driver.check(self.driver.cuMemcpyHtoD_v2(pointer, array.ctypes.data, array.nbytes))
        return pointer

    def norm(self, values, weight):
        with self.lock:
            self._ready()
            rows, width = validate_norm(values, weight)
            with ExitStack() as stack:
                allocate = self._allocator(stack)
                x, w = self._upload(values, allocate), self._upload(weight, allocate)
                sizes = [values.size * 4, rows * 4, rows * 4, rows * 4, values.size * 4, values.size * 4, values.nbytes]
                pointers = [allocate(n) for n in sizes]
                sq, mean, shifted, rstd, normalized, weighted, output = pointers
                self.device.launch(self.module, "norm_square", [C.c_uint64(x), C.c_uint64(sq), C.c_int(values.size)], values.size)
                threads = mean_threads(rows, width)
                self.device.launch(self.module, "norm_mean", [C.c_uint64(sq), C.c_uint64(mean), C.c_int(width), C.c_float(1 / width)], rows * threads, threads=threads)
                self.device.launch(self.module, "norm_rstd", [C.c_uint64(p) for p in (mean, shifted, rstd)] + [C.c_float(NORM_EPS), C.c_int(rows)], rows)
                self.device.launch(self.module, "norm_apply", [C.c_uint64(p) for p in (x, w, rstd, normalized, weighted, output)]
                                   + [C.c_int(width), C.c_int(values.size)], values.size)
                shapes = (values.shape, *([(*values.shape[:-1], 1)] * 3), values.shape, values.shape, values.shape)
                return {name: self.device.download(pointer, shape, np.uint16 if name == "output_bf16" else np.float32)
                        for name, pointer, shape in zip(NORM_STAGES, pointers, shapes)}

    def _frequency_device(self, positions, profile, allocate):
        original, base = ROPE_PROFILES[profile]
        low = max(math.floor(64 * math.log(65536 / (32 * 2 * math.pi)) / (2 * math.log(base))), 0)
        high = min(math.ceil(64 * math.log(65536 / (2 * math.pi)) / (2 * math.log(base))), 63)
        pos = self._upload(positions, allocate)
        pointers = [allocate(128), allocate(positions.size * 32 * 4), allocate(positions.size * 32 * 8)]
        self.device.launch(self.module, "rope_frequencies", [C.c_uint64(p) for p in (pos, *pointers)]
                           + [C.c_int(positions.size), C.c_float(base), C.c_int(low), C.c_int(high), C.c_int(original > 0), C.c_float(16)], positions.size * 32)
        return pointers

    def frequencies(self, positions, profile="window"):
        with self.lock:
            self._ready()
            validate_positions(positions, profile)
            with ExitStack() as stack:
                pointers = self._frequency_device(positions, profile, self._allocator(stack))
                shapes = ((32,), (positions.size, 32), (positions.size, 32, 2))
                return {name: self.device.download(pointer, shape, np.float32)
                        for name, pointer, shape in zip(FREQUENCY_STAGES, pointers, shapes)}

    def rope(self, values, start_pos=0, profile="window", inverse=False, *, positions=None):
        with self.lock:
            self._ready()
            batch, sequence, heads, width = validate_rope(values, start_pos, profile, inverse)
            if positions is None:
                positions = np.arange(start_pos, start_pos + sequence, dtype=np.int32)
            else:
                validate_positions(positions, profile)
                if positions.size != sequence or positions[0] != start_pos:
                    raise ValueError("Explicit rotary positions must match sequence length and first position")
            with ExitStack() as stack:
                allocate = self._allocator(stack)
                frequency = self._frequency_device(positions, profile, allocate)
                source = self._upload(values, allocate)
                output = allocate(values.nbytes)
                rotated = allocate(batch * sequence * heads * 64 * 4)
                self.device.launch(self.module, "rope_rotate", [C.c_uint64(p) for p in (source, frequency[-1], output, rotated)]
                                   + [C.c_int(v) for v in (batch, sequence, heads, width, inverse)], values.size)
                return {"frequency_fp32": self.device.download(frequency[-1], (sequence, 32, 2), np.float32),
                        "rotated_fp32": self.device.download(rotated, (*values.shape[:-1], 64), np.float32),
                        "output_bf16": self.device.download(output, values.shape, np.uint16)}

    def __exit__(self, *_):
        with self.lock:
            self.opened = False
            self.stack.close()

    def stats(self):
        device = getattr(self, "device", None)
        return {"host_allowance_bytes": self.budget.peak, "host_reserved_bytes": self.budget.reserved,
                "device_peak_owned_bytes": device.peak if device else 0,
                "device_owned_bytes": sum(device.allocations.values()) if device else 0,
                "owned_context_active": bool(device.context) if device else False,
                "reservation_is_RSS_limit": False, "driver_module_memory_excluded": True}
