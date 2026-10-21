"""Bounded original Attention projections with NumPy and the CUDA Driver."""

from contextlib import ExitStack
from copy import deepcopy
import ctypes as C
import hashlib
import math
import os
import threading

import numpy as np

from .blas import Cublas, GATE_BLAS_WORKSPACE
from .cuda import Device, Driver, _bind, compile_probe
from .execution_target import resolve_execution_target, compile_execution_kernel
from .resources import GiB, MiB, MemoryBudget

FP8_LAYOUT = {"wq_a": (1280, 5120), "wq_b": (32768, 1280), "wkv": (512, 5120), "wo_b": (5120, 8192)}
MAX_ROWS = 8
PROJECTION_CUBIN_SHA256 = "1ca16c1fa7fab955f88a246f4851aa8ff96b1178afda1af483647f9feb431332"
PROJECTION_STAGES = ("input_fp8", "scale_e8m0", "output_fp32", "output_bf16")


def validate_values(values, width):
    if (not isinstance(values, np.ndarray) or values.dtype != np.uint16 or values.ndim not in (2, 3)
            or values.shape[-1] != width or not values.flags.c_contiguous
            or not 1 <= math.prod(values.shape[:-1]) <= MAX_ROWS
            or np.any((values & 0x7F80) == 0x7F80)):
        raise ValueError("Expected bounded contiguous finite BF16 projection input")
    return math.prod(values.shape[:-1])


def validate_projection_weights(arrays, layer=0, *, profile="qualified"):
    from .attention_topology import REGISTERED_INPUT_LAYERS
    from .prefix_profile import validate_layer
    validate_layer(layer, (0, *REGISTERED_INPUT_LAYERS), profile)
    for name, (rows, width) in {**FP8_LAYOUT, "wo_a": (8192, 4096)}.items():
        for suffix, shape in (("weight", (rows, width)), ("scale", (rows // 32, width // 32))):
            a = arrays.get(f"layers.{layer}.attn.{name}.{suffix}")
            if (not isinstance(a, np.ndarray) or a.dtype != np.uint8 or a.shape != shape or not a.flags.c_contiguous):
                raise ValueError("Expected original finite block-32 Attention weights/scales")
            if np.any((a & 127) == 127) if suffix == "weight" else np.any(a == 255):
                raise ValueError("Original Attention weights/scales must be finite")


class GroupedBF16(Cublas):
    """The pinned ATen bmm layout, host FP32 scalars and BF16 output boundary."""

    def __init__(self, library=None):
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") is not None:
            raise ValueError("Grouped BF16 qualification requires the default cuBLAS workspace policy")
        super().__init__(library)
        try:
            self.bgemm = _bind(self.lib, "cublasGemmStridedBatchedEx", [C.c_void_p, *([C.c_int] * 5),
                C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_longlong,
                C.c_void_p, C.c_int, C.c_int, C.c_longlong, C.c_void_p, C.c_void_p,
                C.c_int, C.c_int, C.c_longlong, C.c_int, C.c_int, C.c_int])
        except BaseException:
            self.close()
            raise

    def grouped(self, weight, values, output, tokens):
        alpha, beta = C.c_float(1), C.c_float(0)
        self.check(self.bgemm(self.handle, 1, 0, 1024, tokens, 4096, C.byref(alpha),
            weight, 14, 4096, 1024 * 4096, values, 14, 4096 if tokens == 1 else 32768,
            4096, C.byref(beta), output, 14, 1024, tokens * 1024, 8, 0, 99))


class NativeProjections:
    """Resident original matrices, serialized calls, scoped temporary allocations."""

    def __init__(self, arrays, ordinal=0, library=None, *, layer=0, profile="qualified", execution_target=None):
        self.execution_target = resolve_execution_target(execution_target)
        self.execution_provenance = self.execution_target.describe()
        validate_projection_weights(arrays, layer, profile=profile)
        self.layer = layer
        self.arrays, self.ordinal, self.library = arrays, ordinal, library
        self.stack, self.lock, self.opened = ExitStack(), threading.Lock(), False
        self.budget = MemoryBudget(cap=512 * MiB)

    def _allocate(self, stack, size):
        pointer = self.device.allocate(size)
        stack.callback(self.device.free, pointer)
        return pointer

    def _upload(self, stack, array):
        pointer = self._allocate(stack, array.nbytes)
        self.driver.check(self.driver.cuMemcpyHtoD_v2(pointer, array.ctypes.data, array.nbytes))
        return pointer

    def __enter__(self):
        layer = self.layer
        with self.lock:
            if self.opened:
                raise RuntimeError("Projection runtime is already open")
            try:
                self.budget.acquire(512 * MiB)
                self.stack.callback(self.budget.release, 512 * MiB)
                self.driver = Driver()
                inventory = self.driver.inventory()
                self.device_info = next(d for d in inventory["devices"] if d["ordinal"] == self.ordinal)
                self.execution_provenance["device"] = self.execution_target.validate_device(self.device_info, inventory)
                self.device = self.stack.enter_context(Device(self.driver, self.ordinal, cap=256 * MiB, free_floor=2 * GiB))
                self.compiler = {}
                binary, provenance = compile_execution_kernel(compile_probe, self.execution_target,
                    projections=True, compiler_info=self.compiler, pc_cubin_sha256=PROJECTION_CUBIN_SHA256)
                self.execution_provenance.update(provenance)
                self.cubin_sha256 = provenance["cubin_sha256"]
                self.module = self.device.load(binary)
                self.blas = GroupedBF16(self.library)
                self.stack.callback(self.blas.close)
                self.execution_provenance["cublas"] = self.execution_target.validate_blas(self.blas.version, self.blas.math_mode)
                workspace = self._allocate(self.stack, GATE_BLAS_WORKSPACE)
                self.blas.workspace(workspace, GATE_BLAS_WORKSPACE)
                self.weights = {f"{n}.{s}": self._upload(self.stack, self.arrays[f"layers.{layer}.attn.{n}.{s}"])
                                for n in FP8_LAYOUT for s in ("weight", "scale")}
                self.wo_a = self._allocate(self.stack, 8192 * 4096 * 2)
                with ExitStack() as temporary:
                    w, s = [self._upload(temporary, self.arrays[f"layers.{layer}.attn.wo_a.{n}"]) for n in ("weight", "scale")]
                    self.device.launch(self.module, "projection_dequant_wo_a", [C.c_uint64(p) for p in (w, s, self.wo_a)] + [C.c_int(8192 * 4096)], 8192 * 4096)
                converted = self.device.download(self.wo_a, (8192, 4096), np.uint16)
                if np.any((converted & 0x7F80) == 0x7F80):
                    raise ValueError("Converted original wo_a leaves the finite BF16 scope")
                self.converted_sha256 = hashlib.sha256(converted.tobytes()).hexdigest()
                self.resident_bytes = sum(self.device.allocations.values())
                self.opened = True
                return self
            except BaseException:
                self.stack.close()
                raise

    def _ready(self):
        if not self.opened:
            raise RuntimeError("Projection runtime is not open")
        self.budget.acquire(0)
        self.device.current()

    def fp8(self, name, values):
        with self.lock:
            self._ready()
            if name not in FP8_LAYOUT:
                raise ValueError("Unknown original FP8 Attention projection")
            rows, width = FP8_LAYOUT[name]
            tokens = validate_values(values, width)
            with ExitStack() as stack:
                x = self._upload(stack, values)
                fp32 = self._allocate(stack, values.size * 4)
                q, s = [self._allocate(stack, n) for n in (values.size, values.size // 32)]
                trace, out = [self._allocate(stack, tokens * rows * n) for n in (4, 2)]
                self.device.launch(self.module, "projection_input", [C.c_uint64(p) for p in (x, fp32)] + [C.c_int(values.size)], values.size)
                self.device.launch(self.module, "activation32", [C.c_uint64(p) for p in (fp32, q, s)] + [C.c_int(values.size // 32)], values.size // 32)
                args = [C.c_uint64(p) for p in (q, s, self.weights[name + ".weight"], self.weights[name + ".scale"], trace, out)]
                self.device.launch(self.module, "projection_fp8", args + [C.c_int(v) for v in (rows, width, tokens)], tokens * rows * 4)
                return dict(zip(PROJECTION_STAGES, (self.device.download(q, values.shape, np.uint8),
                    self.device.download(s, (*values.shape[:-1], width // 32), np.uint8),
                    self.device.download(trace, (*values.shape[:-1], rows), np.float32),
                    self.device.download(out, (*values.shape[:-1], rows), np.uint16))))

    def grouped(self, values):
        with self.lock:
            self._ready()
            if (not isinstance(values, np.ndarray) or values.ndim not in (3, 4) or values.shape[-2:] != (8, 4096)
                    or not values.flags.c_contiguous):
                raise ValueError("Expected original eight-group BF16 output-projection input")
            tokens = validate_values(values.reshape(*values.shape[:-2], 32768), 32768)
            with ExitStack() as stack:
                x = self._upload(stack, values)
                rounded, q, s, out = [self._allocate(stack, n) for n in (values.nbytes, values.size, values.size // 32, tokens * 8192 * 2)]
                self.device.launch(self.module, "kv_round32", [C.c_uint64(p) for p in (x, rounded, q, s)] + [C.c_int(values.size // 32)], values.size)
                self.blas.grouped(self.wo_a, rounded, out, tokens)
                result = self.device.download(out, (8, tokens, 1024), np.uint16).transpose(1, 0, 2).copy().reshape(*values.shape[:-2], 8, 1024)
                return {"rounded_bf16": self.device.download(rounded, values.shape, np.uint16),
                        "input_fp8": self.device.download(q, values.shape, np.uint8),
                        "scale_e8m0": self.device.download(s, (*values.shape[:-1], 128), np.uint8), "output_bf16": result}

    def round_window(self, values):
        """Original per-32 post-RoPE FP8 rounding, without publishing a cache."""
        with self.lock:
            self._ready()
            validate_values(values, 512)
            if values.ndim != 3 or values.shape[0] not in (1, 2):
                raise ValueError("Expected bounded window KV rows")
            with ExitStack() as stack:
                x = self._upload(stack, values)
                rounded, codes, scales = [self._allocate(stack, n) for n in (values.nbytes, values.size, values.size // 32)]
                self.device.launch(self.module, "kv_round32", [C.c_uint64(p) for p in (x, rounded, codes, scales)]
                                   + [C.c_int(values.size // 32)], values.size)
                return {"rounded_bf16": self.device.download(rounded, values.shape, np.uint16),
                        "codes_fp8": self.device.download(codes, values.shape, np.uint8),
                        "scales_e8m0": self.device.download(scales, (*values.shape[:2], 16), np.uint8)}

    def stats(self):
        device = getattr(self, "device", None)
        return {"execution": deepcopy(self.execution_provenance),
                "host_allowance_bytes": self.budget.peak, "host_reserved_bytes": self.budget.reserved,
                "device_peak_owned_bytes": device.peak if device else 0,
                "device_owned_bytes": sum(device.allocations.values()) if device else 0,
                "owned_context_active": bool(device.context) if device else False,
                "reservation_is_RSS_limit": False, "driver_module_memory_excluded": True}

    def __exit__(self, *_):
        with self.lock:
            self.opened = False
            self.stack.close()
