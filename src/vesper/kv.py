"""Native post-RoPE KV rounding and transactional 128-row window state.

Inputs are already projected, normalized and rotated BF16 bits. Those operations
and compressed/shared KV are separate gates. Cache payloads are BF16 containing
the original per-32 FP8 quantize/dequantize result, not a new quantization scheme.
"""

from contextlib import ExitStack
import ctypes as C
from dataclasses import dataclass
import hashlib
from pathlib import Path
import threading

import numpy as np

from .cuda import Device, Driver, compile_probe
from .resources import GiB, MiB, MemoryBudget

WIDTH, WINDOW, MAX_PREFILL, MAX_POSITION = 512, 128, 257, 4096
KV_CUBIN_SHA256 = "65a5efd3670668e6b1c5d748e958282706e826ccaf8bfdaf2d9172ee5cab9342"


def window_indices(batch, sequence, position, *, max_position=MAX_POSITION):
    """Preserve the original slot order, padding and materialized int32 layout."""
    if (type(max_position) is not int or max_position not in (4096, 16392)
            or any(type(v) is not int for v in (batch, sequence, position)) or batch not in (1, 2)
            or not 1 <= sequence <= MAX_PREFILL or position < 0 or position + sequence > max_position
            or (position and sequence != 1)):
        raise ValueError("Unsupported bounded window index shape or position")
    if position == 0:
        end = np.arange(sequence, dtype=np.int32)[:, None]
        indices = np.maximum(end - WINDOW + 1, 0) + np.arange(min(sequence, WINDOW), dtype=np.int32)
        indices = np.where(indices > end, -1, indices)
    else:
        oldest = position % WINDOW + 1
        indices = np.concatenate((np.arange(oldest, WINDOW, dtype=np.int32), np.arange(oldest, dtype=np.int32)))
        indices = np.where(indices > position, -1, indices)[None, :]
    return np.broadcast_to(indices[None, ...], (batch, *indices.shape)).astype(np.int32).copy()


def validate_kv_input(values, batch, position, expected_position):
    if (not isinstance(values, np.ndarray) or values.dtype != np.uint16 or values.ndim != 3
            or values.shape[0] != batch or values.shape[2] != WIDTH or not 1 <= values.shape[1] <= MAX_PREFILL
            or not values.flags.c_contiguous or type(position) is not int or position != expected_position):
        raise ValueError("Expected finite contiguous post-RoPE BF16 bits at the next exact position")
    indices = window_indices(batch, values.shape[1], position)
    if np.any((values & 0x7F80) == 0x7F80):
        raise ValueError("Expected finite post-RoPE BF16 inputs")
    return indices


@dataclass(frozen=True, eq=False)
class WindowSnapshot:
    position: int


class NativeWindowKV:
    """One sequence batch, two publication buffers and one bounded snapshot slot."""

    def __init__(self, batch=1, ordinal=0):
        if type(batch) is not int or batch not in (1, 2):
            raise ValueError("Window KV supports bounded batches of one or two")
        self.batch, self.ordinal = batch, ordinal
        self.lock, self.opened = threading.Lock(), False
        self.budget, self.stack = MemoryBudget(cap=16 * MiB), ExitStack()
        self.position, self.handle = 0, None
        self.cache_bytes = batch * WINDOW * WIDTH * 2

    def __enter__(self):
        with self.lock:
            if self.opened:
                raise RuntimeError("Window KV is already open")
            try:
                self.budget.acquire(16 * MiB)
                self.stack.callback(self.budget.release, 16 * MiB)
                self.driver = Driver()
                self.device_info = next(d for d in self.driver.inventory()["devices"] if d["ordinal"] == self.ordinal)
                if self.device_info["compute_capability"] != "8.9":
                    raise ValueError("Window KV arithmetic is currently qualified only for SM89")
                self.device = self.stack.enter_context(Device(self.driver, self.ordinal, cap=8 * MiB, free_floor=2 * GiB))
                self.compiler = {}
                binary = compile_probe("89", kv=True, compiler_info=self.compiler)
                if self.compiler["nvrtc_version"] != [12, 4]:
                    raise ValueError("Window KV requires the qualified NVRTC 12.4 compiler profile")
                self.cubin_sha256 = hashlib.sha256(binary).hexdigest()
                if self.cubin_sha256 != KV_CUBIN_SHA256:
                    raise ValueError("Unregistered Window KV compiler binary; requalification required")
                self.kernel_source_sha256 = hashlib.sha256((Path(__file__).parent / "kernels/kv_probe.cu").read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                self.module = self.device.load(binary)
                self.buffers = [self.device.allocate(self.cache_bytes) for _ in range(3)]
                self.active, self.position, self.handle = 0, 0, None
                for pointer in self.buffers[:2]:
                    self._zero(pointer)
                self.opened = True
                return self
            except BaseException:
                self.stack.close()
                raise

    def _require_open(self):
        if not self.opened:
            raise RuntimeError("Window KV is not open")
        self.budget.acquire(0)
        self.device.current()

    def _zero(self, pointer):
        self.device.launch(self.module, "kv_zero_window", [C.c_uint64(pointer), C.c_int(self.cache_bytes // 2)], self.cache_bytes // 2)

    def _copy(self, source, target):
        self.device.launch(self.module, "kv_copy_window", [C.c_uint64(source), C.c_uint64(target), C.c_int(self.cache_bytes // 2)], self.cache_bytes // 2)

    def _cache(self, pointer):
        return self.device.download(pointer, (self.batch, WINDOW, WIDTH), np.uint16)

    def append(self, values, *, start_pos=None):
        with self.lock:
            self._require_open()
            position = self.position if start_pos is None else start_pos
            indices = validate_kv_input(values, self.batch, position, self.position)
            sequence = values.shape[1]
            staged = self.buffers[1 - self.active]
            with ExitStack() as temporary:
                def allocate(size):
                    pointer = self.device.allocate(size)
                    temporary.callback(self.device.free, pointer)
                    return pointer
                source = allocate(values.nbytes)
                rounded = allocate(values.nbytes)
                codes = allocate(values.size)
                scales = allocate(values.size // 32)
                self.driver.check(self.driver.cuMemcpyHtoD_v2(source, values.ctypes.data, values.nbytes))
                groups = values.size // 32
                self.device.launch(self.module, "kv_round32", [C.c_uint64(p) for p in (source, rounded, codes, scales)]
                                   + [C.c_int(groups)], groups * 32)
                self.device.launch(self.module, "kv_stage_window", [C.c_uint64(p) for p in (self.buffers[self.active], rounded, staged)]
                                   + [C.c_int(v) for v in (self.batch, sequence, position)], self.cache_bytes // 2)
                cache = self._cache(staged)
                result = {"rounded_bf16": self.device.download(rounded, values.shape, np.uint16),
                          "codes_fp8": self.device.download(codes, values.shape, np.uint8),
                          "scales_e8m0": self.device.download(scales, (*values.shape[:2], WIDTH // 32), np.uint8),
                          "cache_bf16": cache, "indices_i32": indices}
                result["window_bf16"] = result["rounded_bf16"] if position == 0 else cache
            # Publish only after kernels, downloads and temporary-buffer cleanup succeed.
            self.active, self.position = 1 - self.active, position + sequence
            return result

    def inspect(self):
        with self.lock:
            self._require_open()
            return {"position": self.position, "cache_bf16": self._cache(self.buffers[self.active])}

    def reset(self):
        with self.lock:
            self._require_open()
            self._zero(self.buffers[1 - self.active])
            self.active, self.position, self.handle = 1 - self.active, 0, None

    def snapshot(self):
        with self.lock:
            self._require_open()
            if self.handle is not None:
                raise RuntimeError("Window KV has only one bounded live snapshot slot")
            self._copy(self.buffers[self.active], self.buffers[2])
            self.handle = WindowSnapshot(self.position)
            return self.handle

    def _check_handle(self, handle):
        if self.handle is None or handle is not self.handle:
            raise ValueError("Foreign, expired or already consumed Window KV snapshot")

    def rollback(self, handle):
        with self.lock:
            self._require_open()
            self._check_handle(handle)
            self._copy(self.buffers[2], self.buffers[1 - self.active])
            cache = self._cache(self.buffers[1 - self.active])
            self.active, self.position, self.handle = 1 - self.active, handle.position, None
            return {"position": self.position, "cache_bf16": cache}

    def discard(self, handle):
        with self.lock:
            self._require_open()
            self._check_handle(handle)
            self.handle = None

    def __exit__(self, *_):
        with self.lock:
            self.opened, self.handle = False, None
            self.stack.close()

    def stats(self):
        device = getattr(self, "device", None)
        return {"host_allowance_bytes": self.budget.peak, "host_reserved_bytes": self.budget.reserved,
                "device_peak_owned_bytes": device.peak if device else 0,
                "device_owned_bytes": sum(device.allocations.values()) if device else 0,
                "persistent_cache_and_snapshot_bytes": self.cache_bytes * 3,
                "owned_context_active": bool(device.context) if device else False,
                "reservation_is_RSS_limit": False, "driver_module_memory_excluded": True}
