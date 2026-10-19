"""Original compressor projections/pooling and bounded compressed-KV state.

Diagnostic arrays and the small publication cache cross the host boundary.
This fidelity probe is not a resident high-throughput compression executor.
"""

from contextlib import ExitStack
import ctypes as C
from dataclasses import dataclass
import hashlib
import os
import threading

import numpy as np

from .blas import Cublas, GATE_BLAS_WORKSPACE
from .attention_topology import compression_layout, role
from .compression_fixture import COMPRESSION_LAYOUT
from .cuda import Device, Driver, _bind, compile_probe
from .formats import bf16_values
from .normalization import NativeNormRoPE
from .resources import GiB, MiB, MemoryBudget

COMPRESSION_CUBIN_SHA256 = "dc315e262350061173bc2f772a3975dd6f6eee5993d057b5453af7545a01ef5c"
LAYERS = {2: 2, 20: 1}
MAX_POSITION, MAX_PREFILL = 128, 16
POOL_STAGES = ("maximum_fp32", "exponents_fp32", "sum_exp_fp32", "weights_fp32",
               "products_fp32", "pooled_fp32", "pooled_bf16")


def validate_compression_input(values, batch, position, expected, *, max_position=MAX_POSITION):
    if type(max_position) is not int or max_position not in (128, 1152, 16392):
        raise ValueError("Unregistered compression context bound")
    if (type(batch) is not int or batch not in (1, 2) or not isinstance(values, np.ndarray) or values.dtype != np.uint16 or values.ndim != 3
            or values.shape[0] != batch or values.shape[-1] != 5120 or not 1 <= values.shape[1] <= MAX_PREFILL
            or not values.flags.c_contiguous or np.any((values & 0x7f80) == 0x7f80)
            or type(position) is not int or position != expected or position < 0
            or position + values.shape[1] > max_position or (position and values.shape[1] != 1)):
        raise ValueError("Expected bounded finite BF16 compressor input at the next exact position")


def validate_quantized_kv(values):
    if (not isinstance(values, np.ndarray) or values.dtype != np.uint16 or values.ndim != 3
            or values.shape[0] not in (1, 2) or not 1 <= values.shape[1] <= 128 or values.shape[-1] != 512
            or not values.flags.c_contiguous or np.any((values & 0x7fff) > 0x4528)):
        raise ValueError("Expected finite bounded BF16 compressed KV, magnitude at most 2688")


class NativeCompression:
    def __init__(self, arrays, ordinal=0, *, layers=(2, 20)):
        self.layout = compression_layout(layers)
        self.layers = {l: role(l).ratio for l in layers}
        self.arrays, self.ordinal = arrays, ordinal
        for name, shape in self.layout.items():
            a = arrays.get(name)
            if (not isinstance(a, np.ndarray) or a.dtype != np.uint16 or a.shape != shape
                    or not a.flags.c_contiguous or np.any((a & 0x7f80) == 0x7f80)):
                raise ValueError("Expected complete original finite BF16 compressor weights")
        self.stack, self.lock = ExitStack(), threading.Lock()
        self.budget, self.opened = MemoryBudget(cap=128 * MiB), False

    def _allocate(self, stack, size):
        pointer = self.device.allocate(size)
        stack.callback(self.device.free, pointer)
        return pointer

    def _upload(self, stack, array):
        pointer = self._allocate(stack, array.nbytes)
        self.driver.check(self.driver.cuMemcpyHtoD_v2(pointer, array.ctypes.data, array.nbytes))
        return pointer

    def __enter__(self):
        with self.lock:
            if self.opened:
                raise RuntimeError("Compression runtime is already open")
            if os.environ.get("CUBLAS_WORKSPACE_CONFIG") is not None:
                raise ValueError("Compression qualification requires default cuBLAS workspace policy")
            try:
                self.budget.acquire(128 * MiB)
                self.stack.callback(self.budget.release, 128 * MiB)
                self.driver = Driver()
                self.info = next(d for d in self.driver.inventory()["devices"] if d["ordinal"] == self.ordinal)
                if self.info["compute_capability"] != "8.9":
                    raise ValueError("Compression profile is qualified only for SM89")
                self.device = self.stack.enter_context(Device(self.driver, self.ordinal, cap=64 * MiB, free_floor=2 * GiB))
                self.compiler = {}
                binary = compile_probe("89", compression=True, compiler_info=self.compiler)
                self.cubin_sha256 = hashlib.sha256(binary).hexdigest()
                if self.compiler["nvrtc_version"] != [12, 4] or self.cubin_sha256 != COMPRESSION_CUBIN_SHA256:
                    raise ValueError("Unregistered compression compiler/binary; qualification required")
                self.module = self.device.load(binary)
                self.blas = Cublas()
                self.stack.callback(self.blas.close)
                if self.blas.version != 120402 or self.blas.math_mode != 0:
                    raise ValueError("Unregistered compression cuBLAS profile")
                self.gemm_ex = _bind(self.blas.lib, "cublasGemmEx", [C.c_void_p, *([C.c_int] * 5),
                    C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_void_p, C.c_int, C.c_int,
                    C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int])
                self.blas.workspace(self._allocate(self.stack, GATE_BLAS_WORKSPACE), GATE_BLAS_WORKSPACE)
                self.weights = {}
                for layer, ratio in self.layers.items():
                    for part in (("wkv", "wgate") if ratio == 2 else ("wkv",)):
                        a = self.arrays[f"layers.{layer}.attn.compressor.{part}.weight"]
                        # The original ratio-2 module loads these BF16 bytes into FP32 parameters.
                        self.weights[layer, part] = self._upload(self.stack, bf16_values(a) if ratio == 2 else a)
                self.norm = self.stack.enter_context(NativeNormRoPE(self.ordinal))
                self.resident_bytes = sum(self.device.allocations.values())
                self.opened = True
                return self
            except BaseException:
                self.stack.close()
                raise

    def _ready(self):
        if not self.opened:
            raise RuntimeError("Compression runtime is not open")
        self.budget.acquire(0)
        self.device.current()

    def project(self, values, layer):
        with self.lock:
            self._ready()
            if type(layer) is not int or layer not in self.layers:
                raise ValueError("Unsupported original compressor layer")
            ratio = self.layers[layer]
            validate_compression_input(values, values.shape[0] if isinstance(values, np.ndarray) and values.ndim == 3 else 0, 0, 0)
            if values.shape[0] not in (1, 2):
                raise ValueError("Compression supports batches one/two")
            rows = int(np.prod(values.shape[:2]))
            with ExitStack() as temporary:
                source = self._upload(temporary, values)
                if ratio == 2:
                    x = self._allocate(temporary, values.size * 4)
                    self.device.launch(self.module, "compression_promote", [C.c_uint64(source), C.c_uint64(x), C.c_int(values.size)], values.size)
                else:
                    x = source
                result = {}
                for part in (("wkv", "wgate") if ratio == 2 else ("wkv",)):
                    out = self._allocate(temporary, rows * 512 * (4 if ratio == 2 else 2))
                    alpha, beta = C.c_float(1), C.c_float(0)
                    if ratio == 2:
                        self.blas.check(self.blas.cublasSgemm_v2(self.blas.handle, 1, 0, 512, rows, 5120, C.byref(alpha),
                            self.weights[layer, part], 5120, x, 5120, C.byref(beta), out, 512))
                    else:
                        # CUDA_R_16BF inputs/output, original FP32 compute and default Tensor Core algorithm.
                        self.blas.check(self.gemm_ex(self.blas.handle, 1, 0, 512, rows, 5120, C.byref(alpha),
                            self.weights[layer, part], 14, 5120, x, 14, 5120, C.byref(beta), out, 14, 512, 0, 99))
                    result[part + ("_fp32" if ratio == 2 else "_bf16")] = self.device.download(out, (*values.shape[:2], 512), np.float32 if ratio == 2 else np.uint16)
                return result

    def pool(self, kv, scores):
        with self.lock:
            self._ready()
            if (not isinstance(kv, np.ndarray) or kv.dtype != np.float32 or kv.ndim != 4
                    or kv.shape[0] not in (1, 2) or not 1 <= kv.shape[1] <= 8 or kv.shape[-2:] != (2, 512)
                    or not isinstance(scores, np.ndarray) or scores.dtype != np.float32 or scores.shape != kv.shape
                    or any(not a.flags.c_contiguous or not np.isfinite(a).all() for a in (kv, scores))):
                raise ValueError("Expected bounded contiguous finite FP32 pairs for original softmax pooling")
            base = (*kv.shape[:2], 512)
            shapes = (base, kv.shape, base, kv.shape, kv.shape, base, base)
            with ExitStack() as temporary:
                sources = [self._upload(temporary, a) for a in (kv, scores)]
                pointers = [self._allocate(temporary, int(np.prod(shape)) * (2 if i == 6 else 4)) for i, shape in enumerate(shapes)]
                count = int(np.prod(base))
                self.device.launch(self.module, "compression_pool2", [C.c_uint64(p) for p in (*sources, *pointers)] + [C.c_int(count // 512)], count)
                return {name: self.device.download(p, shape, np.uint16 if i == 6 else np.float32)
                        for i, (name, p, shape) in enumerate(zip(POOL_STAGES, pointers, shapes))}

    def round_kv(self, values):
        with self.lock:
            self._ready()
            validate_quantized_kv(values)
            with ExitStack() as temporary:
                source = self._upload(temporary, values)
                rounded, packed, scales = [self._allocate(temporary, n) for n in (values.nbytes, values.size // 2, values.size // 16)]
                self.device.launch(self.module, "compression_round16", [C.c_uint64(p) for p in (source, rounded, packed, scales)] + [C.c_int(values.size // 16)], values.size)
                return {"packed_fp4": self.device.download(packed, (*values.shape[:2], 256), np.uint8),
                        "scales_e4m3": self.device.download(scales, (*values.shape[:2], 32), np.uint8),
                        "rounded_bf16": self.device.download(rounded, values.shape, np.uint16)}

    def __exit__(self, *_):
        with self.lock:
            self.opened = False
            self.stack.close()

    def stats(self):
        device = getattr(self, "device", None)
        return {"host_allowance_bytes": self.budget.peak, "host_reserved_bytes": self.budget.reserved,
                "device_owned_bytes": sum(device.allocations.values()) if device else 0,
                "device_peak_owned_bytes": device.peak if device else 0,
                "owned_context_active": bool(device.context) if device else False,
                "reservation_is_RSS_limit": False, "driver_module_memory_excluded": True}


@dataclass(frozen=True, eq=False)
class CompressionSnapshot:
    position: int


class NativeCompressedKV:
    """Transactional host publication of pre-RoPE latents, partial groups and KV.

    Original model arithmetic runs on CUDA. The bounded cache is a diagnostic
    host copy; snapshots never redirect compression or alter the original bytes.
    """
    def __init__(self, runtime, layer=2, batch=1):
        if type(layer) is not int or layer not in LAYERS or type(batch) is not int or batch not in (1, 2):
            raise ValueError("Unsupported compression state profile")
        self.runtime, self.layer, self.ratio, self.batch = runtime, layer, LAYERS[layer], batch
        self.lock, self.budget, self.opened = threading.Lock(), MemoryBudget(cap=16 * MiB), False

    def __enter__(self):
        with self.lock:
            if self.opened or not self.runtime.opened:
                raise RuntimeError("Compression state requires an opened runtime and single ownership")
            self.budget.acquire(16 * MiB)
            try:
                self._clear()
                self.opened = True
                return self
            except BaseException:
                self.budget.release(16 * MiB)
                raise

    def _clear(self):
        kv = np.zeros((self.batch, 2 if self.ratio == 2 else 0, 512), np.float32)
        score = np.full_like(kv, -np.inf)
        cache = np.zeros((self.batch, MAX_POSITION // self.ratio, 512), np.uint16)
        self.kv_state, self.score_state, self.cache = kv, score, cache
        self.position, self.handle, self.saved = 0, None, None

    def _ready(self):
        if not self.opened or not self.runtime.opened:
            raise RuntimeError("Compression state is not open")
        self.budget.acquire(0)

    def inspect(self):
        with self.lock:
            self._ready()
            return {"position": self.position, "kv_state_fp32": self.kv_state.copy(),
                    "score_state_fp32": self.score_state.copy(), "cache_bf16": self.cache.copy()}

    def step(self, values, *, start_pos=None):
        with self.lock:
            self._ready()
            pos = self.position if start_pos is None else start_pos
            validate_compression_input(values, self.batch, pos, self.position)
            sequence = values.shape[1]
            stages, (kv_state, score_state, cache) = stage_compression(
                self.runtime, self.layer, self.batch, pos, (self.kv_state, self.score_state, self.cache), values)
            self.budget.acquire(0)
            # No publication until all arithmetic, diagnostic downloads and copies succeed.
            self.kv_state, self.score_state, self.cache, self.position = kv_state, score_state, cache, pos + sequence
            return stages

    def reset(self):
        with self.lock:
            self._ready()
            self._clear()

    def snapshot(self):
        with self.lock:
            self._ready()
            if self.handle is not None:
                raise RuntimeError("Only one bounded compression snapshot is available")
            saved = tuple(a.copy() for a in (self.kv_state, self.score_state, self.cache))
            self.handle, self.saved = CompressionSnapshot(self.position), saved
            return self.handle

    def _check_handle(self, handle):
        if self.handle is None or handle is not self.handle:
            raise ValueError("Foreign, expired or consumed compression snapshot")

    def rollback(self, handle):
        with self.lock:
            self._ready()
            self._check_handle(handle)
            kv, score, cache = (a.copy() for a in self.saved)
            result = {"position": handle.position, "kv_state_fp32": kv.copy(), "score_state_fp32": score.copy(), "cache_bf16": cache.copy()}
            self.kv_state, self.score_state, self.cache, self.position = kv, score, cache, handle.position
            self.handle, self.saved = None, None
            return result

    def discard(self, handle):
        with self.lock:
            self._ready()
            self._check_handle(handle)
            self.handle, self.saved = None, None

    def __exit__(self, *_):
        with self.lock:
            if not self.opened:
                return
            self.opened, self.handle, self.saved = False, None, None
            self.kv_state = self.score_state = self.cache = None
            self.budget.release(16 * MiB)


def stage_compression(runtime, layer, batch, pos, source, values, *, max_position=MAX_POSITION):
    """Return original compressor/cache work without changing committed state."""
    validate_compression_input(values, batch, pos, pos, max_position=max_position)
    ratio, sequence = getattr(runtime, "layers", LAYERS)[layer], values.shape[1]
    stages = runtime.project(values, layer)
    kv_state, score_state, cache = (a.copy() for a in source)
    if ratio == 2:
        kv, scores = stages["wkv_fp32"], stages["wgate_fp32"]
        if pos == 0:
            cutoff = sequence - sequence % 2
            if sequence % 2:
                kv_state[:, 0], score_state[:, 0] = kv[:, -1], scores[:, -1]
            pairs = [np.ascontiguousarray(a[:, :cutoff]).reshape(batch, cutoff // 2, 2, 512) for a in (kv, scores)]
        else:
            kv_state[:, pos % 2], score_state[:, pos % 2] = kv[:, 0], scores[:, 0]
            pairs = [np.ascontiguousarray(a[:, None]) for a in (kv_state, score_state)] if (pos + 1) % 2 == 0 else None
        emits = (sequence >= 2) if pos == 0 else (pos + 1) % 2 == 0
        if emits:
            stages.update(("pool_" + name, a) for name, a in runtime.pool(*pairs).items())
            pooled = stages["pool_pooled_bf16"]
        else:
            pooled = None
    else:
        pooled = stages["wkv_bf16"]
    if pooled is not None:
        name = f"layers.{layer}.attn.compressor.norm.weight"
        normalized = runtime.norm.norm(pooled, runtime.arrays[name])
        stages.update(("norm_" + n, a) for n, a in normalized.items())
        latent = normalized["output_bf16"]
        stages["latent_pre_rope_bf16"] = latent.copy()
        positions = (np.arange(latent.shape[1], dtype=np.int32) * ratio if pos == 0
                     else np.array([pos + 1 - ratio], np.int32))
        rotated = runtime.norm.rope(latent, int(positions[0]), "compressed-yarn", positions=positions)
        stages.update(("rope_" + n, a) for n, a in rotated.items())
        rounded = runtime.round_kv(rotated["output_bf16"])
        stages.update(("quant_" + n, a) for n, a in rounded.items())
        start = pos // ratio
        cache[:, start:start + latent.shape[1]] = rounded["rounded_bf16"]
        stages["latent_positions_i32"] = positions
    else:
        stages["latent_pre_rope_bf16"] = np.empty((batch, 0, 512), np.uint16)
        stages["latent_positions_i32"] = np.empty((0,), np.int32)
    stages.update(kv_state_fp32=kv_state.copy(), score_state_fp32=score_state.copy(),
                  cache_bf16=cache.copy(), visible_bf16=cache[:, :(pos + sequence) // ratio].copy(order="C"))
    return stages, (kv_state, score_state, cache)
