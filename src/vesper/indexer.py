"""Bounded original indexer arithmetic with explicit BF16 and FP4 boundaries."""

from contextlib import ExitStack
import ctypes as C
import hashlib
import math
import os
import threading

import numpy as np

from .blas import Cublas, GATE_BLAS_WORKSPACE
from .cuda import Device, Driver, _bind, compile_probe
from .indexer_fixture import DTYPES, INDEXER_LAYOUT
from .normalization import NativeNormRoPE, validate_positions
from .resources import GiB, MiB, MemoryBudget

INDEXER_CUBIN_SHA256 = "411f05580290d098d830bac502f9c6b1c7c98eafc12f65628823a9fdbc3d277f"
MAX_INDEX_POSITIONS, MAX_QUERY_LENGTH = 16392, 8


def finite_bf16(a, width, max_rows=514):
    if (not isinstance(a, np.ndarray) or a.dtype != np.uint16 or not 2 <= a.ndim <= 4
            or a.shape[-1] != width or not a.flags.c_contiguous or not 1 <= math.prod(a.shape[:-1]) <= max_rows
            or np.any((a & 0x7f80) == 0x7f80)):
        raise ValueError("Expected bounded contiguous finite BF16 indexer input")


def validate_query(x, qr, layer, position):
    finite_bf16(x, 5120, 16)
    finite_bf16(qr, 1280, 16)
    if (type(layer) is not int or layer not in (2, 20, 24) or x.ndim != 3 or x.shape[0] not in (1, 2)
            or not 1 <= x.shape[1] <= MAX_QUERY_LENGTH or qr.shape != (*x.shape[:2], 1280)
            or type(position) is not int or not 0 <= position <= 131072 - x.shape[1]):
        raise ValueError("Unsupported original indexer query profile")


def validate_selection(scores, lengths, offset, source, candidates):
    if (not isinstance(scores, np.ndarray) or scores.dtype != np.uint16 or scores.ndim != 3
            or scores.shape[0] not in (1, 2) or not 1 <= scores.shape[1] <= MAX_QUERY_LENGTH
            or not 0 <= scores.shape[2] <= MAX_INDEX_POSITIONS or not scores.flags.c_contiguous
            or np.any((scores & 0x7fff) > 0x7f80)
            or not isinstance(lengths, np.ndarray) or lengths.dtype != np.int32 or lengths.shape != scores.shape[:2]
            or not lengths.flags.c_contiguous or np.any((lengths < 0) | (lengths > scores.shape[2]))
            or type(offset) is not int or not 0 <= offset <= 0x7fffffff - scores.shape[2] or type(source) is not bool):
        raise ValueError("Expected bounded non-NaN BF16 scores, causal lengths and int32 index offset")
    if candidates is not None and (source or not isinstance(candidates, np.ndarray) or candidates.dtype != np.bool_
            or candidates.shape != scores.shape or not candidates.flags.c_contiguous):
        raise ValueError("Invalid original candidate-mask ownership/layout")


class NativeIndexer:
    def __init__(self, arrays, ordinal=0):
        self.arrays, self.ordinal = arrays, ordinal
        for name, (dtype, shape) in INDEXER_LAYOUT.items():
            a = arrays.get(name)
            if not isinstance(a, np.ndarray) or a.dtype != DTYPES[dtype] or a.shape != shape or not a.flags.c_contiguous:
                raise ValueError("Expected complete original indexer weights/scales")
            invalid = (a & 0x7f80) == 0x7f80 if dtype == "BF16" else ((a & 127) == 127 if dtype == "F8_E4M3" else a == 255)
            if np.any(invalid):
                raise ValueError("Original indexer weights/scales must be finite")
        self.stack, self.lock = ExitStack(), threading.RLock()
        self.budget, self.opened = MemoryBudget(cap=256 * MiB), False

    def _allocate(self, stack, size):
        p = self.device.allocate(size)
        stack.callback(self.device.free, p)
        return p

    def _upload(self, stack, a):
        p = self._allocate(stack, a.nbytes)
        self.driver.check(self.driver.cuMemcpyHtoD_v2(p, a.ctypes.data, a.nbytes))
        return p

    def _launch(self, name, pointers, ints, items, threads=128):
        self.device.launch(self.module, name, [C.c_uint64(p) for p in pointers] + [C.c_int(v) for v in ints], items, threads=threads)

    def __enter__(self):
        with self.lock:
            if self.opened:
                raise RuntimeError("Indexer is already open")
            if os.environ.get("CUBLAS_WORKSPACE_CONFIG") is not None:
                raise ValueError("Indexer requires the registered default cuBLAS workspace policy")
            try:
                self.budget.acquire(256 * MiB)
                self.stack.callback(self.budget.release, 256 * MiB)
                self.driver = Driver()
                self.info = next(d for d in self.driver.inventory()["devices"] if d["ordinal"] == self.ordinal)
                if self.info["compute_capability"] != "8.9":
                    raise ValueError("Indexer qualification requires SM89")
                self.device = self.stack.enter_context(Device(self.driver, self.ordinal, cap=128 * MiB, free_floor=2 * GiB))
                self.compiler = {}
                binary = compile_probe("89", indexer=True, compiler_info=self.compiler)
                self.cubin_sha256 = hashlib.sha256(binary).hexdigest()
                if self.compiler["nvrtc_version"] != [12, 4] or self.cubin_sha256 != INDEXER_CUBIN_SHA256:
                    raise ValueError("Unregistered native indexer binary/compiler")
                self.module = self.device.load(binary)
                self.blas = Cublas()
                self.stack.callback(self.blas.close)
                if self.blas.version != 120402 or self.blas.math_mode != 0:
                    raise ValueError("Unregistered native indexer cuBLAS profile")
                self.gemm = _bind(self.blas.lib, "cublasGemmEx", [C.c_void_p, *([C.c_int] * 5),
                    C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_void_p, C.c_int, C.c_int,
                    C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int])
                self.bmm = _bind(self.blas.lib, "cublasGemmStridedBatchedEx", [C.c_void_p, *([C.c_int] * 5),
                    C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_longlong, C.c_void_p, C.c_int, C.c_int, C.c_longlong,
                    C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_longlong, C.c_int, C.c_int, C.c_int])
                self.blas.workspace(self._allocate(self.stack, GATE_BLAS_WORKSPACE), GATE_BLAS_WORKSPACE)
                self.weights = {n: self._upload(self.stack, self.arrays[n]) for n in INDEXER_LAYOUT if not n.endswith("k_norm.weight")}
                self.norm = self.stack.enter_context(NativeNormRoPE(ordinal=self.ordinal))
                self.resident_bytes = sum(self.device.allocations.values())
                self.opened = True
                return self
            except BaseException:
                self.stack.close()
                raise

    def _ready(self):
        if not self.opened:
            raise RuntimeError("Indexer is not open")
        self.budget.acquire(0)
        self.device.current()

    def _bf16_linear(self, stack, values, layer, part, rows, width):
        x = self._upload(stack, values)
        tokens = math.prod(values.shape[:-1])
        out = self._allocate(stack, tokens * rows * 2)
        alpha, beta = C.c_float(1), C.c_float(0)
        self.blas.check(self.gemm(self.blas.handle, 1, 0, rows, tokens, width, C.byref(alpha),
            self.weights[f"layers.{layer}.attn.indexer.{part}.weight"], 14, width, x, 14, width, C.byref(beta), out, 14, rows, 0, 99))
        return out

    def round(self, values):
        with self.lock:
            self._ready()
            finite_bf16(values, 128, 1024)
            with ExitStack() as temp:
                x = self._upload(temp, values)
                out, packed, scales = [self._allocate(temp, n) for n in (values.nbytes, values.size // 2, values.size // 32)]
                self._launch("index_round32", (x, out, packed, scales), (values.size // 32,), values.size)
                return {"packed_fp4": self.device.download(packed, (*values.shape[:-1], 64), np.uint8),
                        "scales_e8m0": self.device.download(scales, (*values.shape[:-1], 4), np.uint8),
                        "rounded_bf16": self.device.download(out, values.shape, np.uint16)}

    def key(self, latent, layer, positions):
        with self.lock:
            self._ready()
            finite_bf16(latent, 512)
            if type(layer) is not int or layer not in (2, 20) or latent.ndim != 3 or latent.shape[0] not in (1, 2) or not 1 <= latent.shape[1] <= 257:
                raise ValueError("Unsupported original index-key owner/profile")
            validate_positions(positions, "compressed-yarn")
            if positions.shape != (latent.shape[1],):
                raise ValueError("Index-key rotary positions must match the latent count")
            with ExitStack() as temp:
                p = self._bf16_linear(temp, latent, layer, "wk", 128, 512)
                projected = self.device.download(p, (*latent.shape[:2], 128), np.uint16)
            stages = {"wk_bf16": projected}
            norm = self.norm.norm(projected, self.arrays[f"layers.{layer}.attn.indexer.k_norm.weight"])
            stages.update(("norm_" + n, a) for n, a in norm.items())
            rope = self.norm.rope(norm["output_bf16"], int(positions[0]), "compressed-yarn", positions=positions)
            stages.update(("rope_" + n, a) for n, a in rope.items())
            stages.update(("quant_" + n, a) for n, a in self.round(rope["output_bf16"]).items())
            return stages

    def queries(self, x, qr, layer, position):
        with self.lock:
            self._ready()
            validate_query(x, qr, layer, position)
            batch, sequence = x.shape[:2]
            tokens = batch * sequence
            with ExitStack() as temp:
                source = self._upload(temp, qr)
                promoted, quant, scale, trace, output = [self._allocate(temp, n) for n in (qr.size * 4, qr.size, qr.size // 32, tokens * 4096 * 4, tokens * 4096 * 2)]
                self._launch("projection_input", (source, promoted), (qr.size,), qr.size)
                self._launch("activation32", (promoted, quant, scale), (qr.size // 32,), qr.size // 32)
                self._launch("projection_fp8", (quant, scale, self.weights[f"layers.{layer}.attn.indexer.wq_b.weight"],
                    self.weights[f"layers.{layer}.attn.indexer.wq_b.scale"], trace, output), (4096, 1280, tokens), tokens * 4096 * 4)
                stages = {"input_fp8": self.device.download(quant, qr.shape, np.uint8),
                          "input_scale_e8m0": self.device.download(scale, (*qr.shape[:2], 40), np.uint8),
                          "projection_fp32": self.device.download(trace, (batch, sequence, 32, 128), np.float32),
                          "projection_bf16": self.device.download(output, (batch, sequence, 32, 128), np.uint16)}
                weight = self._bf16_linear(temp, x, layer, "weights_proj", 32, 5120)
                scaled = self._allocate(temp, tokens * 32 * 2)
                self._launch("index_scale_weights", (weight, scaled), (tokens * 32,), tokens * 32)
                stages["weights_raw_bf16"] = self.device.download(weight, (batch, sequence, 32), np.uint16)
                stages["weights_scaled_bf16"] = self.device.download(scaled, (batch, sequence, 32), np.uint16)
            rotated = self.norm.rope(stages["projection_bf16"], position, "compressed-yarn")
            stages.update(("rope_" + n, a) for n, a in rotated.items())
            stages.update(("quant_" + n, a) for n, a in self.round(rotated["output_bf16"]).items())
            return stages

    def score(self, queries, keys, weights):
        with self.lock:
            self._ready()
            finite_bf16(queries, 128, 512)
            finite_bf16(weights, 32, 16)
            if (queries.ndim != 4 or queries.shape[0] not in (1, 2) or not 1 <= queries.shape[1] <= 8 or queries.shape[2] != 32
                    or weights.shape != (*queries.shape[:2], 32) or not isinstance(keys, np.ndarray) or keys.dtype != np.uint16
                    or keys.ndim != 3 or keys.shape[0] != queries.shape[0] or keys.shape[2] != 128 or not keys.flags.c_contiguous
                    or not 0 <= keys.shape[1] <= MAX_INDEX_POSITIONS or np.any((keys & 0x7f80) == 0x7f80)):
                raise ValueError("Unsupported bounded index score shape")
            batch, sequence = queries.shape[:2]
            n, rows = keys.shape[1], batch * sequence
            shape = (batch, sequence, 32, n)
            if not n:
                return {"dot_bf16": np.empty(shape, np.uint16), "relu_bf16": np.empty(shape, np.uint16),
                        "weighted_bf16": np.empty(shape, np.uint16), "sum_fp32": np.empty((batch, sequence, 0), np.float32),
                        "score_bf16": np.empty((batch, sequence, 0), np.uint16)}
            with ExitStack() as temp:
                q, k, w = [self._upload(temp, a) for a in (queries, keys, weights)]
                dot, relu, weighted, accum, score = [self._allocate(temp, size) for size in (rows * 32 * n * 2,) * 3 + (rows * n * 4, rows * n * 2)]
                alpha, beta = C.c_float(1), C.c_float(0)
                self.blas.check(self.bmm(self.blas.handle, 1, 0, n, sequence * 32, 128, C.byref(alpha),
                    k, 14, 128, n * 128, q, 14, 128, sequence * 32 * 128, C.byref(beta), dot, 14, n,
                    sequence * 32 * n, batch, 0, 99))
                self._launch("index_products", (dot, w, relu, weighted), (n, rows * 32 * n), rows * 32 * n)
                self._launch("index_reduce_heads", (weighted, accum, score), (n, rows * n), rows * n)
                return {"dot_bf16": self.device.download(dot, shape, np.uint16),
                        "relu_bf16": self.device.download(relu, shape, np.uint16),
                        "weighted_bf16": self.device.download(weighted, shape, np.uint16),
                        "sum_fp32": self.device.download(accum, (batch, sequence, n), np.float32),
                        "score_bf16": self.device.download(score, (batch, sequence, n), np.uint16)}

    def select(self, scores, lengths, offset=128, *, candidate_source=False, candidates=None):
        with self.lock:
            self._ready()
            validate_selection(scores, lengths, offset, candidate_source, candidates)
            batch, sequence, n = scores.shape
            rows, k = batch * sequence, min(512, n)
            if not n:
                result = {"causal_bf16": scores.copy(), "masked_bf16": scores.copy(), "indices_i32": np.empty((batch, sequence, 0), np.int32)}
                if candidate_source: result.update(candidate_blocks_bf16=np.empty((batch, sequence, 0), np.uint16), candidates_bool=np.empty(scores.shape, np.bool_))
                elif candidates is not None: result["candidates_bool"] = candidates.copy()
                return result
            with ExitStack() as temp:
                source, lens = self._upload(temp, scores), self._upload(temp, lengths)
                causal = self._allocate(temp, scores.nbytes)
                self._launch("index_causal", (source, causal, lens), (n, scores.size), scores.size)
                result = {"causal_bf16": self.device.download(causal, scores.shape, np.uint16)}
                if candidate_source:
                    count = (n + 7) // 8
                    blocks, mask = self._allocate(temp, rows * count * 2), self._allocate(temp, scores.size)
                    self._launch("index_block_scores", (causal, blocks, lens), (n, count, rows * count), rows * count)
                    self._launch("index_candidates", (blocks, mask), (count, n, rows), rows, threads=32)
                    result["candidate_blocks_bf16"] = self.device.download(blocks, (batch, sequence, count), np.uint16)
                    result["candidates_bool"] = self.device.download(mask, scores.shape, np.bool_)
                elif candidates is not None:
                    mask = self._upload(temp, candidates)
                    masked = self._allocate(temp, scores.nbytes)
                    self._launch("index_candidate_mask", (causal, mask, masked), (scores.size,), scores.size)
                    causal = masked
                    result["candidates_bool"] = candidates.copy()
                result["masked_bf16"] = self.device.download(causal, scores.shape, np.uint16)
                out = self._allocate(temp, rows * k * 4)
                self._launch("index_select", (causal, lens, out), (n, k, rows, offset), rows, threads=32)
                result["indices_i32"] = self.device.download(out, (batch, sequence, k), np.int32)
                return result

    def stats(self):
        device = getattr(self, "device", None)
        return {"host_allowance_bytes": self.budget.peak, "host_reserved_bytes": self.budget.reserved,
                "device_owned_bytes": sum(device.allocations.values()) if device else 0,
                "device_peak_owned_bytes": device.peak if device else 0,
                "owned_context_active": bool(device.context) if device else False,
                "reservation_is_RSS_limit": False, "driver_module_memory_excluded": True}

    def __exit__(self, *_):
        with self.lock:
            self.opened = False
            self.stack.close()
