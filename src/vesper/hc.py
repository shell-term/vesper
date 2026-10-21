"""Bounded original Hyper-Connection mixes, pre-collapse and residual expansion."""

from contextlib import ExitStack
from copy import deepcopy
import ctypes as C
import hashlib
import os
import threading

import numpy as np

from .blas import Cublas, GATE_BLAS_WORKSPACE
from .cuda import Device, Driver, compile_probe
from .execution_target import resolve_execution_target, compile_execution_kernel
from .resources import GiB, MiB, MemoryBudget

HC_CUBIN_SHA256 = "ab71eb8a6ceffec265bb6afb2e3774eaf8fb345c86e0be6ab2a501167143ce5f"
WIDTH, COPIES, MAX_TOKENS = 5120, 4, 8
MIX_STAGES = ("promoted_fp32", "square_fp32", "mean_fp32", "mean_eps_fp32", "rstd_fp32", "projection_fp32", "mixes_fp32",
              "pre_fp32", "post_fp32", "comb_fp32", "affine_fp32", "row_max_fp32", "exp_fp32", "reductions_fp32", "iterations_fp32")


def hidden_shape(x):
    if (not isinstance(x, np.ndarray) or x.dtype != np.uint16 or x.ndim != 4 or x.shape[-2:] != (4, 5120)
            or x.shape[0] not in (1, 2) or not 1 <= x.shape[0] * x.shape[1] <= MAX_TOKENS
            or not x.flags.c_contiguous or np.any((x & 0x7F80) == 0x7F80)):
        raise ValueError("Expected bounded contiguous finite BF16 [batch, sequence, 4, 5120] HC input")
    return x.shape[0] * x.shape[1]


def coefficients(value, shape):
    if (not isinstance(value, np.ndarray) or value.dtype != np.float32 or value.shape != shape
            or not value.flags.c_contiguous or not np.isfinite(value).all()):
        raise ValueError("Expected finite contiguous FP32 Hyper-Connection coefficients")


class NativeHC:
    def __init__(self, arrays, ordinal=0, *, layer=0, profile="qualified", execution_target=None):
        self.execution_target = resolve_execution_target(execution_target)
        self.execution_provenance = self.execution_target.describe()
        from .prefix_profile import validate_layer
        validate_layer(layer, (0, 2, 3, 20, 21, 22, 23, 24, 25), profile)
        self.layer = layer
        self.arrays, self.ordinal, self.opened = arrays, ordinal, False
        self.stack, self.lock, self.budget = ExitStack(), threading.Lock(), MemoryBudget(cap=128 * MiB)
        for part in ("attn", "ffn"):
            for suffix, shape in (("fn", (24, 20480)), ("scale", (3,)), ("base", (24,))):
                coefficients(arrays.get(f"layers.{layer}.hc_{part}_{suffix}"), shape)

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
                raise RuntimeError("HC runtime is already open")
            if os.environ.get("CUBLAS_WORKSPACE_CONFIG") is not None:
                raise ValueError("HC qualification requires the default cuBLAS workspace policy")
            try:
                self.budget.acquire(128 * MiB)
                self.stack.callback(self.budget.release, 128 * MiB)
                self.driver = Driver()
                inventory = self.driver.inventory()
                self.info = next(d for d in inventory["devices"] if d["ordinal"] == self.ordinal)
                self.execution_provenance["device"] = self.execution_target.validate_device(self.info, inventory)
                self.device = self.stack.enter_context(Device(self.driver, self.ordinal, cap=64 * MiB, free_floor=2 * GiB))
                self.compiler = {}
                binary, provenance = compile_execution_kernel(compile_probe, self.execution_target,
                    hc=True, compiler_info=self.compiler, pc_cubin_sha256=HC_CUBIN_SHA256)
                self.execution_provenance.update(provenance)
                self.cubin_sha256 = provenance["cubin_sha256"]
                self.module = self.device.load(binary)
                self.blas = Cublas()
                self.stack.callback(self.blas.close)
                self.execution_provenance["cublas"] = self.execution_target.validate_blas(self.blas.version, self.blas.math_mode)
                self.blas.workspace(self._allocate(self.stack, GATE_BLAS_WORKSPACE), GATE_BLAS_WORKSPACE)
                self.weights = {p + "_" + s: self._upload(self.stack, self.arrays[f"layers.{self.layer}.hc_{p}_{s}"])
                                for p in ("attn", "ffn") for s in ("fn", "scale", "base")}
                self.resident_bytes = sum(self.device.allocations.values())
                self.opened = True
                return self
            except BaseException:
                self.stack.close()
                raise

    def _ready(self):
        if not self.opened:
            raise RuntimeError("HC runtime is not open")
        self.budget.acquire(0)
        self.device.current()

    def _launch(self, name, pointers, ints, items, threads=128):
        self.device.launch(self.module, name, [C.c_uint64(p) for p in pointers] + [C.c_int(i) for i in ints], items, threads=threads)

    def mixes(self, values, part="attn"):
        with self.lock:
            self._ready()
            tokens = hidden_shape(values)
            if part not in ("attn", "ffn"):
                raise ValueError("Unknown original HC parameter group")
            b, s = values.shape[:2]
            shapes = ((b,s,20480), (b,s,20480), (b,s,1), (b,s,1), (b,s,1), (b,s,24), (b,s,24),
                      (b,s,4), (b,s,4), (b,s,4,4), (b,s,4,4), (b,s,4), (b,s,4,4), (b,s,40,4), (b,s,40,4,4))
            with ExitStack() as stack:
                x = self._upload(stack, values)
                pointers = [self._allocate(stack, int(np.prod(shape)) * 4) for shape in shapes]
                promoted, square, mean, shifted, rstd, raw, mixed, *sink = pointers
                self._launch("hc_input", (x, promoted, square), (values.size,), values.size)
                height = 1 << (tokens.bit_length() - 1)
                self._launch("hc_mean", (square, mean), (512 // height,), tokens * 512, threads=512)
                self.device.launch(self.module, "norm_rstd", [C.c_uint64(p) for p in (mean, shifted, rstd)] + [C.c_float(1e-20), C.c_int(tokens)], tokens)
                alpha, beta = C.c_float(1), C.c_float(0)
                self.blas.check(self.blas.cublasSgemm_v2(self.blas.handle, 1, 0, 24, tokens, 20480, C.byref(alpha),
                    self.weights[part + "_fn"], 20480, promoted, 20480, C.byref(beta), raw, 24))
                self._launch("hc_scale_mixes", (raw, rstd, mixed), (tokens * 24,), tokens * 24)
                self._launch("hc_sinkhorn", (mixed, self.weights[part + "_scale"], self.weights[part + "_base"], *sink), (), tokens * 32, threads=32)
                return {n: self.device.download(p, shape, np.float32) for n, p, shape in zip(MIX_STAGES, pointers, shapes)}

    def pre(self, values, pre_mix):
        with self.lock:
            self._ready()
            tokens = hidden_shape(values)
            coefficients(pre_mix, (*values.shape[:2], 4))
            with ExitStack() as stack:
                x, coeff = self._upload(stack, values), self._upload(stack, pre_mix)
                products, sums, out = [self._allocate(stack, n) for n in (values.size * 4, tokens * 5120 * 4, tokens * 5120 * 2)]
                self._launch("hc_pre", (x, coeff, products, sums, out), (tokens,), tokens * 5120)
                shape = (*values.shape[:2], 5120)
                return {"products_fp32": self.device.download(products, values.shape, np.float32),
                        "sum_fp32": self.device.download(sums, shape, np.float32), "output_bf16": self.device.download(out, shape, np.uint16)}

    def post(self, values, residual, post, comb):
        with self.lock:
            self._ready()
            tokens = hidden_shape(residual)
            if (not isinstance(values, np.ndarray) or values.dtype != np.uint16 or values.shape != (*residual.shape[:2], 5120)
                    or not values.flags.c_contiguous or np.any((values & 0x7F80) == 0x7F80)):
                raise ValueError("Expected finite original BF16 sublayer output")
            coefficients(post, (*residual.shape[:2], 4))
            coefficients(comb, (*residual.shape[:2], 4, 4))
            shapes = (residual.shape, (*residual.shape[:2],4,4,5120), residual.shape, residual.shape, residual.shape)
            with ExitStack() as stack:
                sources = [self._upload(stack, a) for a in (values, residual, post, comb)]
                pointers = [self._allocate(stack, int(np.prod(shape)) * (2 if i == 4 else 4)) for i, shape in enumerate(shapes)]
                self._launch("hc_post", (*sources, *pointers), (tokens,), residual.size)
                return {n: self.device.download(p, shape, np.uint16 if i == 4 else np.float32)
                        for i, (n,p,shape) in enumerate(zip(("post_product_fp32", "residual_products_fp32", "residual_sum_fp32", "combined_fp32", "output_bf16"), pointers, shapes))}

    def __exit__(self, *_):
        with self.lock:
            self.opened = False
            self.stack.close()

    def stats(self):
        device = getattr(self, "device", None)
        return {"execution": deepcopy(self.execution_provenance),
                "host_allowance_bytes": self.budget.peak, "host_reserved_bytes": self.budget.reserved,
                "device_owned_bytes": sum(device.allocations.values()) if device else 0,
                "device_peak_owned_bytes": device.peak if device else 0,
                "owned_context_active": bool(device.context) if device else False, "reservation_is_RSS_limit": False,
                "driver_module_memory_excluded": True}
