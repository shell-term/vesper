"""Qualify a storage-only PTX adapter before exposing original Attention math.

Untouched 64x512-kernel execution remains a distinct blocked resource gate.
The adapter preserves literal PTX arithmetic, registers, loops and barriers.
"""

from __future__ import annotations

from contextlib import ExitStack
import ctypes as C
import hashlib
import importlib.util
from pathlib import Path

import numpy as np

from .attention_memory import FUNCTION, inspect_codegen, loader_check_cuda
from .attention_inputs import CASES
from .attention_ptx import virtualize_ptx
from .attention_storage import STAGES, load_variant, trace_shapes
from .attention_storage_reference import _check, _inputs
from .checkpoint import MODEL, REVISION, fetch_reference
from .cuda import Driver, Device, _bind
from .resources import GiB, MiB, MemoryBudget, physical_memory

def missing_dependencies():
    return [name for name in ("torch", "tilelang") if importlib.util.find_spec(name) is None]


class AttentionMemoryReference:
    def __init__(self, directory: Path, ordinal=0):
        self.directory, self.ordinal = Path(directory), ordinal
        self.stack, self.qualified, self.opened = ExitStack(), False, False

    def __enter__(self):
        if self.opened:
            raise RuntimeError("Attention reference is already open")
        try:
            self._open()
            self.opened = True
            return self
        except BaseException:
            self.stack.close()
            raise

    def _open(self):
        import torch
        import tilelang
        from tilelang.contrib.nvrtc import compile_cuda, get_nvrtc_version
        from tilelang.env import TILELANG_TEMPLATE_PATH, CUTLASS_INCLUDE_DIR, CUDA_HOME
        if tilelang.__version__ != "0.1.12" or get_nvrtc_version() != (13, 4):
            raise ValueError("Unqualified Attention compiler profile; requires TileLang 0.1.12 and NVRTC 13.4")
        self.torch, self.compile_cuda, self.tilelang = torch, compile_cuda, tilelang
        self.nvrtc_version = get_nvrtc_version()
        self.budget = MemoryBudget(cap=64 * MiB)
        self.budget.acquire(64 * MiB)
        self.stack.callback(self.budget.release, 64 * MiB)
        self.source = fetch_reference(self.directory)
        with (self.directory / "kernel.py").open("rb") as stream:
            raw = stream.read(MiB)
        self.driver = Driver()
        inventory = self.driver.inventory()
        self.device_info = next(d for d in inventory["devices"] if d["ordinal"] == self.ordinal)
        if self.device_info["compute_capability"] != "8.9" or inventory["driver_api_version"] < 13040:
            raise ValueError("Unqualified Attention execution profile; requires SM 8.9 and a CUDA 13.4-capable driver")
        self.driver_version = inventory["driver_api_version"]
        query = _bind(self.driver.lib, "cuDeviceGetAttribute", [C.POINTER(C.c_int), C.c_int, C.c_int])
        shared_limit = C.c_int()
        self.driver.check(query(C.byref(shared_limit), 97, self.ordinal))
        self.shared_limit = shared_limit.value
        self.device = self.stack.enter_context(Device(self.driver, self.ordinal))
        self.set_attribute = _bind(self.driver.lib, "cuFuncSetAttribute", [C.c_void_p, C.c_int, C.c_int])
        self.options = [f"-I{TILELANG_TEMPLATE_PATH}", f"-I{CUTLASS_INCLUDE_DIR}", f"-I{CUDA_HOME}/include",
                        f"-I{CUDA_HOME}/include", f"-I{CUDA_HOME}/include/cccl", "-D__CUDACC_VER_MAJOR__=13"]
        self.factories = {(storage, diagnostic): load_variant(raw, self.directory / "attention-variants", storage, diagnostic)[0]
                          for storage in ("shared", "register-query") for diagnostic in (False, True)}
        self.profiles, self.kernels, self.originals, self.candidates = [], {}, {}, {}
        with torch.cuda.device(self.ordinal):
            if torch.cuda.mem_get_info(self.ordinal)[0] < 2 * GiB:
                raise MemoryError("Insufficient VRAM reserve for the Attention reference")
            for heads, dimension in ((64, 256), (16, 512), (64, 512)):
                for diagnostic in (False, True):
                    key = heads, dimension, diagnostic
                    original = self.factories["shared", diagnostic](heads, dimension, dimension**-.5)
                    candidate = self.factories["register-query", diagnostic](heads, dimension, dimension**-.5)
                    cuda = original.get_kernel_source()
                    meta = inspect_codegen(cuda)
                    if meta["adapted_shared_bytes"] > self.shared_limit:
                        raise ValueError("Adapted Attention exceeds the device shared-memory limit")
                    flags = original.adapter.lib_generator.compile_flags or []
                    extra = [item for flag in flags for item in flag.split() if item not in self.options]
                    binary = bytes(compile_cuda(cuda, "cubin", 89, self.options + extra))
                    with Path(original.adapter.lib_generator.libpath).open("rb") as stream:
                        cached = stream.read(8 * MiB + 1)
                    if len(cached) > 8 * MiB or binary != cached:
                        raise ValueError("Recompiled original Attention binary does not reproduce the oracle cache")
                    ptx = bytes(compile_cuda(cuda, "ptx", 89, self.options + extra)).decode()
                    adapted, audit = virtualize_ptx(ptx, meta)
                    module = self.device.load(adapted.encode() + b"\0")
                    meta.update(ptx_audit=audit, original_binary_reproduction=True,
                                original_cubin_sha256=hashlib.sha256(binary).hexdigest(), compile_flags=flags)
                    self.profiles.append(meta)
                    self.kernels[key] = module, meta
                    self.originals[key], self.candidates[key] = original, candidate
                    if key == (64, 512, False):
                        self.full_source = cuda
            loader_source, self.loader_meta = loader_check_cuda(self.full_source)
            loader = bytes(compile_cuda(loader_source, "cubin", 89, self.options))
            self.loader_meta["cubin_sha256"] = hashlib.sha256(loader).hexdigest()
            self.loader = self.device.load(loader)

    def _launch(self, module, name, arguments, batch, sequence, shared):
        torch = self.torch
        self.device.current()
        function = C.c_void_p()
        self.driver.check(self.driver.cuModuleGetFunction(C.byref(function), module, name.encode()))
        self.driver.check(self.set_attribute(function, 8, shared))
        values = [C.c_uint64(a.data_ptr()) if isinstance(a, torch.Tensor) else C.c_int(a) for a in arguments]
        pointers = (C.c_void_p * len(values))(*[C.cast(C.byref(a), C.c_void_p) for a in values])
        torch.cuda.synchronize(self.ordinal)
        self.driver.check(self.driver.cuLaunchKernel(function, sequence, batch, 1, 256, 1, 1, shared, None, pointers, None))
        self.driver.check(self.driver.cuCtxSynchronize())

    def _run(self, heads, dimension, diagnostic, inputs):
        torch = self.torch
        q, kv, sink, idx = inputs
        if q.ndim != 4 or idx.ndim != 3:
            raise ValueError("Expected four-dimensional Query and three-dimensional indices")
        batch, sequence = q.shape[:2]
        shapes = trace_shapes(batch, sequence, heads, dimension, idx.shape[-1])
        if (q.shape != (batch, sequence, heads, dimension) or kv.ndim != 3 or kv.shape[0] != batch
                or not 1 <= kv.shape[1] <= 257 or kv.shape[2] != dimension or sink.shape != (heads,)
                or idx.shape[:2] != (batch, sequence)
                or tuple(t.dtype for t in inputs) != (torch.bfloat16, torch.bfloat16, torch.float32, torch.int32)
                or any(t.device != q.device or not t.is_contiguous() for t in inputs)
                or q.device != torch.device(f"cuda:{self.ordinal}") or bool(((idx < -1) | (idx >= kv.shape[1])).any())):
            raise ValueError("Invalid, noncontiguous or unbounded original Attention inputs")
        self.budget.acquire(0)
        result, workspace = torch.empty_like(q), torch.empty_like(q)
        traces = [torch.empty(shape, dtype=torch.float32, device=q.device) for shape in shapes] if diagnostic else []
        args = {"q": q, "kv": kv, "o": result, "attn_sink": sink, "topk_idxs": idx, "b": batch, "m": sequence,
                "n": kv.shape[1], "topk": idx.shape[-1], "q_workspace": workspace, **dict(zip(STAGES, traces))}
        module, meta = self.kernels[heads, dimension, diagnostic]
        self._launch(module, FUNCTION, [args[name] for name in meta["kernel_argument_names"]], batch, sequence, meta["adapted_shared_bytes"])
        return result, traces, workspace

    def __call__(self, q, kv, sink, indices, scale):
        if not self.qualified:
            raise RuntimeError("Attention memory reference is not qualified")
        if q.shape[-2:] != (64, 512) or scale != 512**-.5:
            raise ValueError("Qualified capture requires the original 64x512 Attention profile")
        return self._run(64, 512, False, (q, kv, sink, indices))[0]

    def qualify(self):
        self.qualified = False
        torch = self.torch
        rows, load_checks, store_checks = [], [], []
        minimum = physical_memory()["available"]
        with torch.cuda.device(self.ordinal), torch.inference_mode():
            torch.cuda.reset_peak_memory_stats(self.ordinal)
            # Raw bit transport, including every BF16 encoding, is independent
            # of whether those encodings represent finite arithmetic inputs.
            for pattern in range(4):
                batch, sequence = 2, 3
                data = np.arange(batch * sequence * 32768, dtype=np.uint32).reshape(batch, sequence, 64, 512)
                if pattern == 0:
                    data %= 65536
                elif pattern == 1:
                    data = (data * 177 + 8191) % 65536
                elif pattern == 2:
                    data = 1 << (data % 16)
                else:
                    data = np.random.default_rng(20261018).integers(0, 65536, data.shape, dtype=np.uint32)
                q = torch.from_numpy(data.astype(np.uint16)).view(torch.bfloat16).to(f"cuda:{self.ordinal}")
                workspace = torch.empty_like(q)
                expected = torch.empty((batch, sequence, 32, 256, 4), dtype=torch.int32, device=q.device)
                actual = torch.empty_like(expected)
                self._launch(self.loader, "check_query_load", [q, workspace, expected, actual, sequence, 0], batch, sequence, 65536)
                check = _check(expected, actual, "PTX_load_vs_hardware_ldmatrix")
                check["pattern"] = pattern
                load_checks.append(check)
                kv = torch.zeros(batch, 1, 512, device=q.device, dtype=torch.bfloat16)
                sink = torch.zeros(64, device=q.device, dtype=torch.float32)
                idx = torch.full((batch, sequence, 1), -1, device=q.device, dtype=torch.int32)
                _, _, stored = self._run(64, 512, False, (q, kv, sink, idx))
                check = _check(workspace, stored, "PTX_store_vs_original_Query_copy")
                check["pattern"] = pattern
                store_checks.append(check)
            self._launch(self.loader, "check_query_load", [q, workspace, expected, actual, sequence, 1], batch, sequence, 65536)
            negative = _check(expected, actual, "wrong_column_negative_control")
            for heads, dimension in ((64, 256), (16, 512), (64, 512)):
                for case_index, case in enumerate(CASES):
                    minimum = min(minimum, physical_memory()["available"])
                    self.budget.acquire(0, minimum)
                    inputs, provenance = _inputs(torch, heads, dimension, case, self.ordinal)
                    q, kv, sink, idx = inputs
                    adapted, traces, _ = self._run(heads, dimension, True, inputs)
                    plain, _, _ = self._run(heads, dimension, False, inputs)
                    candidate, cp = torch.empty_like(q), torch.empty_like(q)
                    candidate_trace = [torch.empty_like(t) for t in traces]
                    self.candidates[heads, dimension, True](q, kv, candidate, sink, idx, *candidate_trace)
                    self.candidates[heads, dimension, False](q, kv, cp, sink, idx)
                    checks = [_check(plain, adapted, "PTX_instrumentation_BF16"),
                              _check(cp, candidate, "register_instrumentation_BF16"),
                              _check(plain, cp, "PTX_original_arithmetic_vs_register_BF16")]
                    checks.extend(_check(a, b, "PTX_vs_register_" + name) for name, a, b in zip(STAGES, traces, candidate_trace))
                    if (heads, dimension) != (64, 512):
                        original, op = torch.empty_like(q), torch.empty_like(q)
                        original_trace = [torch.empty_like(t) for t in traces]
                        self.originals[heads, dimension, True](q, kv, original, sink, idx, *original_trace)
                        self.originals[heads, dimension, False](q, kv, op, sink, idx)
                        checks += [_check(op, original, "unmodified_instrumentation_BF16"),
                                   _check(op, plain, "unmodified_vs_PTX_adapter_BF16")]
                        checks.extend(_check(a, b, "unmodified_vs_PTX_" + name) for name, a, b in zip(STAGES, original_trace, traces))
                    rows.append({"heads": heads, "dimension": dimension, "case": case_index, "batch": case[0], "sequence": case[1],
                                 "topk": case[2], "mode": case[3], "input_scale": case[4], "input_sha256": provenance,
                                 "unmodified_kernel_executed": (heads, dimension) != (64, 512), "checks": checks})
            passed = (len(rows) == 36 and all(c["status"] == "pass" for row in rows for c in row["checks"])
                      and all(c["status"] == "pass" for c in load_checks + store_checks)
                      and negative["status"] == "fail" and negative["mismatches"] > 0)
            self.qualified = passed
            return {"schema_version": 1, "probe": "official_attention_PTX_memory_adapter", "status": "pass" if passed else "fail",
                    "model": MODEL, "revision": REVISION, "reference": self.source, "device": self.device_info,
                    "driver_api_version": self.driver_version, "nvrtc_version": self.nvrtc_version, "tilelang_version": self.tilelang.__version__,
                    "torch_version": torch.__version__, "torch_git_version": torch.version.git_version, "torch_cuda_version": torch.version.cuda,
                    "numpy_version": np.__version__, "profiles": self.profiles, "loader": self.loader_meta,
                    "load_checks": load_checks, "store_checks": store_checks, "negative_control": negative,
                    "negative_control_detected": negative["mismatches"] > 0, "cases": rows,
                    "official_arithmetic_memory_adapter_gate": "pass" if passed else "fail",
                    "unmodified_full_kernel_gate": "blocked" if self.shared_limit < 141312 else "not_run", "qualified_real_model_inputs": 0,
                    "device_max_optin_shared_bytes": self.shared_limit,
                    "native_attention_qualification": "not_run", "full_model_quality": "not_run",
                    "memory": {"host_reservation_bytes": self.budget.peak, "reservation_is_RSS_limit": False,
                               "minimum_sampled_host_available_bytes": minimum, "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(self.ordinal),
                               "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(self.ordinal), "driver_compiler_memory_excluded": True},
                    "scope": "Pinned official PTX with integer-only Query memory virtualization, exact arithmetic/control retention and qualified bit transport; not untouched-kernel/full-model validation"}

    def __exit__(self, *_):
        self.qualified = False
        try:
            self.stack.close()
        finally:
            self.opened = False


def compare_attention_memory(directory: Path, ordinal=0):
    missing = missing_dependencies()
    if missing:
        return {"probe": "official_attention_PTX_memory_adapter", "status": "blocked", "missing_dependencies": missing}
    with AttentionMemoryReference(directory, ordinal) as reference:
        return reference.qualify()
