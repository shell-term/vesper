"""Bounded intermediate comparisons of an explicit Query-storage candidate.

A runnable candidate is not a passing original full-profile reference gate.
No automatic substitution into prefix capture or the native runtime occurs.
"""

from __future__ import annotations

import ctypes as C
import hashlib
import importlib.util
from pathlib import Path

import numpy as np

from .attention_storage import STAGES, load_variant, trace_shapes
from .checkpoint import MODEL, REVISION, fetch_reference
from .conformance import compare
from .cuda import Driver, _bind
from .formats import bf16_bits
from .prefix_reference import required_shared_bytes
from .resources import GiB, MemoryBudget, MiB, physical_memory


def _host_source(compiled):
    source = compiled.adapter.host_func
    if not isinstance(source, str):
        source = Path(compiled.adapter.pymodule.__file__).read_text(encoding="utf-8")
    return source


def _numpy(tensor):
    import torch
    return tensor.view(torch.uint16).cpu().numpy().copy() if tensor.dtype == torch.bfloat16 else tensor.cpu().numpy().copy()


def _check(expected, actual, label):
    a, b = _numpy(expected), _numpy(actual)
    # Include signed zeros, infinities and NaN payload bits in these diagnostics.
    row = compare(a.view(np.uint32) if a.dtype == np.float32 else a,
                  b.view(np.uint32) if b.dtype == np.float32 else b, label)
    row.update(expected_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
               actual_sha256=hashlib.sha256(b.tobytes()).hexdigest(), dtype=str(a.dtype))
    return row


def _inputs(torch, heads, dimension, case, ordinal):
    from .attention_inputs import attention_inputs
    (q, kv, sink, idx), provenance = attention_inputs(heads, dimension, case)
    device = f"cuda:{ordinal}"
    tensors = (torch.from_numpy(q).view(torch.bfloat16).to(device), torch.from_numpy(kv).view(torch.bfloat16).to(device),
               torch.from_numpy(sink).to(device), torch.from_numpy(idx).to(device))
    return tensors, provenance


def compare_attention_storage(directory: Path, ordinal=0):
    missing = [name for name in ("torch", "tilelang") if importlib.util.find_spec(name) is None]
    if missing:
        return {"probe": "attention_query_storage", "status": "blocked", "missing_dependencies": missing}
    import torch
    import tilelang
    budget = MemoryBudget(cap=32 * MiB)
    budget.acquire(32 * MiB)
    try:
        return _compare(torch, tilelang, Path(directory), ordinal, budget)
    finally:
        budget.release(32 * MiB)


def _compare(torch, tilelang, directory, ordinal, budget):
    source = fetch_reference(directory)
    with (directory / "kernel.py").open("rb") as stream:
        raw = stream.read(MiB)
    variants = {}
    transforms = []
    for storage in ("shared", "register-query"):
        for diagnostic in (False, True):
            factory, transform = load_variant(raw, directory / "attention-variants", storage, diagnostic)
            variants[storage, diagnostic] = factory
            transforms.append(transform)
    driver = Driver()
    query = _bind(driver.lib, "cuDeviceGetAttribute", [C.POINTER(C.c_int), C.c_int, C.c_int])
    limit = C.c_int()
    driver.check(query(C.byref(limit), 97, ordinal))
    device = next(d for d in driver.inventory()["devices"] if d["ordinal"] == ordinal)
    profiles, kernels = [], {}
    with torch.cuda.device(ordinal), torch.inference_mode():
        if torch.cuda.mem_get_info(ordinal)[0] < 2 * GiB:
            raise MemoryError("Insufficient VRAM reserve for the attention probe")
        torch.cuda.reset_peak_memory_stats(ordinal)
        initial_allocated = torch.cuda.memory_allocated(ordinal)
        for heads, dimension in ((64, 256), (16, 512), (64, 512), (16, 256)):
            for (storage, diagnostic), factory in variants.items():
                # The extra small profile is only used for a diagnostic control.
                if (heads, dimension) == (16, 256) and (storage, diagnostic) != ("shared", True):
                    continue
                compiled = factory(heads, dimension, dimension**-0.5)
                required = required_shared_bytes(_host_source(compiled))
                profiles.append({"heads": heads, "dimension": dimension, "storage": storage, "diagnostics": diagnostic,
                                 "dynamic_shared_bytes": required, "supported": required <= limit.value,
                                 "generated_cuda_sha256": hashlib.sha256(compiled.get_kernel_source().encode()).hexdigest()})
                kernels[storage, diagnostic, heads, dimension] = (compiled, required)

        def execute(storage, diagnostic, inputs, heads, dimension):
            compiled, required = kernels[storage, diagnostic, heads, dimension]
            if required > limit.value:
                raise RuntimeError("Refusing to launch attention above the device shared-memory limit")
            q, kv, sink, idx = inputs
            output = torch.empty_like(q)
            traces = [torch.empty(shape, dtype=torch.float32, device=q.device)
                      for shape in trace_shapes(q.shape[0], q.shape[1], heads, dimension, idx.shape[-1])] if diagnostic else []
            compiled(q, kv, output, sink, idx, *traces)
            return output, traces

        cases = ((1, 1, 1, "normal", .125), (1, 3, 32, "normal", 1.), (1, 2, 64, "masked", 8.),
                 (2, 2, 65, "masked", .5), (1, 2, 128, "empty", 1.), (1, 2, 193, "duplicates", 1.),
                 (1, 1, 128, "zeros", 1.), (1, 2, 65, "signed_zeros", 1.), (1, 1, 127, "normal", 1e-6),
                 (1, 2, 128, "ties", 1.), (1, 1, 129, "sink_positive", .125), (1, 1, 129, "sink_negative", .125))
        results, controls = [], []
        minimum = physical_memory()["available"]
        for heads, dimension in ((64, 256), (16, 512), (64, 512)):
            original_profile_supported = kernels["shared", True, heads, dimension][1] <= limit.value
            for case_index, case in enumerate(cases):
                minimum = min(minimum, physical_memory()["available"])
                budget.acquire(0, minimum)
                inputs, provenance = _inputs(torch, heads, dimension, case, ordinal)
                candidate, candidate_trace = execute("register-query", True, inputs, heads, dimension)
                plain_candidate, _ = execute("register-query", False, inputs, heads, dimension)
                checks = [_check(plain_candidate, candidate, "candidate_instrumentation_preserves_BF16_output")]
                if original_profile_supported:
                    original, original_trace = execute("shared", True, inputs, heads, dimension)
                    plain_original, _ = execute("shared", False, inputs, heads, dimension)
                    checks.extend((_check(plain_original, original, "original_instrumentation_preserves_BF16_output"),
                                   _check(plain_original, plain_candidate, "original_vs_register_query_BF16_output")))
                    checks.extend(_check(a, b, name) for name, a, b in zip(STAGES, original_trace, candidate_trace))
                else:
                    repeated, repeated_trace = execute("register-query", True, inputs, heads, dimension)
                    checks.append(_check(candidate, repeated, "candidate_repeat_BF16_output"))
                    checks.extend(_check(a, b, "candidate_repeat_" + name) for name, a, b in zip(STAGES, candidate_trace, repeated_trace))
                results.append({"heads": heads, "dimension": dimension, "case": case_index, "batch": case[0],
                                "sequence": case[1], "topk": case[2], "mode": case[3], "input_scale": case[4],
                                "input_sha256": provenance, "original_profile_executed": original_profile_supported,
                                "finite_BF16_output": bool(torch.isfinite(candidate).all()), "checks": checks})

                if heads == 64:
                    # A negative/control experiment, never an accepted fallback:
                    # compare four independent 16-head launches before BF16 rounding.
                    pieces, final_pieces, qk_pieces = [], [], []
                    for start in range(0, heads, 16):
                        q, kv, sink, idx = inputs
                        piece_inputs = (q[:, :, start:start + 16].contiguous(), kv, sink[start:start + 16].contiguous(), idx)
                        piece, trace = execute("shared", True, piece_inputs, 16, dimension)
                        pieces.append(piece)
                        final_pieces.append(trace[-1])
                        qk_pieces.append(trace[0])
                    joined = torch.cat(pieces, dim=2)
                    fp32_joined = torch.cat(final_pieces, dim=2)
                    qk_joined = torch.cat(qk_pieces, dim=3)
                    baseline, baseline_trace = (original, original_trace) if original_profile_supported else (candidate, candidate_trace)
                    controls.append({"heads": heads, "dimension": dimension, "case": case_index,
                                     "baseline": "unchanged_original" if original_profile_supported else "register_query_candidate",
                                     "checks": [_check(baseline, joined, "head_partition_BF16_output"),
                                                _check(baseline_trace[-1], fp32_joined, "head_partition_FP32_output"),
                                                _check(baseline_trace[0], qk_joined, "head_partition_QK_FP32")]})
                    if not original_profile_supported:
                        # QK has no cross-head reduction. Keep this component
                        # comparison separate from the unavailable full oracle.
                        checks.append(_check(qk_joined, candidate_trace[0], "QK_against_unchanged_16_head_original"))
        torch.cuda.synchronize(ordinal)
        peak = torch.cuda.max_memory_allocated(ordinal)
        reserved = torch.cuda.max_memory_reserved(ordinal)
    required_full = kernels["shared", True, 64, 512][1]
    passed = all(row["finite_BF16_output"] and all(c["status"] == "pass" for c in row["checks"]) for row in results)
    full_rows = [row for row in results if (row["heads"], row["dimension"]) == (64, 512)]
    full_passed = bool(full_rows) and all(row["original_profile_executed"] and row["finite_BF16_output"]
                                        and all(c["status"] == "pass" for c in row["checks"]) for row in full_rows)
    original_gate = "blocked" if required_full > limit.value else "pass" if full_passed else "fail"
    return {"schema_version": 1, "probe": "attention_query_storage", "status": "fail" if not passed else "pass" if original_gate == "pass" else "partial",
            "model": MODEL, "revision": REVISION, "reference": source, "device": device,
            "torch_version": torch.__version__, "torch_cuda_version": torch.version.cuda,
            "torch_git_version": torch.version.git_version, "tilelang_version": tilelang.__version__, "numpy_version": np.__version__,
            "execution_backend": "TileLang NVRTC", "target_compute_capability": device["compute_capability"],
            "transforms": transforms, "device_max_optin_shared_bytes": limit.value, "profiles": profiles,
            "cases": results, "head_partition_controls": controls,
            "full_profile_original_reference_gate": original_gate, "candidate_scoped_checks": "pass" if passed else "fail",
            "actual_model_activation_gate": "not_run", "native_attention_qualification": "not_run", "full_model_quality": "not_run",
            "memory": {"host_reservation_bytes": budget.peak, "host_reservation_is_RSS_limit": False,
                       "minimum_observed_host_available_bytes": minimum, "initial_torch_allocated_bytes": initial_allocated,
                       "peak_torch_allocated_bytes": peak, "peak_torch_reserved_bytes": reserved,
                       "torch_peak_excludes_driver_and_compiler": True},
            "scope": "Explicit Query-storage candidate and diagnostic original kernels; no prefix substitution, weight download, native Attention or whole-model qualification"}
