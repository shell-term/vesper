"""Shared native-only Attention checks, including failed-call recovery."""

import hashlib

import numpy as np

from .attention_inputs import CASES, PROFILES, attention_inputs
from .resources import physical_memory


def array_digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def check_arrays(expected, actual, name):
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        raise ValueError("Attention boundary shape or dtype differs")
    a = expected.view(np.uint32) if expected.dtype == np.float32 else expected
    b = actual.view(np.uint32) if actual.dtype == np.float32 else actual
    differences = int(np.count_nonzero(a != b))
    return {"check": name, "status": "fail" if differences else "pass", "elements": int(a.size),
            "mismatches": differences, "dtype": str(expected.dtype),
            "expected_sha256": array_digest(expected), "actual_sha256": array_digest(actual)}


def exercise_attention(runtime, checker):
    rows, minimum = [], physical_memory()["available"]
    for heads, dimension in PROFILES:
        for index, case in enumerate(CASES):
            minimum = min(minimum, physical_memory()["available"])
            inputs, provenance = attention_inputs(heads, dimension, case)
            plain = runtime(*inputs)
            traced = runtime(*inputs, diagnostics=True)
            checks = [check_arrays(traced["output_bf16"], plain["output_bf16"], "native_instrumentation_BF16")]
            checks.extend(checker(heads, dimension, index, inputs, provenance, traced))
            if runtime.device.allocations:
                raise RuntimeError("Attention call retained a device allocation")
            rows.append({"heads": heads, "dimension": dimension, "case": index,
                         "batch": case[0], "sequence": case[1], "topk": case[2], "mode": case[3],
                         "input_sha256": provenance, "checks": checks,
                         "status": "pass" if all(c["status"] == "pass" for c in checks) else "fail"})
    faults = exercise_failures(runtime, checker)
    return rows, faults, minimum


def exercise_failures(runtime, checker):
    """Fail before allocation and after three uploads; verify cleanup and retry."""
    inputs, provenance = attention_inputs(64, 512, CASES[3])
    faults = []
    for mode in ("host_pressure", "fourth_device_allocation"):
        original = runtime.budget.acquire if mode == "host_pressure" else runtime.device.allocate
        calls = 0
        def pressured(size, *args, **kwargs):
            nonlocal calls
            calls += 1
            if mode == "host_pressure" or calls == 4:
                raise MemoryError("Injected bounded Attention pressure")
            return original(size, *args, **kwargs)
        target, attribute = (runtime.budget, "acquire") if mode == "host_pressure" else (runtime.device, "allocate")
        setattr(target, attribute, pressured)
        caught = False
        try:
            runtime(*inputs, diagnostics=True)
        except MemoryError:
            caught = True
        finally:
            setattr(target, attribute, original)
        owned = sum(runtime.device.allocations.values())
        if not caught or owned:
            raise RuntimeError("Attention pressure failed to refuse or clean up")
        retried = runtime(*inputs, diagnostics=True)
        checks = checker(64, 512, 3, inputs, provenance, retried)
        faults.append({"condition": mode, "status": "pass" if all(c["status"] == "pass" for c in checks) else "fail",
                       "caught": "MemoryError", "allocation_attempts": calls,
                       "device_owned_bytes_after_failure": owned, "partial_result_returned": False,
                       "retry_checks": checks})
    return faults


def validate_goldens(report):
    """Validate coverage before an independent process accepts golden hashes."""
    from .attention import CANDIDATES
    from .attention_storage import STAGES
    if report.get("probe") != "native_attention_parity" or report.get("status") != "pass":
        raise ValueError("Expected a passing native Attention reference report")
    goldens = report.get("reference_goldens", [])
    keys = {(g.get("heads"), g.get("dimension"), g.get("case")) for g in goldens}
    expected_keys = {(h, d, i) for h, d in PROFILES for i in range(len(CASES))}
    if len(goldens) != len(expected_keys) or keys != expected_keys:
        raise ValueError("Attention golden coverage differs")
    for g in goldens:
        if ((g["heads"], g["dimension"], True) not in CANDIDATES
                or set(g.get("stage_sha256", {})) != {"output_bf16", *STAGES}
                or any(not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v)
                       for v in g["stage_sha256"].values())):
            raise ValueError("Attention golden boundaries differ")
    return {(g["heads"], g["dimension"], g["case"]): g for g in goldens}
