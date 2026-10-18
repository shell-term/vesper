"""Reproducible native KV transitions, Attention integration and recovery."""

import numpy as np

from .attention_validation import array_digest, check_arrays
from .formats import bf16_bits
from .resources import physical_memory

PREFILLS = (1, 2, 31, 32, 33, 127, 128, 129, 255, 256, 257)
ATTENTION_POSITIONS = {1, 2, 31, 32, 63, 64, 126, 127, 128, 129, 130, 254, 255, 256, 257, 258, 259}


def kv_inputs(batch, sequence, seed, mode="random"):
    rng = np.random.default_rng(20261018 + seed)
    values = bf16_bits(rng.standard_normal((batch, sequence, 512)).astype(np.float32))
    if mode == "mixed_scales":
        exponents = rng.integers(-16, 17, (batch, sequence, 16, 1))
        values = bf16_bits(np.ldexp(rng.standard_normal((batch, sequence, 16, 32)).astype(np.float32), exponents).reshape(values.shape))
    elif mode == "zeros":
        values.fill(0)
    elif mode == "signed_zeros":
        values[..., ::2] = 0
        values[..., 1::2] = 0x8000
    elif mode == "finite_bf16_encodings":
        bits = np.arange(65536, dtype=np.uint16)
        bits = bits[(bits & 0x7F80) != 0x7F80]
        values = np.resize(bits, batch * sequence * 512).reshape(batch, sequence, 512).copy()
    elif mode != "random":
        raise ValueError("Unknown KV input generator")
    return values


def query_inputs(batch, sequence, seed):
    rng = np.random.default_rng(20261018 + 1000000 + seed)
    q = bf16_bits(rng.standard_normal((batch, sequence, 64, 512)).astype(np.float32) * np.float32(.25))
    return q, rng.standard_normal(64).astype(np.float32)


def exercise_window(native, attention, checker, control):
    rows, faults, events = [], [], []
    minimum = physical_memory()["available"]
    def reset():
        native.reset()
        control("reset")
    def add(run, sequence, seed, mode="random", attend=False):
        nonlocal minimum
        minimum = min(minimum, physical_memory()["available"])
        values = kv_inputs(native.batch, sequence, seed, mode)
        meta = {"id": f"b{native.batch}/{run}/p{native.position}", "batch": native.batch,
                "position": native.position, "sequence": sequence, "seed": seed, "mode": mode,
                "input_bf16_sha256": array_digest(values), "attention_checked": attend}
        result = native.append(values)
        if attend:
            q, sink = query_inputs(native.batch, sequence, seed)
            meta["query_sha256"], meta["sink_sha256"] = array_digest(q), array_digest(sink)
            output = attention(q, result["window_bf16"], sink, result["indices_i32"], diagnostics=True)
            result.update(("attention_" + name, array) for name, array in output.items())
        checks = checker(meta, values, result)
        if sum(native.device.allocations.values()) != native.cache_bytes * 3:
            raise RuntimeError("KV transition retained temporary allocations")
        rows.append({**meta, "checks": checks, "status": "pass" if all(c["status"] == "pass" for c in checks) else "fail"})
        return rows[-1]
    for length in PREFILLS:
        reset()
        run = f"prefill-{length}"
        add(run, length, 1000 * length + native.batch, attend=length <= 3)
        for offset in range(4):
            add(run, 1, 200000 + length * 10 + native.batch * 10000 + offset, attend=True)
    for mode, length in (("mixed_scales", 33), ("zeros", 31), ("signed_zeros", 32), ("finite_bf16_encodings", 128)):
        reset()
        add("quant-" + mode, length, 600000 + length, mode)
    if native.batch == 1:
        reset()
        add("two-wraps", 1, 300000, attend=True)
        for position in range(1, 260):
            add("two-wraps", 1, 300000 + position, attend=position in ATTENTION_POSITIONS)
    reset()
    add("rollback-base", 127, 700000 + native.batch)
    add("rollback-base", 1, 700100 + native.batch, attend=True)
    before = native.inspect()
    handle = native.snapshot()
    control("snapshot")
    add("discarded-branch", 1, 700200 + native.batch, attend=True)
    add("discarded-branch", 1, 700300 + native.batch, attend=True)
    tentative = native.inspect()
    original_download = native.device.download
    def refuse_download(*args, **kwargs):
        raise OSError("Injected rollback publication read failure")
    native.device.download = refuse_download
    try:
        try:
            native.rollback(handle)
        except OSError:
            pass
        else:
            raise RuntimeError("Expected rollback publication refusal")
    finally:
        native.device.download = original_download
    unchanged = native.inspect()
    if unchanged["position"] != tentative["position"] or array_digest(unchanged["cache_bf16"]) != array_digest(tentative["cache_bf16"]):
        raise RuntimeError("Failed rollback changed active KV state")
    restored = native.rollback(handle)
    control("rollback")
    comparison = check_arrays(before["cache_bf16"], restored["cache_bf16"], "rollback_saved_cache")
    events.append({"condition": "rollback_after_publication_failure", "status": comparison["status"],
                   "failed_attempt_preserved_active_state": True, "restored_position": restored["position"], "check": comparison})
    if restored["position"] != before["position"]:
        raise RuntimeError("Rollback did not restore the exact position")
    for offset in range(2):
        add("replacement-branch", 1, 700400 + native.batch + offset * 100, attend=True)
    try:
        native.rollback(handle)
    except ValueError:
        events.append({"condition": "consumed_snapshot", "status": "pass"})
    else:
        raise RuntimeError("Consumed snapshot was accepted")
    handle = native.snapshot()
    native.discard(handle)
    control("discard")
    try:
        native.rollback(handle)
    except ValueError:
        events.append({"condition": "released_snapshot", "status": "pass"})
    else:
        raise RuntimeError("Released snapshot was accepted")
    handle = native.snapshot()
    reset()
    try:
        native.rollback(handle)
    except ValueError:
        events.append({"condition": "snapshot_invalidated_by_reset", "status": "pass"})
    else:
        raise RuntimeError("Reset did not invalidate a snapshot")
    add("failure-base", 2, 800000 + native.batch, attend=True)
    for index, mode in enumerate(("host_pressure", "third_device_allocation", "staged_download")):
        before = native.inspect()
        values = kv_inputs(native.batch, 1, 800100 + native.batch + index * 100)
        target, attribute = (native.budget, "acquire") if mode == "host_pressure" else (native.device, "allocate" if mode == "third_device_allocation" else "download")
        original = getattr(target, attribute)
        calls = 0
        def refuse(*args, **kwargs):
            nonlocal calls
            calls += 1
            if mode == "staged_download":
                raise OSError("Injected staged cache publication failure")
            if mode == "host_pressure" or calls == 3:
                raise MemoryError("Injected KV pressure")
            return original(*args, **kwargs)
        setattr(target, attribute, refuse)
        caught = None
        try:
            native.append(values)
        except (MemoryError, OSError) as exc:
            caught = type(exc).__name__
        finally:
            setattr(target, attribute, original)
        after = native.inspect()
        if (caught is None or before["position"] != after["position"]
                or array_digest(before["cache_bf16"]) != array_digest(after["cache_bf16"])
                or sum(native.device.allocations.values()) != native.cache_bytes * 3):
            raise RuntimeError("Failed KV append mutated active state or leaked buffers")
        retried = add("retry-" + mode, 1, 800100 + native.batch + index * 100, attend=True)
        faults.append({"condition": mode, "caught": caught, "failed_call_attempts": calls, "active_state_unchanged": True,
                       "active_cache_sha256": array_digest(after["cache_bf16"]), "position_after_failure": after["position"],
                       "partial_result_returned": False, "retry_id": retried["id"], "status": retried["status"]})
    return rows, faults, events, minimum


def validate_goldens(report):
    from .attention_memory import validate_qualification
    from .attention_storage import STAGES
    from .checkpoint import MODEL, REVISION, REFERENCE_HASHES
    from .kv import KV_CUBIN_SHA256
    if (report.get("probe") != "native_window_kv_parity" or report.get("status") != "pass"
            or report.get("model") != MODEL or report.get("revision") != REVISION
            or report.get("native_profile", {}).get("cubin_sha256") != KV_CUBIN_SHA256
            or any(report.get("reference", {}).get("files", {}).get(name) != REFERENCE_HASHES[name]
                   for name in ("model.py", "kernel.py"))):
        raise ValueError("Expected a passing pinned original/native window KV report")
    validate_qualification(report.get("attention_reference_qualification"))
    rows = report.get("reference_goldens", [])
    counts = {b: sum(g.get("meta", {}).get("batch") == b for g in rows) for b in (1, 2)}
    if len(rows) != 398 or counts != {1: 329, 2: 69} or len({g["meta"]["id"] for g in rows}) != len(rows):
        raise ValueError("Window KV golden coverage differs")
    base = {"rounded_bf16", "codes_fp8", "scales_e8m0", "cache_bf16", "window_bf16", "indices_i32"}
    for g in rows:
        expected = base | ({"attention_" + n for n in ("output_bf16", *STAGES)} if g["meta"].get("attention_checked") else set())
        hashes = g.get("stage_sha256", {})
        if (set(hashes) != expected or any(not isinstance(h, str) or len(h) != 64
                                         or any(c not in "0123456789abcdef" for c in h) for h in hashes.values())):
            raise ValueError("Window/Attention golden boundaries differ")
    goldens = {g["meta"]["id"]: g for g in rows}
    cases = report.get("cases", [])
    if len(cases) != len(rows) or {r.get("id") for r in cases} != set(goldens):
        raise ValueError("Window KV reference comparisons are incomplete")
    for row in cases:
        expected = goldens[row["id"]]["stage_sha256"]
        checks = row.get("checks", [])
        if (row.get("status") != "pass" or len(checks) != len(expected) or {c.get("check") for c in checks} != set(expected)
                or any(c.get("status") != "pass" or c.get("mismatches") != 0
                       or c.get("expected_sha256") != expected[c["check"]] or c.get("actual_sha256") != expected[c["check"]]
                       for c in checks)):
            raise ValueError("Window KV reference boundary failed or is missing")
    controls = report.get("negative_controls", [])
    if (len(controls) != 3 or {c.get("check") for c in controls} != {"omitted_rounding", "one_scale_per_512_instead_of_per_32", "omitted_RoPE_tail_rounding"}
            or any(c.get("status") != "fail" or not isinstance(c.get("mismatches"), int) or c["mismatches"] <= 0 for c in controls)):
        raise ValueError("Window KV negative controls are missing or undetected")
    failures = report.get("faults", [])
    if (len(failures) != 6 or {(r.get("batch"), r.get("condition")) for r in failures}
            != {(b, mode) for b in (1, 2) for mode in ("host_pressure", "third_device_allocation", "staged_download")}
            or any(r.get("status") != "pass" or r.get("active_state_unchanged") is not True
                   or r.get("partial_result_returned") is not False for r in failures)):
        raise ValueError("Window KV append failure/recovery evidence is incomplete")
    return goldens
