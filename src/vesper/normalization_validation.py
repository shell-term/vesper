"""Framework-free normalization/rotary corpus and a projected-input KV chain."""

import numpy as np

from .attention_validation import array_digest
from .formats import bf16_bits
from .kv_validation import kv_inputs, query_inputs
from .normalization import MAX_POSITION, NORM_STAGES
from .normalization_fixture import NORM_WEIGHTS
from .resources import physical_memory

ROWS = (1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 64, 127, 128, 129, 257, 514)
MODES = ("zeros", "signed_zeros", "mixed_scales", "near_eps", "subnormal_squares", "bf16_subnormal", "large", "finite_encodings")
ROPE_SHAPES = ((1, 1, 512), (2, 3, 512), (1, 3, 64, 512), (2, 2, 16, 512), (1, 1, 32, 64), (1, 257, 512))
STARTS = (0, 1, 31, 127, 128, 129, 4095, 65535, 65536, 131069)
CHAIN_PREFILLS = (1, 2, 127, 128, 129, 255, 257)


def inputs(shape, seed, mode="normal"):
    rng = np.random.default_rng(20261018 + seed)
    values = rng.standard_normal(shape).astype(np.float32)
    if mode == "mixed_scales":
        values = np.ldexp(values, rng.integers(-100, 101, (*shape[:-1], 1)))
    elif mode == "near_eps":
        values *= np.float32(1e-10)
    elif mode == "subnormal_squares":
        values *= np.float32(1e-20)
    elif mode == "large":
        values *= np.float32(1e20)
    result = bf16_bits(values)
    if mode == "zeros":
        result.fill(0)
    elif mode == "signed_zeros":
        result[..., ::2] = 0
        result[..., 1::2] = 0x8000
    elif mode == "bf16_subnormal":
        result = rng.integers(1, 128, shape, dtype=np.uint16)
        result[..., 1::2] |= np.uint16(0x8000)
    elif mode == "finite_encodings":
        bits = np.arange(65536, dtype=np.uint16)
        result = np.resize(bits[(bits & 0x7F80) != 0x7F80], int(np.prod(shape))).reshape(shape).copy()
    return result


def norm_cases():
    for ordinal, (name, width) in enumerate(NORM_WEIGHTS.items()):
        for rows in ROWS:
            shape = (2, 257, width) if rows == 514 else ((1, 2, 2, width) if rows == 4 else (rows, width))
            yield name, shape, ordinal * 10000 + rows, "normal"
        for ordinal_mode, mode in enumerate(MODES):
            for rows in (1, 17):
                yield name, (rows, width), ordinal * 10000 + ordinal_mode * 100 + rows, mode


def rope_cases():
    for profile in ("window", "compressed-yarn"):
        for inverse in (False, True):
            for start in STARTS:
                for shape in ROPE_SHAPES:
                    if start + shape[1] <= MAX_POSITION:
                        yield profile, inverse, start, shape, start + int(np.prod(shape[:-1])), "normal"
            for start in (0, 65535):
                for ordinal, mode in enumerate(("zeros", "signed_zeros", "mixed_scales", "bf16_subnormal", "large")):
                    yield profile, inverse, start, (1, 2, 64, 512), 400000 + ordinal + start, mode


def exercise_components(native, weights, checker):
    records, minimum = [], physical_memory()["available"]
    def record(meta, value, result):
        nonlocal minimum
        minimum = min(minimum, physical_memory()["available"])
        checks = checker(meta, value, result)
        records.append({**meta, "checks": checks, "status": "pass" if all(c["status"] == "pass" for c in checks) else "fail"})
        if native.device.allocations:
            raise RuntimeError("Normalization/rotary call retained GPU buffers")
    for name, shape, seed, mode in norm_cases():
        values = inputs(shape, seed, mode)
        meta = {"id": f"norm/{name}/{mode}/{shape[0] if len(shape)==2 else 'x'.join(map(str,shape[:-1]))}",
                "operation": "norm", "weight": name, "shape": list(shape), "seed": seed, "mode": mode,
                "input_sha256": array_digest(values), "weight_sha256": array_digest(weights[name])}
        record(meta, values, native.norm(values, weights[name]))
    for profile in ("window", "compressed-yarn"):
        for start in range(0, MAX_POSITION, 4096):
            positions = np.arange(start, start + 4096, dtype=np.int32)
            meta = {"id": f"frequencies/{profile}/{start}", "operation": "frequencies", "profile": profile,
                    "start": start, "count": len(positions), "input_sha256": array_digest(positions)}
            record(meta, positions, native.frequencies(positions, profile))
    for profile, inverse, start, shape, seed, mode in rope_cases():
        values = inputs(shape, seed, mode)
        meta = {"id": f"rope/{profile}/{int(inverse)}/{start}/{'x'.join(map(str,shape))}/{mode}",
                "operation": "rope", "profile": profile, "inverse": inverse, "start": start,
                "shape": list(shape), "seed": seed, "mode": mode, "input_sha256": array_digest(values)}
        record(meta, values, native.rope(values, start, profile, inverse))
    return records, minimum


def exercise_chain(native, kv, attention, weights, checker, reset_reference):
    records, minimum = [], physical_memory()["available"]
    for length in CHAIN_PREFILLS:
        kv.reset()
        reset_reference()
        for ordinal in range(4):
            sequence = length if ordinal == 0 else 1
            position = kv.position
            seed = 900000 + kv.batch * 10000 + length * 10 + ordinal
            projected = kv_inputs(kv.batch, sequence, seed)
            meta = {"id": f"chain/b{kv.batch}/prefill-{length}/p{position}", "operation": "chain",
                    "batch": kv.batch, "sequence": sequence, "position": position, "seed": seed,
                    "input_sha256": array_digest(projected), "attention_checked": sequence <= 3}
            normalized = native.norm(projected, weights["layers.0.attn.kv_norm.weight"])
            rotated = native.rope(normalized["output_bf16"], position)
            cached = kv.append(rotated["output_bf16"])
            actual = {"norm_" + name: a for name, a in normalized.items()}
            actual.update(("rope_" + name, a) for name, a in rotated.items())
            actual.update(("kv_" + name, a) for name, a in cached.items())
            if meta["attention_checked"]:
                q, sink = query_inputs(kv.batch, sequence, seed)
                meta["query_sha256"], meta["sink_sha256"] = array_digest(q), array_digest(sink)
                query = native.rope(q, position)
                attn = attention(query["output_bf16"], cached["window_bf16"], sink, cached["indices_i32"], diagnostics=True)
                inverse = native.rope(attn["output_bf16"], position, inverse=True)
                for prefix, result in (("query_rope_", query), ("attention_", attn), ("inverse_rope_", inverse)):
                    actual.update((prefix + name, a) for name, a in result.items())
            checks = checker(meta, projected, actual)
            minimum = min(minimum, physical_memory()["available"])
            records.append({**meta, "checks": checks, "status": "pass" if all(c["status"] == "pass" for c in checks) else "fail"})
    return records, minimum


def exercise_failures(native, weights, checker):
    failures = []
    name = "layers.0.attn.q_norm.weight"
    value = inputs((2, 1280), 1234)
    for mode in ("host_pressure", "third_allocation", "download"):
        target, attribute = (native.budget, "acquire") if mode == "host_pressure" else (native.device, "allocate" if mode == "third_allocation" else "download")
        original = getattr(target, attribute)
        calls = 0
        def refuse(*args, **kwargs):
            nonlocal calls
            calls += 1
            if mode == "download":
                raise OSError("Injected normalization download failure")
            if mode == "host_pressure" or calls == 3:
                raise MemoryError("Injected normalization allocation pressure")
            return original(*args, **kwargs)
        setattr(target, attribute, refuse)
        caught = None
        try:
            native.norm(value, weights[name])
        except (MemoryError, OSError) as exc:
            caught = type(exc).__name__
        finally:
            setattr(target, attribute, original)
        if caught is None or native.device.allocations:
            raise RuntimeError("Normalization failed call returned or leaked GPU buffers")
        meta = {"id": "retry/" + mode, "operation": "norm", "weight": name, "shape": [2, 1280],
                "seed": 1234, "mode": "normal", "input_sha256": array_digest(value), "weight_sha256": array_digest(weights[name])}
        checks = checker(meta, value, native.norm(value, weights[name]))
        failures.append({"condition": mode, "caught": caught, "attempts": calls, "device_owned_after_failure": 0,
                         "partial_result_returned": False, "retry_meta": meta, "retry_checks": checks,
                         "status": "pass" if all(c["status"] == "pass" for c in checks) else "fail"})
    return failures


def validate_goldens(report):
    from .attention_memory import validate_qualification
    from .checkpoint import MODEL, REVISION, REFERENCE_HASHES
    from .normalization import NORM_CUBIN_SHA256
    from .normalization_reference import PYTORCH_SOURCES
    if (report.get("probe") != "native_normalization_rotary_parity" or report.get("status") != "pass"
            or report.get("model") != MODEL or report.get("revision") != REVISION
            or report.get("native_profile", {}).get("cubin_sha256") != NORM_CUBIN_SHA256
            or any(report.get("reference", {}).get("files", {}).get(n) != REFERENCE_HASHES[n] for n in ("model.py", "kernel.py"))
            or any(report.get("pytorch_source", {}).get("files", {}).get(n, {}).get("sha256") != spec[1] for n, spec in PYTORCH_SOURCES.items())):
        raise ValueError("Expected passing pinned native normalization/rotary evidence")
    validate_qualification(report.get("attention_reference_qualification"))
    rows = report.get("component_cases", []) + report.get("chain_cases", [])
    faults = report.get("faults", [])
    records = report.get("reference_goldens", [])
    count = {op: sum(r.get("operation") == op for r in rows) for op in ("norm", "frequencies", "rope", "chain")}
    if count != {"norm": 136, "frequencies": 64, "rope": 276, "chain": 56} or len(records) != 535:
        raise ValueError("Normalization/rotary corpus coverage differs")
    if len(faults) != 3 or {f.get("condition") for f in faults} != {"host_pressure", "third_allocation", "download"}:
        raise ValueError("Normalization failure/retry coverage differs")
    frequencies = {(r["profile"], r["start"]) for r in rows if r["operation"] == "frequencies"}
    if frequencies != {(p, s) for p in ("window", "compressed-yarn") for s in range(0, MAX_POSITION, 4096)}:
        raise ValueError("Rotary all-position coverage is incomplete")
    goldens = {g["meta"]["id"]: g for g in records}
    if len(goldens) != len(records) or {r["id"] for r in rows} | {f["retry_meta"]["id"] for f in faults} != set(goldens):
        raise ValueError("Normalization/rotary golden identities differ")
    for row in rows + [{**f["retry_meta"], "status": f["status"], "checks": f["retry_checks"]} for f in faults]:
        hashes = goldens[row["id"]]["stage_sha256"]
        checks = row.get("checks", [])
        if (row.get("status") != "pass" or len(checks) != len(hashes) or {c.get("check") for c in checks} != set(hashes)
                or any(c.get("status") != "pass" or c.get("mismatches") != 0
                       or c.get("expected_sha256") != hashes[c["check"]] or c.get("actual_sha256") != hashes[c["check"]]
                       for c in checks)):
            raise ValueError("Normalization/rotary checked boundary failed or is missing")
    negative = report.get("negative_controls", [])
    expected_negative = {"higher_precision_mean_replacement", "default_epsilon_replacement", "unfused_rotary_replacement", "direct_YaRN_division_replacement"}
    if (len(negative) != 4 or {n.get("check") for n in negative} != expected_negative
            or any(n.get("status") != "fail" or not isinstance(n.get("mismatches"), int) or n["mismatches"] <= 0 for n in negative)):
        raise ValueError("Normalization/rotary negative controls were not detected")
    if any(f.get("partial_result_returned") is not False or f.get("device_owned_after_failure") != 0 for f in faults):
        raise ValueError("Normalization failure did not refuse/clean up")
    return goldens
