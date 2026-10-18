"""Framework-free projection/layer corpus and publication-failure checks."""

import base64
import hashlib

import numpy as np

from .attention_validation import array_digest
from .normalization_validation import inputs
from .projections import FP8_LAYOUT
from .resources import physical_memory

ROWS = (1, 2, 3, 4, 6, 8)
MODES = ("zeros", "signed_zeros", "mixed_scales", "near_eps", "subnormal_squares", "large")
FAULTS = ("host_pressure", "third_allocation", "post_KV_download", "grouped_compute", "rollback_download")


def component_plan():
    for name, width in [(n, shape[1]) for n, shape in FP8_LAYOUT.items()] + [("wo_a", 4096)]:
        for mode in ("normal", *MODES):
            for rows in ROWS if mode == "normal" else (1, 3):
                shape = (1, rows, 8, width) if name == "wo_a" else (rows, width)
                seed = 700000 + width + rows + len(mode) * 31
                value = inputs(shape, seed, mode)
                yield {"id": f"projection/{name}/{mode}/{rows}", "kind": "projection", "projection": name,
                       "shape": list(shape), "seed": seed, "mode": mode, "input_sha256": array_digest(value)}, value


def exercise_components(native, checker):
    records, minimum = [], physical_memory()["available"]
    for meta, values in component_plan():
        actual = native.grouped(values) if meta["projection"] == "wo_a" else native.fp8(meta["projection"], values)
        checks = checker(meta, values, actual)
        if sum(native.device.allocations.values()) != native.resident_bytes:
            raise RuntimeError("Projection retained temporary GPU buffers")
        records.append({**meta, "checks": checks, "status": "pass" if all(c["status"] == "pass" for c in checks) else "fail"})
        minimum = min(minimum, physical_memory()["available"])
    return records, minimum


def decode_real_inputs(records):
    if not isinstance(records, list) or len(records) != 4:
        raise ValueError("Expected four captured original-prefix Attention inputs")
    values = []
    for ordinal, row in enumerate(records):
        encoded = row.get("input_bf16_base64")
        if (not isinstance(encoded, str) or len(encoded) != 13656 or row.get("shape") != [1, 1, 5120]
                or row.get("sequence") != ordinal // 2 or row.get("position") != ordinal % 2
                or row.get("token_id") != (1, 42, 1000, 4096)[ordinal]):
            raise ValueError("Captured Attention input identity/layout differs")
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) != 10240 or hashlib.sha256(raw).hexdigest() != row.get("input_bf16_sha256"):
            raise ValueError("Captured Attention input bits differ")
        bits = np.frombuffer(raw, dtype="<u2").reshape(1, 1, 5120).copy()
        if np.any((bits & 0x7F80) == 0x7F80):
            raise ValueError("Captured Attention inputs must be finite BF16")
        values.append(bits)
    return values


def layer_plan(batch, real):
    def generated(group, mode, prefill, count):
        for step in range(count):
            position, length = (0, prefill) if step == 0 else (prefill + step - 1, 1)
            seed = 1000000 + batch * 100000 + prefill * 1000 + len(group) * 100 + step
            value = inputs((batch, length, 5120), seed, mode)
            meta = {"id": f"layer/b{batch}/{group}/p{position}", "kind": "layer", "batch": batch,
                    "position": position, "sequence": length, "shape": list(value.shape), "seed": seed,
                    "mode": mode, "input_sha256": array_digest(value), "reset": step == 0}
            yield meta, value
    for prefill in (1, 2, 3):
        yield from generated(f"prefill-{prefill}", "normal", prefill, 4)
    for mode in ("zeros", "signed_zeros", "mixed_scales", "near_eps"):
        yield from generated(mode, mode, 3, 2)
    if batch == 1:
        yield from generated("two-window-wraps", "normal", 1, 260)
        for ordinal, (row, value) in enumerate(zip(real, decode_real_inputs(real))):
            yield {"id": f"layer/original-prefix/sequence{row['sequence']}/p{row['position']}", "kind": "layer",
                "batch": 1, "position": row["position"], "sequence": 1, "shape": list(value.shape),
                "source": "unchanged_original_Block_prefix_Attention_input", "captured_input": ordinal,
                "input_sha256": array_digest(value), "reset": row["position"] == 0}, value


def exercise_layer(layer, real_inputs, checker, reset_reference):
    records, minimum = [], physical_memory()["available"]
    for meta, values in layer_plan(layer.batch, real_inputs):
        if meta["reset"]:
            layer.reset()
            reset_reference()
        actual = layer.step(values, start_pos=meta["position"])
        checks = checker(meta, values, actual)
        records.append({**meta, "checks": checks, "status": "pass" if all(c["status"] == "pass" for c in checks) else "fail"})
        minimum = min(minimum, physical_memory()["available"])
    return records, minimum


def exercise_failures(layer, checker):
    rows, faults = [], []
    projection = layer.projections
    for ordinal, mode in enumerate(FAULTS):
        before = layer.kv.inspect()
        position = layer.position
        value = inputs((layer.batch, 1, 5120), 1800000 + ordinal)
        original_fp8, original_grouped = projection.fp8, projection.blas.grouped
        original_acquire, original_allocate = projection.budget.acquire, projection.device.allocate
        original_download, original_kv_download = projection.device.download, layer.kv.device.download
        calls = 0
        def refuse(*args, **kwargs):
            nonlocal calls
            calls += 1
            if mode == "host_pressure" or mode == "third_allocation" and calls == 3:
                raise MemoryError("Injected projection memory pressure")
            return original_acquire(*args, **kwargs) if mode == "host_pressure" else original_allocate(*args, **kwargs)
        def download_failure(*args, **kwargs):
            raise OSError("Injected post-KV publication download failure")
        def fp8(name, values):
            if name == "wo_b":
                if mode == "rollback_download":
                    layer.kv.device.download = download_failure
                    raise OSError("Injected output projection failure before rollback failure")
                projection.device.download = download_failure
            return original_fp8(name, values)
        def compute_failure(*args, **kwargs):
            raise OSError("Injected grouped projection compute failure")
        if mode == "host_pressure":
            projection.budget.acquire = refuse
        elif mode == "third_allocation":
            projection.device.allocate = refuse
        elif mode == "grouped_compute":
            projection.blas.grouped = compute_failure
        else:
            projection.fp8 = fp8
        caught = None
        try:
            layer.step(value)
        except (MemoryError, OSError, RuntimeError) as exc:
            caught = type(exc).__name__
        finally:
            projection.fp8, projection.blas.grouped = original_fp8, original_grouped
            projection.budget.acquire, projection.device.allocate = original_acquire, original_allocate
            projection.device.download, layer.kv.device.download = original_download, original_kv_download
        blocked = False
        if mode == "rollback_download":
            if layer.pending is None:
                raise RuntimeError("Failed rollback lost its recovery handle")
            try:
                layer.step(value)
            except RuntimeError:
                blocked = True
            if not blocked:
                raise RuntimeError("Layer published after a failed rollback")
            layer.recover()
        after = layer.kv.inspect()
        if (caught is None or layer.position != position or before["position"] != after["position"]
                or array_digest(before["cache_bf16"]) != array_digest(after["cache_bf16"])
                or layer.pending is not None or sum(projection.device.allocations.values()) != projection.resident_bytes
                or layer.norm.device.allocations or layer.attention.device.allocations
                or sum(layer.kv.device.allocations.values()) != 3 * layer.kv.cache_bytes):
            raise RuntimeError("Failed layer call returned partial output, changed committed KV or leaked buffers")
        meta = {"id": "layer/retry/" + mode, "kind": "layer", "batch": layer.batch, "position": position,
                "sequence": 1, "shape": list(value.shape), "seed": 1800000 + ordinal, "mode": "normal",
                "input_sha256": array_digest(value), "reset": False}
        checks = checker(meta, value, layer.step(value))
        status = "pass" if all(c["status"] == "pass" for c in checks) else "fail"
        rows.append({**meta, "checks": checks, "status": status})
        faults.append({"condition": mode, "caught": caught, "committed_position_preserved": position,
            "committed_cache_preserved": True, "partial_result_returned": False, "temporary_buffers_released": True,
            "calls_blocked_until_recovery": blocked, "retry_id": meta["id"], "status": status})
    return rows, faults


def validate_goldens(report):
    from .attention_memory import validate_qualification
    from .checkpoint import MODEL, REVISION, REFERENCE_HASHES
    from .gate_reference import GATE_SOURCE_HASHES
    from .prefix_reference import PREFIX_SOURCE_HASHES
    from .projections import PROJECTION_CUBIN_SHA256
    from .router import ROUTER_TORCH_REVISION
    if (report.get("status") != "pass" or report.get("probe") != "native_attention_projection_parity"
            or report.get("model") != MODEL or report.get("revision") != REVISION
            or report.get("native_profile", {}).get("cubin_sha256") != PROJECTION_CUBIN_SHA256
            or report.get("native_profile", {}).get("cublas_version") != 120402
            or report.get("native_profile", {}).get("cublas_math_mode") != 0
            or report.get("native_profile", {}).get("compiler", {}).get("concatenated_source_sha256") != "56ca03fb20f1208b226ddf810f2d508f4b35037e4ecb45b4e91a9c7384397262"
            or report.get("native_profile", {}).get("converted_wo_a_bf16_sha256") != "6969c8d5adbf4ed357443f417f6adfb75ee0ebe559d3776bd7d44762b452c8f3"
            or report.get("torch_git_version") != ROUTER_TORCH_REVISION
            or any(report.get("blas_reference", {}).get("files", {}).get(n, {}).get("sha256") != h for n, h in GATE_SOURCE_HASHES.items())
            or any(report.get("reference", {}).get("files", {}).get(n) != h for n, h in PREFIX_SOURCE_HASHES.items())
            or any(report.get("reference", {}).get("files", {}).get(n) != REFERENCE_HASHES[n] for n in ("model.py", "kernel.py"))):
        raise ValueError("Expected passing pinned original/native projection evidence")
    validate_qualification(report.get("attention_reference_qualification"))
    real = report.get("captured_original_prefix_Attention_inputs")
    decode_real_inputs(real)
    expected = [m for m, _ in component_plan()]
    for batch in (1, 2):
        expected.extend(m for m, _ in layer_plan(batch, real))
    for ordinal, mode in enumerate(FAULTS):
        value = inputs((1, 1, 5120), 1800000 + ordinal)
        expected.append({"id": "layer/retry/" + mode, "kind": "layer", "batch": 1, "position": 2 + ordinal,
                         "sequence": 1, "shape": [1, 1, 5120], "seed": 1800000 + ordinal, "mode": "normal",
                         "input_sha256": array_digest(value), "reset": False})
    component, layers, failures = report.get("component_cases", []), report.get("layer_cases", []), report.get("faults", [])
    if len(component) != 90 or len(layers) != 309 or [f.get("condition") for f in failures] != list(FAULTS):
        raise ValueError("Projection/layer/failure coverage is incomplete")
    if any(not f.get("committed_cache_preserved") or not f.get("temporary_buffers_released")
           or f.get("partial_result_returned") or f.get("status") != "pass" for f in failures):
        raise ValueError("Layer failure evidence differs")
    if failures[-1].get("calls_blocked_until_recovery") is not True:
        raise ValueError("Failed-rollback recovery control is missing")
    goldens = report.get("reference_goldens", [])
    if len(goldens) != 399 or len({g["meta"]["id"] for g in goldens}) != 399:
        raise ValueError("Projection golden IDs are incomplete")
    indexed = {g["meta"]["id"]: g for g in goldens}
    if len({r["id"] for r in component + layers}) != len(goldens):
        raise ValueError("Duplicate projection/layer comparison IDs")
    for meta in expected:
        if indexed.get(meta["id"], {}).get("meta") != meta:
            raise ValueError("Projection/layer golden inputs differ")
    from .normalization import NORM_STAGES
    from .attention_storage import STAGES
    from .projections import PROJECTION_STAGES
    layer_names = {p + n for p in ("wq_a_", "wq_b_", "wkv_", "wo_b_") for n in PROJECTION_STAGES}
    layer_names |= {p + n for p in ("q_norm_", "kv_norm_") for n in NORM_STAGES}
    layer_names |= {p + n for p in ("query_rope_", "kv_rope_", "inverse_rope_") for n in ("frequency_fp32", "rotated_fp32", "output_bf16")}
    layer_names |= {"kv_" + n for n in ("rounded_bf16", "codes_fp8", "scales_e8m0", "cache_bf16", "window_bf16", "indices_i32")}
    layer_names |= {"attention_" + n for n in ("output_bf16", *STAGES)}
    layer_names |= {"wo_a_" + n for n in ("input_fp8", "scale_e8m0", "rounded_bf16", "output_bf16")}
    for row in component + layers:
        golden = indexed.get(row["id"], {})
        hashes = golden.get("stage_sha256", {})
        names = layer_names if row["kind"] == "layer" else (set(PROJECTION_STAGES) if row["projection"] != "wo_a" else {"input_fp8", "scale_e8m0", "rounded_bf16", "output_bf16"})
        checks = row.get("checks", [])
        if ({k: v for k, v in row.items() if k not in ("checks", "status")} != golden.get("meta")
                or row.get("status") != "pass" or set(hashes) != names or len(checks) != len(names)
                or {c["check"] for c in checks} != names
                or any(c.get("mismatches") != 0 or c.get("status") != "pass" or c.get("expected_sha256") != c.get("actual_sha256")
                       or c.get("expected_sha256") != hashes.get(c["check"]) for c in checks)
                or any(not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h) for h in hashes.values())):
            raise ValueError("Projection/layer golden boundaries or comparisons differ")
    negatives = report.get("negative_controls", [])
    if ([n.get("check") for n in negatives] != ["omit_output_activation_rounding", "replace_grouped_BF16_with_FP32"]
            or any(n.get("mismatches", 0) <= 0 for n in negatives)):
        raise ValueError("Projection negative controls are missing")
    controls = report.get("instrumentation_controls", {}).get("unchanged_Attention_forward_output_and_KV", [])
    if (len(controls) != 309 or any(len(row) != 2 or any(c.get("mismatches") != 0 for c in row) for row in controls)
            or report.get("instrumentation_controls", {}).get("FP32_projection_to_original_BF16") != 1308):
        raise ValueError("Unchanged original Attention.forward controls are incomplete")
    memory = report.get("memory", {})
    for record in [memory.get(n, {}) for n in ("native_cleanup", "normalization_cleanup", "attention_cleanup")] + [p["KV_cleanup"] for p in memory.get("KV_phases", [])]:
        if record.get("device_owned_bytes") != 0 or record.get("host_reserved_bytes") != 0 or record.get("owned_context_active") is not False:
            raise ValueError("Owned projection/layer resources were not released")
    return indexed
