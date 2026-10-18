"""Replay original-weight window Attention without Torch or TileLang."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from vesper.attention import NativeAttention, read_bounded
from vesper.attention_validation import array_digest
from vesper.normalization import NativeNormRoPE
from vesper.prefix import load_prefix_fixture
from vesper.projections import NativeProjections
from vesper.projection_validation import exercise_components, exercise_layer, exercise_failures, validate_goldens
from vesper.resources import MemoryBudget, MiB
from vesper.window_attention import NativeWindowAttention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--prefix-fixture", type=Path, required=True)
    parser.add_argument("--attention-bundle", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = read_bounded(args.reference_report, 32 * MiB)
    report = json.loads(raw)
    goldens = validate_goldens(report)
    budget = MemoryBudget(cap=512 * MiB)
    budget.acquire(512 * MiB)
    try:
        arrays, fixture = load_prefix_fixture(args.prefix_fixture)
    finally:
        budget.release(512 * MiB)
    if fixture != report["fixture"]:
        raise ValueError("Original Attention/prefix fixture differs")
    def checker(meta, values, actual):
        golden = goldens[meta["id"]]
        if meta != golden["meta"] or set(actual) != set(golden["stage_sha256"]):
            raise ValueError("Projection/layer input metadata or boundary set differs")
        return [{"check": name, "status": "pass" if array_digest(a) == golden["stage_sha256"][name] else "fail",
                 "elements": int(a.size), "expected_sha256": golden["stage_sha256"][name], "actual_sha256": array_digest(a)}
                for name, a in actual.items()]
    layer_rows, faults, phases = [], [], []
    with NativeProjections(arrays, args.device) as native, NativeNormRoPE(args.device) as norm, NativeAttention(args.attention_bundle, args.device) as attention:
        profile = {"compiler": native.compiler, "cubin_sha256": native.cubin_sha256, "device": native.device_info,
                   "cublas_version": native.blas.version, "cublas_math_mode": native.blas.math_mode,
                   "converted_wo_a_bf16_sha256": native.converted_sha256}
        if profile != report["native_profile"] or attention.manifest_sha256 != report["attention_bundle_manifest_sha256"]:
            raise ValueError("Projection compiler/binary/BLAS or Attention bundle differs")
        components, minimum = exercise_components(native, checker)
        for batch in (1, 2):
            with NativeWindowAttention(arrays, native, norm, attention, batch) as layer:
                rows, low = exercise_layer(layer, report["captured_original_prefix_Attention_inputs"], checker, lambda: None)
                if batch == 1:
                    retries, failures = exercise_failures(layer, checker)
                    rows.extend(retries)
                    faults.extend(failures)
            layer_rows.extend(rows)
            minimum = min(minimum, low)
            phases.append({"batch": batch, "KV_cleanup": layer.kv.stats()})
    frameworks = {n + "_installed": importlib.util.find_spec(n) is not None for n in ("torch", "tilelang")}
    frameworks.update({n + "_imported": any(m == n or m.startswith(n + ".") for m in sys.modules) for n in ("torch", "tilelang")})
    passed = (all(r["status"] == "pass" for r in components + layer_rows + faults)
              and not frameworks["torch_imported"] and not frameworks["tilelang_imported"]
              and all(not p["KV_cleanup"]["device_owned_bytes"] for p in phases)
              and not native.stats()["device_owned_bytes"] and not norm.stats()["device_owned_bytes"] and not attention.stats()["device_owned_bytes"])
    result = {"schema_version": 1, "probe": "native_attention_projection_recheck", "status": "pass" if passed else "fail",
        "reference_report_sha256": hashlib.sha256(raw).hexdigest(), "native_profile": profile, **frameworks,
        "component_cases": components, "layer_cases": layer_rows, "faults": faults,
        "memory": {"native_cleanup": native.stats(), "normalization_cleanup": norm.stats(), "attention_cleanup": attention.stats(),
            "KV_phases": phases, "combined_native_host_allowance_bytes": 656 * MiB, "minimum_sampled_host_available_bytes": minimum,
            "reservation_is_RSS_limit": False}, "weights_fetched_bytes": 0, "full_model_quality": "not_run", "end_to_end_speed": "not_measured",
        "scope": "Independent framework-free original-weight layer-0 window Attention replay, including real prefix inputs and failed-call recovery"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode())
    print(json.dumps({"status": result["status"], "components": len(components), "layers": len(layer_rows), **frameworks}))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
