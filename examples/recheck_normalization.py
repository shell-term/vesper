"""Replay original-weight RMS/RoPE and the projected-input chain without Torch."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from vesper.attention import NativeAttention, read_bounded
from vesper.attention_validation import array_digest
from vesper.kv import NativeWindowKV
from vesper.normalization import NativeNormRoPE
from vesper.normalization_fixture import load_norm_weights
from vesper.normalization_validation import exercise_chain, exercise_components, exercise_failures, validate_goldens
from vesper.resources import MiB


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--prefix-fixture", type=Path, required=True)
    parser.add_argument("--attention-bundle", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = read_bounded(args.reference_report, 8 * MiB)
    report = json.loads(raw)
    goldens = validate_goldens(report)
    weights, provenance = load_norm_weights(args.prefix_fixture)
    if provenance != report["original_weight_fixture"]:
        raise ValueError("Original normalization fixture differs")
    def checker(meta, values, actual):
        golden = goldens[meta["id"]]
        if meta != golden["meta"] or set(actual) != set(golden["stage_sha256"]):
            raise ValueError("Normalization/rotary input, profile or boundary set differs")
        return [{"check": name, "status": "pass" if array_digest(a) == golden["stage_sha256"][name] else "fail",
                 "expected_sha256": golden["stage_sha256"][name], "actual_sha256": array_digest(a), "elements": int(a.size)}
                for name, a in actual.items()]
    chains, phases = [], []
    with NativeNormRoPE(args.device) as native:
        if (native.cubin_sha256 != report["native_profile"]["cubin_sha256"] or native.compiler != report["native_profile"]["compiler"]
                or native.kernel_source_sha256 != report["native_profile"]["kernel_source_sha256"]):
            raise ValueError("Normalization compiler/source/binary provenance differs")
        records, minimum = exercise_components(native, weights, checker)
        failures = exercise_failures(native, weights, checker)
        with NativeAttention(args.attention_bundle, args.device) as attention:
            if attention.manifest_sha256 != report["attention_bundle_manifest_sha256"]:
                raise ValueError("Attention bundle differs")
            for batch in (1, 2):
                with NativeWindowKV(batch, args.device) as kv:
                    rows, chain_minimum = exercise_chain(native, kv, attention, weights, checker, lambda: None)
                chains.extend(rows)
                minimum = min(minimum, chain_minimum)
                phases.append({"batch": batch, "KV_cleanup": kv.stats()})
    frameworks = {n + "_installed": importlib.util.find_spec(n) is not None for n in ("torch", "tilelang")}
    frameworks.update({n + "_imported": any(m == n or m.startswith(n + ".") for m in sys.modules) for n in ("torch", "tilelang")})
    passed = (all(r["status"] == "pass" for r in records + chains + failures)
              and not frameworks["torch_imported"] and not frameworks["tilelang_imported"]
              and not native.stats()["device_owned_bytes"] and not native.stats()["host_reserved_bytes"]
              and all(not p["KV_cleanup"]["device_owned_bytes"] for p in phases) and not attention.stats()["device_owned_bytes"])
    result = {"schema_version": 1, "probe": "native_normalization_rotary_recheck", "status": "pass" if passed else "fail",
              "reference_report_sha256": hashlib.sha256(raw).hexdigest(), "native_profile": report["native_profile"], **frameworks,
              "component_cases": records, "chain_cases": chains, "faults": failures,
              "memory": {"native_cleanup": native.stats(), "attention_cleanup": attention.stats(), "KV_phases": phases,
                         "combined_native_host_allowance_bytes": 144 * MiB, "minimum_sampled_host_available_bytes": minimum},
              "original_weight_fixture": provenance, "native_projections_and_compressed_KV": "not_run", "full_model_quality": "not_run",
              "scope": "Independent framework-free replay of original-weight RMS, rotary arithmetic and projected-input KV/Attention/inverse-RoPE boundaries"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode())
    print(json.dumps({"status": result["status"], "components": len(records), "chains": len(chains), **frameworks}))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
