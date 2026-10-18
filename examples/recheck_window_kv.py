"""Replay native KV state/Attention goldens without Torch or TileLang."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from vesper.attention import NativeAttention, read_bounded
from vesper.attention_validation import array_digest
from vesper.kv import NativeWindowKV
from vesper.kv_validation import exercise_window, validate_goldens
from vesper.resources import MiB


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--attention-bundle", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = read_bounded(args.reference_report, 8 * MiB)
    report = json.loads(raw)
    goldens = validate_goldens(report)
    records, faults, events, phases = [], [], [], []
    with NativeAttention(args.attention_bundle, args.device) as attention:
        if attention.manifest_sha256 != report["attention_bundle_manifest_sha256"]:
            raise ValueError("Attention bundle differs from the paired report")
        def checker(meta, values, actual):
            expected = goldens[meta["id"]]
            if meta != expected["meta"] or set(actual) != set(expected["stage_sha256"]):
                raise ValueError("Window KV inputs, positions or boundary set differ")
            return [{"check": name, "status": "pass" if array_digest(a) == expected["stage_sha256"][name] else "fail",
                     "elements": int(a.size), "expected_sha256": expected["stage_sha256"][name], "actual_sha256": array_digest(a)}
                    for name, a in actual.items()]
        for batch in (1, 2):
            with NativeWindowKV(batch, args.device) as native:
                if (native.cubin_sha256 != report["native_profile"]["cubin_sha256"]
                        or native.kernel_source_sha256 != report["native_profile"]["kernel_source_sha256"]
                        or native.compiler != report["native_profile"]["compiler"]):
                    raise ValueError("Window KV compiler/source/binary provenance differs")
                rows, failed, operations, minimum = exercise_window(native, attention, checker, lambda event: None)
            records.extend(rows)
            faults.extend({"batch": batch, **f} for f in failed)
            events.extend({"batch": batch, **e} for e in operations)
            phases.append({"batch": batch, "minimum_sampled_host_available_bytes": minimum, "cleanup": native.stats()})
    frameworks = {name + "_installed": importlib.util.find_spec(name) is not None for name in ("torch", "tilelang")}
    frameworks.update({name + "_imported": any(n == name or n.startswith(name + ".") for n in sys.modules) for name in ("torch", "tilelang")})
    passed = (len(records) == len(goldens) and all(r["status"] == "pass" for r in records + faults + events)
              and not frameworks["torch_imported"] and not frameworks["tilelang_imported"]
              and all(not p["cleanup"]["host_reserved_bytes"] and not p["cleanup"]["device_owned_bytes"] for p in phases)
              and not attention.stats()["device_owned_bytes"])
    result = {"schema_version": 1, "probe": "native_window_kv_recheck", "status": "pass" if passed else "fail",
              "reference_report_sha256": hashlib.sha256(raw).hexdigest(), "cases": records, "faults": faults, "state_events": events,
              **frameworks, "native_profile": report["native_profile"],
              "memory": {"phases": phases, "attention_cleanup": attention.stats(), "combined_native_host_allowance_bytes": 48 * MiB},
              "projection_normalization_and_RoPE": "not_run", "compressed_KV_indexing_and_sharing": "not_run", "full_model_quality": "not_run",
              "scope": "Same bounded post-RoPE rounding, ring state, selected Attention and failure/rollback matrix; independent framework-free process"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode())
    print(json.dumps({"status": result["status"], "cases": len(records), **frameworks}))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
