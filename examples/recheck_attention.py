"""Replay bounded Attention against oracle hashes without Torch or TileLang."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from vesper.attention import NativeAttention, read_bounded
from vesper.attention_validation import array_digest, exercise_attention, validate_goldens
from vesper.resources import MiB


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = read_bounded(args.reference_report, 2 * MiB)
    report = json.loads(raw)
    goldens = validate_goldens(report)
    def checker(heads, dimension, index, inputs, provenance, actual):
        golden = goldens[heads, dimension, index]
        if provenance != golden["input_sha256"] or set(actual) != set(golden["stage_sha256"]):
            raise ValueError("Attention inputs or boundary set differ from the oracle")
        return [{"check": name, "status": "pass" if array_digest(a) == golden["stage_sha256"][name] else "fail",
                 "expected_sha256": golden["stage_sha256"][name], "actual_sha256": array_digest(a), "elements": int(a.size)}
                for name, a in actual.items()]
    with NativeAttention(args.bundle, args.device) as native:
        if native.manifest != report["bundle_manifest"] or native.manifest_sha256 != report["bundle_manifest_sha256"]:
            raise ValueError("Native Attention bundle differs from the reference")
        rows, faults, minimum = exercise_attention(native, checker)
        device = native.device_info
    cleanup = native.stats()
    torch_imported = any(n == "torch" or n.startswith("torch.") for n in sys.modules)
    tilelang_imported = any(n == "tilelang" or n.startswith("tilelang.") for n in sys.modules)
    passed = (all(r["status"] == "pass" for r in rows + faults) and not torch_imported and not tilelang_imported
              and not cleanup["host_reserved_bytes"] and not cleanup["device_owned_bytes"])
    result = {"schema_version": 1, "probe": "native_attention_recheck", "status": "pass" if passed else "fail",
              "reference_report_sha256": hashlib.sha256(raw).hexdigest(), "bundle_manifest_sha256": native.manifest_sha256,
              "device": device, "cases": rows, "faults": faults,
              "torch_installed": importlib.util.find_spec("torch") is not None, "torch_imported": torch_imported,
              "tilelang_installed": importlib.util.find_spec("tilelang") is not None, "tilelang_imported": tilelang_imported,
              "numpy_version": np.__version__, "memory": {**cleanup, "minimum_sampled_host_available_bytes": minimum},
              "full_model_quality": "not_run", "kv_state_and_projections": "not_run",
              "scope": "Independent framework-free replay of the same bounded sparse Attention and pressure/recovery matrix"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode())
    print(json.dumps({k: result[k] for k in ("status", "torch_installed", "torch_imported", "tilelang_installed", "tilelang_imported")}))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
