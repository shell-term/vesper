"""Recheck sealed all-layer original goldens without original frameworks."""

import argparse
import json
import importlib.util
import sys
from pathlib import Path
from vesper.attention import read_bounded
from vesper.backbone_validation import run_native, validate_original
from vesper.memory import MemoryPolicy, RuntimeMemory
from vesper.model_artifacts import digest
from vesper.resources import GiB, MiB


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(importlib.util.find_spec(n) is not None or n in sys.modules for n in ("torch", "tilelang", "transformers", "sympy")):
        raise RuntimeError("Use a native environment without reference frameworks installed or imported")
    raw = read_bounded(args.reference, 32 * MiB); original = validate_original(json.loads(raw))
    bindings = json.loads((args.artifacts / "backbone-controls.json").read_bytes())
    with RuntimeMemory(MemoryPolicy(host_cap_bytes=64 * GiB)) as policy:
        report = run_native(args.artifacts, bindings, original, policy, lambda r: print(json.dumps(r), flush=True))
        report["memory_before_policy_close"] = policy.snapshot()
    report["memory_after_policy_close"] = policy.snapshot()
    report["reference_report_sha256"] = digest(raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(report, indent=2) + "\n").encode())


if __name__ == "__main__": main()
