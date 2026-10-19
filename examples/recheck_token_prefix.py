"""Replay complete original prefix goldens/faults with no reference frameworks."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

from vesper.attention import read_bounded
from vesper.memory import RuntimeMemory, MemoryPolicy
from vesper.model_artifacts import digest
from vesper.resources import GiB, MiB
from vesper.token_checkpoint_probe import probe
from vesper.token_prefix_validation import run_native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--reference-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(importlib.util.find_spec(n) is not None or n in sys.modules for n in ("torch", "tilelang", "transformers", "sympy")):
        raise RuntimeError("Use the native environment without reference frameworks installed/imported")
    raw = read_bounded(args.reference_result, 16 * MiB)
    original = json.loads(raw)
    with RuntimeMemory(MemoryPolicy(host_cap_bytes=24 * GiB)) as policy:
        report = run_native(args.artifacts, original, policy, lambda r: print(json.dumps(r), flush=True))
        report["outer_KV_checkpoint"] = probe()
        report["reference_report_sha256"] = digest(raw)
        report["memory_before_policy_close"] = policy.snapshot()
    report["memory_after_policy_close"] = policy.snapshot()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(report, indent=2) + "\n").encode())


if __name__ == "__main__": main()
