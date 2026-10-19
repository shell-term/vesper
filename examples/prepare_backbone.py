"""Prepare bounded two-token, forty-layer original control/expert artifacts."""

import argparse
import json
from pathlib import Path

from vesper.backbone_assets import prepare as prepare_controls
from vesper.backbone_prepare import prepare
from vesper.memory import MemoryPolicy, RuntimeMemory
from vesper.resources import GiB


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--controls", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    progress = lambda row: print(json.dumps(row), flush=True)
    with RuntimeMemory(MemoryPolicy(host_cap_bytes=64 * GiB)) as policy:
        if args.controls.exists():
            bindings = json.loads(args.controls.read_bytes())
        else:
            bindings = prepare_controls(args.artifacts, progress)
            args.controls.parent.mkdir(parents=True, exist_ok=True)
            args.controls.write_bytes((json.dumps(bindings, indent=2) + "\n").encode())
        report = prepare(args.artifacts, bindings, progress)
        report["memory_before_policy_close"] = policy.snapshot()
    report["memory_after_policy_close"] = policy.snapshot()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(report, indent=2) + "\n").encode())


if __name__ == "__main__": main()
