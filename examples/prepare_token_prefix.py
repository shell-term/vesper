"""Prepare only the measured 0..3 prefix's additional original artifacts."""

import argparse
import json
from pathlib import Path

from vesper.memory import RuntimeMemory, MemoryPolicy
from vesper.resources import GiB
from vesper.token_prefix_fixture import prepare_layer1
from vesper.token_prefix_prepare import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    def progress(row): print(json.dumps(row), flush=True)
    with RuntimeMemory(MemoryPolicy(host_cap_bytes=16 * GiB)) as policy:
        prepare_layer1(args.artifacts / "token-prefix-fixture/layer1", progress)
        report = prepare(args.artifacts, progress)
        report["memory_before_policy_close"] = policy.snapshot()
    report["memory_after_policy_close"] = policy.snapshot()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(report, indent=2) + "\n").encode())


if __name__ == "__main__": main()
