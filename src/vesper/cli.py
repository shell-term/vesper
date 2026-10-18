from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="vesper", description="Vesper native-precision MoE research probes")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("inventory", "validate", "bench-transfer", "fetch-fixture", "bench-disk", "compare-reference",
                 "trace-reduction", "fetch-expert-fixture", "compare-expert"):
        command = sub.add_parser(name)
        command.add_argument("--output", type=Path)
        if name in ("validate", "bench-transfer"):
            command.add_argument("--devices", type=int, nargs="+")
        if name in ("validate", "compare-reference"):
            command.add_argument("--fixture", type=Path)
        if name in ("trace-reduction", "compare-expert"):
            command.add_argument("--fixture", type=Path, required=True)
        if name == "fetch-fixture":
            command.add_argument("--directory", type=Path, default=Path("artifacts/fixture"))
        if name == "fetch-expert-fixture":
            command.add_argument("--directory", type=Path, default=Path("artifacts/expert-fixture"))
        if name == "bench-disk":
            command.add_argument("--directory", type=Path, default=Path("artifacts/disk-probe"))
            command.add_argument("--file", type=Path, help="Read an existing model file without creating scratch")
        if name in ("compare-reference", "compare-expert"):
            command.add_argument("--reference-directory", type=Path, default=Path("artifacts/reference"))
        if name in ("compare-reference", "compare-expert", "trace-reduction"):
            command.add_argument("--device", type=int, default=0)
            command.add_argument("--reference-device", type=int)
        if name in ("compare-reference", "trace-reduction"):
            command.add_argument("--seed", type=int, default=20261018)
        if name == "compare-expert":
            command.add_argument("--seeds", type=int, nargs="+", default=[20261018])
            command.add_argument("--input-scales", type=float, nargs="+", default=[1.0])
            command.add_argument("--routing-weights", type=float, nargs="+", default=[0.25, 1.5, 0.0])
    args = parser.parse_args(argv)
    try:
        if args.command == "inventory":
            from .cuda import Driver
            from .resources import physical_memory
            result = {"probe": "inventory", "status": "pass", "cuda": Driver().inventory(), "host_memory": physical_memory()}
        elif args.command == "validate":
            from .conformance import validate
            result = validate(args.devices, args.fixture)
        elif args.command == "bench-transfer":
            from .bandwidth import benchmark
            result = benchmark(args.devices)
        elif args.command == "fetch-fixture":
            from .checkpoint import fetch_fixture
            result = fetch_fixture(args.directory)
        elif args.command == "fetch-expert-fixture":
            from .checkpoint import fetch_expert_fixture
            result = fetch_expert_fixture(args.directory)
        elif args.command == "bench-disk":
            from .disk import benchmark
            result = benchmark(args.directory, args.file)
        elif args.command == "trace-reduction":
            from .reduction import trace_reduction
            reference = 0 if args.reference_device is None else args.reference_device
            result = trace_reduction(args.fixture, args.device, reference, args.seed)
        elif args.command == "compare-expert":
            from .expert_reference import compare_expert
            reference = args.device if args.reference_device is None else args.reference_device
            result = compare_expert(args.fixture, args.reference_directory, args.device, reference,
                                    args.seeds, (None, *args.routing_weights), args.input_scales)
        else:
            from .reference import compare_official
            result = compare_official(args.fixture, args.reference_directory, args.device, args.reference_device, args.seed)
    except Exception as exc:
        result = {"probe": args.command, "status": "error", "error_type": type(exc).__name__, "error": str(exc)}
    result["timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rendered = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result.get("status") == "pass" else 2


if __name__ == "__main__":
    sys.exit(main())
