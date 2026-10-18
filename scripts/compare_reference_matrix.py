"""Repeat the bounded official comparison with the first experiment's inputs."""

import argparse
import json
from pathlib import Path
import time

from vesper.reference import compare_official


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--reference-device", type=int, default=0)
    parser.add_argument("--reference-directory", type=Path, default=Path("artifacts/reference"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/official-matrix.json"))
    args = parser.parse_args()
    records = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for device in args.devices:
        for seed in (20261018, 0, 1, 42, 127, 512, 4096, 65535):
            try:
                result = compare_official(args.fixture, args.reference_directory, device,
                                          args.reference_device, seed)
            except Exception as exc:
                result = {"probe": "official_kernel_parity", "status": "error",
                          "device_ordinal": device, "seed": seed,
                          "error_type": type(exc).__name__, "error": str(exc)}
            result["timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            records.append(result)
            args.output.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            print(f"device={device} seed={seed}: {result['status']}", flush=True)
            if result["status"] in ("blocked", "error"):
                return 2
    return 0 if all(record["status"] == "pass" for record in records) else 2


if __name__ == "__main__":
    raise SystemExit(main())
