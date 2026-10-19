"""Compare complete consecutive blocks 2/3 with unchanged original methods."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("reference-directory", "topology-fixtures", "source-fixtures", "consumer-fixtures", "attention-bundle",
              "plan", "source-store", "consumer-store", "source-experts", "consumer-experts", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    args = p.parse_args()
    from vesper.block_pair_reference import compare_pair
    result = compare_pair(args.reference_directory, args.topology_fixtures, args.source_fixtures, args.consumer_fixtures,
        args.attention_bundle, args.plan, args.source_store, args.consumer_store, args.source_experts, args.consumer_experts,
        progress=lambda x: print(json.dumps(x), flush=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(result, indent=2, allow_nan=False) + "\n").encode())
    print(json.dumps({"status": result["status"], "pairs": len(result.get("pair_cases", []))}))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__": raise SystemExit(main())
