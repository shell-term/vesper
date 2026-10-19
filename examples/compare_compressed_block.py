"""Compare the complete layer-2 Transformer block with pinned original sources."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("reference-directory", "topology-fixtures", "fixtures", "attention-bundle", "plan", "store", "expert-directory", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    from vesper.compressed_block_reference import compare_block
    result = compare_block(args.reference_directory, args.topology_fixtures, args.fixtures, args.attention_bundle,
        args.plan, args.store, args.expert_directory, progress=lambda p: print(json.dumps(p), flush=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(result, indent=2, allow_nan=False) + "\n").encode())
    print(json.dumps({"status": result["status"], "blocks": len(result.get("block_cases", []))}))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__": raise SystemExit(main())
