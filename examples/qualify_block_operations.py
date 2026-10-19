"""Audit all five original Issue #7 checks against pinned published evidence."""

import argparse
import json
from pathlib import Path


def main():
    from vesper.block_ops_qualification import qualify, RESULT_ROOT
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--manifest", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    manifest = args.manifest or args.repository / RESULT_ROOT / "evidence-manifest.json"
    result = qualify(args.repository, manifest, lambda x: print(json.dumps(x), flush=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(result, indent=2, allow_nan=False) + "\n").encode())
    print(json.dumps({"status": result["status"], "criteria": len(result["acceptance_checks"]), "evidence_groups": result["qualified_evidence_groups"]}))
    return 0


if __name__ == "__main__": raise SystemExit(main())
