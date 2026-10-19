"""Compare complete original blocks 20..25, including actual index refresh24."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("reference-directory", "topology-fixtures", "block-root", "attention-bundle", "plan-directory", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    a = p.parse_args()
    from vesper.block_chain_reference import compare_chain
    r = compare_chain(a.reference_directory, a.topology_fixtures, a.block_root, a.attention_bundle, a.plan_directory,
        progress=lambda x: print(json.dumps(x), flush=True))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_bytes((json.dumps(r, indent=2, allow_nan=False) + "\n").encode())
    print(json.dumps({"status": r["status"], "chain_steps": len(r.get("chain_cases", []))}))
    return 0 if r["status"] == "pass" else 2


if __name__ == "__main__": raise SystemExit(main())
