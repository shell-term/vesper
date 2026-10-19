"""Prepare the bounded actual consecutive 20..25 chain without skipping blocks."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("fixtures", "plan", "experts"))
    p.add_argument("--topology-fixtures", type=Path, required=True)
    p.add_argument("--block-root", type=Path, required=True)
    p.add_argument("--attention-bundle", type=Path)
    p.add_argument("--plan", type=Path)
    p.add_argument("--target-layer", type=int, choices=(22, 23, 24, 25))
    args = p.parse_args()
    from vesper.block_chain_fixture import prepare_fixtures, plan_next, prepare_store, load_plan
    progress = lambda x: print(json.dumps(x), flush=True)
    if args.stage == "fixtures": result = prepare_fixtures(args.topology_fixtures, args.block_root, progress)
    elif args.stage == "plan":
        if any(getattr(args, n) is None for n in ("attention_bundle", "plan", "target_layer")): p.error("Planning requires --attention-bundle, --plan and --target-layer")
        if args.plan.exists(): raise FileExistsError("Refusing to replace an immutable chain import plan")
        result = plan_next(args.topology_fixtures, args.block_root, args.attention_bundle, args.target_layer, progress=progress)
        args.plan.parent.mkdir(parents=True, exist_ok=True)
        args.plan.write_bytes((json.dumps(result, indent=2) + "\n").encode())
    else:
        if args.plan is None or args.target_layer is None: p.error("Expert import requires --plan and --target-layer")
        if load_plan(args.plan)["target_layer"] != args.target_layer: raise ValueError("Requested plan layer differs")
        result = prepare_store(args.plan, args.block_root, progress)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
