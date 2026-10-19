"""Explicit bounded fixture, route-plan and expert-store preparation for a source."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fixtures", "plan", "experts"))
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--layer", type=int, choices=(2, 20), default=2)
    for flag in ("topology-fixtures", "attention-bundle", "plan", "expert-directory", "store"):
        parser.add_argument("--" + flag, type=Path)
    args = parser.parse_args()
    def require(*names):
        for name in names:
            if getattr(args, name.replace("-", "_")) is None: parser.error("--" + name + " is required for " + args.stage)
    if args.stage == "fixtures":
        from vesper.block_fixture import fetch_block_fixture, load_block_fixture
        from vesper.gate import fetch_gate_fixture, load_gate_fixture
        from vesper.router import fetch_router_fixture, load_router_fixture
        from vesper.shared import fetch_shared_fixture, load_shared_fixture
        records = []
        for name, filename, fetch, load in (
            ("hc-norm", "weights.npz", lambda p: fetch_block_fixture(p, layer=args.layer), lambda p: load_block_fixture(p, layer=args.layer)),
            ("gate", "native-gate.npz", lambda p: fetch_gate_fixture(p, layer=args.layer), load_gate_fixture),
            ("router", "native-router.npz", lambda p: fetch_router_fixture(p, layer=args.layer), load_router_fixture),
            ("shared", "native-shared.npz", lambda p: fetch_shared_fixture(p, layer=args.layer), load_shared_fixture)):
            folder = args.fixtures / name
            if (folder / filename).exists():
                arrays, manifest = load(folder if name == "hc-norm" else folder / filename)
                del arrays
                if manifest["layer"] != args.layer: raise ValueError("Existing block fixture layer differs")
            else: manifest = fetch(folder)
            records.append(manifest)
        result = {"probe": "prepare_compressed_block_fixtures", "status": "pass", "fixtures": records}
    elif args.stage == "plan":
        require("topology-fixtures", "attention-bundle", "plan")
        from vesper.compressed_block_fixture import plan_block
        if args.plan.exists(): raise FileExistsError("Refusing to replace an immutable block import plan")
        result = plan_block(args.topology_fixtures, args.fixtures, args.attention_bundle, layer=args.layer)
        args.plan.parent.mkdir(parents=True, exist_ok=True)
        args.plan.write_bytes((json.dumps(result, indent=2) + "\n").encode())
    else:
        require("plan", "expert-directory", "store")
        from vesper.compressed_block_fixture import prepare_block_store
        from vesper.compressed_block_fixture import load_plan
        if load_plan(args.plan)["layer"] != args.layer: raise ValueError("Source plan layer differs from --layer")
        result = prepare_block_store(args.plan, args.expert_directory, args.store)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
