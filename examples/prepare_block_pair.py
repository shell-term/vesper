"""Fetch only consumer control tensors and experts selected after the real source block."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("fixtures", "plan", "experts"))
    p.add_argument("--source-layer", type=int, choices=(2, 20), default=2)
    for n in ("source-fixtures", "consumer-fixtures"): p.add_argument("--" + n, type=Path, required=True)
    for n in ("topology-fixtures", "attention-bundle", "source-store", "plan", "consumer-experts", "consumer-store"):
        p.add_argument("--" + n, type=Path)
    args = p.parse_args()
    layers = (args.source_layer, args.source_layer + 1)
    consumer_layer = layers[1]
    def require(*names):
        for n in names:
            if getattr(args, n.replace("-", "_")) is None: p.error("--" + n + " is required for " + args.stage)
    if args.stage == "fixtures":
        from vesper.block_fixture import fetch_block_fixture, load_block_fixture
        from vesper.gate import fetch_gate_fixture, load_gate_fixture
        from vesper.router import fetch_router_fixture, load_router_fixture
        from vesper.shared import fetch_shared_fixture, load_shared_fixture
        records = []
        for name, filename, fetch, load in (
            ("hc-norm", "weights.npz", lambda f: fetch_block_fixture(f, layer=consumer_layer), lambda f: load_block_fixture(f, layer=consumer_layer)),
            ("gate", "native-gate.npz", lambda f: fetch_gate_fixture(f, layer=consumer_layer), load_gate_fixture),
            ("router", "native-router.npz", lambda f: fetch_router_fixture(f, layer=consumer_layer), load_router_fixture),
            ("shared", "native-shared.npz", lambda f: fetch_shared_fixture(f, layer=consumer_layer), load_shared_fixture)):
            root = args.consumer_fixtures / name
            if (root / filename).exists():
                arrays, manifest = load(root if name == "hc-norm" else root / filename); del arrays
                if manifest["layer"] != consumer_layer: raise ValueError("Consumer fixture layer differs")
            else: manifest = fetch(root)
            records.append(manifest)
        result = {"probe": "prepare_block_pair_control_fixtures", "status": "pass", "fixtures": records}
    elif args.stage == "plan":
        require("topology-fixtures", "attention-bundle", "source-store", "plan")
        from vesper.block_pair_fixture import plan_pair
        if args.plan.exists(): raise FileExistsError("Refusing to replace an immutable pair import plan")
        result = plan_pair(args.topology_fixtures, args.source_fixtures, args.consumer_fixtures, args.attention_bundle,
            args.source_store, progress=lambda x: print(json.dumps(x), flush=True), layers=layers)
        args.plan.parent.mkdir(parents=True, exist_ok=True)
        args.plan.write_bytes((json.dumps(result, indent=2) + "\n").encode())
    else:
        require("plan", "consumer-experts", "consumer-store")
        from vesper.block_pair_fixture import prepare_consumer_store
        from vesper.block_pair_fixture import load_plan
        if load_plan(args.plan)["layers"] != list(layers): raise ValueError("Pair plan layers differ from --source-layer")
        result = prepare_consumer_store(args.plan, args.consumer_experts, args.consumer_store, progress=lambda x: print(json.dumps(x), flush=True))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
