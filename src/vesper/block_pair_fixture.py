"""Bounded consumer expert import using the actual preceding complete block."""

from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np

from .attention_validation import array_digest
from .bandwidth import EXPERT_BYTES
from .block_pair import BlockStages, LAYERS
from .attention_topology import validate_block_pair
from .block_pair_validation import load_arrays, open_operators, open_moe, block_inputs, origin, FIXTURE_HOST_MIB
from .checkpoint import MODEL, REVISION, fetch_expert_fixture, load_expert_fixture
from .gate import load_gate_fixture, GateProjection
from .router import load_router_fixture, RouterProbe
from .resources import MemoryBudget, MiB
from .shared_attention import SharedAttentionFrame
from .storage import build_store, validate_store_limit
from .formats import bf16_values


def plan_pair(topology, source, consumer, bundle, source_store, ordinal=0, *, max_experts=48, progress=None, layers=LAYERS):
    source_layer, consumer_layer = validate_block_pair(layers)
    root, budget = Path(consumer), MemoryBudget(cap=FIXTURE_HOST_MIB * MiB)
    validate_store_limit(max_experts)
    budget.acquire(budget.cap)
    try:
        arrays, fixtures = load_arrays(topology, source, consumer, layers=layers)
        weight, gate_manifest = load_gate_fixture(root / "gate/native-gate.npz")
        biases, router_manifest = load_router_fixture(root / "router/native-router.npz")
        if any(m["layer"] != consumer_layer for m in (gate_manifest, router_manifest)): raise ValueError("Consumer plan requires its original gate artifacts")
        cases, selected = [], {str(l): set() for l in layers}
        with ExitStack() as stack:
            hc, ops = open_operators(stack, arrays, bundle, ordinal, layers=layers)
            ps, norm, comp, index, attn = ops
            source_moe = open_moe(stack, source, source_store, source_layer, ordinal)
            gate = stack.enter_context(GateProjection(weight, ordinal))
            router = stack.enter_context(RouterProbe(biases, ordinal))
            frame = stack.enter_context(SharedAttentionFrame(arrays, ps, norm, comp, index, attn, layers))
            for image in (False, True):
                frame.reset()
                for pos in range(4):
                    hidden, pre = block_inputs(pos)
                    phases, carry, decisions = {}, [hidden, pre], {}
                    def prepare(l):
                        phases[l] = BlockStages(arrays, hc[l], norm, l, *carry)
                        return phases[l].input
                    def finish(l, attention):
                        x = phases[l].after_attention(attention["wo_b_output_bf16"]).reshape(5120)
                        if l == source_layer:
                            result = source_moe.forward(x, image)
                            carry[:] = phases[l].finish(result)
                            logits, decision = result["logits"], result["routing"]
                        else:
                            logits = gate.project(bf16_values(x)); decision = router.inspect(logits, image)
                        ids = decision["indices"].tolist()
                        selected[str(l)].update(ids)
                        if len(selected[str(l)]) > max_experts: raise ValueError("Pair expert union exceeds the import cap; no consumer weights fetched")
                        decisions[str(l)] = {"indices": ids, "weight_bits": decision["weights"].view(np.uint32).tolist(),
                            "moe_input_sha256": array_digest(x), "logits_sha256": array_digest(logits)}
                    frame.step_connected(prepare, finish, start_pos=pos)
                    row = {"position": pos, "image_span": image, "decisions": decisions,
                        "initial_hidden_sha256": array_digest(hidden), "initial_pre_sha256": array_digest(pre),
                        "handoff_hidden_sha256": array_digest(carry[0]), "handoff_pre_sha256": array_digest(carry[1])}
                    cases.append(row)
                    if progress: progress({"position": pos, "image_span": image, "selected_consumer_experts": len(selected[str(consumer_layer)])})
        return {"probe": "consecutive_block_expert_plan", "status": "pass", "model": MODEL, "revision": REVISION,
            "layers": list(layers), "input_origin": origin(layers), "fixtures": fixtures, "consumer_gate": gate_manifest, "consumer_router": router_manifest,
            "cases": cases, "expert_ids": {l: sorted(ids) for l, ids in selected.items()}, "max_experts_per_layer": max_experts,
            "consumer_weight_bytes": len(selected[str(consumer_layer)]) * EXPERT_BYTES, "consumer_weights_fetched_bytes": 0,
            "scope": f"Complete source-{source_layer} output and actual HC handoff through consumer-{consumer_layer} FFN input/gate; consumer FFN output is not evaluated during planning"}
    finally: budget.release(budget.cap)


def validate_plan(plan):
    layers = validate_block_pair(tuple(plan.get("layers", [])))
    if (plan.get("probe"), plan.get("status"), plan.get("model"), plan.get("revision"), plan.get("layers"), plan.get("input_origin")) != (
            "consecutive_block_expert_plan", "pass", MODEL, REVISION, list(layers), origin(layers)): raise ValueError("Consecutive-block plan identity differs")
    if set(plan["fixtures"]) != {f"{part}{l}" for l in layers for part in ("attention", "hc_norm")} or any(
            (plan["fixtures"][f"{part}{l}"].get("model"), plan["fixtures"][f"{part}{l}"].get("revision"), plan["fixtures"][f"{part}{l}"].get("layer")) != (MODEL, REVISION, l)
            for l in layers for part in ("attention", "hc_norm")):
        raise ValueError("Pair plan contains cross-layer or unpinned artifacts")
    validate_store_limit(plan["max_experts_per_layer"])
    rows = plan.get("cases", [])
    if [(c["image_span"], c["position"]) for c in rows] != [(i, p) for i in (False, True) for p in range(4)]: raise ValueError("Pair plan corpus differs")
    union = {str(l): set() for l in layers}
    for c in rows:
        h, p = block_inputs(c["position"])
        if type(c["image_span"]) is not bool or c["initial_hidden_sha256"] != array_digest(h) or c["initial_pre_sha256"] != array_digest(p):
            raise ValueError("Pair plan initial input differs")
        if set(c["decisions"]) != set(union): raise ValueError("Incomplete pair route plan")
        for l, d in c["decisions"].items():
            if (len(d["indices"]) != 6 or len(set(d["indices"])) != 6 or any(type(e) is not int or not 0 <= e < 384 for e in d["indices"])
                    or len(d["weight_bits"]) != 6 or any(type(b) is not int or not 0 <= b < 2**32 for b in d["weight_bits"])):
                raise ValueError("Pair plan ordered routing decision differs")
            union[l].update(d["indices"])
    if (any(not 1 <= len(ids) <= plan["max_experts_per_layer"] or sorted(ids) != plan["expert_ids"][l] for l, ids in union.items())
            or set(plan["expert_ids"]) != set(union) or plan.get("consumer_weights_fetched_bytes") != 0
            or plan.get("consumer_weight_bytes") != len(union[str(layers[1])]) * EXPERT_BYTES
            or any(plan[k].get("layer") != layers[1] for k in ("consumer_gate", "consumer_router"))):
        raise ValueError("Pair selected expert import scope differs")


def load_plan(path):
    path = Path(path)
    if path.stat().st_size > 256 * 1024: raise ValueError("Excessive pair import plan")
    data = json.loads(path.read_bytes()); validate_plan(data)
    return data


def prepare_consumer_store(plan_file, experts, store, progress=None):
    plan, experts, store = load_plan(plan_file), Path(experts), Path(store)
    layer = plan["layers"][1]
    if store.exists(): raise FileExistsError("Refusing to replace a consumer native store")
    budget, paths, fetched = MemoryBudget(cap=96 * MiB), [], 0
    budget.acquire(budget.cap)
    try:
        for e in plan["expert_ids"][str(layer)]:
            folder = experts / f"expert-{e:03d}"
            path = folder / "native-expert.npz"
            if path.exists():
                arrays, manifest = load_expert_fixture(path); del arrays
                if (manifest["layer"], manifest["expert"]) != (layer, e): raise ValueError("Cached consumer expert identity differs")
            else:
                fetch_expert_fixture(folder, layer=layer, expert=e); fetched += EXPERT_BYTES
            paths.append(path)
            if progress: progress({"prepared_consumer_experts": len(paths), "selected": len(plan["expert_ids"][str(layer)])})
        result = build_store(paths, store, max_experts=plan["max_experts_per_layer"])
        return {"probe": "prepare_block_pair_consumer_store", "status": "pass", "layer": layer,
            "expert_ids": plan["expert_ids"][str(layer)], "new_weight_bytes_fetched": fetched,
            "verified_existing_weight_bytes_reused": len(paths) * EXPERT_BYTES - fetched, "store": result}
    finally: budget.release(budget.cap)
