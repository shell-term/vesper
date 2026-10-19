"""Bounded immutable artifacts and actual-prefix expert plans for blocks 20..25."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np

from .attention_topology import REFRESH_CHAIN, validate_block_chain
from .attention_validation import array_digest
from .bandwidth import EXPERT_BYTES
from .block_pair import BlockStages
from .checkpoint import MODEL, REVISION, fetch_expert_fixture, load_expert_fixture
from .compressed_block_validation import block_inputs
from .formats import bf16_values
from .resources import MemoryBudget, MiB
from .shared_attention import SharedAttentionFrame
from .storage import build_store, validate_store_limit

ORIGIN = "generated_layer20_HC_entry_with_actual_consecutive_hidden_and_pre_handoffs_through_index_refresh24"
PREPARATION_HOST_MIB = 224  # two 64-MiB fetch/validation workers, then a 96-MiB store builder


def folder(root, layer, kind="fixture"):
    if type(layer) is not int or layer not in REFRESH_CHAIN: raise ValueError("Unregistered chain artifact layer")
    return Path(root) / f"block{layer}-{kind}"


def prepare_fixtures(topology, root, progress=None):
    from .topology_fixture import fetch_layer, load_layer
    from .block_fixture import fetch_block_fixture, load_block_fixture
    from .gate import fetch_gate_fixture, load_gate_fixture
    from .router import fetch_router_fixture, load_router_fixture
    from .shared import fetch_shared_fixture, load_shared_fixture
    records, fetched, reused = [], 0, 0
    for l in REFRESH_CHAIN:
        path = Path(topology) / str(l)
        if path.exists():
            arrays, m = load_layer(path, l); del arrays
            reused += m["tensor_bytes"]
        else:
            m = fetch_layer(path, l); fetched += m["tensor_bytes"]
        records.append(m)
        for part, name, fetch, load in (
            ("hc-norm", "weights.npz", lambda p: fetch_block_fixture(p, layer=l), lambda p: load_block_fixture(p, layer=l)),
            ("gate", "native-gate.npz", lambda p: fetch_gate_fixture(p, layer=l), load_gate_fixture),
            ("router", "native-router.npz", lambda p: fetch_router_fixture(p, layer=l), load_router_fixture),
            ("shared", "native-shared.npz", lambda p: fetch_shared_fixture(p, layer=l), load_shared_fixture)):
            path = folder(root, l) / part
            if (path / name).exists():
                arrays, m = load(path if part == "hc-norm" else path / name); del arrays
                if m["layer"] != l: raise ValueError("Existing chain control artifact layer differs")
                reused += m["weights_fetched_bytes"]
            else:
                m = fetch(path); fetched += m["weights_fetched_bytes"]
            records.append(m)
        if progress: progress({"prepared_layer": l, "new_weight_bytes": fetched, "verified_reused_bytes": reused})
    return {"probe": "prepare_complete_refresh_chain_fixtures", "status": "pass", "model": MODEL, "revision": REVISION,
        "layers": list(REFRESH_CHAIN), "fixtures": records, "new_weight_bytes_fetched": fetched,
        "verified_original_bytes_reused": reused, "maximum_range_bytes": 8 * MiB}


def plan_next(topology, root, bundle, target, ordinal=0, *, max_experts=48, progress=None):
    from .block_chain_validation import load_arrays, open_operators, FIXTURE_HOST_MIB
    from .block_pair_validation import open_moe
    from .gate import GateProjection, load_gate_fixture
    from .router import RouterProbe, load_router_fixture
    if type(target) is not int or target not in (22, 23, 24, 25): raise ValueError("Expected the next registered chain layer")
    layers = REFRESH_CHAIN[:REFRESH_CHAIN.index(target) + 1]
    validate_store_limit(max_experts)
    budget = MemoryBudget(cap=FIXTURE_HOST_MIB * MiB); budget.acquire(budget.cap)
    try:
        arrays, fixtures = load_arrays(topology, root, layers=layers)
        weight, gm = load_gate_fixture(folder(root, target) / "gate/native-gate.npz")
        bias, rm = load_router_fixture(folder(root, target) / "router/native-router.npz")
        if any(m["layer"] != target for m in (gm, rm)): raise ValueError("Original planned gate/router layer differs")
        selected, cases = set(), []
        with ExitStack() as stack:
            hc, ops = open_operators(stack, arrays, bundle, ordinal, layers=layers)
            ps, norm, comp, index, sparse = ops
            moes = {l: open_moe(stack, folder(root, l), folder(root, l, "store"), l, ordinal) for l in layers[:-1]}
            gate = stack.enter_context(GateProjection(weight, ordinal))
            router = stack.enter_context(RouterProbe(bias, ordinal))
            frame = stack.enter_context(SharedAttentionFrame(arrays, ps, norm, comp, index, sparse, layers))
            for image in (False, True):
                frame.reset()
                for pos in range(4):
                    initial = block_inputs(pos)
                    carry, phases, decisions = list(initial), {}, {}
                    def prepare(l):
                        phases[l] = BlockStages(arrays, hc[l], norm, l, *carry)
                        return phases[l].input
                    def finish(l, attention):
                        x = phases[l].after_attention(attention["wo_b_output_bf16"]).reshape(5120)
                        if l != target:
                            result = moes[l].forward(x, image)
                            carry[:] = phases[l].finish(result)
                            logits, decision = result["logits"], result["routing"]
                        else:
                            logits = gate.project(bf16_values(x)); decision = router.inspect(logits, image)
                            selected.update(decision["indices"].tolist())
                            if len(selected) > max_experts: raise ValueError("Actual chain expert union exceeds its bounded import cap")
                        decisions[str(l)] = {"indices": decision["indices"].tolist(), "weight_bits": decision["weights"].view(np.uint32).tolist(),
                            "moe_input_sha256": array_digest(x), "logits_sha256": array_digest(logits)}
                    frame.step_connected(prepare, finish, start_pos=pos)
                    cases.append({"position": pos, "image_span": image, "initial_hidden_sha256": array_digest(initial[0]),
                        "initial_pre_sha256": array_digest(initial[1]), "decisions": decisions,
                        "handoffs": {str(l): {"hidden_sha256": array_digest(phases[l].hidden),
                            "pre_sha256": array_digest(phases[l].stages["incoming_pre_fp32"])} for l in layers[1:]}})
                    if progress: progress({"target_layer": target, "position": pos, "image_span": image, "selected_experts": len(selected)})
        return {"probe": "complete_refresh_chain_expert_plan", "status": "pass", "model": MODEL, "revision": REVISION,
            "layers": list(layers), "target_layer": target, "origin": ORIGIN, "fixtures": fixtures, "gate": gm, "router": rm,
            "cases": cases, "expert_ids": sorted(selected), "max_experts": max_experts,
            "selected_weight_bytes": len(selected) * EXPERT_BYTES, "target_expert_weights_fetched_bytes": 0,
            "scope": "All preceding complete blocks execute before the target gate; target FFN output is never invented"}
    finally: budget.release(budget.cap)


def validate_plan(plan):
    layers = validate_block_chain(tuple(plan.get("layers", [])), prefix=True)
    target = plan.get("target_layer")
    if (target not in (22, 23, 24, 25) or type(target) is not int or layers[-1] != target
            or (plan.get("probe"), plan.get("status"), plan.get("model"), plan.get("revision"), plan.get("origin")) !=
                ("complete_refresh_chain_expert_plan", "pass", MODEL, REVISION, ORIGIN)):
        raise ValueError("Complete chain expert plan identity differs")
    validate_store_limit(plan["max_experts"])
    if set(plan["fixtures"]) != {f"{p}{l}" for l in layers for p in ("attention", "hc_norm")} or any(
            (m.get("model"), m.get("revision"), m.get("layer")) != (MODEL, REVISION, l)
            for l in layers for m in (plan["fixtures"][f"attention{l}"], plan["fixtures"][f"hc_norm{l}"])):
        raise ValueError("Complete chain plan contains unpinned or cross-layer artifacts")
    if any((plan[k].get("model"), plan[k].get("revision"), plan[k].get("layer")) != (MODEL, REVISION, target) for k in ("gate", "router")):
        raise ValueError("Target gate/router identity differs")
    cases, selected = plan.get("cases", []), set()
    if [(c["image_span"], c["position"]) for c in cases] != [(i, p) for i in (False, True) for p in range(4)]: raise ValueError("Chain plan corpus differs")
    digest = lambda s: type(s) is str and len(s) == 64 and all(c in "0123456789abcdef" for c in s)
    for c in cases:
        h, p = block_inputs(c["position"])
        if type(c["image_span"]) is not bool or c["initial_hidden_sha256"] != array_digest(h) or c["initial_pre_sha256"] != array_digest(p):
            raise ValueError("Chain plan generated entry differs")
        if set(c["decisions"]) != {str(l) for l in layers} or set(c["handoffs"]) != {str(l) for l in layers[1:]}:
            raise ValueError("Chain plan omits a real preceding block or handoff")
        for d in c["decisions"].values():
            if (len(d["indices"]) != 6 or len(set(d["indices"])) != 6 or any(type(e) is not int or not 0 <= e < 384 for e in d["indices"])
                    or len(d["weight_bits"]) != 6 or any(type(b) is not int or not 0 <= b < 2**32 for b in d["weight_bits"])
                    or any(not digest(d[k]) for k in ("moe_input_sha256", "logits_sha256"))):
                raise ValueError("Actual ordered chain route differs")
        if any(set(h) != {"hidden_sha256", "pre_sha256"} or any(not digest(d) for d in h.values()) for h in c["handoffs"].values()):
            raise ValueError("Chain plan handoff evidence differs")
        selected.update(c["decisions"][str(target)]["indices"])
    if (not 1 <= len(selected) <= plan["max_experts"] or sorted(selected) != plan["expert_ids"]
            or plan.get("selected_weight_bytes") != len(selected) * EXPERT_BYTES or plan.get("target_expert_weights_fetched_bytes") != 0):
        raise ValueError("Complete chain selected-expert import scope differs")


def load_plan(path):
    path = Path(path)
    if path.stat().st_size > 512 * 1024: raise ValueError("Excessive chain expert plan")
    result = json.loads(path.read_bytes()); validate_plan(result)
    return result


def prepare_store(plan_file, root, progress=None):
    plan = load_plan(plan_file); layer = plan["target_layer"]
    experts, store = folder(root, layer, "experts"), folder(root, layer, "store")
    if store.exists(): raise FileExistsError("Refusing to replace an immutable chain expert store")
    budget = MemoryBudget(cap=PREPARATION_HOST_MIB * MiB); budget.acquire(budget.cap)
    pool, fetched, paths = ThreadPoolExecutor(max_workers=2), 0, []
    def prepare(e):
        path = experts / f"expert-{e:03d}" / "native-expert.npz"
        if path.exists():
            allowance = MemoryBudget(cap=64 * MiB); allowance.acquire(allowance.cap)
            try:
                arrays, manifest = load_expert_fixture(path); del arrays
                if (manifest["layer"], manifest["expert"]) != (layer, e): raise ValueError("Existing chain expert identity differs")
            finally: allowance.release(allowance.cap)
            return path, 0
        fetch_expert_fixture(path.parent, layer=layer, expert=e)
        return path, EXPERT_BYTES
    futures = []
    try:
        futures = [pool.submit(prepare, e) for e in plan["expert_ids"]]
        for future in as_completed(futures):
            path, count = future.result(); paths.append(path); fetched += count
            if progress: progress({"layer": layer, "prepared_experts": len(paths), "selected": len(plan["expert_ids"])})
        pool.shutdown(wait=True)
        result = build_store(sorted(paths), store, max_experts=plan["max_experts"])
        return {"probe": "prepare_complete_refresh_chain_store", "status": "pass", "layer": layer, "expert_ids": plan["expert_ids"],
            "new_weight_bytes_fetched": fetched, "verified_existing_bytes_reused": len(paths) * EXPERT_BYTES - fetched,
            "host_allowance_bytes": PREPARATION_HOST_MIB * MiB, "maximum_fetch_workers": 2, "reservation_is_RSS_limit": False, "store": result}
    finally:
        for future in futures: future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
        budget.release(budget.cap)
