"""Plan a bounded layer-2 expert union from the connected HC/Attention prefix."""

from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np

from .attention_validation import array_digest
from .bandwidth import EXPERT_BYTES
from .checkpoint import MODEL, REVISION, fetch_expert_fixture, load_expert_fixture
from .compressed_block_validation import ORIGIN, block_inputs, load_arrays, open_operators
from .gate import GateProjection, load_gate_fixture
from .resources import MemoryBudget, MiB
from .router import RouterProbe, load_router_fixture
from .shared_attention import SharedAttentionFrame
from .storage import build_store, validate_store_limit


def plan_block(topology, fixtures, bundle, ordinal=0, *, max_experts=48):
    validate_store_limit(max_experts)
    root, budget = Path(fixtures), MemoryBudget(cap=768 * MiB)
    budget.acquire(768 * MiB)
    try:
        arrays, manifests = load_arrays(topology, fixtures)
        weight, gate_manifest = load_gate_fixture(root / "gate/native-gate.npz")
        biases, router_manifest = load_router_fixture(root / "router/native-router.npz")
        if any(m["layer"] != 2 for m in (gate_manifest, router_manifest)): raise ValueError("Block planning requires original layer-2 gate artifacts")
        cases, selected = [], set()
        with ExitStack() as stack:
            hc, ops = open_operators(stack, arrays, bundle, ordinal)
            ps, norm, comp, index, attn = ops
            gate = stack.enter_context(GateProjection(weight, ordinal))
            router = stack.enter_context(RouterProbe(biases, ordinal))
            frame = stack.enter_context(SharedAttentionFrame(arrays, ps, norm, comp, index, attn, (2,)))
            for image in (False, True):
                frame.reset()
                for pos in range(4):
                    hidden, pre = block_inputs(pos)
                    mixes = hc.mixes(hidden, "attn")
                    x = hc.pre(hidden, pre)["output_bf16"]
                    x = norm.norm(x, arrays["layers.2.attn_norm.weight"])["output_bf16"]
                    def prefix(stages):
                        after = hc.post(stages["layer2_wo_b_output_bf16"], hidden, mixes["post_fp32"], mixes["comb_fp32"])["output_bf16"]
                        ffn = hc.pre(after, mixes["pre_fp32"])["output_bf16"]
                        ffn = norm.norm(ffn, arrays["layers.2.ffn_norm.weight"])["output_bf16"].reshape(5120)
                        from .formats import bf16_values
                        logits = gate.project(bf16_values(ffn))
                        decision = router.inspect(logits, image)
                        ids = decision["indices"].tolist()
                        selected.update(ids)
                        if len(selected) > max_experts: raise ValueError("Connected block expert union exceeds the import cap; no expert weights fetched")
                        cases.append({"position": pos, "image_span": image, "hidden_sha256": array_digest(hidden),
                            "incoming_pre_sha256": array_digest(pre), "moe_input_sha256": array_digest(ffn), "logits_sha256": array_digest(logits),
                            "indices": ids, "weight_bits": decision["weights"].view(np.uint32).tolist()})
                    frame.step({2: x}, start_pos=pos, finalize=prefix)
            profile = {"HC_cubin_sha256": hc.cubin_sha256, "router_cubin_sha256": router.cubin_sha256, "cublas_version": gate.blas.version}
        return {"probe": "compressed_block_expert_plan", "status": "pass", "model": MODEL, "revision": REVISION, "layer": 2,
            "input_origin": ORIGIN, "fixtures": manifests, "gate_fixture": gate_manifest, "router_fixture": router_manifest,
            "cases": cases, "expert_ids": sorted(selected), "max_experts": max_experts, "selected_weight_bytes": len(selected) * EXPERT_BYTES,
            "maximum_weight_bytes": max_experts * EXPERT_BYTES, "native_profile": profile, "expert_weights_fetched_bytes": 0,
            "scope": "Actual connected HC/Attention/RMS input to the original gate; generated block input, not an original full-model activation"}
    finally: budget.release(768 * MiB)


def load_plan(path):
    path = Path(path)
    if path.stat().st_size > 128 * 1024: raise ValueError("Excessive block expert plan")
    plan = json.loads(path.read_bytes())
    validate_plan(plan)
    return plan


def validate_plan(plan):
    if (plan.get("probe"), plan.get("status"), plan.get("model"), plan.get("revision"), plan.get("layer"), plan.get("input_origin")) != (
            "compressed_block_expert_plan", "pass", MODEL, REVISION, 2, ORIGIN): raise ValueError("Block plan source identity differs")
    validate_store_limit(plan["max_experts"])
    cases, union = plan.get("cases", []), set()
    if [(c["image_span"], c["position"]) for c in cases] != [(i, p) for i in (False, True) for p in range(4)]:
        raise ValueError("Block expert plan corpus differs")
    for c in cases:
        h, p = block_inputs(c["position"])
        if (type(c["image_span"]) is not bool or c["hidden_sha256"] != array_digest(h) or c["incoming_pre_sha256"] != array_digest(p)
                or len(c["indices"]) != 6 or len(set(c["indices"])) != 6
                or any(type(e) is not int or not 0 <= e < 384 for e in c["indices"])
                or len(c["weight_bits"]) != 6 or any(type(b) is not int or not 0 <= b < 2**32 for b in c["weight_bits"])):
            raise ValueError("Block plan input/routing decision differs")
        union.update(c["indices"])
    if (not 1 <= len(union) <= plan["max_experts"] or sorted(union) != plan["expert_ids"]
            or plan["selected_weight_bytes"] != len(union) * EXPERT_BYTES or plan["maximum_weight_bytes"] != plan["max_experts"] * EXPERT_BYTES
            or plan.get("expert_weights_fetched_bytes") != 0): raise ValueError("Block plan selected expert import scope differs")


def prepare_block_store(plan_file, expert_directory, store_directory):
    plan = load_plan(plan_file)
    experts, target = Path(expert_directory), Path(store_directory)
    if target.exists(): raise FileExistsError("Refusing to replace a block expert store")
    budget = MemoryBudget(cap=96 * MiB)
    budget.acquire(96 * MiB)
    paths, fetched = [], 0
    try:
        for e in plan["expert_ids"]:
            folder = experts / f"expert-{e:03d}"
            path = folder / "native-expert.npz"
            if path.exists():
                arrays, manifest = load_expert_fixture(path)
                del arrays
                if (manifest["layer"], manifest["expert"]) != (2, e): raise ValueError("Cached original block expert identity differs")
            else:
                fetch_expert_fixture(folder, layer=2, expert=e)
                fetched += EXPERT_BYTES
            paths.append(path)
        result = build_store(paths, target, max_experts=plan["max_experts"])
        return {"probe": "prepare_compressed_block_store", "status": "pass", "layer": 2, "expert_ids": plan["expert_ids"],
            "new_weight_bytes_fetched": fetched, "verified_existing_weight_bytes_reused": len(paths) * EXPERT_BYTES - fetched, "store": result}
    finally: budget.release(96 * MiB)
