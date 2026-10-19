"""Complete source-20 to refreshed-index-24/consumer-25 qualification corpus."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .attention_topology import REFRESH_CHAIN, validate_block_chain
from .block_pair_validation import _open_operators, open_moe
from .block_chain import NativeBlockChain
from .block_chain_fixture import folder
from .block_chain_fixture import ORIGIN
from .attention_validation import array_digest
from .block_validation import block_stage_names
from .compressed_block_validation import block_inputs
from .long_context_validation import compact_comparison, replay_comparison
from .topology_validation import flatten, state_digest, owned_clean, boundary_names as attention_names
from .resources import MiB, GiB, physical_memory

FIXTURE_HOST_MIB, REFERENCE_HOST_MIB, NATIVE_HOST_MIB = 2048, 6144, 9728
FAULTS = ("host_pressure", "source20_after_three_experts", "handoff21", "refresh24_score_download",
    "refresh24_after_three_experts", "consumer25_attention_output", "consumer25_after_three_experts",
    "consumer25_final_download", "combined_primary_cap", "publication_pressure")
QUALIFIED_IDENTITY = "a4417ad56b7cfcad8b5be1c22d4b69fe79ffb2f13468fbbc5e743c51ae41cb9e"
QUALIFIED_GOLDENS = "6962fd3dc5a2f446697080a96bdb3988113a46edd75496bf02b1ec0f222c12b9"


def load_arrays(topology, root, *, layers=REFRESH_CHAIN):
    from .topology_fixture import load_layer
    from .block_fixture import load_block_fixture
    validate_block_chain(layers, prefix=True)
    arrays, fixtures = {}, {}
    for l in layers:
        a, m = load_layer(Path(topology) / str(l), l)
        arrays.update(a); fixtures[f"attention{l}"] = m
        a, m = load_block_fixture(folder(root, l) / "hc-norm", layer=l)
        arrays.update(a); fixtures[f"hc_norm{l}"] = m
    return arrays, fixtures


def open_operators(stack, arrays, bundle, ordinal=0, *, layers=REFRESH_CHAIN):
    validate_block_chain(layers, prefix=True)
    return _open_operators(stack, arrays, bundle, ordinal, layers)


def open_chain(stack, arrays, hc, ops, root, ordinal=0):
    moe = {l: open_moe(stack, folder(root, l), folder(root, l, "store"), l, ordinal) for l in REFRESH_CHAIN}
    return stack.enter_context(NativeBlockChain(arrays, hc, *ops, moe))


def meta(name, position, image):
    h, p = block_inputs(position)
    return {"id": name, "layers": list(REFRESH_CHAIN), "position": position, "sequence": 1, "image_span": image,
        "origin": ORIGIN, "initial_hidden_sha256": array_digest(h), "initial_pre_sha256": array_digest(p)}


def expected_metadata():
    for repeat in (0, 1):
        for image in (False, True):
            for pos in range(4): yield meta(f"chain/repeat{repeat}/image{int(image)}/p{pos}", pos, image)
    for i, mode in enumerate(FAULTS):
        for phase, pos in (("warm", 0), ("retry", 1)): yield meta(f"fault/{mode}/{phase}", pos, bool(i % 2))
    for name, pos in (("warm", 0), ("advance", 1), ("advance-again", 2), ("after-rollback", 1)):
        yield meta("snapshot/" + name, pos, False)


def values(result, state):
    return {**flatten(result), **{"state/" + n: a for n, a in flatten(state).items()}}


def names_for(position, decisions):
    names = {"output_bf16/", "next_pre_mix_fp32/"}
    for l in REFRESH_CHAIN:
        names.update(f"layer{l}/" + n + "/" for n in block_stage_names(decisions[str(l)]["indices"]) if not n.startswith("attention_"))
        names.update(f"layer{l}/{n}/" for n in ("input_hidden_bf16", "incoming_pre_fp32"))
        names.add(f"state/windows/{l}/")
        if l != 20: names.update(f"handoffs/{l}/{n}/" for n in ("hidden_bf16", "pre_mix_fp32"))
    names.update("attention/" + n + "/" for n in attention_names(REFRESH_CHAIN, {"position": position, "sequence": 1}))
    names.update(f"state/compressors/20/{i}/" for i in range(3))
    names.update(("state/index_keys/20/", "state/sharing/indices/", "state/sharing/candidates/"))
    return names


def registry(plan): return sorted(set().union(*(names_for(c["position"], c["decisions"]) for c in plan["cases"])))


def plan_case(plan, m): return next(c for c in plan["cases"] if (c["position"], c["image_span"]) == (m["position"], m["image_span"]))


def decisions_and_handoffs(result):
    decisions = {str(l): {"indices": result[f"layer{l}"]["moe_gate.indices"].tolist(),
        "weight_bits": result[f"layer{l}"]["moe_gate.weights"].view(np.uint32).tolist(),
        "moe_input_sha256": array_digest(result[f"layer{l}"]["ffn_norm_output_bf16"].reshape(5120)),
        "logits_sha256": array_digest(result[f"layer{l}"]["moe_gate.logits"])} for l in REFRESH_CHAIN}
    handoffs = {str(l): {"hidden_sha256": array_digest(h["hidden_bf16"]), "pre_sha256": array_digest(h["pre_mix_fp32"])} for l, h in result["handoffs"].items()}
    return decisions, handoffs


def exercise(chain, ops, plan, checker, reset_reference, lifecycle, progress=None):
    from .compressed_attention import CompressedAttentionSnapshot
    rows, faults, events = [], [], []
    minimum = physical_memory()["available"]
    resident = {l: {n: sum(d.allocations.values()) for n, d in m.devices.items()} for l, m in chain.moe.items()}
    def clean():
        return (owned_clean(ops) and all(sum(h.device.allocations.values()) == h.resident_bytes for h in chain.hc.values())
            and all({n: sum(d.allocations.values()) for n, d in m.devices.items()} == resident[l] for l, m in chain.moe.items()))
    def record(m):
        nonlocal minimum
        h, p = block_inputs(m["position"])
        result = chain.step(h, p, m["image_span"], start_pos=m["position"])
        state = chain.attention.inspect()["state"]
        actual = values(result, state)
        decisions, handoffs = decisions_and_handoffs(result)
        planned = plan_case(plan, m)
        reuse = {21: 20, 22: 20, 23: 20, 25: 24}
        if (decisions != planned["decisions"] or handoffs != planned["handoffs"] or set(actual) != names_for(m["position"], decisions)
                or any(not np.array_equal(result[f"layer{l-1}"]["output_bf16"], result[f"layer{l}"]["input_hidden_bf16"])
                       or not np.array_equal(result[f"layer{l-1}"]["next_pre_mix_fp32"].view(np.uint32), result[f"layer{l}"]["incoming_pre_fp32"].view(np.uint32)) for l in REFRESH_CHAIN[1:])
                or any(not np.array_equal(result["attention"][f"layer{l}_selected_indices_i32"], result["attention"][f"layer{o}_selected_indices_i32"]) for l, o in reuse.items())
                or any(not np.array_equal(result["attention"][f"layer{l}_compressed_visible_bf16"], result["attention"]["layer20_compressed_visible_bf16"]) for l in REFRESH_CHAIN)
                or state["sharing"]["index_owner"] != 24 or state["sharing"]["candidate_owner"] != 20):
            raise RuntimeError("Complete chain changed actual routes, a preceding hidden/coefficient handoff, KV ownership or refreshed-index reuse")
        queries = [l for l in REFRESH_CHAIN if f"layer{l}_index_query_projection_bf16" in result["attention"]]
        if queries != [20, 24]: raise RuntimeError("Complete chain used a wrong index-query layer")
        checked = checker(m, h, p, actual, state)
        rows.append({"meta": m, **checked, "decisions": decisions, "handoffs": handoffs,
            "committed_state_sha256": state_digest(state), "committed_position": chain.position,
            "index_query_layers": queries, "reused_index_owners": {str(l): o for l, o in reuse.items()},
            "all_shared_KV_from_layer20": True, "candidate_owner": 20, "final_index_owner": 24})
        if not clean(): raise RuntimeError("Complete chain leaked temporary GPU buffers")
        minimum = min(minimum, physical_memory()["available"])
        if progress: progress({"case": m["id"], "chains": len(rows), "status": checked["status"]})
        return actual
    for repeat in (0, 1):
        for image in (False, True):
            chain.reset(); reset_reference()
            for pos in range(4): record(meta(f"chain/repeat{repeat}/image{int(image)}/p{pos}", pos, image))
    expected_counts = ((0, 0, 0, 0, 0, 0), (3, 0, 0, 0, 0, 0), (6, 0, 0, 0, 0, 0),
        (6, 6, 6, 6, 0, 0), (6, 6, 6, 6, 3, 0), (6, 6, 6, 6, 6, 0),
        (6, 6, 6, 6, 6, 3), (6, 6, 6, 6, 6, 6), (6, 6, 6, 6, 6, 6), (6, 6, 6, 6, 6, 6))
    for i, mode in enumerate(FAULTS):
        chain.reset(); reset_reference()
        image = bool(i % 2)
        record(meta(f"fault/{mode}/warm", 0, image))
        before = chain.attention.inspect()
        calls = {l: m.stats["calls_started"] for l, m in chain.moe.items()}
        originals = (chain.budget.acquire, chain.moe[20].experts.forward, chain.hc[21].mixes, chain.index.device.download,
            chain.moe[24].experts.forward, chain.ps[25].fp8, chain.moe[25].experts.forward,
            chain.hc[25].post, chain.hc[25].device.download, chain.moe[25].forward, chain.attention.budget.acquire)
        count = 0
        def fail(*a, **k): raise OSError("Injected complete-chain boundary failure")
        def refuse(*a, **k): raise MemoryError("Injected complete-chain reserve refusal")
        def expert(*a, **k):
            nonlocal count
            count += 1
            if count == 4: raise OSError("Injected chain failure after three experts")
            return originals[{1: 1, 4: 4, 6: 6}[i]](*a, **k)
        def score(q, keys, weights):
            chain.index.device.download = fail
            return original_score(q, keys, weights)
        original_score = chain.index.score
        def conditional_score(q, keys, weights):
            nonlocal count
            count += 1
            return score(q, keys, weights) if count == 2 else original_score(q, keys, weights)
        def projection(name, *a, **k):
            if name == "wo_b": raise OSError("Injected last-consumer Attention output failure")
            return originals[5](name, *a, **k)
        def post(*a, **k):
            nonlocal count
            count += 1
            if count == 2: chain.hc[25].device.download = fail
            return originals[7](*a, **k)
        def after_moe(*a, **k):
            result = originals[9](*a, **k)
            if mode == "combined_primary_cap": chain.hc[25].device.allocate(chain.primary_cap)
            else: chain.attention.budget.acquire = refuse
            return result
        if i == 0: chain.budget.acquire = refuse
        elif i == 1: chain.moe[20].experts.forward = expert
        elif i == 2: chain.hc[21].mixes = fail
        elif i == 3: chain.index.score = conditional_score
        elif i == 4: chain.moe[24].experts.forward = expert
        elif i == 5: chain.ps[25].fp8 = projection
        elif i == 6: chain.moe[25].experts.forward = expert
        elif i == 7: chain.hc[25].post = post
        else: chain.moe[25].forward = after_moe
        caught = None
        try:
            h, p = block_inputs(1); chain.step(h, p, image, start_pos=1)
        except (MemoryError, OSError) as e: caught = type(e).__name__
        finally:
            (chain.budget.acquire, chain.moe[20].experts.forward, chain.hc[21].mixes, chain.index.device.download,
             chain.moe[24].experts.forward, chain.ps[25].fp8, chain.moe[25].experts.forward,
             chain.hc[25].post, chain.hc[25].device.download, chain.moe[25].forward, chain.attention.budget.acquire) = originals
            chain.index.score = original_score
        completed = {str(l): len(m.last_attempt["completed_expert_ids"]) if m.stats["calls_started"] > calls[l] else 0 for l, m in chain.moe.items()}
        unchanged = state_digest(before) == state_digest(chain.attention.inspect())
        passed = caught is not None and unchanged and clean() and tuple(completed[str(l)] for l in REFRESH_CHAIN) == expected_counts[i]
        faults.append({"name": mode, "status": "pass" if passed else "fail", "error_type": caught,
            "no_chain_output_returned": caught is not None, "all_six_states_sharing_candidates_and_position_unchanged": unchanged,
            "temporary_GPU_buffers_released": clean(), "completed_experts_in_failed_attempt": completed})
        if not passed: raise RuntimeError("Complete chain failed atomic refusal, expert progress or cleanup")
        record(meta(f"fault/{mode}/retry", 1, image))
    chain.reset(); reset_reference()
    record(meta("snapshot/warm", 0, False))
    before = chain.attention.inspect(); handle = chain.attention.snapshot(); lifecycle("snapshot")
    first = record(meta("snapshot/advance", 1, False)); record(meta("snapshot/advance-again", 2, False))
    chain.attention.rollback(handle); lifecycle("rollback")
    restored = state_digest(before) == state_digest(chain.attention.inspect())
    again = record(meta("snapshot/after-rollback", 1, False))
    same = state_digest(first) == state_digest(again)
    refused = 0
    for action, token in ((chain.attention.rollback, handle), (chain.attention.discard, handle),
                           (chain.attention.rollback, CompressedAttentionSnapshot(chain.position))):
        try: action(token)
        except ValueError: refused += 1
    copy = chain.attention.inspect(); digest = state_digest(chain.attention.inspect())
    copy["state"]["windows"][24].fill(123); copy["state"]["sharing"]["candidates"].fill(False)
    isolated = digest == state_digest(chain.attention.inspect())
    handle = chain.attention.snapshot(); chain.reset(); reset_reference()
    try: chain.attention.rollback(handle)
    except ValueError: refused += 1
    events.append({"name": "whole_chain_snapshot", "status": "pass" if restored and same and isolated and refused == 4 else "fail",
        "all_six_states_restored": restored, "all_chain_boundaries_reproduced": same,
        "diagnostic_copy_isolated": isolated, "consumed_foreign_expired_handle_refusals": refused})
    return rows, faults, events, minimum


def identity(report):
    from .block_pair_validation import identity as pair_identity
    data = dict(report)
    # Bind every staged expert plan, including preceding complete-block inputs.
    data["plan"] = {"final": report["plan"], "staged": report["expert_plans"]}
    return pair_identity(data)


def golden_digest(report):
    fields = ("meta", "boundaries", "expected_layout_sha256", "decisions", "handoffs", "committed_state_sha256")
    return hashlib.sha256(json.dumps([[r[k] for k in fields] for r in report["chain_cases"]], sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_goldens(report):
    from .attention_memory import validate_qualification
    from .checkpoint import MODEL, REVISION
    from .block_chain_fixture import validate_plan
    from .block_pair_validation import block_control_names
    from .topology_validation import original_control_names
    if ((report.get("probe"), report.get("status"), report.get("model"), report.get("revision"), report.get("layers")) !=
            ("complete_refresh_chain_parity", "pass", MODEL, REVISION, list(REFRESH_CHAIN))):
        raise ValueError("Complete chain report identity differs")
    if not QUALIFIED_IDENTITY or identity(report) != QUALIFIED_IDENTITY or report.get("source_fixture_compiler_identity_sha256") != QUALIFIED_IDENTITY:
        raise ValueError("Unqualified complete-chain source/fixture/compiler identity")
    if not QUALIFIED_GOLDENS or golden_digest(report) != QUALIFIED_GOLDENS or report.get("complete_goldens_sha256") != QUALIFIED_GOLDENS:
        raise ValueError("Incomplete complete-chain original goldens")
    plans = report.get("expert_plans", [])
    if [p["target_layer"] for p in plans] != [22, 23, 24, 25] or plans[-1] != report["plan"]: raise ValueError("Chain staged plans differ")
    for plan in plans:
        validate_plan(plan)
        if any(m != report["fixtures"][n] for n, m in plan["fixtures"].items()): raise ValueError("Chain plan fixtures differ from execution")
        for a, b in zip(plan["cases"], report["plan"]["cases"]):
            if any(d != b["decisions"][l] for l, d in a["decisions"].items()) or any(h != b["handoffs"][l] for l, h in a["handoffs"].items()):
                raise ValueError("A chain plan skipped or changed an actual preceding block")
    validate_qualification(report["attention_reference_qualification"])
    if report["boundary_names"] != registry(report["plan"]): raise ValueError("Complete chain boundary registry differs")
    rows = report.get("chain_cases", [])
    if [r["meta"] for r in rows] != list(expected_metadata()): raise ValueError("Complete chain corpus differs")
    for r in rows:
        names = [report["boundary_names"][b[0]] for b in r["boundaries"]]
        p = plan_case(report["plan"], r["meta"])
        if (r.get("status") != "pass" or r.get("failures") or set(names) != names_for(r["meta"]["position"], r["decisions"])
                or len(set(names)) != len(names) or r["decisions"] != p["decisions"] or r["handoffs"] != p["handoffs"]
                or r["index_query_layers"] != [20, 24] or r["reused_index_owners"] != {"21": 20, "22": 20, "23": 20, "25": 24}
                or r["all_shared_KV_from_layer20"] is not True or r["candidate_owner"] != 20 or r["final_index_owner"] != 24
                or r["committed_position"] != r["meta"]["position"] + 1
                or r["actual_layout_sha256"] != r["expected_layout_sha256"] or r["actual_boundary_root_sha256"] != r["expected_boundary_root_sha256"]
                or any(b[2] or len(b[3]) != 64 for b in r["boundaries"])):
            raise ValueError("Complete-chain exact handoffs, routes, index refresh, layouts or boundaries differ")
        root = hashlib.sha256(json.dumps([[b[0], b[3]] for b in r["boundaries"]], separators=(",", ":")).encode()).hexdigest()
        if root != r["expected_boundary_root_sha256"]: raise ValueError("Complete chain boundary root differs")
    for key, names in (("unchanged_Block_forward_controls", block_control_names(REFRESH_CHAIN)),
                       ("unchanged_Attention_forward_controls", original_control_names(REFRESH_CHAIN))):
        controls = report.get(key, [])
        if len(controls) != len(rows) or any(len(c) != len(names) or sorted(x["check"] for x in c) != names or any(x["status"] != "pass" or x["mismatches"] for x in c) for c in controls):
            raise ValueError("Unchanged original consecutive-chain controls differ")
    controls = report.get("whole_chain_state_controls", [])
    if len(controls) != len(rows) or any(c["case"] != r["meta"]["id"] or c["status"] != "pass" or c["expected_sha256"] != r["committed_state_sha256"] or c["actual_sha256"] != c["expected_sha256"] for c, r in zip(controls, rows)):
        raise ValueError("Whole-chain original state controls differ")
    counts = ((0, 0, 0, 0, 0, 0), (3, 0, 0, 0, 0, 0), (6, 0, 0, 0, 0, 0),
        (6, 6, 6, 6, 0, 0), (6, 6, 6, 6, 3, 0), (6, 6, 6, 6, 6, 0),
        (6, 6, 6, 6, 6, 3), (6, 6, 6, 6, 6, 6), (6, 6, 6, 6, 6, 6), (6, 6, 6, 6, 6, 6))
    faults = report.get("faults", [])
    if [f["name"] for f in faults] != list(FAULTS) or any(f["status"] != "pass" or not all(f[k] is True for k in
            ("no_chain_output_returned", "all_six_states_sharing_candidates_and_position_unchanged", "temporary_GPU_buffers_released"))
            or tuple(f["completed_experts_in_failed_attempt"][str(l)] for l in REFRESH_CHAIN) != n for f, n in zip(faults, counts)):
        raise ValueError("Complete-chain atomic failure evidence differs")
    events = report.get("snapshot_events", [])
    if (len(events) != 1 or events[0]["status"] != "pass" or events[0]["consumed_foreign_expired_handle_refusals"] != 4
            or not all(events[0][k] is True for k in ("all_six_states_restored", "all_chain_boundaries_reproduced", "diagnostic_copy_isolated"))):
        raise ValueError("Complete-chain snapshot evidence differs")
    negatives = report.get("negative_controls", [])
    if [c["check"] for c in negatives] != ["replace_transferred_pre_with_initial_pre", "bypass_preceding_complete_blocks", "replace_shared_KV_with_consumer_window",
            "drop_reused_source_selected_indices", "clear_source20_candidates", "reuse_source20_index_query_at24", "reuse_source20_index_scores_at24"] or any(c["status"] != "fail" or c["mismatches"] <= 0 for c in negatives):
        raise ValueError("Complete-chain changed-handoff/index controls differ")
    memory = report["memory"]
    expected_ops = {*(f"projection_{l}" for l in REFRESH_CHAIN), "normalization", "compression", "compression_normalization", "indexer", "indexer_normalization", "attention"}
    expected_moe = {"parent_reservation_bytes", "accumulator_reservation_bytes", "cache_reservation_bytes", "primary_explicit_live_bytes", "helper_explicit_live_bytes"}
    if (memory.get("combined_native_host_allowance_bytes") != NATIVE_HOST_MIB * MiB or memory.get("additional_reference_host_allowance_bytes") != REFERENCE_HOST_MIB * MiB
            or memory.get("retained_fixture_host_allowance_bytes") != FIXTURE_HOST_MIB * MiB or memory.get("all_frame_reservations_released") is not True
            or type(memory.get("minimum_sampled_host_available_bytes")) is not int or memory["minimum_sampled_host_available_bytes"] < 32 * GiB
            or set(memory["operator_cleanup"]) != expected_ops or set(memory["HC_cleanup"]) != {str(l) for l in REFRESH_CHAIN}
            or set(memory["moe_cleanup"]) != {str(l) for l in REFRESH_CHAIN} or any(set(c) != expected_moe for c in memory["moe_cleanup"].values())
            or any(c["device_owned_bytes"] or c["host_reserved_bytes"] or c["owned_context_active"] for c in list(memory["operator_cleanup"].values()) + list(memory["HC_cleanup"].values()))
            or any(any(c.values()) for c in memory["moe_cleanup"].values()) or memory["chain_cleanup"]["host_reserved_bytes"]
            or memory["chain_active_stats"]["primary_cap_bytes"] != NativeBlockChain.primary_cap or memory["chain_active_stats"]["helper_cap_bytes"] != NativeBlockChain.helper_cap
            or not 0 < memory["chain_active_stats"]["peak_owned_primary_bytes"] <= NativeBlockChain.primary_cap
            or not 0 < memory["chain_active_stats"]["peak_owned_helper_bytes"] <= NativeBlockChain.helper_cap):
        raise ValueError("Complete-chain native resource limits or cleanup differ")
    return {r["meta"]["id"]: r for r in rows}
