"""Consecutive source/consumer block corpus and complete native-only evidence."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from .attention_validation import array_digest
from .block_pair import LAYERS, NativeBlockPair
from .attention_topology import role, validate_block_pair
from .compressed_block_validation import block_inputs
from .resources import MemoryBudget, MiB, GiB, physical_memory
from .topology_validation import flatten, state_digest, owned_clean, boundary_names as attention_names
from .block_validation import block_stage_names
from .long_context_validation import compact_comparison, replay_comparison

NATIVE_HOST_MIB, REFERENCE_HOST_MIB, FIXTURE_HOST_MIB = 4832, 3072, 1024
FAULTS = ("host_pressure", "source2_after_three_experts", "handoff_to_consumer3", "consumer3_attention_output",
          "consumer3_after_three_experts", "consumer3_final_download", "combined_primary_cap", "publication_pressure")
ORIGIN = "generated_layer2_HC_input_and_actual_layer2_to3_hidden_and_pre_mix_handoff"
QUALIFIED_IDENTITY = "b2f05dbfd0af5d35d214c2f17127ef2e64013d9366e858d3b216788e1946370a"
QUALIFIED_GOLDENS = "b7f0e88990485560c5261f27df8f71017f5ced1fdf02e1b0134f90520af195b4"
QUALIFIED_PAIRS = {LAYERS: (QUALIFIED_IDENTITY, QUALIFIED_GOLDENS),
    (20, 21): ("f98c6541b689a300c36d7461ad79e5072366e064b0a45d0e35dc67e5cd99e3db",
               "f2d8753055d69d5df0759dd5dd6f24ad662e323846371a47815e654ff18ba351")}


def origin(layers):
    source, consumer = validate_block_pair(layers)
    return f"generated_layer{source}_HC_input_and_actual_layer{source}_to{consumer}_hidden_and_pre_mix_handoff"


def fault_names(layers):
    source, consumer = validate_block_pair(layers)
    return ("host_pressure", f"source{source}_after_three_experts", f"handoff_to_consumer{consumer}",
        f"consumer{consumer}_attention_output", f"consumer{consumer}_after_three_experts",
        f"consumer{consumer}_final_download", "combined_primary_cap", "publication_pressure")


def load_arrays(topology, source, consumer, *, layers=LAYERS):
    from .topology_fixture import load_layer
    from .block_fixture import load_block_fixture
    arrays, manifests = {}, {}
    for l, fixture in zip(validate_block_pair(layers), (source, consumer)):
        a, m = load_layer(Path(topology) / str(l), l)
        arrays.update(a); manifests[f"attention{l}"] = m
        a, m = load_block_fixture(Path(fixture) / "hc-norm", layer=l)
        arrays.update(a); manifests[f"hc_norm{l}"] = m
    return arrays, manifests


def open_operators(stack, arrays, bundle, ordinal=0, *, layers=LAYERS):
    validate_block_pair(layers)
    return _open_operators(stack, arrays, bundle, ordinal, layers)


def _open_operators(stack, arrays, bundle, ordinal, layers):
    from .hc import NativeHC
    from .projections import NativeProjections
    from .normalization import NativeNormRoPE
    from .compression import NativeCompression
    from .indexer import NativeIndexer
    from .attention import NativeAttention
    hc = {l: stack.enter_context(NativeHC(arrays, ordinal, layer=l)) for l in layers}
    ps = {l: stack.enter_context(NativeProjections(arrays, ordinal, layer=l)) for l in layers}
    norm = stack.enter_context(NativeNormRoPE(ordinal))
    comp = stack.enter_context(NativeCompression(arrays, ordinal, layers=tuple(l for l in layers if role(l).owns_kv)))
    index = stack.enter_context(NativeIndexer(arrays, ordinal, layers=tuple(l for l in layers if role(l).owns_index)))
    sparse = stack.enter_context(NativeAttention(bundle, ordinal, profile="long_context"))
    return hc, (ps, norm, comp, index, sparse)


def open_moe(stack, fixtures, store, layer, ordinal=0):
    from .connected import ConnectedMoE, ConnectedConfig
    root, config = Path(fixtures), ConnectedConfig()
    config = replace(config, cache=replace(config.cache, primary_device=ordinal, helper_device=1 if ordinal == 0 else 0))
    return stack.enter_context(ConnectedMoE(store, root / "gate/native-gate.npz", root / "router/native-router.npz",
        root / "shared/native-shared.npz", config, layer=layer))


def open_pair(stack, arrays, hc, ops, source, consumer, source_store, consumer_store, ordinal=0, *, layers=LAYERS):
    validate_block_pair(layers)
    moe = {l: open_moe(stack, f, s, l, ordinal) for l, f, s in zip(layers, (source, consumer), (source_store, consumer_store))}
    ps, norm, comp, index, sparse = ops
    return stack.enter_context(NativeBlockPair(arrays, hc, ps, norm, comp, index, sparse, moe, layers=layers))


def meta(identity, position, image, layers=LAYERS):
    h, p = block_inputs(position)
    return {"id": identity, "layers": list(layers), "position": position, "sequence": 1, "image_span": image,
        "origin": origin(layers), "initial_hidden_sha256": array_digest(h), "initial_pre_sha256": array_digest(p)}


def expected_metadata(layers=LAYERS):
    for repeat in (0, 1):
        for image in (False, True):
            for pos in range(4): yield meta(f"pair/repeat{repeat}/image{int(image)}/p{pos}", pos, image, layers)
    for i, mode in enumerate(fault_names(layers)):
        for label, pos in (("warm", 0), ("retry", 1)): yield meta(f"fault/{mode}/{label}", pos, bool(i % 2), layers)
    for label, pos in (("warm", 0), ("advance", 1), ("advance-again", 2), ("after-rollback", 1)):
        yield meta("snapshot/" + label, pos, False, layers)


def values(result, state):
    flat = flatten(result)
    flat.update({"state/" + n: a for n, a in flatten(state).items()})
    return flat


def names_for(position, decisions, layers=LAYERS):
    names = set()
    for l in layers:
        names.update(f"layer{l}/" + n + "/" for n in block_stage_names(decisions[str(l)]["indices"]) if not n.startswith("attention_"))
        names.update(f"layer{l}/" + n + "/" for n in ("input_hidden_bf16", "incoming_pre_fp32"))
    names.update("attention/" + n + "/" for n in attention_names(layers, {"position": position, "sequence": 1}))
    names.update(n + "/" for n in ("handoff_hidden_bf16", "handoff_pre_mix_fp32", "output_bf16", "next_pre_mix_fp32"))
    source, consumer = layers
    names.update("state/" + n for n in (f"windows/{source}/", f"windows/{consumer}/", *(f"compressors/{source}/{i}/" for i in range(3)), f"index_keys/{source}/", "sharing/indices/"))
    if source == 20: names.add("state/sharing/candidates/")
    return names


def registry(plan): return sorted(set().union(*(names_for(c["position"], c["decisions"], tuple(plan["layers"])) for c in plan["cases"])))


def plan_case(plan, m): return next(c for c in plan["cases"] if (c["position"], c["image_span"]) == (m["position"], m["image_span"]))


def exercise(pair, ops, plan, checker, reset_reference, lifecycle, progress=None):
    from .block_pair import PRIMARY_CAP
    from .compressed_attention import CompressedAttentionSnapshot
    layers = validate_block_pair(tuple(plan["layers"]))
    source, consumer = layers
    if pair.layers != layers: raise ValueError("Complete pair and routing plan layers differ")
    rows, faults, events, minimum = [], [], [], physical_memory()["available"]
    resident = {l: {n: sum(d.allocations.values()) for n, d in m.devices.items()} for l, m in pair.moe.items()}
    def clean():
        return (owned_clean(ops) and all(sum(h.device.allocations.values()) == h.resident_bytes for h in pair.hc.values())
            and all({n: sum(d.allocations.values()) for n, d in m.devices.items()} == resident[l] for l, m in pair.moe.items()))
    def record(m):
        nonlocal minimum
        h, p = block_inputs(m["position"])
        result = pair.step(h, p, m["image_span"], start_pos=m["position"])
        state = pair.attention.inspect()["state"]
        actual, planned = values(result, state), plan_case(plan, m)
        decisions = {}
        for l in layers:
            data = result[f"layer{l}"]
            decisions[str(l)] = {"indices": data["moe_gate.indices"].tolist(), "weight_bits": data["moe_gate.weights"].view(np.uint32).tolist(),
                "moe_input_sha256": array_digest(data["ffn_norm_output_bf16"].reshape(5120)), "logits_sha256": array_digest(data["moe_gate.logits"])}
        handoff = {"hidden_sha256": array_digest(result["handoff_hidden_bf16"]), "pre_sha256": array_digest(result["handoff_pre_mix_fp32"])}
        if (decisions != planned["decisions"] or set(actual) != names_for(m["position"], decisions, layers)
                or handoff["hidden_sha256"] != planned["handoff_hidden_sha256"] or handoff["pre_sha256"] != planned["handoff_pre_sha256"]
                or not np.array_equal(result[f"layer{source}"]["output_bf16"], result[f"layer{consumer}"]["input_hidden_bf16"])
                or not np.array_equal(result[f"layer{source}"]["next_pre_mix_fp32"].view(np.uint32), result[f"layer{consumer}"]["incoming_pre_fp32"].view(np.uint32))
                or not np.array_equal(result["attention"][f"layer{source}_compressed_visible_bf16"], result["attention"][f"layer{consumer}_compressed_visible_bf16"])
                or not np.array_equal(result["attention"][f"layer{source}_selected_indices_i32"], result["attention"][f"layer{consumer}_selected_indices_i32"])):
            raise RuntimeError("Consecutive blocks changed planned routing, HC handoff, index reuse or boundary coverage")
        checked = checker(m, h, p, actual, state)
        rows.append({"meta": m, **checked, "decisions": decisions, "handoff": handoff,
            "committed_state_sha256": state_digest(state), "committed_position": pair.position,
            "consumer_reused_source_indices": True, "consumer_reused_source_KV": True,
            "index_query_layers": [l for l in layers if f"layer{l}_index_query_projection_bf16" in result["attention"]],
            "selected_width": result["attention"][f"layer{consumer}_selected_indices_i32"].shape[-1]})
        if not clean(): raise RuntimeError("Consecutive pair leaked temporary GPU buffers")
        minimum = min(minimum, physical_memory()["available"])
        if progress: progress({"case": m["id"], "pairs": len(rows), "status": checked["status"]})
        return actual
    for repeat in (0, 1):
        for image in (False, True):
            pair.reset(); reset_reference()
            for pos in range(4): record(meta(f"pair/repeat{repeat}/image{int(image)}/p{pos}", pos, image, layers))
    for i, mode in enumerate(fault_names(layers)):
        pair.reset(); reset_reference()
        image = bool(i % 2)
        record(meta(f"fault/{mode}/warm", 0, image, layers))
        before = pair.attention.inspect()
        call_counts = {l: m.stats["calls_started"] for l, m in pair.moe.items()}
        complete_counts = {l: m.stats["calls_completed"] for l, m in pair.moe.items()}
        originals = (pair.budget.acquire, pair.moe[source].experts.forward, pair.hc[consumer].mixes, pair.ps[consumer].fp8,
            pair.moe[consumer].experts.forward, pair.hc[consumer].post, pair.hc[consumer].device.download, pair.moe[consumer].forward, pair.attention.budget.acquire)
        count = 0
        def refuse(*a, **k): raise MemoryError("Injected pair reserve refusal")
        def fail(*a, **k): raise OSError("Injected complete-pair boundary failure")
        def expert_failure(*a, **k):
            nonlocal count
            count += 1
            if count == 4: raise OSError("Injected pair expert failure after three completed experts")
            return originals[1 if i == 1 else 4](*a, **k)
        def projection(name, *a, **k):
            if name == "wo_b": raise OSError("Injected consumer Attention output failure")
            return originals[3](name, *a, **k)
        def final_post(*a, **k):
            nonlocal count
            count += 1
            if count == 2: pair.hc[consumer].device.download = fail
            return originals[5](*a, **k)
        def after_moe(*a, **k):
            result = originals[7](*a, **k)
            if mode == "combined_primary_cap": pair.hc[consumer].device.allocate(PRIMARY_CAP)
            else: pair.attention.budget.acquire = refuse
            return result
        if mode == "host_pressure": pair.budget.acquire = refuse
        elif i == 1: pair.moe[source].experts.forward = expert_failure
        elif i == 2: pair.hc[consumer].mixes = fail
        elif i == 3: pair.ps[consumer].fp8 = projection
        elif i == 4: pair.moe[consumer].experts.forward = expert_failure
        elif i == 5: pair.hc[consumer].post = final_post
        else: pair.moe[consumer].forward = after_moe
        caught = None
        try:
            h, p = block_inputs(1); pair.step(h, p, image, start_pos=1)
        except (MemoryError, OSError) as e: caught = type(e).__name__
        finally:
            (pair.budget.acquire, pair.moe[source].experts.forward, pair.hc[consumer].mixes, pair.ps[consumer].fp8,
             pair.moe[consumer].experts.forward, pair.hc[consumer].post, pair.hc[consumer].device.download, pair.moe[consumer].forward, pair.attention.budget.acquire) = originals
        unchanged = state_digest(before) == state_digest(pair.attention.inspect())
        completed = {str(l): len(m.last_attempt["completed_expert_ids"]) if m.stats["calls_started"] > call_counts[l] else 0 for l, m in pair.moe.items()}
        completed_source = pair.moe[source].stats["calls_completed"] > complete_counts[source]
        expected_counts = ((0, 0), (3, 0), (6, 0), (6, 0), (6, 3), (6, 6), (6, 6), (6, 6))[i]
        passed = caught is not None and unchanged and clean() and tuple(completed[str(l)] for l in layers) == expected_counts
        faults.append({"name": mode, "status": "pass" if passed else "fail", "error_type": caught, "no_pair_output_returned": caught is not None,
            "both_block_states_and_position_unchanged": unchanged, "temporary_GPU_buffers_released": clean(),
            "completed_experts_in_failed_attempt": completed, "source_block_completed_before_failure": completed_source})
        if not passed: raise RuntimeError("Consecutive pair failed atomic refusal/cleanup")
        record(meta(f"fault/{mode}/retry", 1, image, layers))
    pair.reset(); reset_reference()
    record(meta("snapshot/warm", 0, False, layers))
    saved = pair.attention.inspect(); handle = pair.attention.snapshot(); lifecycle("snapshot")
    first = record(meta("snapshot/advance", 1, False, layers))
    record(meta("snapshot/advance-again", 2, False, layers))
    pair.attention.rollback(handle); lifecycle("rollback")
    restored = state_digest(saved) == state_digest(pair.attention.inspect())
    again = record(meta("snapshot/after-rollback", 1, False, layers))
    identical = state_digest(first) == state_digest(again)
    refused = 0
    for action, token in ((pair.attention.rollback, handle), (pair.attention.discard, handle),
                          (pair.attention.rollback, CompressedAttentionSnapshot(pair.position))):
        try: action(token)
        except ValueError: refused += 1
    copy = pair.attention.inspect(); baseline = state_digest(pair.attention.inspect())
    copy["state"]["windows"][source].fill(123)
    if source == 20: copy["state"]["sharing"]["candidates"].fill(False)
    isolated = baseline == state_digest(pair.attention.inspect())
    expired = pair.attention.snapshot(); pair.reset(); reset_reference()
    try: pair.attention.rollback(expired)
    except ValueError: refused += 1
    events.append({"name": "whole_pair_snapshot", "status": "pass" if restored and identical and isolated and refused == 4 else "fail",
        "both_states_restored": restored, "all_pair_boundaries_reproduced": identical, "diagnostic_copy_isolated": isolated,
        "consumed_foreign_expired_handle_refusals": refused})
    return rows, faults, events, minimum


def identity(report):
    fields = ("model", "revision", "fixtures", "moe_fixtures", "plan", "native_profiles", "hc_profiles", "reference_identity", "original_methods", "hc_instrumentation")
    data = {k: report[k] for k in fields}
    q = report["attention_reference_qualification"]
    data["attention_reference_qualification"] = {**q, "memory": {k: v for k, v in q["memory"].items() if k != "minimum_sampled_host_available_bytes"}}
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def golden_digest(report):
    data = [[r["meta"], r["boundaries"], r["expected_layout_sha256"], r["decisions"], r["handoff"], r["committed_state_sha256"]] for r in report["pair_cases"]]
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def block_control_names(layers=LAYERS):
    return sorted(f"unchanged_layer{l}_Block_" + n for l in layers for n in
        ("input_hidden", "incoming_pre", "attention_input", "MoE_input", "gate_logits", "gate_indices", "gate_weights", "output", "next_pre"))


def validate_goldens(report):
    from .attention_memory import validate_qualification
    from .checkpoint import MODEL, REVISION
    from .topology_validation import original_control_names
    from .block_pair import PRIMARY_CAP, HELPER_CAP
    from .block_pair_fixture import validate_plan
    layers = validate_block_pair(tuple(report.get("layers", [])))
    qualified_identity, qualified_goldens = QUALIFIED_PAIRS[layers]
    if (report.get("probe"), report.get("status"), report.get("model"), report.get("revision"), report.get("layers")) != (
            "consecutive_complete_block_parity", "pass", MODEL, REVISION, list(layers)): raise ValueError("Consecutive-block report identity differs")
    if not qualified_identity or identity(report) != qualified_identity or report.get("source_fixture_compiler_identity_sha256") != qualified_identity:
        raise ValueError("Unqualified consecutive-block source/fixture/compiler identity")
    if not qualified_goldens or golden_digest(report) != qualified_goldens or report.get("complete_goldens_sha256") != qualified_goldens:
        raise ValueError("Incomplete consecutive-block original golden coverage")
    validate_plan(report["plan"]); validate_qualification(report["attention_reference_qualification"])
    if report["boundary_names"] != registry(report["plan"]): raise ValueError("Pair boundary registry differs")
    rows = report.get("pair_cases", [])
    if report["plan"]["layers"] != list(layers) or [r["meta"] for r in rows] != list(expected_metadata(layers)): raise ValueError("Consecutive-block corpus differs")
    for r in rows:
        planned = plan_case(report["plan"], r["meta"])
        named = [report["boundary_names"][b[0]] for b in r["boundaries"]]
        if (r.get("status") != "pass" or r.get("failures") or len(set(named)) != len(named)
                or set(named) != names_for(r["meta"]["position"], r["decisions"], layers) or r["decisions"] != planned["decisions"]
                or r["handoff"] != {"hidden_sha256": planned["handoff_hidden_sha256"], "pre_sha256": planned["handoff_pre_sha256"]}
                or r.get("consumer_reused_source_indices") is not True or r.get("consumer_reused_source_KV") is not True
                or r.get("index_query_layers") != ([layers[0]] if (r["meta"]["position"] + 1) // role(layers[0]).ratio else []) or r["committed_position"] != r["meta"]["position"] + 1
                or r["actual_layout_sha256"] != r["expected_layout_sha256"] or r["actual_boundary_root_sha256"] != r["expected_boundary_root_sha256"]
                or any(b[2] != 0 or len(b[3]) != 64 for b in r["boundaries"])):
            raise ValueError("Consecutive-block exact boundaries, handoff or actual routing differ")
        root = hashlib.sha256(json.dumps([[b[0], b[3]] for b in r["boundaries"]], separators=(",", ":")).encode()).hexdigest()
        if root != r["expected_boundary_root_sha256"]: raise ValueError("Pair golden boundary root differs")
    for key, names in (("unchanged_Block_forward_controls", block_control_names(layers)), ("unchanged_Attention_forward_controls", original_control_names(layers))):
        controls = report.get(key, [])
        if len(controls) != len(rows) or any(len(c) != len(names) or sorted(x["check"] for x in c) != names or any(x["status"] != "pass" or x["mismatches"] != 0 for x in c) for c in controls):
            raise ValueError("Unchanged original consecutive-block controls differ")
    states = report.get("whole_pair_state_controls", [])
    if len(states) != len(rows) or any(c.get("case") != r["meta"]["id"] or c.get("status") != "pass" or c.get("expected_sha256") != r["committed_state_sha256"] or c.get("actual_sha256") != c.get("expected_sha256") for c, r in zip(states, rows)):
        raise ValueError("Whole-pair original state controls differ")
    faults = report.get("faults", [])
    expected = ((0, 0), (3, 0), (6, 0), (6, 0), (6, 3), (6, 6), (6, 6), (6, 6))
    if ([f["name"] for f in faults] != list(fault_names(layers)) or any(f.get("status") != "pass" or not all(f.get(k) is True for k in
            ("no_pair_output_returned", "both_block_states_and_position_unchanged", "temporary_GPU_buffers_released"))
            or tuple(f["completed_experts_in_failed_attempt"][str(l)] for l in layers) != e or f.get("source_block_completed_before_failure") is not bool(e[0] == 6)
            for f, e in zip(faults, expected))): raise ValueError("Whole-pair atomic failure evidence differs")
    events = report.get("snapshot_events", [])
    if (len(events) != 1 or events[0].get("status") != "pass" or events[0].get("consumed_foreign_expired_handle_refusals") != 4
            or not all(events[0].get(k) is True for k in ("both_states_restored", "all_pair_boundaries_reproduced", "diagnostic_copy_isolated"))):
        raise ValueError("Whole-pair snapshot evidence differs")
    negatives = report.get("negative_controls", [])
    negative_names = ["replace_transferred_pre_with_initial_pre", "bypass_source_block_hidden_output", "replace_shared_KV_with_consumer_window", "drop_reused_source_selected_indices"]
    if layers[0] == 20: negative_names.append("clear_source20_candidates")
    if [c["check"] for c in negatives] != negative_names or any(c.get("status") != "fail" or c.get("mismatches", 0) <= 0 for c in negatives):
        raise ValueError("Consecutive handoff/sharing negative controls differ")
    memory, q = report["memory"], report["attention_reference_qualification"]["memory"]
    expected_operators = {*(f"projection_{l}" for l in layers), "normalization", "compression", "indexer", "compression_normalization", "indexer_normalization", "attention"}
    expected_moe = {"parent_reservation_bytes", "accumulator_reservation_bytes", "cache_reservation_bytes", "primary_explicit_live_bytes", "helper_explicit_live_bytes"}
    if (any(type(m.get("minimum_sampled_host_available_bytes")) is not int or m["minimum_sampled_host_available_bytes"] < 32 * GiB for m in (memory, q))
            or memory.get("combined_native_host_allowance_bytes") != NATIVE_HOST_MIB * MiB or memory.get("additional_reference_host_allowance_bytes") != REFERENCE_HOST_MIB * MiB
            or memory.get("retained_fixture_host_allowance_bytes") != FIXTURE_HOST_MIB * MiB or memory.get("all_frame_reservations_released") is not True
            or memory["pair_active_stats"].get("primary_cap_bytes") != PRIMARY_CAP or memory["pair_active_stats"].get("helper_cap_bytes") != HELPER_CAP
            or not 0 < memory["pair_active_stats"]["peak_owned_primary_bytes"] <= PRIMARY_CAP or not 0 < memory["pair_active_stats"]["peak_owned_helper_bytes"] <= HELPER_CAP
            or set(memory["operator_cleanup"]) != expected_operators or set(memory["HC_cleanup"]) != {str(l) for l in layers} or set(memory["moe_cleanup"]) != {str(l) for l in layers}
            or any(set(c) != expected_moe for c in memory["moe_cleanup"].values())
            or any(c["device_owned_bytes"] or c["host_reserved_bytes"] or c["owned_context_active"] for c in memory["operator_cleanup"].values())
            or any(c["device_owned_bytes"] or c["host_reserved_bytes"] or c["owned_context_active"] for c in memory["HC_cleanup"].values())
            or any(any(c.values()) for c in memory["moe_cleanup"].values()) or memory["pair_cleanup"]["host_reserved_bytes"]):
        raise ValueError("Consecutive-block resource caps, headroom or cleanup evidence differs")
    return {r["meta"]["id"]: r for r in rows}
