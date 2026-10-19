"""Complete source-2 block corpus, atomic failures and framework-free replay."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .attention_validation import array_digest
from .block_validation import block_stage_names
from .formats import bf16_bits
from .long_context_validation import compact_comparison, replay_comparison
from .resources import GiB, MiB, physical_memory
from .topology_validation import boundary_names as attention_names, flatten, state_digest, owned_clean

FAULTS = ("host_pressure", "index_key_download", "MoE_after_three_experts", "final_residual_download",
          "combined_device_cap", "publication_pressure")
ORIGIN = "generated_independent_HC_lanes_and_incoming_coefficients_not_a_full_model_activation"
# Sealed after all 32 original/native complete-block comparisons passed.
QUALIFIED_IDENTITY = "d971ac25aac2d85b366d4bc7f8eeda842cc5b2f8263ac9b6da913639f85dcd0a"
QUALIFIED_COVERAGE = "f167c9544d6feeaaa738bb4c9e6fdb98f11e97288dd4d1f0298f10979eabe0b4"


def block_inputs(position):
    if type(position) is not int or not 0 <= position < 4: raise ValueError("Unregistered block input position")
    rng = np.random.default_rng(8100 + position)
    hidden = bf16_bits(rng.standard_normal((1, 1, 4, 5120)).astype(np.float32) * np.float32(0.125))
    pre = rng.uniform(0.125, 0.875, (1, 1, 4)).astype(np.float32)
    return hidden, pre


def meta(identity, position, image_span):
    hidden, pre = block_inputs(position)
    return {"id": identity, "layer": 2, "position": position, "sequence": 1, "image_span": image_span,
        "origin": ORIGIN, "hidden_sha256": array_digest(hidden), "incoming_pre_sha256": array_digest(pre)}


def expected_metadata():
    for repeat in (0, 1):
        for image in (False, True):
            for pos in range(4): yield meta(f"block/repeat{repeat}/image{int(image)}/p{pos}", pos, image)
    for i, mode in enumerate(FAULTS):
        for label, pos in (("warm", 0), ("retry", 1)): yield meta(f"fault/{mode}/{label}", pos, bool(i % 2))
    for label, pos in (("warm", 0), ("advance", 1), ("advance-again", 2), ("after-rollback", 1)):
        yield meta("snapshot/" + label, pos, False)


def load_arrays(topology, fixtures, *, layer=2):
    from .block_fixture import load_block_fixture
    from .topology_fixture import load_layer
    if type(layer) is not int or layer not in (2, 20): raise ValueError("Unregistered complete-block source")
    arrays, attention = load_layer(Path(topology) / str(layer), layer)
    control, hc = load_block_fixture(Path(fixtures) / "hc-norm", layer=layer)
    arrays.update(control)
    return arrays, {"attention": attention, "hc_norm": hc}


def open_operators(stack, arrays, bundle, ordinal=0, *, layer=2):
    from .attention import NativeAttention
    from .compression import NativeCompression
    from .hc import NativeHC
    from .indexer import NativeIndexer
    from .normalization import NativeNormRoPE
    from .projections import NativeProjections
    if type(layer) is not int or layer not in (2, 20): raise ValueError("Unregistered complete-block source")
    hc = stack.enter_context(NativeHC(arrays, ordinal, layer=layer))
    p = stack.enter_context(NativeProjections(arrays, ordinal, layer=layer))
    norm = stack.enter_context(NativeNormRoPE(ordinal))
    comp = stack.enter_context(NativeCompression(arrays, ordinal, layers=(layer,)))
    index = stack.enter_context(NativeIndexer(arrays, ordinal, layers=(layer,)))
    attn = stack.enter_context(NativeAttention(bundle, ordinal, profile="long_context"))
    return hc, ({layer: p}, norm, comp, index, attn)


def open_block(stack, arrays, hc, ops, store, fixtures, ordinal=0):
    from dataclasses import replace
    from .block import NativeBlock
    from .connected import ConnectedMoE, ConnectedConfig
    root, (ps, norm, comp, index, attn) = Path(fixtures), ops
    config = ConnectedConfig()
    config = replace(config, cache=replace(config.cache, primary_device=ordinal, helper_device=1 if ordinal == 0 else 0))
    moe = stack.enter_context(ConnectedMoE(store, root / "gate/native-gate.npz", root / "router/native-router.npz",
        root / "shared/native-shared.npz", config, layer=2))
    return stack.enter_context(NativeBlock(arrays, hc, norm, ps[2], attn, moe, layer=2, compressor=comp, indexer=index))


def frame_values(result, state):
    return {**result, **{"state/" + n: a for n, a in flatten(state).items()}}


def expected_names(position, ids):
    base = {n for n in block_stage_names(ids) if not n.startswith("attention_")}
    base.update("attention_" + n for n in attention_names((2,), {"position": position, "sequence": 1}))
    base.update("state/" + n for n in ("windows/2/", "compressors/2/0/", "compressors/2/1/", "compressors/2/2/",
        "index_keys/2/", "sharing/indices/"))
    return base


def registry(plan):
    return sorted(set().union(*(expected_names(c["position"], c["indices"]) for c in plan["cases"])))


def fixture_identity(fixtures):
    from .topology_validation import fixture_identities
    return {name: fixture_identities([value])[0] for name, value in fixtures.items()}


def plan_case(plan, m):
    return next(c for c in plan["cases"] if (c["position"], c["image_span"]) == (m["position"], m["image_span"]))


def exercise(block, ops, plan, checker, reset_reference, lifecycle, progress=None):
    rows, faults, events = [], [], []
    minimum = physical_memory()["available"]
    moe_resident = {n: sum(d.allocations.values()) for n, d in block.moe.devices.items()}
    def clean():
        return (owned_clean(ops) and sum(block.hc.device.allocations.values()) == block.hc.resident_bytes
            and {n: sum(d.allocations.values()) for n, d in block.moe.devices.items()} == moe_resident)
    def record(m):
        nonlocal minimum
        hidden, pre = block_inputs(m["position"])
        actual = block.step(hidden, pre, m["image_span"], start_pos=m["position"])
        state = block.attention.inspect()["state"]
        values, planned = frame_values(actual, state), plan_case(plan, m)
        ids = actual["moe_gate.indices"].tolist()
        weights = actual["moe_gate.weights"].view(np.uint32).tolist()
        if (ids != planned["indices"] or weights != planned["weight_bits"]
                or array_digest(actual["ffn_norm_output_bf16"].reshape(5120)) != planned["moe_input_sha256"]
                or array_digest(actual["moe_gate.logits"]) != planned["logits_sha256"]
                or set(values) != expected_names(m["position"], ids)):
            raise RuntimeError("Complete block input, actual route or boundary coverage differs from its import plan")
        checked = checker(m, hidden, pre, values, state)
        rows.append({"meta": m, **checked, "selected_expert_ids": ids, "routing_weight_bits": weights,
            "committed_state_sha256": state_digest(state), "committed_position": block.position,
            "selected_compressed_width": actual["attention_layer2_selected_indices_i32"].shape[-1]})
        if not clean():
            raise RuntimeError("Complete block leaked temporary GPU allocations")
        minimum = min(minimum, physical_memory()["available"])
        if progress: progress({"case": m["id"], "status": checked["status"], "position": block.position})
        return values
    for repeat in (0, 1):
        for image in (False, True):
            block.reset(); reset_reference()
            for pos in range(4): record(meta(f"block/repeat{repeat}/image{int(image)}/p{pos}", pos, image))
    for i, mode in enumerate(FAULTS):
        block.reset(); reset_reference()
        image = bool(i % 2)
        record(meta(f"fault/{mode}/warm", 0, image))
        before = block.attention.inspect()
        moe_calls_before = block.moe.stats["calls_started"]
        originals = block.budget.acquire, block.indexer.device.download, block.moe.experts.forward, block.hc.device.download, block.moe.forward, block.attention.budget.acquire
        post = block.hc.post
        count = 0
        def refuse(*a, **k): raise MemoryError("Injected block headroom refusal")
        def download_failure(*a, **k): raise OSError("Injected block diagnostic download failure")
        def expert_failure(*a, **k):
            nonlocal count
            count += 1
            if count == 4: raise OSError("Injected failure after three completed original experts")
            return originals[2](*a, **k)
        def final_post(*a, **k):
            nonlocal count
            count += 1
            if count == 2: block.hc.device.download = download_failure
            return post(*a, **k)
        def after_moe(*a, **k):
            result = originals[4](*a, **k)
            if mode == "combined_device_cap": block.hc.device.allocate(512 * MiB)
            else: block.attention.budget.acquire = refuse
            return result
        if mode == "host_pressure": block.budget.acquire = refuse
        elif mode == "index_key_download": block.indexer.device.download = download_failure
        elif mode == "MoE_after_three_experts": block.moe.experts.forward = expert_failure
        elif mode == "final_residual_download": block.hc.post = final_post
        else: block.moe.forward = after_moe
        caught = None
        try:
            h, p = block_inputs(1)
            block.step(h, p, image, start_pos=1)
        except (OSError, MemoryError) as e: caught = type(e).__name__
        finally:
            (block.budget.acquire, block.indexer.device.download, block.moe.experts.forward,
             block.hc.device.download, block.moe.forward, block.attention.budget.acquire) = originals
            block.hc.post = post
        after = block.attention.inspect()
        unchanged = state_digest(before) == state_digest(after)
        cleaned = clean()
        completed = (len(block.moe.last_attempt["completed_expert_ids"])
            if block.moe.stats["calls_started"] > moe_calls_before else 0)
        passed = caught is not None and unchanged and cleaned and (mode != "MoE_after_three_experts" or completed == 3)
        faults.append({"name": mode, "status": "pass" if passed else "fail", "error_type": caught,
            "no_output_returned": caught is not None, "all_committed_state_unchanged": unchanged,
            "temporary_GPU_buffers_released": cleaned, "completed_experts_before_failure": completed})
        if not passed: raise RuntimeError("Complete block failed atomic refusal/cleanup")
        record(meta(f"fault/{mode}/retry", 1, image))
    block.reset(); reset_reference()
    record(meta("snapshot/warm", 0, False))
    saved_state = block.attention.inspect()
    handle = block.attention.snapshot(); lifecycle("snapshot")
    first = record(meta("snapshot/advance", 1, False))
    record(meta("snapshot/advance-again", 2, False))
    block.attention.rollback(handle); lifecycle("rollback")
    restored = state_digest(saved_state) == state_digest(block.attention.inspect())
    again = record(meta("snapshot/after-rollback", 1, False))
    identical = state_digest(first) == state_digest(again)
    rejected = 0
    for action in (block.attention.rollback, block.attention.discard):
        try: action(handle)
        except ValueError: rejected += 1
    event = {"name": "rollback_replays_complete_block", "status": "pass" if restored and identical and rejected == 2 else "fail",
        "state_restored": restored, "all_replayed_boundaries_identical": identical, "consumed_handle_refusals": rejected}
    events.append(event)
    return rows, faults, events, minimum


def identity(report):
    fields = ("model", "revision", "fixtures", "moe_fixtures", "plan", "native_profiles", "hc_profile",
        "reference_identity", "original_methods", "hc_instrumentation", "attention_reference_qualification")
    values = {k: report[k] for k in fields}
    # Available RAM is an observation, not compiler/source identity. Retain it
    # in the report and validate its floor separately; do not seal its value.
    q = report["attention_reference_qualification"]
    values["attention_reference_qualification"] = {**q, "memory": {k: v for k, v in q["memory"].items()
        if k != "minimum_sampled_host_available_bytes"}}
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def coverage(report):
    return hashlib.sha256(json.dumps([[r["meta"], r["expected_layout_sha256"],
        [[c[0], c[1], c[3]] for c in r["boundaries"]], r["selected_expert_ids"], r["routing_weight_bits"], r["committed_state_sha256"]]
        for r in report["block_cases"]], sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_goldens(report):
    from .checkpoint import MODEL, REVISION
    from .attention_memory import validate_qualification
    if (report.get("probe"), report.get("status"), report.get("model"), report.get("revision"), report.get("layer")) != (
            "native_compressed_block_parity", "pass", MODEL, REVISION, 2): raise ValueError("Complete compressed block report identity differs")
    if not QUALIFIED_IDENTITY or identity(report) != QUALIFIED_IDENTITY or report.get("qualification_identity_sha256") != QUALIFIED_IDENTITY:
        raise ValueError("Unqualified complete-block original/compiler/fixture identity")
    if not QUALIFIED_COVERAGE or coverage(report) != QUALIFIED_COVERAGE: raise ValueError("Incomplete compressed-block corpus/shape/state coverage")
    validate_qualification(report["attention_reference_qualification"])
    observed = report["attention_reference_qualification"]["memory"].get("minimum_sampled_host_available_bytes")
    if type(observed) is not int or observed < 32 * GiB:
        raise ValueError("Original Attention qualification did not retain the host-memory reserve")
    if report["boundary_names"] != registry(report["plan"]): raise ValueError("Compressed block boundary registry differs")
    rows = report.get("block_cases", [])
    if [r["meta"] for r in rows] != list(expected_metadata()): raise ValueError("Complete block corpus differs")
    for r in rows:
        named = [report["boundary_names"][c[0]] for c in r["boundaries"]]
        if (r.get("status") != "pass" or r.get("failures") or len(named) != len(set(named))
                or set(named) != expected_names(r["meta"]["position"], r["selected_expert_ids"])
                or any(c[2] != 0 or len(c[3]) != 64 for c in r["boundaries"])
                or r["actual_layout_sha256"] != r["expected_layout_sha256"]
                or r["actual_boundary_root_sha256"] != r["expected_boundary_root_sha256"]
                or r["committed_position"] != r["meta"]["position"] + 1): raise ValueError("Complete block boundary parity differs")
        root = hashlib.sha256(json.dumps([[c[0], c[3]] for c in r["boundaries"]], separators=(",", ":")).encode()).hexdigest()
        if root != r["expected_boundary_root_sha256"]: raise ValueError("Compressed block golden root differs")
        p = plan_case(report["plan"], r["meta"])
        if r["selected_expert_ids"] != p["indices"] or r["routing_weight_bits"] != p["weight_bits"]: raise ValueError("Complete block actual routing differs")
    controls = report.get("unchanged_Block_forward_controls", [])
    block_controls = {"unchanged_Block_forward_" + n for n in ("output_BF16", "next_pre_FP32", "window_KV", "compressed_KV", "index_keys", "selected_indices", "kv_state", "score_state")}
    if len(controls) != len(rows) or any(len(c) != 8 or {x["check"] for x in c} != block_controls or any(x["status"] != "pass" or x["mismatches"] != 0 for x in c) for c in controls):
        raise ValueError("Unchanged complete Block.forward controls differ")
    attention = report.get("unchanged_Attention_forward_controls", [])
    from .topology_validation import original_control_names
    if len(attention) != len(rows) or any(len(c) != 12 or sorted(x["check"] for x in c) != original_control_names((2,))
            or any(x["status"] != "pass" or x["mismatches"] != 0 for x in c) for c in attention):
        raise ValueError("Unchanged complete Attention.forward controls differ")
    states = report.get("whole_committed_state_controls", [])
    if len(states) != len(rows) or any(c.get("case") != r["meta"]["id"] or c.get("status") != "pass"
            or c.get("expected_sha256") != r["committed_state_sha256"] or c.get("actual_sha256") != c.get("expected_sha256") for c, r in zip(states, rows)):
        raise ValueError("Whole committed block state controls differ")
    faults = report.get("faults", [])
    if [f["name"] for f in faults] != list(FAULTS) or any(f.get("status") != "pass" or not all(f.get(k) is True for k in
            ("no_output_returned", "all_committed_state_unchanged", "temporary_GPU_buffers_released")) for f in faults):
        raise ValueError("Complete block fault evidence differs")
    if [f.get("completed_experts_before_failure") for f in faults] != [0, 0, 3, 6, 6, 6]:
        raise ValueError("Completed expert counts must describe the failed attempt, not the preceding warm-up")
    events = report.get("snapshot_events", [])
    if (len(events) != 1 or events[0].get("status") != "pass" or events[0].get("consumed_handle_refusals") != 2
            or events[0].get("state_restored") is not True or events[0].get("all_replayed_boundaries_identical") is not True):
        raise ValueError("Complete block snapshot evidence differs")
    negatives = report.get("negative_controls", [])
    if [n["check"] for n in negatives] != ["use_FFN_own_pre_for_current_FFN", "omit_compressed_KV_in_block_Attention"] or any(n.get("status") != "fail" or n.get("mismatches", 0) <= 0 for n in negatives):
        raise ValueError("Complete block negative controls differ")
    memory = report["memory"]
    if (type(memory.get("minimum_sampled_host_available_bytes")) is not int
            or memory["minimum_sampled_host_available_bytes"] < 32 * GiB
            or memory.get("combined_native_host_allowance_bytes") != 3584 * MiB
            or memory.get("retained_fixture_host_allowance_bytes") != 768 * MiB
            or memory.get("additional_reference_host_allowance_bytes") != 2048 * MiB
            or memory["block_active_stats"].get("primary_device_cap_bytes") != 512 * MiB
            or not 0 < memory["block_active_stats"].get("peak_owned_primary_device_bytes", 0) <= 512 * MiB):
        raise ValueError("Complete block memory admission, observation or GPU-cap evidence differs")
    if (memory.get("all_frame_reservations_released") is not True or any(c["device_owned_bytes"] or c["host_reserved_bytes"] or c["owned_context_active"]
            for c in memory["operator_cleanup"].values()) or any(memory["moe_cleanup"].values())
            or memory["block_cleanup"]["host_reserved_bytes"] or memory["HC_cleanup"]["device_owned_bytes"] or memory["HC_cleanup"]["host_reserved_bytes"]):
        raise ValueError("Complete block cleanup evidence differs")
    return {r["meta"]["id"]: r for r in rows}
