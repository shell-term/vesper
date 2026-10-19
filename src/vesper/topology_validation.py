"""Sharing-role corpus and full compact evidence, independent of Torch."""

import copy
import hashlib
import json

import numpy as np

from .attention_topology import GROUPS, role
from .attention_validation import array_digest, check_arrays
from .indexer_validation import bounded_inputs
from .long_context_validation import compact_comparison, replay_comparison
from .resources import physical_memory
from .shared_attention import CAPACITY, SharedAttentionFrame, initial_state

FAULTS = ("host_pressure", "index_key_download", "consumer_attention", "consumer_output", "publication_pressure")
SOURCE_FIELDS = ("reference", "blas_reference", "topk_reference", "reduction_reference", "index_reference", "softmax_reference",
    "original_methods_ast_sha256", "original_wo_a_conversion_ast_sha256", "original_index_quant_cuda_sha256",
    "original_compression_quant_cuda_sha256", "original_index_query_trace_cuda_sha256", "torch_git_version", "tilelang_version")
SOURCE_IDENTITY = "6a7bcc8b16dffdfee2b8ed1caa18523710da52dc3914629d57954c418ebbac51"
IDENTITIES = {
    GROUPS[0]: "02610b54ecc9a35ea13a6a3ee7de89aef04e60aa11691d072b626c1b23119797",
    GROUPS[1]: "75307d32c0765b984846dd7b567051c48100f4ae2ca3f273d5e9bc4e25b0b74c",
    GROUPS[2]: "fb3338b939fc32fd1e45b5e945205944636d5b15f5c47540d82a7acb32b4e032",
    GROUPS[3]: "e91153f971d1980eccc4b2766eef13214f7f1c4eb7719f65e48141365b39a85b",
}
COVERAGE = {g: "f32f2a94b1b8240db15bed09796399499c29877b414b88247e5c16dfb85f4cf0" for g in GROUPS[:3]}
COVERAGE[GROUPS[3]] = "31a7fa3e82eb4e0cc52193d6b2a522b4b192a45605641799ce3d820fffc03d31"


def inputs(layers, batch, sequence, position, seed):
    return {l: bounded_inputs((batch, sequence, 5120), seed + l * 101 + position * 17) for l in layers}


def meta(layers, identity, batch, position, sequence, seed, initial=0):
    xs = inputs(layers, batch, sequence, position, seed)
    return {"id": identity, "layers": list(layers), "batch": batch, "position": position, "sequence": sequence, "seed": seed,
            "initial_diagnostic_position": initial, "capacity": CAPACITY, "input_sha256": {str(l): array_digest(x) for l, x in xs.items()}}


def configurations():
    for batch in (1, 2):
        for prefill in (1, 2, 3): yield f"short/b{batch}/s{prefill}", batch, 0, prefill, 2, 7100 + batch * 10 + prefill
        yield f"continuous/b{batch}", batch, 0, 3, 257, 7200 + batch
        yield f"primed/b{batch}", batch, 1024, 1, 3, 7300 + batch


def expected_metadata(layers):
    for identity, batch, start, sequence, decodes, seed in configurations():
        for i, (pos, s) in enumerate([(start, sequence)] + [(start + sequence + n, 1) for n in range(decodes)]):
            yield meta(layers, identity + f"/{i}", batch, pos, s, seed, start)
    for mode in FAULTS:
        for label, pos in (("warm", 1024), ("retry", 1025)):
            yield meta(layers, f"fault/{mode}/{label}", 1, pos, 1, 7400, 1024)
    for label, pos in (("warm", 1024), ("advance", 1025), ("advance-again", 1026), ("after-rollback", 1025)):
        yield meta(layers, "snapshot/" + label, 2, pos, 1, 7500, 1024)


def flatten(value, prefix=""):
    if isinstance(value, np.ndarray): return {prefix: value}
    result = {}
    if isinstance(value, (dict, tuple)):
        for k, v in (value.items() if isinstance(value, dict) else enumerate(value)):
            result.update(flatten(v, prefix + str(k) + "/"))
    return result


def state_digest(value):
    def encode(v):
        if isinstance(v, np.ndarray): return {"dtype": v.dtype.str, "shape": list(v.shape), "sha256": array_digest(v)}
        if isinstance(v, dict): return {str(k): encode(a) for k, a in v.items()}
        if isinstance(v, tuple): return [encode(a) for a in v]
        return v
    return hashlib.sha256(json.dumps(encode(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def make_seed(arrays, ops, layers, batch, position, seed):
    ps, norm, comp, index, _ = ops
    state, banks = initial_state(layers, batch), {}
    for l in layers:
        if role(l).owns_kv:
            r = role(l).ratio
            x = bounded_inputs((batch, 16, 5120), seed + l + 8000)
            p = comp.project(x, l)
            pooled = comp.pool(p["wkv_fp32"].reshape(batch, 8, 2, 512), p["wgate_fp32"].reshape(batch, 8, 2, 512))["pooled_bf16"] if r == 2 else p["wkv_bf16"]
            latent = norm.norm(pooled, arrays[f"layers.{l}.attn.compressor.norm.weight"])["output_bf16"]
            positions = np.arange(latent.shape[1], dtype=np.int32) * r
            keys = index.key(latent, l, positions)["quant_rounded_bf16"]
            kv = comp.round_kv(comp.norm.rope(latent, 0, "compressed-yarn", positions=positions)["output_bf16"])["rounded_bf16"]
            n = position // r
            ids = np.arange(n) % latent.shape[1]
            state["index_keys"][l][:, :n] = keys[:, ids]
            state["compressors"][l][2][:, :n] = kv[:, ids]
            banks.update({f"{l}/latent": latent, f"{l}/keys": keys, f"{l}/compressed_KV": kv})
        s = 8 // batch
        x = bounded_inputs((batch, s, 5120), seed + l + 9000)
        kv = ps[l].fp8("wkv", x)["output_bf16"]
        kv = norm.norm(kv, arrays[f"layers.{l}.attn.kv_norm.weight"])["output_bf16"]
        kv = norm.rope(kv, 0, "compressed-yarn")["output_bf16"]
        window = ps[l].round_window(kv)["rounded_bf16"]
        state["windows"][l][:] = np.tile(window, (1, 128 // s, 1))
        banks[f"{l}/window"] = window
    return state, {"batch": batch, "position": position, "seed": seed, "capacity": CAPACITY,
        "bank_sha256": {n: array_digest(a) for n, a in banks.items()}, "state_sha256": {n: array_digest(a) for n, a in flatten(state).items()},
        "scope": "Explicit synthetic original-component history, not a full-model autoregressive past"}


def boundary_names(layers, m):
    from .compressed_attention_validation import boundary_names as legacy_names
    from .compression_validation import state_stage_names
    from .indexer_validation import key_names, query_names, SCORE_NAMES, selection_names
    base = {n.removeprefix("layer24_") for n in legacy_names(m) if n.startswith("layer24_")}
    names = set()
    pos, seq = m["position"], m["sequence"]
    for l in layers:
        r, p = role(l), f"layer{l}_"
        names.update(p + n for n in base)
        names.add(p + "selected_indices_i32")
        n = (pos + seq) // r.ratio
        emits = seq >= r.ratio if pos == 0 else (pos + 1) % r.ratio == 0
        if r.owns_kv:
            names.update(p + "compressor_" + k for k in state_stage_names(2 if r.ratio == 2 else 20, pos, seq))
            names.add(f"owner{l}_index_cache_bf16")
            if n and emits: names.update(p + "index_key_" + k for k in key_names())
        if r.owns_index and n:
            names.add(p + "index_keys_bf16")
            names.update(p + "index_query_" + k for k in query_names())
            names.update(p + "index_score_" + k for k in SCORE_NAMES)
            names.update(p + "index_selection_" + k for k in selection_names("source" if l == 20 else ("consumer" if l > 20 else "plain")))
    return names


def registry(layers):
    return sorted(boundary_names(layers, {"position": 0, "sequence": 3}) | boundary_names(layers, {"position": 2, "sequence": 1}))


def observe(layers, stages, m):
    queries = [l for l in layers if f"layer{l}_index_query_projection_bf16" in stages]
    expected = [l for l in layers if role(l).owns_index and (m["position"] + m["sequence"]) // role(l).ratio]
    reused = {str(l): role(l).index_owner for l in layers if not role(l).owns_index}
    reuse_matches = all(np.array_equal(stages[f"layer{l}_selected_indices_i32"], stages[f"layer{owner}_selected_indices_i32"]) for l, owner in ((int(k), v) for k, v in reused.items()))
    return {"index_query_layers": queries, "expected_index_query_layers": expected,
            "reused_index_owners": reused, "reused_indices_identical": reuse_matches,
            "KV_owners": {str(l): role(l).kv_owner for l in layers},
            "window_wrap": bool(m["position"] and m["position"] % 128 == 0),
            "selected_widths": {str(l): int(stages[f"layer{l}_selected_indices_i32"].shape[-1]) for l in layers}}


def owned_clean(ops):
    ps, norm, comp, index, attn = ops
    return (all(sum(p.device.allocations.values()) == p.resident_bytes for p in ps.values())
        and not norm.device.allocations and sum(comp.device.allocations.values()) == comp.resident_bytes
        and sum(index.device.allocations.values()) == index.resident_bytes and not comp.norm.device.allocations
        and not index.norm.device.allocations and not attn.device.allocations)


def exercise(layers, arrays, ops, checker, reset_reference, seed_reference, lifecycle, progress=None):
    ps, norm, comp, index, attn = ops
    rows, faults, events, seeds, minimum = [], [], [], [], physical_memory()["available"]
    def prime(frame, identity, batch, seed):
        state, m = make_seed(arrays, ops, layers, batch, 1024, seed)
        seed_reference(identity, state, m)
        frame.prime_diagnostic(state, 1024)
        seeds.append({"id": identity, "meta": m})
    def record(frame, m):
        nonlocal minimum
        xs = inputs(layers, m["batch"], m["sequence"], m["position"], m["seed"])
        actual = frame.step(xs, start_pos=m["position"])
        check = checker(m, xs, actual)
        observed = observe(layers, actual, m)
        if observed["index_query_layers"] != observed["expected_index_query_layers"] or not observed["reused_indices_identical"]:
            raise RuntimeError("Native index computation/reuse role differs")
        rows.append({"meta": m, **check, "observations": observed})
        if not owned_clean(ops): raise RuntimeError("Sharing frame leaked native scratch")
        minimum = min(minimum, physical_memory()["available"])
        if progress and (len(rows) % 64 == 0 or m["id"].startswith(("fault/", "snapshot/"))): progress({"layers": list(layers), "frames": len(rows), "case": m["id"], "status": check["status"]})
        return actual
    for identity, batch, start, sequence, decodes, seed in configurations():
        reset_reference(batch)
        with SharedAttentionFrame(arrays, *ops, layers, batch) as frame:
            if start: prime(frame, identity, batch, seed)
            for i, (pos, s) in enumerate([(start, sequence)] + [(start + sequence + n, 1) for n in range(decodes)]):
                record(frame, meta(layers, identity + f"/{i}", batch, pos, s, seed, start))
        if frame.budget.reserved: raise RuntimeError("Sharing frame reservation leaked")
    for mode in FAULTS:
        reset_reference(1)
        with SharedAttentionFrame(arrays, *ops, layers) as frame:
            prime(frame, "fault/" + mode, 1, 7400)
            record(frame, meta(layers, f"fault/{mode}/warm", 1, 1024, 1, 7400, 1024))
            before = frame.inspect()
            target, attribute = frame.budget, "acquire"
            if mode == "index_key_download": target, attribute = index.device, "download"
            elif mode == "consumer_output": target, attribute = ps[layers[-1]], "fp8"
            original, calls = getattr(target, attribute), 0
            def injected(*a, **k):
                nonlocal calls
                calls += 1
                if mode == "index_key_download": raise OSError("Injected shared key readback failure")
                if mode == "consumer_attention":
                    if calls == len(layers): raise OSError("Injected final consumer Attention failure")
                    return attn(*a, **k)
                if mode == "consumer_output":
                    value = original(*a, **k)
                    if a[0] == "wo_b": raise OSError("Injected consumer final output failure")
                    return value
                if mode != "publication_pressure" or calls == 2: raise MemoryError("Injected sharing frame pressure")
                return original(*a, **k)
            if mode == "consumer_attention":
                class FailingAttention:
                    def __call__(self, *a, **k): return injected(*a, **k)
                frame.attention = FailingAttention()
            else: setattr(target, attribute, injected)
            try:
                try: frame.step(inputs(layers, 1, 1, 1025, 7400))
                except (MemoryError, OSError): pass
                else: raise RuntimeError("Injected sharing failure was not refused")
            finally:
                frame.attention = attn
                if mode != "consumer_attention": setattr(target, attribute, original)
            after = frame.inspect()
            faults.append({"mode": mode, "state_before_sha256": state_digest(before), "state_after_sha256": state_digest(after),
                "position_preserved": before["position"] == after["position"], "temporary_buffers_released": owned_clean(ops),
                "status": "pass" if state_digest(before) == state_digest(after) and owned_clean(ops) else "fail"})
            record(frame, meta(layers, f"fault/{mode}/retry", 1, 1025, 1, 7400, 1024))
        if frame.budget.reserved: raise RuntimeError("Failed sharing frame reservation leaked")
    reset_reference(2)
    with SharedAttentionFrame(arrays, *ops, layers, 2) as frame:
        prime(frame, "snapshot", 2, 7500)
        result = record(frame, meta(layers, "snapshot/warm", 2, 1024, 1, 7500, 1024))
        before = frame.inspect()
        for a in result.values(): a.fill(0)
        isolation = state_digest(before) == state_digest(frame.inspect())
        handle = frame.snapshot()
        lifecycle("snapshot")
        record(frame, meta(layers, "snapshot/advance", 2, 1025, 1, 7500, 1024))
        record(frame, meta(layers, "snapshot/advance-again", 2, 1026, 1, 7500, 1024))
        restored = frame.rollback(handle)
        lifecycle("rollback")
        record(frame, meta(layers, "snapshot/after-rollback", 2, 1025, 1, 7500, 1024))
        refusals = 0
        try: frame.discard(handle)
        except ValueError: refusals += 1
        handle = frame.snapshot()
        try: frame.snapshot()
        except RuntimeError: refusals += 1
        frame.reset()
        try: frame.rollback(handle)
        except ValueError: refusals += 1
        events.append({"state_before_sha256": state_digest(before), "state_restored_sha256": state_digest(restored),
            "restored_position": restored["position"], "diagnostic_copy_isolated": isolation, "expired_or_duplicate_refusals": refusals,
            "status": "pass" if state_digest(before) == state_digest(restored) and isolation and refusals == 3 else "fail"})
    if frame.budget.reserved: raise RuntimeError("Sharing snapshot reservation leaked")
    return rows, faults, events, seeds, minimum


def identity_digest(report):
    profiles = copy.deepcopy(report["native_profiles"])
    for p in profiles["projections"].values(): p.pop("device", None)
    fixtures = fixture_identities(report["fixtures"])
    return hashlib.sha256(json.dumps([profiles, fixtures], sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def fixture_identities(fixtures):
    # Retrieval/reuse counters may differ when another user recreates identical bytes.
    return [{"model": f["model"], "revision": f["revision"], "layer": f["layer"], "tensors": f["tensors"]} for f in fixtures]


def coverage_digest(report):
    values = [[r["meta"]["id"], [[b[0], b[1]] for b in r["boundaries"]], r["expected_layout_sha256"]] for r in report["frame_cases"]]
    values.extend([["control", i, [[b[0], b[1]] for b in r["boundaries"]]] for i, r in enumerate(report["unchanged_complete_Attention_controls"])])
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def original_control_names(layers):
    names = {f"unchanged_layer{l}_" + n for l in layers for n in ("output", "window", "query", "combined_KV", "combined_indices", "Attention_output", "selected", "compressed_visible")}
    for l in layers:
        if role(l).owns_kv:
            names.update(f"unchanged_layer{l}_" + n for n in ("compressed_cache", "key_cache"))
            if role(l).ratio == 2: names.update(f"unchanged_layer{l}_" + n for n in ("kv_state", "score_state"))
    if 20 in layers: names.add("unchanged_source20_candidates")
    return sorted(names)


def validate_goldens(report):
    from .attention_memory import validate_qualification
    from .attention_topology import validate_group
    from .checkpoint import MODEL, REVISION
    from .long_context_validation import validate_compact_group, is_digest
    from .resources import MiB, GiB
    layers = validate_group(tuple(report.get("layers", [])))
    if (report.get("probe") != "native_attention_topology_parity" or report.get("status") != "pass"
            or report.get("model") != MODEL or report.get("revision") != REVISION
            or hashlib.sha256(json.dumps({k: report.get(k) for k in SOURCE_FIELDS}, sort_keys=True, separators=(",", ":")).encode()).hexdigest() != SOURCE_IDENTITY):
        raise ValueError("Unqualified original Attention-sharing source identity")
    validate_qualification(report.get("attention_reference_qualification"))
    if identity_digest(report) != IDENTITIES.get(layers): raise ValueError("Registered sharing fixture/compiler/conversion identity differs")
    from .checkpoint import LIMIT
    if any(f.get("kind") != "original_topology_layer_fixture" or f.get("maximum_range_bytes") != LIMIT
            or f.get("tensor_bytes") != sum(t["bytes"] for t in f["tensors"])
            or any(type(f.get(k)) is not int or f[k] < 0 for k in ("weights_fetched_bytes", "weights_reused_bytes"))
            or f["weights_fetched_bytes"] + f["weights_reused_bytes"] != f["tensor_bytes"] for f in report["fixtures"]):
        raise ValueError("Original sharing fixture retrieval scope differs")
    names, rows = registry(layers), report.get("frame_cases", [])
    if report.get("boundary_names") != names or report.get("boundary_columns") != ["name_id", "storage_elements", "mismatches", "expected_sha256"]:
        raise ValueError("Incomplete sharing boundary registry/schema")
    expected = list(expected_metadata(layers))
    if len(rows) != 556: raise ValueError("Sharing frame coverage is incomplete")
    ids = {n: i for i, n in enumerate(names)}
    for row, m in zip(rows, expected):
        if row.get("meta") != m or row.get("failures") != []: raise ValueError("Sharing input/frame metadata differs")
        validate_compact_group(row, sorted(ids[n] for n in boundary_names(layers, m)), layout=True)
        obs = row.get("observations", {})
        q = [l for l in layers if role(l).owns_index and (m["position"] + m["sequence"]) // role(l).ratio]
        if (obs.get("index_query_layers") != q or obs.get("expected_index_query_layers") != q
                or obs.get("reused_index_owners") != {str(l): role(l).index_owner for l in layers if not role(l).owns_index}
                or obs.get("reused_indices_identical") is not True or obs.get("KV_owners") != {str(l): role(l).kv_owner for l in layers}
                or obs.get("window_wrap") is not bool(m["position"] and m["position"] % 128 == 0)
                or obs.get("selected_widths") != {str(l): min(512, (m["position"] + m["sequence"]) // role(l).ratio) for l in layers}):
            raise ValueError("Original index computation/reuse or KV ownership differs")
    cn = original_control_names(layers)
    controls = report.get("unchanged_complete_Attention_controls", [])
    if report.get("original_control_names") != cn or len(controls) != 556: raise ValueError("Original full sharing controls are incomplete")
    for control in controls: validate_compact_group(control, range(len(cn)), layout=False)
    if coverage_digest(report) != COVERAGE.get(layers): raise ValueError("Registered sharing boundary sizes/layout coverage differs")
    seeds, seed_controls = report.get("diagnostic_seeds", []), report.get("original_seed_controls", [])
    specs = [("primed/b1", 1, 7301), ("primed/b2", 2, 7302)] + [("fault/" + f, 1, 7400) for f in FAULTS] + [("snapshot", 2, 7500)]
    if len(seeds) != 8 or len(seed_controls) != 8: raise ValueError("Explicit sharing seed coverage is incomplete")
    owner = layers[0]
    state_names = {f"windows/{l}/" for l in layers} | {f"compressors/{owner}/{i}/" for i in range(3)} | {f"index_keys/{owner}/"}
    banks = {f"{owner}/" + n for n in ("latent", "keys", "compressed_KV")} | {f"{l}/window" for l in layers}
    for seed, control, (identity, batch, seed_id) in zip(seeds, seed_controls, specs):
        m, checks = seed.get("meta", {}), control.get("checks", [])
        if (seed.get("id") != identity or control.get("id") != identity or control.get("meta") != m
                or (m.get("batch"), m.get("position"), m.get("seed"), m.get("capacity")) != (batch, 1024, seed_id, 1152)
                or set(m.get("state_sha256", {})) != state_names or set(m.get("bank_sha256", {})) != banks
                or any(not is_digest(h) for h in [*m["state_sha256"].values(), *m["bank_sha256"].values()])
                or len(checks) != len(state_names) or {c.get("check") for c in checks} != state_names
                or any(c.get("status") != "pass" or c.get("mismatches") != 0 or c.get("actual_sha256") != c.get("expected_sha256")
                       or c.get("expected_sha256") != m["state_sha256"].get(c["check"]) for c in checks)):
            raise ValueError("Original synthetic sharing history identity differs")
    faults, events = report.get("faults", []), report.get("snapshot_events", [])
    if ([f.get("mode") for f in faults] != list(FAULTS) or any(f.get("status") != "pass" or f.get("position_preserved") is not True
            or f.get("temporary_buffers_released") is not True or not is_digest(f.get("state_before_sha256"))
            or f.get("state_before_sha256") != f.get("state_after_sha256") for f in faults)):
        raise ValueError("Atomic sharing failure evidence differs")
    if (len(events) != 1 or events[0].get("status") != "pass" or events[0].get("restored_position") != 1025
            or events[0].get("diagnostic_copy_isolated") is not True or events[0].get("expired_or_duplicate_refusals") != 3
            or not is_digest(events[0].get("state_before_sha256")) or events[0].get("state_before_sha256") != events[0].get("state_restored_sha256")):
        raise ValueError("Sharing snapshot/copy evidence differs")
    ratio2 = owner != 20
    count = {"projection": {str(l): 2232 for l in layers}, "index_projection": 554 if ratio2 else 2780,
        "index_sum": 554 if ratio2 else 2780, "index_packed_vs_inplace": 843 if ratio2 else 3344,
        "compressed_packed_vs_inplace": 289 if ratio2 else 564}
    if report.get("instrumentation_counts") != count: raise ValueError("Original sharing diagnostic control counts differ")
    negatives = report.get("negative_controls", [])
    if ([n.get("check") for n in negatives] != ["omit_reused_compressed_indices", "replace_shared_KV_with_consumer_window", "omit_shared_index_offset"]
            or any(n.get("status") != "fail" or n.get("mismatches", 0) <= 0 for n in negatives)):
        raise ValueError("Shared-consumer negative controls are incomplete")
    memory, cleaned = report.get("memory", {}), report.get("memory", {}).get("cleanup", {})
    if (set(cleaned) != {f"projection_{l}" for l in layers} | {"normalization", "compression", "indexer", "compression_normalization", "indexer_normalization", "attention"}
            or any(c.get("device_owned_bytes") != 0 or c.get("host_reserved_bytes") != 0 or c.get("owned_context_active") is not False for c in cleaned.values())
            or memory.get("combined_native_host_allowance_bytes") != (512 * len(layers) + 3872) * MiB
            or memory.get("combined_native_and_reference_host_allowance_bytes") != (512 * len(layers) + 4960) * MiB
            or memory.get("retained_fixture_host_allowance_bytes") != 2048 * MiB or memory.get("all_frame_reservations_released") is not True
            or memory.get("state_storage") != "bounded_host_diagnostic_frame"
            or memory.get("minimum_sampled_host_available_bytes", 0) < 32 * GiB or memory.get("reservation_is_RSS_limit") is not False):
        raise ValueError("Sharing owned-resource evidence differs")
    if (report.get("weights_fetched_during_comparison_bytes") != 0 or report.get("capacity") != 1152
            or report.get("maximum_unprimed_continuous_context") != 260 or report.get("maximum_seeded_context") != 1028
            or report.get("full_model_quality") != "not_run" or report.get("end_to_end_speed") != "not_measured"
            or report.get("untouched_full_Attention_kernel") != "blocked_by_shared_memory_limit"):
        raise ValueError("Sharing qualification scope differs")
    return {r["meta"]["id"]: r for r in rows}
