"""Long-context corpus, explicit synthetic seeds and compact complete goldens."""

import hashlib
import json

import numpy as np

from .attention_validation import array_digest, check_arrays
from .compressed_attention import LAYERS, NativeCompressedAttention, initial_state
from .compressed_attention_validation import frame_inputs, metadata, boundary_names, flatten_state, inspect_checks, clean
from .formats import bf16_values
from .indexer_validation import bounded_inputs
from .resources import physical_memory

PROFILE = "long_context"
FAULTS = ("host_pressure", "index_key_download", "consumer_attention", "consumer_output", "publication_pressure")
# Filled from the qualified per-case sizes/layouts; values remain separately
# bound to every published original golden and native-only replay.
LAYOUT_COVERAGE_SHA256 = "8f0d45c11479a3b07eeaa711249519723e19d2777aa1309bc61a100606f4e1c1"


def configurations():
    yield {"id": "continuous/1152", "batch": 1, "position": 0, "prefill": 3, "decodes": 1149, "seed": 6100, "capacity": 1152}
    yield {"id": "continuous/260/batch2", "batch": 2, "position": 0, "prefill": 3, "decodes": 257, "seed": 6200, "capacity": 1152}
    yield {"id": "primed/1024/batch2", "batch": 2, "position": 1024, "prefill": 1, "decodes": 2, "seed": 6300, "capacity": 1152}
    yield {"id": "primed/16384", "batch": 1, "position": 16384, "prefill": 1, "decodes": 7, "seed": 6400, "capacity": 16392}
    yield {"id": "primed/16384/batch2", "batch": 2, "position": 16384, "prefill": 1, "decodes": 7, "seed": 6450, "capacity": 16392}


def case_meta(identity, batch, position, sequence, seed, *, initial_position=0, capacity=16392):
    return {**metadata(identity, batch, position, sequence, seed), "initial_diagnostic_position": initial_position, "cache_capacity": capacity}


def stream_cases(config):
    pos, prefill = config["position"], config["prefill"]
    for index, (p, s) in enumerate([(pos, prefill)] + [(pos + prefill + n, 1) for n in range(config["decodes"])]):
        yield case_meta(f"frame/{config['id']}/{index}", config["batch"], p, s, config["seed"], initial_position=pos, capacity=config["capacity"])


def expected_metadata():
    for config in configurations(): yield from stream_cases(config)
    for mode in FAULTS:
        for label, pos in (("warm", 16384), ("retry", 16385)):
            yield case_meta(f"fault/{mode}/{label}", 1, pos, 1, 6500, initial_position=16384)
    for label, pos in (("warm", 16384), ("advance", 16385), ("advance-again", 16386), ("after-rollback", 16385)):
        yield case_meta(f"snapshot/{label}", 2, pos, 1, 6600, initial_position=16384)


def seed_specs():
    for c in configurations():
        if c["position"]: yield c["id"], c["batch"], c["position"], c["seed"], c["capacity"]
    for mode in FAULTS: yield "fault/" + mode, 1, 16384, 6500, 16392
    yield "snapshot", 2, 16384, 6600, 16392


def make_seed(arrays, ops, batch, position, seed, *, capacity=16392):
    """Build original-weight component banks, then explicit artificial history.

    This is an adversarial diagnostic state, not a captured/autoregressive past.
    Source-20 keys include a low-20/high-24 block to make candidate pruning real.
    """
    ps, norm, comp, index, _ = ops
    state = initial_state(batch, profile=PROFILE, capacity=capacity)
    banks, diagnostics = {}, {}
    for layer in (2, 20):
        chunks = []
        for chunk in range(1 if layer == 2 else 4):
            x = bounded_inputs((batch, 16, 5120), seed + layer + 7000 + chunk)
            projected = comp.project(x, layer)
            if layer == 2:
                pooled = comp.pool(projected["wkv_fp32"].reshape(batch, 8, 2, 512), projected["wgate_fp32"].reshape(batch, 8, 2, 512))["pooled_bf16"]
            else: pooled = projected["wkv_bf16"]
            chunks.append(pooled)
        latent = comp.norm.norm(np.concatenate(chunks, axis=1), comp.arrays[f"layers.{layer}.attn.compressor.norm.weight"])["output_bf16"]
        positions = np.arange(latent.shape[1], dtype=np.int32) * (2 if layer == 2 else 1)
        keys = index.key(latent, layer, positions)["quant_rounded_bf16"]
        rotated = comp.norm.rope(latent, 0, "compressed-yarn", positions=positions)["output_bf16"]
        kv = comp.round_kv(rotated)["rounded_bf16"]
        banks[layer] = keys, kv
        diagnostics[f"bank{layer}_pre_RoPE_latent_bf16"], diagnostics[f"bank{layer}_index_keys_bf16"], diagnostics[f"bank{layer}_compressed_KV_bf16"] = latent, keys, kv
    current = frame_inputs(batch, 1, position, seed)
    scores = {}
    for layer in (20, 24):
        qr = ps[layer].fp8("wq_a", current[layer])["output_bf16"]
        qr = norm.norm(qr, arrays[f"layers.{layer}.attn.q_norm.weight"])["output_bf16"]
        q = index.queries(current[layer], qr, layer, position)
        scores[layer] = index.score(q["quant_rounded_bf16"], banks[20][0], q["weights_scaled_bf16"])["score_bf16"]
        diagnostics[f"seed_query{layer}_score_bf16"] = scores[layer]
    pairs = []
    for b in range(batch):
        s20, s24 = (bf16_values(scores[l][b, 0]).astype(np.float64) for l in (20, 24))
        gap20, gap24 = s20[None, :] - s20[:, None], s24[:, None] - s24[None, :]
        products = np.where((gap20 > 0) & (gap24 > 0), gap20 * gap24, -1)
        if products.max() <= 0: raise RuntimeError("Original seed bank lacks a source/consumer score inversion")
        a, z = np.unravel_index(int(products.argmax()), products.shape)
        pairs.append([int(a), int(z)])
        for layer, ratio in ((2, 2), (20, 1)):
            keys, kv = banks[layer]
            n = position // ratio
            if layer == 2:
                ids = np.arange(n) % keys.shape[1]
            else:
                ids = np.full(n, z, np.int64)
                ids[:8] = a
            state["index_keys"][layer][b, :n] = keys[b, ids]
            state["compressors"][layer][2][b, :n] = kv[b, ids]
    for layer in LAYERS:
        s = 8 // batch
        x = bounded_inputs((batch, s, 5120), seed + layer + 8000)
        kv = ps[layer].fp8("wkv", x)["output_bf16"]
        kv = norm.norm(kv, arrays[f"layers.{layer}.attn.kv_norm.weight"])["output_bf16"]
        kv = norm.rope(kv, 0, "compressed-yarn")["output_bf16"]
        rounded = ps[layer].round_window(kv)["rounded_bf16"]
        state["windows"][layer][:] = np.tile(rounded, (1, 128 // s, 1))
        diagnostics[f"bank{layer}_window_bf16"] = rounded
    meta = {"batch": batch, "position": position, "seed": seed, "cache_capacity": capacity, "source20_low_consumer24_high_pairs": pairs,
            "bank_sha256": {n: array_digest(a) for n, a in diagnostics.items()},
            "state_sha256": {n: array_digest(a) for n, a in flatten_state(state).items()},
            "scope": "Explicit synthetic history from original component banks, not a real autoregressive/model history"}
    return state, meta


def registry():
    return sorted(boundary_names({"position": 0, "sequence": 3}) | boundary_names({"position": 0, "sequence": 1}))


def compact_comparison(expected, actual, names):
    """Compare every boundary; store each golden once with integer name IDs."""
    if set(expected) != set(actual): raise ValueError("Long-context boundary sets differ")
    ids = {n: i for i, n in enumerate(names)}
    rows, actual_hashes, failures = [], [], []
    for name in sorted(expected):
        c = check_arrays(expected[name], actual[name], name)
        rows.append([ids[name], c["elements"], c["mismatches"], c["expected_sha256"]])
        actual_hashes.append([ids[name], c["actual_sha256"]])
        if c["status"] != "pass": failures.append(c)
    actual_root = hashlib.sha256(json.dumps(actual_hashes, separators=(",", ":")).encode()).hexdigest()
    expected_root = hashlib.sha256(json.dumps([[r[0], r[3]] for r in rows], separators=(",", ":")).encode()).hexdigest()
    def layout(values):
        return hashlib.sha256(json.dumps([[ids[n], a.dtype.str, list(a.shape)] for n, a in sorted(values.items())], separators=(",", ":")).encode()).hexdigest()
    return {"status": "pass" if not failures else "fail", "boundaries": rows,
            "expected_layout_sha256": layout(expected), "actual_layout_sha256": layout(actual), "actual_boundary_root_sha256": actual_root,
            "expected_boundary_root_sha256": expected_root, "failures": failures}


def observations(stages, meta):
    pos = meta["position"]
    selected = {str(l): int(stages[f"index_layer{l}_selection_indices_i32"].shape[-1]) for l in LAYERS}
    mask = stages["index_shared20_to24_candidates_bool"]
    n = mask.shape[-1]
    kept = mask.sum(axis=-1)
    latest = bool(mask[..., -1].all()) if n else True
    idx = stages["index_layer24_selection_indices_i32"]
    offset = meta["sequence"] if pos == 0 else 128
    valid = idx >= 0
    respected = bool(np.all(~valid | np.take_along_axis(mask, np.maximum(idx - offset, 0), axis=-1))) if n else True
    return {"window_wrap": bool(pos > 0 and pos % 128 == 0), "selected_widths": selected,
            "candidate_positions": n, "reachable_excluded_positions": (n - kept).tolist(),
            "latest_candidate_kept": latest, "consumer_respects_candidates": respected}


def replay_comparison(golden, actual, names):
    """Check every native boundary, size, dtype and shape against the oracle."""
    rows = golden["boundaries"]
    if set(actual) != {names[r[0]] for r in rows}:
        raise ValueError("Complete long-context replay boundaries differ")
    hashes, layout, failures = [], [], []
    elements = 0
    for identity, size, _, expected in rows:
        name = names[identity]
        a = actual[name]
        digest = array_digest(a)
        hashes.append([identity, digest])
        layout.append([identity, a.dtype.str, list(a.shape)])
        elements += int(a.size)
        if digest != expected or int(a.size) != size:
            failures.append({"check": name, "elements": int(a.size), "expected_elements": size,
                             "expected_sha256": expected, "actual_sha256": digest})
    root = hashlib.sha256(json.dumps(hashes, separators=(",", ":")).encode()).hexdigest()
    shape_root = hashlib.sha256(json.dumps(layout, separators=(",", ":")).encode()).hexdigest()
    passed = not failures and root == golden["expected_boundary_root_sha256"] and shape_root == golden["expected_layout_sha256"]
    return {"status": "pass" if passed else "fail", "boundary_comparisons": len(rows), "compared_storage_elements": elements,
            "expected_boundary_root_sha256": golden["expected_boundary_root_sha256"], "actual_boundary_root_sha256": root,
            "expected_layout_sha256": golden["expected_layout_sha256"], "actual_layout_sha256": shape_root, "failures": failures}


def exercise_long(arrays, ops, checker, reset_reference, seed_reference, lifecycle, progress=None):
    ps, norm, comp, index, attn = ops
    rows, faults, events, seeds = [], [], [], []
    minimum = physical_memory()["available"]
    def start_seed(frame, identity, batch, position, seed):
        state, m = make_seed(arrays, ops, batch, position, seed, capacity=frame.capacity)
        seed_reference(identity, state, m)
        frame.prime_diagnostic(state, position)
        seeds.append({"id": identity, "meta": m})
    def record(frame, m):
        nonlocal minimum
        x = frame_inputs(m["batch"], m["sequence"], m["position"], m["seed"])
        actual = frame.step(x, start_pos=m["position"])
        check = checker(m, x, actual)
        observed = observations(actual, m)
        if not observed["latest_candidate_kept"] or not observed["consumer_respects_candidates"]: raise RuntimeError("Connected candidate ownership or newest block differs")
        rows.append({"meta": m, **check, "observations": observed})
        if not clean(ops): raise RuntimeError("Long-context frame leaked native scratch")
        minimum = min(minimum, physical_memory()["available"])
        if progress and (len(rows) % 64 == 0 or m["initial_diagnostic_position"]): progress({"frames": len(rows), "case": m["id"], "position": m["position"], "status": check["status"]})
        return actual
    for c in configurations():
        reset_reference(c["batch"], c["capacity"])
        with NativeCompressedAttention(arrays, *ops, c["batch"], profile=PROFILE, capacity=c["capacity"]) as frame:
            if c["position"]: start_seed(frame, c["id"], c["batch"], c["position"], c["seed"])
            for m in stream_cases(c): record(frame, m)
        if frame.budget.reserved: raise RuntimeError("Long frame reservation was not released")
    for mode in FAULTS:
        reset_reference(1, 16392)
        with NativeCompressedAttention(arrays, *ops, 1, profile=PROFILE) as frame:
            start_seed(frame, "fault/" + mode, 1, 16384, 6500)
            record(frame, case_meta(f"fault/{mode}/warm", 1, 16384, 1, 6500, initial_position=16384))
            before = frame.inspect()
            target, attribute = frame.budget, "acquire"
            if mode == "index_key_download": target, attribute = index.device, "download"
            elif mode == "consumer_output": target, attribute = ps[24], "fp8"
            original, calls = getattr(target, attribute), 0
            def injected(*a, **k):
                nonlocal calls
                calls += 1
                if mode == "index_key_download": raise OSError("Injected long index-key readback failure")
                if mode == "consumer_attention":
                    if calls == 3: raise OSError("Injected long consumer Attention failure")
                    return attn(*a, **k)
                if mode == "consumer_output":
                    result = original(*a, **k)
                    if a[0] == "wo_b": raise OSError("Injected long final output failure")
                    return result
                if mode != "publication_pressure" or calls == 2: raise MemoryError("Injected long frame pressure")
                return original(*a, **k)
            if mode == "consumer_attention":
                class FailingAttention:
                    def __call__(self, *a, **k): return injected(*a, **k)
                frame.attention = FailingAttention()
            else: setattr(target, attribute, injected)
            try:
                try: frame.step(frame_inputs(1, 1, 16385, 6500))
                except (MemoryError, OSError): pass
                else: raise RuntimeError("Long injected failure published a partial frame")
            finally:
                frame.attention = attn
                if mode != "consumer_attention": setattr(target, attribute, original)
            after = frame.inspect()
            checks = inspect_checks(before, after)
            passed = before["position"] == after["position"] and all(c["status"] == "pass" for c in checks) and clean(ops)
            faults.append({"mode": mode, "position_preserved": before["position"] == after["position"], "checks": checks,
                "temporary_buffers_released": clean(ops), "status": "pass" if passed else "fail"})
            record(frame, case_meta(f"fault/{mode}/retry", 1, 16385, 1, 6500, initial_position=16384))
        if frame.budget.reserved: raise RuntimeError("Long failure frame reservation was not released")
    reset_reference(2, 16392)
    with NativeCompressedAttention(arrays, *ops, 2, profile=PROFILE) as frame:
        start_seed(frame, "snapshot", 2, 16384, 6600)
        result = record(frame, case_meta("snapshot/warm", 2, 16384, 1, 6600, initial_position=16384))
        before = frame.inspect()
        for a in result.values(): a.fill(0)
        copied = inspect_checks(before, frame.inspect())
        handle = frame.snapshot()
        lifecycle("snapshot")
        record(frame, case_meta("snapshot/advance", 2, 16385, 1, 6600, initial_position=16384))
        record(frame, case_meta("snapshot/advance-again", 2, 16386, 1, 6600, initial_position=16384))
        restored = frame.rollback(handle)
        lifecycle("rollback")
        checks = inspect_checks(before, restored)
        record(frame, case_meta("snapshot/after-rollback", 2, 16385, 1, 6600, initial_position=16384))
        refused = 0
        try: frame.discard(handle)
        except ValueError: refused += 1
        handle = frame.snapshot()
        try: frame.snapshot()
        except RuntimeError: refused += 1
        frame.reset()
        try: frame.rollback(handle)
        except ValueError: refused += 1
        events.append({"checks": checks, "diagnostic_copy_checks": copied, "restored_position": restored["position"],
            "expired_or_duplicate_refusals": refused, "status": "pass" if restored["position"] == 16385 and refused == 3
                and all(c["status"] == "pass" for c in checks + copied) else "fail"})
    if frame.budget.reserved: raise RuntimeError("Long snapshot frame reservation was not released")
    return rows, faults, events, seeds, minimum


def coverage_digest(report):
    data = [[r["meta"]["id"], [[b[0], b[1]] for b in r["boundaries"]], r["expected_layout_sha256"]] for r in report["frame_cases"]]
    data.extend([["control", i, [[b[0], b[1]] for b in r["boundaries"]]] for i, r in enumerate(report["unchanged_complete_Attention_controls"])])
    return hashlib.sha256(json.dumps(data, separators=(",", ":")).encode()).hexdigest()


def original_control_names():
    names = {"unchanged_shared_compressed_KV", "unchanged_shared_index_K", "unchanged_shared_candidates", "unchanged_shared_indices",
             "unchanged_compressor_kv_state", "unchanged_compressor_score_state"}
    for layer in LAYERS:
        names.update(f"unchanged_layer{layer}_" + n for n in ("output", "window", "query", "combined_KV", "combined_indices", "Attention_output"))
        if layer != 24: names.update(f"unchanged_layer{layer}_" + n for n in ("compressed_cache", "index_cache"))
    return sorted(names)


def is_digest(h):
    return isinstance(h, str) and len(h) == 64 and all(c in "0123456789abcdef" for c in h)


def validate_compact_group(group, ids, *, layout):
    rows = group.get("boundaries", [])
    if (group.get("status") != "pass" or len(rows) != len(ids)
            or any(not isinstance(r, list) or len(r) != 4 or type(r[0]) is not int or type(r[1]) is not int or r[1] < 0
                   or type(r[2]) is not int or r[2] != 0 or not is_digest(r[3]) for r in rows)
            or [r[0] for r in rows] != list(ids)):
        raise ValueError("Incomplete or failed compact long-context boundaries")
    root = hashlib.sha256(json.dumps([[r[0], r[3]] for r in rows], separators=(",", ":")).encode()).hexdigest()
    if group.get("expected_boundary_root_sha256") != root or group.get("actual_boundary_root_sha256") != root:
        raise ValueError("Original/native long-context boundary roots differ")
    if layout and (not is_digest(group.get("expected_layout_sha256")) or group.get("actual_layout_sha256") != group["expected_layout_sha256"]):
        raise ValueError("Long-context boundary dtype/shape identity differs")


def validate_goldens(report):
    from .compressed_attention_validation import validate_provenance
    from .resources import GiB, MiB
    validate_provenance(report, probe="native_long_context_parity")
    if (report.get("boundary_names") != registry() or report.get("boundary_columns") != ["name_id", "storage_elements", "mismatches", "expected_sha256"]
            or report.get("execution_limits") != {"profile": PROFILE, "context_capacities": [1152, 16392], "attention_KV_rows": 16520,
                "attention_index_width": 640, "long_queries_per_call": 1, "prefill_queries": 3, "batch": [1, 2], "attention_owned_device_cap_bytes": 64 * MiB}):
        raise ValueError("Unregistered long-context execution or record schema")
    rows, expected = report.get("frame_cases", []), list(expected_metadata())
    names = registry()
    ids = {n: i for i, n in enumerate(names)}
    if len(rows) != 1441 or len({r.get("meta", {}).get("id") for r in rows}) != 1441:
        raise ValueError("Long-context frame coverage is incomplete")
    for row, meta in zip(rows, expected):
        if row.get("meta") != meta or row.get("failures") != []: raise ValueError("Long-context input, position or failures differ")
        validate_compact_group(row, sorted(ids[n] for n in boundary_names(meta)), layout=True)
        observed = row.get("observations", {})
        n = meta["position"] + meta["sequence"]
        if (observed.get("window_wrap") is not bool(meta["position"] > 0 and meta["position"] % 128 == 0)
                or observed.get("selected_widths") != {"2": min(512, n // 2), "20": min(512, n), "24": min(512, n)}
                or observed.get("candidate_positions") != n or observed.get("reachable_excluded_positions") != [[8 if n > 16384 else 0] * meta["sequence"] for _ in range(meta["batch"])]
                or observed.get("latest_candidate_kept") is not True or observed.get("consumer_respects_candidates") is not True):
            raise ValueError("Connected window/top-512/candidate observations differ")
    if report.get("original_control_names") != original_control_names(): raise ValueError("Original long Attention control identities differ")
    controls = report.get("unchanged_complete_Attention_controls", [])
    if len(controls) != 1441: raise ValueError("Complete original long Attention controls are incomplete")
    for group in controls: validate_compact_group(group, range(28), layout=False)
    if coverage_digest(report) != LAYOUT_COVERAGE_SHA256: raise ValueError("Qualified long-context boundary sizes/layout coverage differs")
    seeds, seed_controls = report.get("diagnostic_seeds", []), report.get("original_seed_controls", [])
    specs = list(seed_specs())
    state_names = {f"windows/{l}/" for l in LAYERS} | {f"compressors/{l}/{i}/" for l in (2, 20) for i in range(3)} | {f"index_keys/{l}/" for l in (2, 20)}
    bank_names = {f"bank{l}_" + n for l in (2, 20) for n in ("pre_RoPE_latent_bf16", "index_keys_bf16", "compressed_KV_bf16")} | {f"bank{l}_window_bf16" for l in LAYERS} | {f"seed_query{l}_score_bf16" for l in (20, 24)}
    if len(seeds) != 9 or len(seed_controls) != 9: raise ValueError("Explicit diagnostic seed coverage is incomplete")
    for seed, control, (identity, batch, position, seed_id, capacity) in zip(seeds, seed_controls, specs):
        m, checks = seed.get("meta", {}), control.get("checks", [])
        pairs = m.get("source20_low_consumer24_high_pairs", [])
        if (seed.get("id") != identity or control.get("id") != identity or control.get("meta") != m
                or (m.get("batch"), m.get("position"), m.get("seed"), m.get("cache_capacity")) != (batch, position, seed_id, capacity)
                or len(pairs) != batch or any(len(p) != 2 or any(type(v) is not int or not 0 <= v < 64 for v in p) or p[0] == p[1] for p in pairs)
                or set(m.get("state_sha256", {})) != state_names or set(m.get("bank_sha256", {})) != bank_names
                or any(not is_digest(h) for h in [*m["state_sha256"].values(), *m["bank_sha256"].values()])
                or len(checks) != 11 or {c.get("check") for c in checks} != state_names
                or any(c.get("status") != "pass" or c.get("mismatches") != 0 or c.get("expected_sha256") != c.get("actual_sha256")
                       or c.get("expected_sha256") != m["state_sha256"].get(c["check"]) for c in checks)):
            raise ValueError("Original/native synthetic history provenance differs")
    faults, events = report.get("faults", []), report.get("snapshot_events", [])
    if ([f.get("mode") for f in faults] != list(FAULTS) or any(f.get("status") != "pass" or f.get("position_preserved") is not True
            or f.get("temporary_buffers_released") is not True or len(f.get("checks", [])) != 17
            or any(c.get("status") != "pass" or c.get("mismatches") != 0 for c in f["checks"]) for f in faults)):
        raise ValueError("Long-context atomic failure evidence differs")
    if (len(events) != 1 or events[0].get("status") != "pass" or events[0].get("restored_position") != 16385 or events[0].get("expired_or_duplicate_refusals") != 3
            or any(len(events[0].get(n, [])) != 17 or any(c.get("status") != "pass" or c.get("mismatches") != 0 for c in events[0][n])
                   for n in ("checks", "diagnostic_copy_checks"))):
        raise ValueError("Long-context snapshot/copy evidence differs")
    if report.get("instrumentation_counts") != {"projection": {"2": 5773, "20": 5782, "24": 5782}, "index_projection": 4341,
            "index_sum": 4341, "index_packed_vs_inplace": 6522, "compressed_packed_vs_inplace": 2181}:
        raise ValueError("Long original diagnostic control counts differ")
    negatives = report.get("negative_controls", [])
    if ([n.get("check") for n in negatives] != ["ignore_source_candidates_selected_indices", "ignore_source_candidates_Attention_output",
            "omit_compressed_index_window_offset", "reverse_window_slot_order_QK"]
            or any(n.get("status") != "fail" or n.get("mismatches", 0) <= 0 for n in negatives)):
        raise ValueError("Long connected negative controls are incomplete")
    memory = report.get("memory", {})
    cleaned = memory.get("cleanup", {})
    if (set(cleaned) != {"projection_2", "projection_20", "projection_24", "normalization", "compression", "indexer", "compression_normalization", "indexer_normalization", "attention"}
            or any(c.get("device_owned_bytes") != 0 or c.get("host_reserved_bytes") != 0 or c.get("owned_context_active") is not False for c in cleaned.values())
            or cleaned.get("attention", {}).get("host_allowance_bytes") != 128 * MiB
            or memory.get("combined_native_host_allowance_bytes") != 3488 * MiB
            or memory.get("combined_native_and_reference_host_allowance_bytes") != 5728 * MiB
            or memory.get("all_frame_reservations_released") is not True or memory.get("state_storage") != "bounded_host_diagnostic_frame"
            or memory.get("reservation_is_RSS_limit") is not False or memory.get("minimum_sampled_host_available_bytes", 0) < 32 * GiB):
        raise ValueError("Long-context owned-resource evidence differs")
    if (report.get("weights_fetched_during_comparison_bytes") != 0 or report.get("new_weight_bytes") != 0
            or report.get("full_model_quality") != "not_run" or report.get("end_to_end_speed") != "not_measured"
            or report.get("other_layers") != "not_qualified" or report.get("untouched_full_Attention_kernel") != "blocked_by_shared_memory_limit"):
        raise ValueError("Long-context fidelity scope differs")
    return {r["meta"]["id"]: r for r in rows}
