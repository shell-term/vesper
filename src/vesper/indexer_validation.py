"""Deterministic native/original indexer and sharing-frame corpus."""

import copy

import numpy as np

from .attention_validation import array_digest, check_arrays
from .formats import bf16_bits
from .indexer_state import NativeIndexSession
from .normalization_validation import inputs
from .resources import physical_memory

FAULTS = ("host_pressure", "key_download", "consumer_queries", "consumer_selection", "publication_pressure")


def bounded_inputs(shape, seed, mode="normal"):
    if mode != "mixed_scales": return inputs(shape, seed, mode)
    rng = np.random.default_rng(20261018 + seed)
    x = rng.standard_normal(shape).astype(np.float32)
    return bf16_bits(np.ldexp(x, rng.integers(-8, 9, (*shape[:-1], 1))))


def component_cases():
    bits = np.arange(65536, dtype=np.uint16)
    bits = bits[(bits & 0x7f80) != 0x7f80]
    for start in range(0, bits.size, 16384):
        yield {"id": f"round/finite/{start}", "operation": "round"}, (np.resize(bits[start:start + 16384], 16384).reshape(1, 128, 128).copy(),)
    for batch in (1, 2):
        for mode in ("normal", "zeros", "signed_zeros", "near_eps", "bf16_subnormal", "mixed_scales", "large"):
            yield {"id": f"round/{batch}/{mode}", "operation": "round"}, (inputs((batch, 3, 32, 128), batch, mode),)
    levels = np.array([0, .25, .5, .75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 5, 6, -0.], np.float32)
    for exponent in (-126, -20, 0, 20, 100):
        values = np.resize(np.ldexp(levels, exponent), (1, 4, 128)).copy()
        values[..., 16:32] *= np.float32(-1)
        yield {"id": f"round/ties/{exponent}", "operation": "round"}, (bf16_bits(values),)
    for layer in (2, 20):
        for rows in (1, 3, 8, 17, 33, 128, 257, 514):
            yield {"id": f"norm/{layer}/{rows}", "operation": "norm", "layer": layer}, (inputs((rows, 128), rows),)
        for mode in ("zeros", "signed_zeros", "bf16_subnormal", "mixed_scales"):
            yield {"id": f"norm/{layer}/{mode}", "operation": "norm", "layer": layer}, (inputs((17, 128), layer, mode),)
        ratio = 2 if layer == 2 else 1
        for batch, sequence in ((1, 1), (1, 3), (2, 8), (1, 257)):
            for start in (0, 127, 131071 - (sequence - 1) * ratio):
                pos = start + np.arange(sequence, dtype=np.int32) * ratio
                yield {"id": f"key/{layer}/{batch}/{sequence}/{start}", "operation": "key", "layer": layer}, (inputs((batch, sequence, 512), layer + sequence), pos)
        for mode in ("zeros", "signed_zeros", "near_eps"):
            yield {"id": f"key/{layer}/{mode}", "operation": "key", "layer": layer}, (inputs((1, 3, 512), layer, mode), np.arange(3, dtype=np.int32) * ratio)
    for layer in (2, 20, 24):
        for batch, sequence in ((1, 1), (1, 3), (2, 2), (2, 8)):
            for start in (0, 127, 131072 - sequence):
                yield {"id": f"query/{layer}/{batch}/{sequence}/{start}", "operation": "queries", "layer": layer, "position": start}, (
                    inputs((batch, sequence, 5120), 20 + layer), inputs((batch, sequence, 1280), 40 + layer))
        for mode in ("zeros", "signed_zeros", "mixed_scales"):
            yield {"id": f"query/{layer}/{mode}", "operation": "queries", "layer": layer, "position": 17}, (
                bounded_inputs((1, 3, 5120), 20 + layer, mode), bounded_inputs((1, 3, 1280), 40 + layer, mode))
    sizes = (0, 1, 2, 3, 7, 8, 31, 128, 511, 512, 513, 1025, 16385, 16392)
    for batch, sequence in ((1, 2), (2, 3)):
        for n in sizes:
            yield {"id": f"score/{batch}/{sequence}/{n}", "operation": "score"}, (
                inputs((batch, sequence, 32, 128), 100), inputs((batch, n, 128), 101), inputs((batch, sequence, 32), 102))
    for n in (1, 3, 513, 16385):
        for mode in ("zeros", "near_eps", "mixed_scales"):
            yield {"id": f"score/{n}/{mode}", "operation": "score"}, (
                bounded_inputs((1, 2, 32, 128), 100, mode), bounded_inputs((1, n, 128), 101, mode), bounded_inputs((1, 2, 32), 102, mode))
    for n in (0, 1, 3, 8, 511, 512, 513, 1025, 16384, 16385, 16392):
        for mode in ("normal", "ties", "signed_zeros"):
            values = inputs((2, 2, n), 700, mode if mode != "ties" else "normal")
            if mode == "ties": values.fill(0x3f80)
            lengths = np.array([[n, n // 2], [min(1, n), 0]], np.int32)
            for style in ("plain", "source", "consumer"):
                meta = {"id": f"select/{n}/{mode}/{style}", "operation": "select", "style": style, "offset": 128}
                args = (values, lengths)
                if style == "consumer": args += (np.broadcast_to((np.arange(n) % 3 != 1)[None, None, :], values.shape).copy(),)
                yield meta, args


def component_meta(meta, args):
    return {**meta, "input_sha256": [array_digest(a) for a in args], "input_shapes": [list(a.shape) for a in args]}


def execute_component(runtime, meta, args):
    op = meta["operation"]
    if op == "norm":
        name = f"layers.{meta['layer']}.attn.indexer.k_norm.weight"
        return runtime.norm.norm(args[0], runtime.arrays[name]) if hasattr(runtime, "norm") else runtime.math.norm(args[0], name)
    if op == "key": return runtime.key(args[0], meta["layer"], args[1])
    if op == "queries": return runtime.queries(*args, meta["layer"], meta["position"])
    if op == "select":
        return runtime.select(*args[:2], meta["offset"], candidate_source=meta["style"] == "source", candidates=args[2] if len(args) == 3 else None)
    return getattr(runtime, op)(*args)


def exercise_components(native, checker):
    rows, minimum = [], physical_memory()["available"]
    for meta, args in component_cases():
        meta = component_meta(meta, args)
        actual = execute_component(native, meta, args)
        checks = checker(meta, args, actual)
        rows.append({**meta, "checks": checks, "status": "pass" if all(c["status"] == "pass" for c in checks) else "fail"})
        minimum = min(minimum, physical_memory()["available"])
        if sum(native.device.allocations.values()) != native.resident_bytes or native.norm.device.allocations:
            raise RuntimeError("Indexer component leaked temporary device buffers")
    return rows, minimum


def frame_inputs(batch, sequence, position, seed, mode="normal"):
    xs = {layer: bounded_inputs((batch, sequence, 5120), seed + position + layer, mode) for layer in (2, 20, 24)}
    qrs = {layer: bounded_inputs((batch, sequence, 1280), seed + 100 + position + layer, mode) for layer in (2, 20, 24)}
    latents = {}
    for layer, ratio in ((2, 2), (20, 1)):
        n = sequence // ratio if position == 0 else (position + 1) // ratio - position // ratio
        latents[layer] = bounded_inputs((batch, n, 512), seed + 200 + position + layer, mode) if n else None
    return xs, qrs, latents


def frame_meta(identity, batch, capacity, position, sequence, seed, mode="normal", primed=False):
    xs, qrs, latents = frame_inputs(batch, sequence, position, seed, mode)
    arrays = [xs[n] for n in (2, 20, 24)] + [qrs[n] for n in (2, 20, 24)] + [latents[n] for n in (2, 20)]
    return {"id": identity, "batch": batch, "capacity": capacity, "position": position, "sequence": sequence, "seed": seed,
            "mode": mode, "primed_diagnostic_history": primed, "input_sha256": [None if a is None else array_digest(a) for a in arrays]}


def streams():
    for batch in (1, 2):
        for prefill in (1, 2, 3, 7, 8):
            yield {"id": f"core/{batch}/{prefill}", "batch": batch, "capacity": 128, "prefill": prefill, "decodes": 2, "mode": "normal", "seed": 2000}
        for mode in ("zeros", "signed_zeros", "near_eps", "mixed_scales"):
            yield {"id": f"stress/{batch}/{mode}", "batch": batch, "capacity": 128, "prefill": 3, "decodes": 2, "mode": mode, "seed": 3000}
    yield {"id": "continuous/32", "batch": 1, "capacity": 128, "prefill": 8, "decodes": 24, "mode": "normal", "seed": 4000}
    for position in (512, 1024, 16384):
        yield {"id": f"primed/{position}", "batch": 1, "capacity": position + 2, "initial_position": position,
               "prefill": 1, "decodes": 1, "mode": "normal", "seed": 5000}


def stream_steps(config):
    start = config.get("initial_position", 0)
    steps = [(start, config["prefill"])] + [(start + config["prefill"] + j, 1) for j in range(config["decodes"])]
    for index, (position, sequence) in enumerate(steps):
        yield frame_meta(f"frame/{config['id']}/{index}", config["batch"], config["capacity"], position, sequence, config["seed"], config["mode"], bool(start))


def prime_keys(runtime, batch, position):
    result = []
    for layer, count in ((2, position // 2), (20, position)):
        latent = inputs((batch, 1, 512), 6000 + layer)
        row = runtime.key(latent, layer, np.array([0], np.int32))["quant_rounded_bf16"]
        result.append(np.repeat(row, count, axis=1))
    return result


def exercise_frames(native, checker, reset, prime):
    rows, minimum = [], physical_memory()["available"]
    for config in streams():
        reset(config["batch"], config["capacity"])
        with NativeIndexSession(native, config["batch"], config["capacity"]) as session:
            initial = config.get("initial_position", 0)
            if initial:
                keys = prime_keys(native, config["batch"], initial)
                session.prime(*keys, initial)
                prime(keys, initial)
            for meta in stream_steps(config):
                args = frame_inputs(meta["batch"], meta["sequence"], meta["position"], meta["seed"], meta["mode"])
                actual = session.step(*args, start_pos=meta["position"])
                checks = checker(meta, args, actual)
                rows.append({**meta, "checks": checks, "status": "pass" if all(c["status"] == "pass" for c in checks) else "fail"})
                minimum = min(minimum, physical_memory()["available"])
                for consumer, name in ((3, "shared2_to3_indices_i32"), (25, "shared24_to25_indices_i32")):
                    if array_digest(session.shared_indices(consumer, meta["position"], meta["sequence"])) != array_digest(actual[name]):
                        raise RuntimeError("Shared index consumer differs from the committed frame")
                if sum(native.device.allocations.values()) != native.resident_bytes or native.norm.device.allocations:
                    raise RuntimeError("Native index frame leaked temporary buffers")
        if session.budget.reserved: raise RuntimeError("Index frame reservation was not released")
    return rows, minimum


def inspect_checks(before, after):
    checks = [check_arrays(before[n], after[n], n) for n in ("cache2_bf16", "cache20_bf16")]
    if before["shared"] is not None:
        for n in ("keys20", "candidates20", "indices3", "indices25"):
            checks.append(check_arrays(before["shared"][n], after["shared"][n], "shared_" + n))
    return checks


def exercise_failures(native, checker, reset):
    rows, faults = [], []
    for mode in FAULTS:
        reset(1, 128)
        with NativeIndexSession(native) as session:
            def record(label, position, sequence):
                meta = frame_meta(f"fault/{mode}/{label}", 1, 128, position, sequence, 7000)
                args = frame_inputs(1, sequence, position, 7000)
                actual = session.step(*args, start_pos=position)
                checks = checker(meta, args, actual)
                rows.append({**meta, "checks": checks, "status": "pass" if all(c["status"] == "pass" for c in checks) else "fail"})
            record("prefill", 0, 3)
            before = session.inspect()
            target, attribute = session.budget, "acquire"
            if mode == "key_download": target, attribute = native, "key"
            elif mode == "consumer_queries": target, attribute = native, "queries"
            elif mode == "consumer_selection": target, attribute = native, "select"
            original, calls = getattr(target, attribute), 0
            def injected(*args, **kwargs):
                nonlocal calls
                calls += 1
                if mode == "key_download":
                    download = native.device.download
                    def broken(*a, **k): raise OSError("Injected index-key diagnostic download failure")
                    native.device.download = broken
                    try: return original(*args, **kwargs)
                    finally: native.device.download = download
                if mode == "consumer_queries":
                    if args[2] == 24: raise OSError("Injected consumer query failure after owner publication staging")
                    return original(*args, **kwargs)
                if mode == "consumer_selection":
                    result = original(*args, **kwargs)
                    if kwargs.get("candidates") is not None: raise OSError("Injected consumer selection failure before frame commit")
                    return result
                if mode != "publication_pressure" or calls == 2: raise MemoryError("Injected index frame pressure")
                return original(*args, **kwargs)
            setattr(target, attribute, injected)
            try:
                try: session.step(*frame_inputs(1, 1, 3, 7000))
                except (MemoryError, OSError): pass
                else: raise RuntimeError("Injected index frame failure did not refuse publication")
            finally: setattr(target, attribute, original)
            after = session.inspect()
            checks = inspect_checks(before, after)
            passed = before["position"] == after["position"] and all(c["status"] == "pass" for c in checks)
            clean = sum(native.device.allocations.values()) == native.resident_bytes and not native.norm.device.allocations
            faults.append({"mode": mode, "checks": checks, "position_preserved": before["position"] == after["position"],
                           "temporary_buffers_released": clean, "status": "pass" if passed and clean else "fail"})
            record("retry", 3, 1)
    return rows, faults


def exercise_snapshot(native, checker, reference):
    rows = []
    reference("reset", 2, 128)
    with NativeIndexSession(native, 2) as session:
        def record(label, position, sequence):
            meta = frame_meta(f"snapshot/{label}", 2, 128, position, sequence, 8000)
            args = frame_inputs(2, sequence, position, 8000)
            actual = session.step(*args, start_pos=position)
            checks = checker(meta, args, actual)
            rows.append({**meta, "checks": checks, "status": "pass" if all(c["status"] == "pass" for c in checks) else "fail"})
        record("prefill", 0, 3)
        before, handle = session.inspect(), session.snapshot()
        reference("snapshot")
        record("advance", 3, 1)
        record("advance-again", 4, 1)
        restored = session.rollback(handle)
        reference("rollback")
        checks = inspect_checks(before, restored)
        record("after-rollback", 3, 1)
        refused = 0
        for action in (lambda: session.discard(handle), lambda: session.shared_indices(3, 0, 3)):
            try: action()
            except ValueError: refused += 1
        handle = session.snapshot()
        try: session.snapshot()
        except RuntimeError: refused += 1
        session.reset()
        try: session.rollback(handle)
        except ValueError: refused += 1
        event = {"checks": checks, "restored_position": restored["position"], "stale_or_expired_or_duplicate_refusals": refused,
                 "status": "pass" if restored["position"] == 3 and refused == 4 and all(c["status"] == "pass" for c in checks) else "fail"}
    return rows, [event]


def key_names():
    from .normalization import NORM_STAGES
    return {"wk_bf16"} | {"norm_" + n for n in NORM_STAGES} | {"rope_" + n for n in ("frequency_fp32", "rotated_fp32", "output_bf16")} | {"quant_" + n for n in ("packed_fp4", "scales_e8m0", "rounded_bf16")}


def query_names():
    return {"input_fp8", "input_scale_e8m0", "projection_fp32", "projection_bf16", "weights_raw_bf16", "weights_scaled_bf16"} | {"rope_" + n for n in ("frequency_fp32", "rotated_fp32", "output_bf16")} | {"quant_" + n for n in ("packed_fp4", "scales_e8m0", "rounded_bf16")}


SCORE_NAMES = {"dot_bf16", "relu_bf16", "weighted_bf16", "sum_fp32", "score_bf16"}


def selection_names(style):
    names = {"causal_bf16", "masked_bf16", "indices_i32"}
    if style != "plain": names.add("candidates_bool")
    if style == "source": names.add("candidate_blocks_bf16")
    return names


def frame_names(meta):
    position, sequence = meta["position"], meta["sequence"]
    names = {"shared20_to24_keys_bf16", "shared20_to24_candidates_bool", "shared2_to3_indices_i32", "shared24_to25_indices_i32", "cache2_bf16", "cache20_bf16"}
    for layer in (2, 20, 24):
        prefix = f"layer{layer}_"
        names.add(prefix + "index_keys_bf16")
        if layer == 20 or (layer == 2 and (sequence >= 2 if position == 0 else (position + 1) % 2 == 0)):
            names.update(prefix + "key_" + n for n in key_names())
        names.update(prefix + "query_" + n for n in query_names())
        names.update(prefix + "score_" + n for n in SCORE_NAMES)
        names.update(prefix + "selection_" + n for n in selection_names("plain" if layer == 2 else ("source" if layer == 20 else "consumer")))
    return names


def expected_metadata():
    for meta, args in component_cases(): yield component_meta(meta, args)
    for config in streams(): yield from stream_steps(config)
    for mode in FAULTS:
        yield frame_meta(f"fault/{mode}/prefill", 1, 128, 0, 3, 7000)
        yield frame_meta(f"fault/{mode}/retry", 1, 128, 3, 1, 7000)
    for label, pos, seq in (("prefill", 0, 3), ("advance", 3, 1), ("advance-again", 4, 1), ("after-rollback", 3, 1)):
        yield frame_meta(f"snapshot/{label}", 2, 128, pos, seq, 8000)


def validate_goldens(report):
    from .checkpoint import MODEL, REVISION, REFERENCE_HASHES
    from .gate_reference import GATE_SOURCE_HASHES
    from .indexer import INDEXER_CUBIN_SHA256
    from .indexer_fixture import INDEXER_BYTES, INDEXER_LAYOUT, DTYPES
    from .indexer_reference import INDEX_METHODS, INDEX_QUANT_CUDA, INDEX_QUERY_TRACE_CUDA, INDEX_SOURCE_HASHES
    from .normalization import NORM_CUBIN_SHA256, NORM_STAGES
    from .normalization_reference import PYTORCH_SOURCES
    from .prefix_reference import PREFIX_SOURCE_HASHES
    from .router import ROUTER_TORCH_REVISION
    from .router_reference import TOPK_SOURCE_HASHES
    profile = report.get("indexer_profile", {})
    if (report.get("status") != "pass" or report.get("probe") != "native_indexer_parity" or report.get("model") != MODEL or report.get("revision") != REVISION
            or report.get("torch_git_version") != ROUTER_TORCH_REVISION or report.get("tilelang_version") != "0.1.12"
            or profile.get("cubin_sha256") != INDEXER_CUBIN_SHA256 or profile.get("compiler", {}).get("nvrtc_version") != [12, 4]
            or profile.get("compiler", {}).get("concatenated_source_sha256") != "aaeade922ba2f4efb56e23f3320d8f45716e3d72aab0cefef908463b7f0d63e5"
            or profile.get("cublas_version") != 120402 or profile.get("cublas_math_mode") != 0
            or report.get("normalization_profile", {}).get("cubin_sha256") != NORM_CUBIN_SHA256
            or report.get("normalization_profile", {}).get("compiler", {}).get("nvrtc_version") != [12, 4]
            or report.get("original_methods_ast_sha256") != INDEX_METHODS or report.get("original_quant_cuda_sha256") != INDEX_QUANT_CUDA
            or report.get("original_query_FP32_cuda_sha256") != INDEX_QUERY_TRACE_CUDA):
        raise ValueError("Expected passing pinned original/native indexer evidence")
    if (any(report.get("reference", {}).get("files", {}).get(n) != h for n, h in {**REFERENCE_HASHES, **PREFIX_SOURCE_HASHES}.items() if n in ("kernel.py", "model.py", "convert.py", "config.json", "generate.py"))
            or any(report.get("blas_reference", {}).get("files", {}).get(n, {}).get("sha256") != h for n, h in GATE_SOURCE_HASHES.items())
            or any(report.get("reduction_reference", {}).get("files", {}).get(n, {}).get("sha256") != h for n, (_, h) in PYTORCH_SOURCES.items())
            or any(report.get("topk_reference", {}).get("files", {}).get(n, {}).get("sha256") != h for n, h in TOPK_SOURCE_HASHES.items())
            or any(report.get("index_reference", {}).get("files", {}).get(n, {}).get("sha256") != h for n, h in INDEX_SOURCE_HASHES.items())):
        raise ValueError("Pinned indexer source provenance differs")
    fixture = report.get("fixture", {})
    tensors = fixture.get("tensors", [])
    if (fixture.get("kind") != "original_indexer_fixture" or fixture.get("model") != MODEL or fixture.get("revision") != REVISION
            or fixture.get("layers") != [2, 20, 24] or fixture.get("weights_fetched_bytes") != INDEXER_BYTES
            or len(tensors) != 13 or {t.get("name") for t in tensors} != set(INDEXER_LAYOUT)):
        raise ValueError("Original indexer fixture provenance is incomplete")
    for t in tensors:
        dtype, shape = INDEXER_LAYOUT[t["name"]]
        if t.get("dtype") != dtype or t.get("shape") != list(shape) or t.get("bytes") != int(np.prod(shape)) * DTYPES[dtype].itemsize:
            raise ValueError("Original indexer fixture layout differs")
    components, frames = report.get("component_cases", []), report.get("frame_cases", [])
    goldens = report.get("reference_goldens", [])
    expected = {m["id"]: m for m in expected_metadata()}
    indexed = {g["meta"]["id"]: g for g in goldens}
    if (len(components) != 261 or len(frames) != 99 or len(goldens) != 360 or len(indexed) != 360
            or set(indexed) != set(expected) or len({r["id"] for r in components + frames}) != 360):
        raise ValueError("Indexer component/frame/golden coverage is incomplete")
    for row in components + frames:
        golden = indexed[row["id"]]
        meta = golden["meta"]
        if meta != expected[row["id"]] or {k: v for k, v in row.items() if k not in ("checks", "status")} != meta:
            raise ValueError("Indexer input, position or primed-state metadata differs")
        op = meta.get("operation")
        if op == "round": names = {"packed_fp4", "scales_e8m0", "rounded_bf16"}
        elif op == "norm": names = set(NORM_STAGES)
        elif op == "key": names = key_names()
        elif op == "queries": names = query_names()
        elif op == "score": names = SCORE_NAMES
        elif op == "select": names = selection_names(meta["style"])
        else: names = frame_names(meta)
        hashes, checks = golden.get("stage_sha256", {}), row.get("checks", [])
        if (row.get("status") != "pass" or set(hashes) != names or len(checks) != len(names) or {c.get("check") for c in checks} != names
                or any(c.get("status") != "pass" or c.get("mismatches") != 0 or c.get("expected_sha256") != c.get("actual_sha256")
                       or c.get("expected_sha256") != hashes.get(c["check"]) for c in checks)
                or any(not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h) for h in hashes.values())):
            raise ValueError("Indexer exact comparison or golden boundary set differs")
    faults, events = report.get("faults", []), report.get("snapshot_events", [])
    if ([f.get("mode") for f in faults] != list(FAULTS) or any(f.get("status") != "pass" or f.get("position_preserved") is not True
            or f.get("temporary_buffers_released") is not True or len(f.get("checks", [])) != 6
            or any(c.get("status") != "pass" or c.get("mismatches") != 0 for c in f["checks"]) for f in faults)):
        raise ValueError("Index-frame whole-call refusal evidence differs")
    if (len(events) != 1 or events[0].get("status") != "pass" or events[0].get("restored_position") != 3
            or events[0].get("stale_or_expired_or_duplicate_refusals") != 4 or len(events[0].get("checks", [])) != 6
            or any(c.get("status") != "pass" or c.get("mismatches") != 0 for c in events[0]["checks"])):
        raise ValueError("Index snapshot/sharing restoration evidence differs")
    controls = report.get("unchanged_original_controls", {})
    for name, count in (("projection", 342), ("sum", 337), ("packed_vs_inplace", 563), ("forward_sharing", 99), ("primed_keys", 3)):
        rows = controls.get(name, [])
        if len(rows) != count: raise ValueError("Unchanged indexer controls are incomplete")
        if name in ("forward_sharing", "primed_keys"):
            if any(len(row) != (9 if name == "forward_sharing" else 2) for row in rows): raise ValueError("Original sharing-control boundaries differ")
            rows = [c for row in rows for c in row]
        if any(c.get("status") != "pass" or c.get("mismatches") != 0 for c in rows): raise ValueError("Original indexer controls differ")
    negatives = report.get("negative_controls", [])
    if ([n.get("check") for n in negatives] != ["replace_index_FP4_with_FP8", "remove_BF16_weighted_product_boundary", "replace_ATen_head_reduction_with_sequential_sum",
            "replace_radix_signed_zero_membership_with_stable_float_ties", "omit_newest_candidate_block_pin", "rotate_index_key_at_last_group_token"]
            or any(n.get("status") != "fail" or n.get("mismatches", 0) <= 0 for n in negatives)):
        raise ValueError("Indexer arithmetic/membership negative controls are incomplete")
    memory = report.get("memory", {})
    for name in ("indexer_cleanup", "normalization_cleanup"):
        c = memory.get(name, {})
        if c.get("device_owned_bytes") != 0 or c.get("host_reserved_bytes") != 0 or c.get("owned_context_active") is not False:
            raise ValueError("Indexer owned buffers/context retains were not released")
    if memory.get("all_session_reservations_released") is not True or memory.get("state_storage") != "bounded_host_diagnostic_key_caches":
        raise ValueError("Index session resource scope differs")
    return indexed
