"""Deterministic connected compressed-Attention corpus and evidence checks."""

import copy

import numpy as np

from .attention_validation import array_digest, check_arrays
from .compressed_attention import LAYERS, NativeCompressedAttention
from .indexer_validation import bounded_inputs
from .resources import physical_memory

FAULTS = ("host_pressure", "index_key_download", "consumer_attention", "consumer_output", "publication_pressure")
CONVERTED_WO_A = {"2": "329e0e1cb44d22d715e82aafb4b868474519f1d04842d7a30b13ab1ba282b77c",
                  "20": "bce6e1d5b2391dc811e17aa2c5b87288669bc8d2949c3a26d8d90a733bb8f78f",
                  "24": "3d65cf3450e72ed36ca6cd0a3cc3212c327d1b6a81648674c807aa208bdf1080"}


def frame_inputs(batch, sequence, position, seed, mode="normal"):
    return {l: bounded_inputs((batch, sequence, 5120), seed + position + l, mode) for l in LAYERS}


def metadata(identity, batch, position, sequence, seed, mode="normal"):
    xs = frame_inputs(batch, sequence, position, seed, mode)
    return {"id": identity, "batch": batch, "position": position, "sequence": sequence, "seed": seed, "mode": mode,
            "input_sha256": {str(l): array_digest(xs[l]) for l in LAYERS}}


def streams():
    for batch in (1, 2):
        for prefill in (1, 2, 3): yield (f"core/{batch}/{prefill}", batch, prefill, 2, 1000, "normal")
    for mode in ("zeros", "signed_zeros", "mixed_scales"): yield (f"stress/{mode}", 1, 3, 2, 2000, mode)
    yield ("continuous/64", 1, 3, 61, 3000, "normal")


def stream_metadata(config):
    name, batch, prefill, decodes, seed, mode = config
    for i, (pos, seq) in enumerate([(0, prefill)] + [(prefill + j, 1) for j in range(decodes)]):
        yield metadata(f"frame/{name}/{i}", batch, pos, seq, seed, mode)


def expected_metadata():
    for config in streams(): yield from stream_metadata(config)
    for mode in FAULTS:
        for label, pos, seq in (("prefill", 0, 3), ("retry", 3, 1)):
            yield metadata(f"fault/{mode}/{label}", 1, pos, seq, 4000)
    for label, pos, seq in (("prefill", 0, 3), ("advance", 3, 1), ("advance-again", 4, 1), ("after-rollback", 3, 1)):
        yield metadata(f"snapshot/{label}", 2, pos, seq, 5000)


def flatten_state(value, prefix=""):
    result = {}
    if isinstance(value, np.ndarray): result[prefix] = value
    elif type(value) is int: result[prefix] = np.array([value], np.int64)
    elif isinstance(value, dict):
        for name, item in value.items(): result.update(flatten_state(item, prefix + str(name) + "/"))
    elif isinstance(value, tuple):
        for name, item in enumerate(value): result.update(flatten_state(item, prefix + str(name) + "/"))
    return result


def inspect_checks(before, after):
    left, right = flatten_state(before["state"]), flatten_state(after["state"])
    if set(left) != set(right): raise ValueError("Compressed frame state members differ")
    return [check_arrays(a, right[n], n) for n, a in left.items()]


def clean(ops):
    ps, norm, comp, index, attn = ops
    return (all(sum(p.device.allocations.values()) == p.resident_bytes for p in ps.values())
            and sum(comp.device.allocations.values()) == comp.resident_bytes
            and sum(index.device.allocations.values()) == index.resident_bytes
            and not norm.device.allocations and not comp.norm.device.allocations and not index.norm.device.allocations and not attn.device.allocations)


def exercise(arrays, ops, checker, reset_reference, lifecycle):
    ps, norm, comp, index, attn = ops
    rows, faults, events = [], [], []
    minimum = physical_memory()["available"]
    def record(frame, meta):
        nonlocal minimum
        xs = frame_inputs(meta["batch"], meta["sequence"], meta["position"], meta["seed"], meta["mode"])
        actual = frame.step(xs, start_pos=meta["position"])
        checks = checker(meta, xs, actual)
        rows.append({**meta, "checks": checks, "status": "pass" if all(c["status"] == "pass" for c in checks) else "fail"})
        if not clean(ops): raise RuntimeError("Connected compressed Attention leaked GPU scratch")
        minimum = min(minimum, physical_memory()["available"])
        return actual
    for config in streams():
        reset_reference(config[1])
        with NativeCompressedAttention(arrays, *ops, config[1]) as frame:
            for meta in stream_metadata(config): record(frame, meta)
        if frame.budget.reserved: raise RuntimeError("Frame host reservation was not released")
    for mode in FAULTS:
        reset_reference(1)
        with NativeCompressedAttention(arrays, *ops, 1) as frame:
            record(frame, metadata(f"fault/{mode}/prefill", 1, 0, 3, 4000))
            before = frame.inspect()
            target, attribute = frame.budget, "acquire"
            if mode == "index_key_download": target, attribute = index.device, "download"
            elif mode == "consumer_attention": target, attribute = attn, "__call__"
            elif mode == "consumer_output": target, attribute = ps[24], "fp8"
            original, calls = getattr(target, attribute), 0
            def injected(*args, **kwargs):
                nonlocal calls
                calls += 1
                if mode == "index_key_download": raise OSError("Injected index-key readback failure")
                if mode == "consumer_attention":
                    if calls == 3: raise OSError("Injected source-KV consumer Attention failure")
                    return original(*args, **kwargs)
                if mode == "consumer_output":
                    if args[0] == "wo_b":
                        original(*args, **kwargs)
                        raise OSError("Injected consumer output failure before frame publication")
                    return original(*args, **kwargs)
                if mode != "publication_pressure" or calls == 2: raise MemoryError("Injected compressed frame pressure")
                return original(*args, **kwargs)
            # Special methods are resolved on the class; use an explicit wrapper.
            if mode == "consumer_attention":
                class FailedAttention:
                    def __call__(self, *a, **k): return injected(*a, **k)
                frame.attention = FailedAttention()
            else: setattr(target, attribute, injected)
            try:
                try: frame.step(frame_inputs(1, 1, 3, 4000))
                except (MemoryError, OSError): pass
                else: raise RuntimeError("Injected compressed Attention failure did not refuse")
            finally:
                if mode == "consumer_attention": frame.attention = attn
                else: setattr(target, attribute, original)
            after, checks = frame.inspect(), None
            checks = inspect_checks(before, after)
            passed = before["position"] == after["position"] and all(c["status"] == "pass" for c in checks) and clean(ops)
            faults.append({"mode": mode, "checks": checks, "position_preserved": before["position"] == after["position"],
                           "temporary_buffers_released": clean(ops), "status": "pass" if passed else "fail"})
            record(frame, metadata(f"fault/{mode}/retry", 1, 3, 1, 4000))
    reset_reference(2)
    with NativeCompressedAttention(arrays, *ops, 2) as frame:
        actual = record(frame, metadata("snapshot/prefill", 2, 0, 3, 5000))
        before = frame.inspect()
        for a in actual.values(): a.fill(0)
        diagnostic_checks = inspect_checks(before, frame.inspect())
        handle = frame.snapshot()
        lifecycle("snapshot")
        record(frame, metadata("snapshot/advance", 2, 3, 1, 5000))
        record(frame, metadata("snapshot/advance-again", 2, 4, 1, 5000))
        restored = frame.rollback(handle)
        lifecycle("rollback")
        checks = inspect_checks(before, restored)
        record(frame, metadata("snapshot/after-rollback", 2, 3, 1, 5000))
        refused = 0
        try: frame.discard(handle)
        except ValueError: refused += 1
        handle = frame.snapshot()
        try: frame.snapshot()
        except RuntimeError: refused += 1
        frame.reset()
        try: frame.rollback(handle)
        except ValueError: refused += 1
        events.append({"checks": checks, "diagnostic_copy_checks": diagnostic_checks, "restored_position": restored["position"],
            "expired_or_duplicate_refusals": refused, "status": "pass" if restored["position"] == 3 and refused == 3
                and all(c["status"] == "pass" for c in checks + diagnostic_checks) else "fail"})
    return rows, faults, events, minimum


def boundary_names(meta):
    from .attention_storage import STAGES
    from .compression_validation import state_stage_names
    from .indexer_validation import frame_names
    from .normalization import NORM_STAGES
    from .projections import PROJECTION_STAGES
    names = {"index_" + n for n in frame_names(meta)} | {"shared20_to24_compressed_KV_bf16"}
    for layer in LAYERS:
        p = f"layer{layer}_"
        for projection in ("wq_a", "wq_b", "wkv", "wo_b"): names.update(p + projection + "_" + n for n in PROJECTION_STAGES)
        for norm in ("q_norm", "kv_norm"): names.update(p + norm + "_" + n for n in NORM_STAGES)
        for rope in ("query_rope", "kv_rope", "inverse_rope"): names.update(p + rope + "_" + n for n in ("frequency_fp32", "rotated_fp32", "output_bf16"))
        names.update(p + "window_" + n for n in ("rounded_bf16", "codes_fp8", "scales_e8m0", "cache_bf16", "visible_bf16", "indices_i32"))
        names.update(p + "attention_" + n for n in ("output_bf16", *STAGES))
        names.update(p + "wo_a_" + n for n in ("rounded_bf16", "input_fp8", "scale_e8m0", "output_bf16"))
        names.update(p + n for n in ("compressed_visible_bf16", "combined_KV_bf16", "combined_indices_i32"))
        if layer != 24: names.update(p + "compressor_" + n for n in state_stage_names(layer, meta["position"], meta["sequence"]))
    return names


def profiles(ops):
    ps, norm, comp, index, attn = ops
    def projection(p):
        return {"compiler": p.compiler, "cubin_sha256": p.cubin_sha256, "device": p.device_info,
            "cublas_version": p.blas.version, "cublas_math_mode": p.blas.math_mode, "converted_wo_a_bf16_sha256": p.converted_sha256}
    def simple(op): return {"compiler": op.compiler, "cubin_sha256": op.cubin_sha256}
    return {"projections": {str(l): projection(p) for l, p in ps.items()}, "normalization": simple(norm),
            "compression": {**simple(comp), "cublas_version": comp.blas.version, "cublas_math_mode": comp.blas.math_mode},
            "indexer": {**simple(index), "cublas_version": index.blas.version, "cublas_math_mode": index.blas.math_mode},
            "compression_normalization": simple(comp.norm), "indexer_normalization": simple(index.norm),
            "attention_bundle_manifest_sha256": attn.manifest_sha256}


def cleanup(ops):
    ps, norm, comp, index, attn = ops
    return {**{f"projection_{l}": p.stats() for l, p in ps.items()}, "normalization": norm.stats(),
            "compression": comp.stats(), "indexer": index.stats(), "compression_normalization": comp.norm.stats(),
            "indexer_normalization": index.norm.stats(), "attention": attn.stats()}


def validate_goldens(report):
    from .attention_memory import validate_qualification
    from .checkpoint import MODEL, REVISION, REFERENCE_HASHES
    from .compressed_attention_fixture import LAYOUT, FIXTURE_BYTES
    from .compressed_attention_reference import METHODS
    from .compression import COMPRESSION_CUBIN_SHA256
    from .compression_fixture import COMPRESSION_BYTES, COMPRESSION_LAYOUT
    from .compression_reference import SOFTMAX_SOURCES, QUANTIZATION_CUDA
    from .gate_reference import GATE_SOURCE_HASHES
    from .indexer import INDEXER_CUBIN_SHA256
    from .indexer_fixture import INDEXER_BYTES, INDEXER_LAYOUT
    from .indexer_reference import INDEX_SOURCE_HASHES, INDEX_QUANT_CUDA, INDEX_QUERY_TRACE_CUDA
    from .normalization import NORM_CUBIN_SHA256
    from .normalization_reference import PYTORCH_SOURCES
    from .prefix import STORAGE_DTYPES
    from .prefix_reference import PREFIX_SOURCE_HASHES
    from .projections import PROJECTION_CUBIN_SHA256
    from .router import ROUTER_TORCH_REVISION
    from .router_reference import TOPK_SOURCE_HASHES
    if (report.get("probe") != "native_compressed_attention_parity" or report.get("status") != "pass"
            or report.get("model") != MODEL or report.get("revision") != REVISION or report.get("torch_git_version") != ROUTER_TORCH_REVISION
            or report.get("tilelang_version") != "0.1.12" or report.get("original_methods_ast_sha256") != METHODS
            or report.get("original_index_quant_cuda_sha256") != INDEX_QUANT_CUDA
            or report.get("original_compression_quant_cuda_sha256") != QUANTIZATION_CUDA
            or report.get("original_index_query_trace_cuda_sha256") != INDEX_QUERY_TRACE_CUDA
            or report.get("original_wo_a_conversion_ast_sha256") != "b818b92b3fda9929147cceeed5cde398b05b1c0c12e2e03adefca3a6a32d9df0"):
        raise ValueError("Expected passing pinned complete compressed Attention evidence")
    if any(report.get("reference", {}).get("files", {}).get(n) != h for n, h in {**REFERENCE_HASHES, **PREFIX_SOURCE_HASHES}.items()
           if n in ("kernel.py", "model.py", "convert.py", "config.json", "generate.py")):
        raise ValueError("Compressed Attention original source provenance differs")
    for field, hashes in (("blas_reference", GATE_SOURCE_HASHES), ("topk_reference", TOPK_SOURCE_HASHES),
            ("reduction_reference", {n: h for n, (_, h) in PYTORCH_SOURCES.items()}),
            ("index_reference", INDEX_SOURCE_HASHES), ("softmax_reference", SOFTMAX_SOURCES)):
        if any(report.get(field, {}).get("files", {}).get(n, {}).get("sha256") != h for n, h in hashes.items()):
            raise ValueError("Pinned compressed Attention dependency source differs")
    validate_qualification(report.get("attention_reference_qualification"))
    profile = report.get("native_profiles", {})
    p = profile.get("projections", {})
    if set(p) != {"2", "20", "24"} or profile.get("attention_bundle_manifest_sha256") != "b0e50760224c68bb2206bceff1f17d5ef40748e7b0df60b56efe96c7320a98c4":
        raise ValueError("Connected Attention binary bundle/projection profiles differ")
    if any(p[l].get("converted_wo_a_bf16_sha256") != h for l, h in CONVERTED_WO_A.items()):
        raise ValueError("Original converted grouped Attention weights differ")
    for name, row, cubin, source, blas in (
            *[("projection_" + l, r, PROJECTION_CUBIN_SHA256, "56ca03fb20f1208b226ddf810f2d508f4b35037e4ecb45b4e91a9c7384397262", True) for l, r in p.items()],
            ("compression", profile.get("compression", {}), COMPRESSION_CUBIN_SHA256, "4e4eee19934c224a2f4fb8bbf3eaf2c7213c1fb00dc349665a028c57641d898a", True),
            ("indexer", profile.get("indexer", {}), INDEXER_CUBIN_SHA256, "aaeade922ba2f4efb56e23f3320d8f45716e3d72aab0cefef908463b7f0d63e5", True),
            *[(n, profile.get(n, {}), NORM_CUBIN_SHA256, "357dc794add97f0d98a00dc72e10dfb054bdfc911dc4fda6c724c1098c764ff1", False)
              for n in ("normalization", "compression_normalization", "indexer_normalization")]):
        if (row.get("cubin_sha256") != cubin or row.get("compiler", {}).get("nvrtc_version") != [12, 4]
                or row.get("compiler", {}).get("concatenated_source_sha256") != source
                or (blas and (row.get("cublas_version") != 120402 or row.get("cublas_math_mode") != 0))):
            raise ValueError("Unregistered connected native profile: " + name)
    fixtures = report.get("fixtures", {})
    comp_layout = {n: ("BF16", s) for n, s in COMPRESSION_LAYOUT.items()}
    for name, kind, layout, size in (("attention", "original_compressed_attention_fixture", LAYOUT, FIXTURE_BYTES),
            ("compression", "original_compression_fixture", comp_layout, COMPRESSION_BYTES),
            ("indexer", "original_indexer_fixture", INDEXER_LAYOUT, INDEXER_BYTES)):
        f = fixtures.get(name, {})
        tensors = f.get("tensors", [])
        if (f.get("kind") != kind or f.get("model") != MODEL or f.get("revision") != REVISION or f.get("weights_fetched_bytes") != size
                or len(tensors) != len(layout) or {t.get("name") for t in tensors} != set(layout)):
            raise ValueError("Complete Attention fixture provenance is incomplete")
        for t in tensors:
            dtype, shape = layout[t["name"]]
            if t.get("dtype") != dtype or t.get("shape") != list(shape) or t.get("bytes") != int(np.prod(shape)) * STORAGE_DTYPES[dtype].itemsize:
                raise ValueError("Connected fixture tensor layout differs")
    rows, goldens = report.get("frame_cases", []), report.get("reference_goldens", [])
    expected = {m["id"]: m for m in expected_metadata()}
    indexed = {g["meta"]["id"]: g for g in goldens}
    if len(rows) != 103 or len(goldens) != 103 or len(indexed) != 103 or set(indexed) != set(expected) or len({r["id"] for r in rows}) != 103:
        raise ValueError("Complete compressed Attention corpus/goldens are incomplete")
    for row in rows:
        g = indexed[row["id"]]
        meta, hashes, checks = g["meta"], g.get("stage_sha256", {}), row.get("checks", [])
        names = boundary_names(expected[row["id"]])
        if (meta != expected[row["id"]] or {n: v for n, v in row.items() if n not in ("checks", "status")} != meta
                or row.get("status") != "pass" or set(hashes) != names or len(checks) != len(names) or {c.get("check") for c in checks} != names
                or any(c.get("status") != "pass" or c.get("mismatches") != 0 or c.get("expected_sha256") != c.get("actual_sha256")
                       or c.get("expected_sha256") != hashes.get(c["check"]) for c in checks)
                or any(not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h) for h in hashes.values())):
            raise ValueError("Complete Attention input, boundary coverage or exact comparisons differ")
    faults, events = report.get("faults", []), report.get("snapshot_events", [])
    if ([f.get("mode") for f in faults] != list(FAULTS) or any(f.get("status") != "pass" or f.get("position_preserved") is not True
            or f.get("temporary_buffers_released") is not True or len(f.get("checks", [])) != 17
            or any(c.get("status") != "pass" or c.get("mismatches") != 0 for c in f["checks"]) for f in faults)):
        raise ValueError("Atomic complete Attention failure evidence differs")
    if (len(events) != 1 or events[0].get("status") != "pass" or events[0].get("restored_position") != 3 or events[0].get("expired_or_duplicate_refusals") != 3
            or any(len(events[0].get(name, [])) != 17 or any(c.get("status") != "pass" or c.get("mismatches") != 0 for c in events[0][name])
                   for name in ("checks", "diagnostic_copy_checks"))):
        raise ValueError("Complete Attention snapshot/copy evidence differs")
    controls = report.get("unchanged_complete_Attention_controls", [])
    if (len(controls) != 103 or any(len(r) != 28 or any(c.get("status") != "pass" or c.get("mismatches") != 0 for c in r) for r in controls)
            or report.get("projection_instrumentation_controls") != {"2": 412, "20": 412, "24": 412}):
        raise ValueError("Unchanged full Attention controls are incomplete")
    for name, count in (("projection", 309), ("sum", 309), ("packed_vs_inplace", 473)):
        rows = report.get("indexer_instrumentation_controls", {}).get(name, [])
        if len(rows) != count or any(c.get("status") != "pass" or c.get("mismatches") != 0 for c in rows):
            raise ValueError("Original indexer diagnostic controls differ")
    rows = report.get("compression_packed_vs_inplace_controls", [])
    if len(rows) != 164 or any(c.get("status") != "pass" or c.get("mismatches") != 0 for c in rows):
        raise ValueError("Original compressed-KV packed/in-place controls differ")
    negatives = report.get("negative_controls", [])
    if ([n.get("check") for n in negatives] != ["omit_compressed_positions_from_Attention", "read_layer2_KV_instead_of_layer20_for_consumer24",
            "use_window_frequencies_for_compressed_query", "index_key_from_post_RoPE_rounded_latent"]
            or any(n.get("status") != "fail" or n.get("mismatches", 0) <= 0 for n in negatives)):
        raise ValueError("Connected compressed Attention negative controls are incomplete")
    memory = report.get("memory", {})
    cleaned = memory.get("cleanup", {})
    if (set(cleaned) != {"projection_2", "projection_20", "projection_24", "normalization", "compression", "indexer", "compression_normalization", "indexer_normalization", "attention"}
            or any(c.get("device_owned_bytes") != 0 or c.get("host_reserved_bytes") != 0 or c.get("owned_context_active") is not False for c in cleaned.values())
            or memory.get("all_frame_reservations_released") is not True or memory.get("state_storage") != "bounded_host_diagnostic_frame"):
        raise ValueError("Connected compressed Attention resource evidence differs")
    return indexed
