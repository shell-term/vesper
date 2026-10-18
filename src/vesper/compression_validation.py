"""Shared deterministic compressor/quantization corpus and recovery checks."""

import numpy as np

from .attention_validation import array_digest, check_arrays
from .compression import NativeCompressedKV
from .formats import bf16_bits
from .normalization_validation import inputs
from .resources import physical_memory


def quantization_cases():
    for batch in (1, 2):
        for mode in ("normal", "zeros", "signed_zeros", "near_eps", "bf16_subnormal"):
            x = inputs((batch, 3, 512), 400 + batch, mode)
            yield {"id": f"quant/{batch}/{mode}", "operation": "quant", "mode": mode, "batch": batch}, (x,)
    bits = np.arange(65536, dtype=np.uint16)
    bits = bits[(bits & 0x7fff) <= 0x4528]
    for start in range(0, bits.size, 8192):
        x = np.resize(bits[start:start + 8192], 8192).reshape(1, 16, 512).copy()
        yield {"id": f"quant/finite-encodings/{start}", "operation": "quant", "encoding_start": start}, (x,)
    levels = np.array([0., .25, .5, .75, 1., 1.25, 1.5, 1.75, 2., 2.5, 3., 3.5, 4., 5., 6., -0.], np.float32)
    for scale in (2**-9, 2**-8, .5, 1., 1.5, 32., 448.):
        x = np.resize(levels * np.float32(scale), (1, 2, 512)).copy()
        x[:, 1] *= np.float32(-1)
        x = bf16_bits(x)
        yield {"id": f"quant/FP4-ties/{scale}", "operation": "quant", "scale": scale}, (x,)
    # Alternating adjacent BF16 magnitudes also cross E4M3 scale rounding steps.
    for seed in (1, 2, 3):
        rng = np.random.default_rng(seed)
        a = rng.integers(0x3b00, 0x4529, (2, 16, 512), dtype=np.uint16)
        a[..., 1::2] |= np.uint16(0x8000)
        yield {"id": f"quant/scale-boundaries/{seed}", "operation": "quant", "seed": seed}, (a,)


def pool_cases():
    for batch, groups in ((1, 1), (1, 3), (2, 1), (2, 8)):
        for mode in ("normal", "zeros", "signed_zeros", "saturated_scores", "near_ties", "cancellation"):
            rng = np.random.default_rng(20261018 + batch * 100 + groups)
            shape = (batch, groups, 2, 512)
            kv, score = [rng.standard_normal(shape).astype(np.float32) for _ in range(2)]
            if mode in ("zeros", "signed_zeros"):
                kv.fill(0)
                score.fill(0)
                if mode == "signed_zeros":
                    kv.view(np.uint32)[..., 1::2] = 0x80000000
            elif mode == "saturated_scores":
                score[:, :, 0], score[:, :, 1] = np.float32(1000), np.float32(-1000)
            elif mode == "near_ties":
                score[:, :, 1] = np.nextafter(score[:, :, 0], np.float32(np.inf))
            elif mode == "cancellation":
                kv[:, :, 0], kv[:, :, 1] = np.float32(2**24), np.float32(-2**24)
                score *= np.float32(1e-7)
            yield {"id": f"pool/{batch}/{groups}/{mode}", "operation": "pool", "batch": batch, "groups": groups, "mode": mode}, (kv, score)


def component_cases():
    yield from quantization_cases()
    yield from pool_cases()


def component_meta(meta, args):
    return {**meta, "inputs_sha256": [array_digest(a) for a in args], "input_shapes": [list(a.shape) for a in args]}


def exercise_components(native, checker):
    rows, minimum = [], physical_memory()["available"]
    owned = sum(native.device.allocations.values())
    for meta, args in component_cases():
        meta = component_meta(meta, args)
        actual = native.round_kv(*args) if meta["operation"] == "quant" else native.pool(*args)
        checks = checker(meta, args, actual)
        rows.append({**meta, "checks": checks, "status": "pass" if all(c["status"] == "pass" for c in checks) else "fail"})
        minimum = min(minimum, physical_memory()["available"])
        if sum(native.device.allocations.values()) != owned or native.norm.device.allocations:
            raise RuntimeError("Compression component retained temporary allocations")
    return rows, minimum


def streams():
    for layer in (2, 20):
        for batch in (1, 2):
            for prefill in (1, 2, 3, 7, 8, 15, 16):
                yield {"stream": f"core/{layer}/{batch}/{prefill}", "layer": layer, "batch": batch,
                       "prefill": prefill, "decodes": 4, "mode": "normal"}
            for mode in ("zeros", "signed_zeros", "near_eps", "mixed_scales"):
                yield {"stream": f"stress/{layer}/{batch}/{mode}", "layer": layer, "batch": batch,
                       "prefill": 3, "decodes": 2, "mode": mode}
        yield {"stream": f"continuous/{layer}", "layer": layer, "batch": 1,
               "prefill": 16, "decodes": 112, "mode": "normal"}


def stream_steps(config):
    positions = [(0, config["prefill"])] + [(config["prefill"] + d, 1) for d in range(config["decodes"])]
    for index, (position, length) in enumerate(positions):
        value = inputs((config["batch"], length, 5120), 8000 + config["layer"] * 100 + position, config["mode"])
        yield {"id": f"state/{config['stream']}/{index}", "stream": config["stream"], "layer": config["layer"],
               "batch": config["batch"], "position": position, "sequence": length, "mode": config["mode"],
               "input_sha256": array_digest(value)}, value


def state_stage_names(layer, position, sequence):
    result = {"wkv_fp32", "wgate_fp32"} if layer == 2 else {"wkv_bf16"}
    ratio = 2 if layer == 2 else 1
    emits = sequence >= ratio if position == 0 else (position + 1) % ratio == 0
    if emits:
        if layer == 2:
            result.update("pool_" + n for n in ("maximum_fp32", "exponents_fp32", "sum_exp_fp32", "weights_fp32", "products_fp32", "pooled_fp32", "pooled_bf16"))
        result.update("norm_" + n for n in ("square_fp32", "mean_fp32", "mean_eps_fp32", "rstd_fp32", "normalized_fp32", "weighted_fp32", "output_bf16"))
        result.update(("rope_frequency_fp32", "rope_rotated_fp32", "rope_output_bf16", "quant_packed_fp4", "quant_scales_e4m3", "quant_rounded_bf16"))
    result.update(("latent_pre_rope_bf16", "latent_positions_i32", "kv_state_fp32", "score_state_fp32", "cache_bf16", "visible_bf16"))
    return result


def exercise_states(native, checker, reset):
    rows, minimum = [], physical_memory()["available"]
    owned = sum(native.device.allocations.values())
    for config in streams():
        reset(config["layer"], config["batch"])
        with NativeCompressedKV(native, config["layer"], config["batch"]) as state:
            for meta, value in stream_steps(config):
                actual = state.step(value, start_pos=meta["position"])
                if set(actual) != state_stage_names(meta["layer"], meta["position"], meta["sequence"]):
                    raise RuntimeError("Incomplete native compression trace")
                checks = checker(meta, value, actual)
                rows.append({**meta, "checks": checks, "status": "pass" if all(c["status"] == "pass" for c in checks) else "fail"})
                minimum = min(minimum, physical_memory()["available"])
                if sum(native.device.allocations.values()) != owned or native.norm.device.allocations:
                    raise RuntimeError("Compression state call retained temporary allocations")
        if state.budget.reserved:
            raise RuntimeError("Compression state reservation was not released")
    return rows, minimum


def exercise_failures(native, checker, reset):
    rows, faults = [], []
    owned = sum(native.device.allocations.values())
    for mode in ("host_pressure", "projection_allocation", "rotary_failure", "rounded_download", "publication_pressure"):
        reset(2, 1)
        with NativeCompressedKV(native, 2, 1) as state:
            prefix = inputs((1, 3, 5120), 19000)
            value = inputs((1, 1, 5120), 19003)
            def record(suffix, x, pos):
                meta = {"id": f"fault/{mode}/{suffix}", "stream": f"fault/{mode}", "layer": 2, "batch": 1,
                        "position": pos, "sequence": x.shape[1], "mode": "normal", "input_sha256": array_digest(x)}
                actual = state.step(x, start_pos=pos)
                checks = checker(meta, x, actual)
                rows.append({**meta, "checks": checks, "status": "pass" if all(c["status"] == "pass" for c in checks) else "fail"})
            record("prefill", prefix, 0)
            before = state.inspect()
            target, attribute = (state.budget, "acquire")
            if mode == "projection_allocation": target, attribute = native.device, "allocate"
            elif mode == "rotary_failure": target, attribute = native.norm, "rope"
            elif mode == "rounded_download": target, attribute = native, "round_kv"
            original, calls = getattr(target, attribute), 0
            def injected(*args, **kwargs):
                nonlocal calls
                calls += 1
                if mode == "rounded_download":
                    download = native.device.download
                    def refuse(pointer, shape, dtype):
                        if dtype == np.uint16:
                            raise OSError("Injected compressed-KV rounded download failure")
                        return download(pointer, shape, dtype)
                    native.device.download = refuse
                    try: return original(*args, **kwargs)
                    finally: native.device.download = download
                if mode != "publication_pressure" or calls == 2:
                    raise MemoryError("Injected bounded compression failure")
                return original(*args, **kwargs)
            setattr(target, attribute, injected)
            try:
                try:
                    state.step(value)
                except (MemoryError, OSError):
                    pass
                else:
                    raise RuntimeError("Injected compression failure did not refuse publication")
            finally:
                setattr(target, attribute, original)
            after = state.inspect()
            checks = [check_arrays(before[n], after[n], n) for n in ("kv_state_fp32", "score_state_fp32", "cache_bf16")]
            passed = before["position"] == after["position"] and all(c["status"] == "pass" for c in checks) and sum(native.device.allocations.values()) == owned
            faults.append({"mode": mode, "checks": checks, "position_unchanged": before["position"] == after["position"],
                           "temporary_allocations_released": sum(native.device.allocations.values()) == owned, "status": "pass" if passed else "fail"})
            record("retry", value, 3)
    return rows, faults


def exercise_snapshots(native, checker, reset, restore_reference):
    rows, events = [], []
    for layer in (2, 20):
        reset(layer, 1)
        with NativeCompressedKV(native, layer, 1) as state:
            prefix = inputs((1, 3, 5120), 22000 + layer)
            def record(label, value, pos):
                meta = {"id": f"snapshot/{layer}/{label}", "stream": f"snapshot/{layer}", "layer": layer, "batch": 1,
                        "position": pos, "sequence": value.shape[1], "mode": "normal", "input_sha256": array_digest(value)}
                actual = state.step(value, start_pos=pos)
                checks = checker(meta, value, actual)
                rows.append({**meta, "checks": checks, "status": "pass" if all(c["status"] == "pass" for c in checks) else "fail"})
            record("prefill", prefix, 0)
            saved = state.inspect()
            handle = state.snapshot()
            for label, pos in (("advance", 3), ("advance-again", 4)):
                record(label, inputs((1, 1, 5120), 22000 + layer + pos), pos)
            restored = state.rollback(handle)
            checks = [check_arrays(saved[n], restored[n], n) for n in ("kv_state_fp32", "score_state_fp32", "cache_bf16")]
            reset(layer, 1)
            restore_reference(prefix)
            record("after-rollback", inputs((1, 1, 5120), 22000 + layer + 3), 3)
            invalidations = 0
            try: state.discard(handle)
            except ValueError: invalidations += 1
            other = state.snapshot()
            try: state.snapshot()
            except RuntimeError: invalidations += 1
            state.reset()
            try: state.rollback(other)
            except ValueError: invalidations += 1
            events.append({"layer": layer, "checks": checks, "restored_position": restored["position"],
                           "invalid_or_duplicate_snapshot_refusals": invalidations,
                           "status": "pass" if restored["position"] == saved["position"] and invalidations == 3 and all(c["status"] == "pass" for c in checks) else "fail"})
    return rows, events


def expected_metadata():
    for meta, args in component_cases():
        yield component_meta(meta, args)
    for config in streams():
        for meta, _ in stream_steps(config):
            yield meta
    for mode in ("host_pressure", "projection_allocation", "rotary_failure", "rounded_download", "publication_pressure"):
        for label, length, position, seed in (("prefill", 3, 0, 19000), ("retry", 1, 3, 19003)):
            yield {"id": f"fault/{mode}/{label}", "stream": f"fault/{mode}", "layer": 2, "batch": 1,
                   "position": position, "sequence": length, "mode": "normal", "input_sha256": array_digest(inputs((1, length, 5120), seed))}
    for layer in (2, 20):
        for label, length, position in (("prefill", 3, 0), ("advance", 1, 3), ("advance-again", 1, 4), ("after-rollback", 1, 3)):
            yield {"id": f"snapshot/{layer}/{label}", "stream": f"snapshot/{layer}", "layer": layer, "batch": 1,
                   "position": position, "sequence": length, "mode": "normal",
                   "input_sha256": array_digest(inputs((1, length, 5120), 22000 + layer + position))}


def validate_goldens(report):
    from .checkpoint import MODEL, REVISION, REFERENCE_HASHES
    from .compression import COMPRESSION_CUBIN_SHA256, POOL_STAGES
    from .compression_fixture import COMPRESSION_BYTES, COMPRESSION_LAYOUT
    from .compression_reference import SOFTMAX_SOURCES, COMPRESSION_METHODS, QUANTIZATION_CUDA
    from .gate_reference import GATE_SOURCE_HASHES
    from .normalization import NORM_CUBIN_SHA256
    from .normalization_reference import PYTORCH_SOURCES
    from .prefix_reference import PREFIX_SOURCE_HASHES
    from .router import ROUTER_TORCH_REVISION
    profile = report.get("compression_profile", {})
    if (report.get("status") != "pass" or report.get("probe") != "native_compression_parity"
            or report.get("model") != MODEL or report.get("revision") != REVISION
            or report.get("torch_git_version") != ROUTER_TORCH_REVISION or report.get("tilelang_version") != "0.1.12"
            or profile.get("cubin_sha256") != COMPRESSION_CUBIN_SHA256
            or profile.get("cublas_version") != 120402 or profile.get("cublas_math_mode") != 0
            or profile.get("compiler", {}).get("nvrtc_version") != [12, 4]
            or profile.get("compiler", {}).get("concatenated_source_sha256") != "4e4eee19934c224a2f4fb8bbf3eaf2c7213c1fb00dc349665a028c57641d898a"
            or report.get("normalization_profile", {}).get("cubin_sha256") != NORM_CUBIN_SHA256
            or report.get("normalization_profile", {}).get("compiler", {}).get("nvrtc_version") != [12, 4]
            or report.get("original_methods_ast_sha256") != COMPRESSION_METHODS
            or report.get("original_quant_cuda_sha256") != QUANTIZATION_CUDA):
        raise ValueError("Expected passing registered original/native compression evidence")
    if (any(report.get("reference", {}).get("files", {}).get(n) != h for n, h in {**REFERENCE_HASHES, **PREFIX_SOURCE_HASHES}.items() if n in ("kernel.py", "model.py", "convert.py", "generate.py", "config.json"))
            or any(report.get("blas_reference", {}).get("files", {}).get(n, {}).get("sha256") != h for n, h in GATE_SOURCE_HASHES.items())
            or any(report.get("reduction_reference", {}).get("files", {}).get(n, {}).get("sha256") != h for n, (_, h) in PYTORCH_SOURCES.items())
            or any(report.get("softmax_reference", {}).get("files", {}).get(n, {}).get("sha256") != h for n, h in SOFTMAX_SOURCES.items())):
        raise ValueError("Original compression source provenance differs")
    fixture = report.get("fixture", {})
    tensors = fixture.get("tensors", [])
    if (fixture.get("kind") != "original_compression_fixture" or fixture.get("model") != MODEL or fixture.get("revision") != REVISION
            or fixture.get("layers_and_ratios") != {"2": 2, "20": 1} or fixture.get("weights_fetched_bytes") != COMPRESSION_BYTES
            or len(tensors) != 5 or {e.get("name") for e in tensors} != set(COMPRESSION_LAYOUT)):
        raise ValueError("Original bounded compressor fixture provenance is incomplete")
    for tensor in tensors:
        shape = COMPRESSION_LAYOUT[tensor["name"]]
        if tensor.get("dtype") != "BF16" or tensor.get("shape") != list(shape) or tensor.get("bytes") != int(np.prod(shape)) * 2:
            raise ValueError("Original compressor fixture layout differs")
    components, states = report.get("component_cases", []), report.get("state_cases", [])
    goldens = report.get("reference_goldens", [])
    expected = {m["id"]: m for m in expected_metadata()}
    indexed = {g["meta"]["id"]: g for g in goldens}
    if (len(components) != 49 or len(states) != 432 or len(goldens) != 481 or len(indexed) != 481
            or set(indexed) != set(expected) or len({r["id"] for r in components + states}) != 481):
        raise ValueError("Compression corpus or original golden coverage is incomplete")
    for row in components + states:
        golden = indexed[row["id"]]
        meta = golden["meta"]
        if meta != expected[row["id"]] or {k: v for k, v in row.items() if k not in ("checks", "status")} != meta:
            raise ValueError("Compression input/position metadata differs")
        if meta.get("operation") == "quant":
            names = {"packed_fp4", "scales_e4m3", "rounded_bf16"}
        elif meta.get("operation") == "pool":
            names = set(POOL_STAGES)
        else:
            names = state_stage_names(meta["layer"], meta["position"], meta["sequence"])
        hashes, checks = golden.get("stage_sha256", {}), row.get("checks", [])
        if (row.get("status") != "pass" or set(hashes) != names or len(checks) != len(names) or {c.get("check") for c in checks} != names
                or any(c.get("status") != "pass" or c.get("mismatches") != 0 or c.get("expected_sha256") != c.get("actual_sha256")
                       or c.get("expected_sha256") != hashes.get(c["check"]) for c in checks)
                or any(not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h) for h in hashes.values())):
            raise ValueError("Compression golden boundary set or exact comparisons differ")
    faults = report.get("faults", [])
    if ([f.get("mode") for f in faults] != ["host_pressure", "projection_allocation", "rotary_failure", "rounded_download", "publication_pressure"]
            or any(f.get("status") != "pass" or f.get("position_unchanged") is not True or f.get("temporary_allocations_released") is not True
                   or len(f.get("checks", [])) != 3 or any(c.get("status") != "pass" or c.get("mismatches") != 0 for c in f["checks"]) for f in faults)):
        raise ValueError("Compression failure/refusal evidence is incomplete")
    events = report.get("snapshot_events", [])
    if ([e.get("layer") for e in events] != [2, 20] or any(e.get("status") != "pass" or e.get("restored_position") != 3
            or e.get("invalid_or_duplicate_snapshot_refusals") != 3 or len(e.get("checks", [])) != 3
            or any(c.get("status") != "pass" or c.get("mismatches") != 0 for c in e["checks"]) for e in events)):
        raise ValueError("Compression snapshot evidence is incomplete")
    controls, quant = report.get("unchanged_original_controls", []), report.get("original_packed_vs_inplace_controls", [])
    if (len(controls) != 434 or len(quant) != 364 or any(c.get("empty_index_callback") is not True or not c.get("checks")
            or any(k.get("status") != "pass" or k.get("mismatches") != 0 for k in c["checks"]) for c in controls)
            or any(c.get("status") != "pass" or c.get("mismatches") != 0 for c in quant)):
        raise ValueError("Unchanged original compression controls are incomplete")
    negatives = report.get("negative_controls", [])
    if ([n.get("check") for n in negatives] != ["replace_ratio2_FP32_projection_with_BF16", "round_pool_products_to_BF16_early",
            "replace_compressed_KV_group16_with_group32", "replace_compressed_KV_E4M3_scales_with_E8M0", "rotate_compressed_group_at_last_token"]
            or any(n.get("status") != "fail" or n.get("mismatches", 0) <= 0 for n in negatives)):
        raise ValueError("Compression arithmetic negative controls are incomplete")
    memory = report.get("memory", {})
    for key in ("compression_cleanup", "normalization_cleanup"):
        c = memory.get(key, {})
        if c.get("host_reserved_bytes") != 0 or c.get("device_owned_bytes") != 0 or c.get("owned_context_active") is not False:
            raise ValueError("Compression owned buffers or context retains were not released")
    if memory.get("all_state_reservations_released") is not True or memory.get("publication_cache") != "bounded_host_diagnostic_copy":
        raise ValueError("Compression state memory accounting differs")
    return indexed
