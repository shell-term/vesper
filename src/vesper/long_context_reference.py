"""Pinned original controls for long-context connected compressed Attention."""

from contextlib import ExitStack
import hashlib
import json
from pathlib import Path

import numpy as np

from .attention_storage_reference import _numpy
from .attention_validation import check_arrays
from .long_context_validation import PROFILE, compact_comparison, exercise_long, make_seed, registry


def compact_controls(checks, names):
    ids = {n: i for i, n in enumerate(names)}
    rows = [[ids[c["check"]], c["elements"], c["mismatches"], c["expected_sha256"]] for c in sorted(checks, key=lambda c: c["check"])]
    actual = [[ids[c["check"]], c["actual_sha256"]] for c in sorted(checks, key=lambda c: c["check"])]
    expected = [[r[0], r[3]] for r in rows]
    return {"boundaries": rows, "expected_boundary_root_sha256": hashlib.sha256(json.dumps(expected, separators=(",", ":")).encode()).hexdigest(),
            "actual_boundary_root_sha256": hashlib.sha256(json.dumps(actual, separators=(",", ":")).encode()).hexdigest(),
            "status": "pass" if all(c["status"] == "pass" for c in checks) else "fail"}


def connection_negatives(reference):
    """This seed forces a source/consumer inversion so pruning reaches output."""
    torch, stage = reference.torch, reference.last_stages
    q, kv = (reference.index.tensor(reference.captured[n]) for n in ("query_bf16", "KV_bf16"))
    indices = torch.from_numpy(stage["layer24_combined_indices_i32"]).to(q.device)
    sink = reference.p[24].sink
    score = stage["index_layer24_score_score_bf16"]
    lengths = np.full(score.shape[:2], score.shape[-1], np.int32)
    unfiltered = reference.index.select(score, lengths, 128)["indices_i32"]
    wrong = torch.cat((indices[..., :128], torch.from_numpy(unfiltered).to(q.device)), dim=-1)
    out = _numpy(reference.memory(q, kv, sink, wrong, 512**-0.5))
    unshifted = indices.clone()
    unshifted[..., 128:] -= 128
    offset = _numpy(reference.memory(q, kv, sink, unshifted, 512**-0.5))
    reversed_window = indices.clone()
    reversed_window[..., :128] = reversed_window[..., :128].flip(-1)
    _, traces, _ = reference.memory._run(64, 512, True, (q, kv, sink, reversed_window))
    return [check_arrays(stage["index_layer24_selection_indices_i32"], unfiltered, "ignore_source_candidates_selected_indices"),
            check_arrays(stage["layer24_attention_output_bf16"], out, "ignore_source_candidates_Attention_output"),
            check_arrays(stage["layer24_attention_output_bf16"], offset, "omit_compressed_index_window_offset"),
            check_arrays(stage["layer24_attention_qk_fp32"], _numpy(traces[0]), "reverse_window_slot_order_QK")]


def compare_long_context(directory, fixture_path, compression_path, index_path, bundle, ordinal=0, progress=None):
    from .attention import NativeAttention
    from .attention_memory_reference import AttentionMemoryReference, missing_dependencies
    from .checkpoint import MODEL, REVISION
    from .compressed_attention_fixture import load_compressed_attention_fixture
    from .compressed_attention_reference import CompressedAttentionOracle
    from .compressed_attention_validation import profiles, cleanup, flatten_state
    from .compression import NativeCompression
    from .compression_fixture import load_compression_fixture
    from .compression_reference import QUANTIZATION_CUDA
    from .indexer import NativeIndexer
    from .indexer_fixture import load_indexer_fixture
    from .indexer_reference import INDEX_QUANT_CUDA, INDEX_QUERY_TRACE_CUDA
    from .normalization import NativeNormRoPE
    from .projections import NativeProjections
    from .resources import MemoryBudget, MiB
    missing = missing_dependencies()
    if missing: return {"probe": "native_long_context_parity", "status": "blocked", "missing_dependencies": missing}
    import torch
    import tilelang
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    budget = MemoryBudget(cap=1024 * MiB)
    try:
        budget.acquire(1024 * MiB)
        arrays, fixture = load_compressed_attention_fixture(fixture_path)
        comp_arrays, comp_fixture = load_compression_fixture(compression_path)
        index_arrays, index_fixture = load_indexer_fixture(index_path)
        names, full_controls, control_names, seed_controls, negatives = registry(), [], [], [], []
        saved = None
        with torch.inference_mode(), torch.cuda.device(ordinal), torch.device(f"cuda:{ordinal}"), ExitStack() as stack:
            memory = stack.enter_context(AttentionMemoryReference(directory, ordinal, profile=PROFILE))
            qualification = memory.qualify()
            if qualification["status"] != "pass": raise RuntimeError("Original Attention memory qualification failed")
            reference = CompressedAttentionOracle(directory, arrays, comp_arrays, index_arrays, memory, ordinal)
            stack.callback(reference.close)
            ps = {l: stack.enter_context(NativeProjections(arrays, ordinal, layer=l)) for l in (2, 20, 24)}
            norm = stack.enter_context(NativeNormRoPE(ordinal))
            comp = stack.enter_context(NativeCompression(comp_arrays, ordinal))
            index = stack.enter_context(NativeIndexer(index_arrays, ordinal))
            attn = stack.enter_context(NativeAttention(bundle, ordinal, profile=PROFILE))
            ops = ps, norm, comp, index, attn
            if any(ps[l].converted_sha256 != reference.p[l].converted_sha256 for l in ps): raise RuntimeError("Original grouped weight conversion differs")
            def reset(batch, capacity): reference.reset(batch, profile=PROFILE, capacity=capacity)
            def seed_reference(identity, actual, meta):
                refops = reference.p, reference.norm, reference.comp_adapter, reference.index, reference.diagnostic.attention
                expected, expected_meta = make_seed(arrays, refops, meta["batch"], meta["position"], meta["seed"], capacity=meta["cache_capacity"])
                if meta != expected_meta: raise RuntimeError("Native/original diagnostic seed bank identity differs")
                right = flatten_state(actual)
                checks = [check_arrays(a, right[n], n) for n, a in flatten_state(expected).items()]
                if any(c["status"] != "pass" for c in checks): raise RuntimeError("Native/original diagnostic history bits differ")
                seed_controls.append({"id": identity, "meta": expected_meta, "checks": checks})
                reference.prime_diagnostic(expected, meta["position"])
            def checker(meta, xs, actual):
                expected = reference.step(xs, meta["position"])
                control = reference.controls.pop()
                nonlocal control_names, negatives
                current = sorted(c["check"] for c in control)
                if not control_names: control_names = current
                if current != control_names: raise ValueError("Complete original long control set differs")
                full_controls.append(compact_controls(control, control_names))
                row = compact_comparison(expected, actual, names)
                if meta["id"] == "frame/primed/16384/0": negatives = connection_negatives(reference)
                return row
            def lifecycle(event):
                nonlocal saved
                if event == "snapshot": saved = reference.snapshot()
                elif event == "rollback": reference.rollback(saved)
            rows, faults, events, seeds, minimum = exercise_long(arrays, ops, checker, reset, seed_reference, lifecycle, progress)
            index_quant = {str(f): hashlib.sha256(reference.index.kernel.fp4_quant_kernel(128, 32, inplace=f).get_kernel_source().encode()).hexdigest() for f in (False, True)}
            comp_quant = {str(f): hashlib.sha256(reference.comp.kernel.fp4_quant_kernel(512, 16, scale_dtype="float8_e4m3fn", inplace=f).get_kernel_source().encode()).hexdigest() for f in (False, True)}
            trace = hashlib.sha256(reference.index.trace_kernel.get_kernel_source().encode()).hexdigest()
            if index_quant != INDEX_QUANT_CUDA or comp_quant != QUANTIZATION_CUDA or trace != INDEX_QUERY_TRACE_CUDA: raise ValueError("Original generated quantizer/query profile differs")
            native_profiles = profiles(ops)
            counters = {"projection": {str(l): p.fp8_controls for l, p in reference.p.items()},
                "index_projection": len(reference.index.projection_controls), "index_sum": len(reference.index.sum_controls),
                "index_packed_vs_inplace": len(reference.index.quant_controls), "compressed_packed_vs_inplace": len(reference.comp.quantization_controls)}
        cleaned = cleanup(ops)
        passed = (all(r["status"] == "pass" for r in rows + faults + events + full_controls)
                  and len(negatives) == 4 and all(n["mismatches"] > 0 for n in negatives)
                  and all(not c["device_owned_bytes"] and not c["host_reserved_bytes"] and not c["owned_context_active"] for c in cleaned.values()))
        return {"schema_version": 1, "probe": "native_long_context_parity", "status": "pass" if passed else "fail", "model": MODEL, "revision": REVISION,
            "reference": reference.p[2].source, "blas_reference": reference.p[2].blas_source, "topk_reference": reference.index.topk_source,
            "reduction_reference": reference.index.reduction_source, "index_reference": reference.index.index_source, "softmax_reference": reference.comp.softmax_source,
            "fixtures": {"attention": fixture, "compression": comp_fixture, "indexer": index_fixture}, "native_profiles": native_profiles,
            "original_methods_ast_sha256": reference.methods, "original_wo_a_conversion_ast_sha256": reference.p[2].methods["wo_a_conversion"],
            "original_index_quant_cuda_sha256": index_quant, "original_compression_quant_cuda_sha256": comp_quant, "original_index_query_trace_cuda_sha256": trace,
            "attention_reference_qualification": qualification, "boundary_names": names,
            "boundary_columns": ["name_id", "storage_elements", "mismatches", "expected_sha256"], "frame_cases": rows,
            "original_control_names": control_names, "unchanged_complete_Attention_controls": full_controls, "instrumentation_counts": counters,
            "diagnostic_seeds": seeds, "original_seed_controls": seed_controls, "faults": faults, "snapshot_events": events, "negative_controls": negatives,
            "execution_limits": {"profile": PROFILE, "context_capacities": [1152, 16392], "attention_KV_rows": 16520, "attention_index_width": 640,
                "long_queries_per_call": 1, "prefill_queries": 3, "batch": [1, 2], "attention_owned_device_cap_bytes": 64 * MiB},
            "torch_version": torch.__version__, "torch_git_version": torch.version.git_version, "torch_cuda_version": torch.version.cuda, "tilelang_version": tilelang.__version__,
            "reference_runtime_workarounds": reference.index.workarounds,
            "memory": {"cleanup": cleaned, "combined_native_host_allowance_bytes": 3488 * MiB,
                "combined_native_and_reference_host_allowance_bytes": 5728 * MiB, "minimum_sampled_host_available_bytes": minimum,
                "all_frame_reservations_released": True, "state_storage": "bounded_host_diagnostic_frame", "reservation_is_RSS_limit": False},
            "weights_fetched_during_comparison_bytes": 0, "new_weight_bytes": 0,
            "scope": "Original Attention 2/20/24, unprimed generated-input continuous 1152/260-token streams and explicitly seeded histories up to 16392 positions; not full model activations/quality",
            "untouched_full_Attention_kernel": "blocked_by_shared_memory_limit", "other_layers": "not_qualified", "full_model_quality": "not_run", "end_to_end_speed": "not_measured"}
    finally:
        if budget.reserved: budget.release(budget.reserved)
        torch.set_default_dtype(previous)


def write_report(path, report):
    """Keep every boundary while avoiding repeated field labels/whitespace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    keys = list(report)
    for key in keys:
        value = report[key]
        if key in ("frame_cases", "unchanged_complete_Attention_controls"):
            rendered = "[\n" + ",\n".join("  " + json.dumps(r, separators=(",", ":"), allow_nan=False) for r in value) + "\n]"
        else: rendered = json.dumps(value, indent=2, allow_nan=False)
        lines.append(json.dumps(key) + ": " + rendered)
    path.write_bytes(("{\n" + ",\n".join(lines) + "\n}\n").encode())
