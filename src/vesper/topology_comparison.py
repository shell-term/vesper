"""Optional original/native sharing-topology qualification."""

from contextlib import ExitStack
import hashlib
import json
from pathlib import Path

from .attention_topology import GROUPS, role
from .attention_validation import check_arrays
from .topology_validation import exercise, registry, make_seed, flatten, compact_comparison, boundary_names


def group_name(layers): return "group-" + "-".join(map(str, layers))


def load_group(directory, layers):
    from .topology_fixture import load_layer
    from .resources import MemoryBudget, MiB
    budget = MemoryBudget(cap=2048 * MiB)
    budget.acquire(2048 * MiB)
    try:
        arrays, manifests = {}, []
        for layer in layers:
            values, manifest = load_layer(Path(directory) / str(layer), layer)
            arrays.update(values)
            manifests.append(manifest)
        return arrays, manifests
    finally: budget.release(2048 * MiB)


def connection_negatives(reference):
    from .attention_storage_reference import _numpy
    torch, values = reference.torch, reference.captured
    q, kv, indices = (reference.index.tensor(values[n]) for n in ("query", "combined_KV", "combined_indices"))
    sink = reference.p[reference.layers[-1]].sink
    expected = values["Attention_output"]
    omitted = indices.clone()
    omitted[..., 128:] = -1
    wrong_kv = kv.clone()
    n = kv.shape[1] - 128
    wrong_kv[:, 128:] = kv[:, :128].repeat(1, (n + 127) // 128, 1)[:, :n]
    unshifted = indices.clone()
    unshifted[..., 128:] -= 128
    return [check_arrays(expected, _numpy(reference.memory(q, kv, sink, omitted, 512**-0.5)), "omit_reused_compressed_indices"),
            check_arrays(expected, _numpy(reference.memory(q, wrong_kv, sink, indices, 512**-0.5)), "replace_shared_KV_with_consumer_window"),
            check_arrays(expected, _numpy(reference.memory(q, kv, sink, unshifted, 512**-0.5)), "omit_shared_index_offset")]


def compare_group(directory, fixtures, bundle, layers, ordinal=0, progress=None):
    from .attention import NativeAttention
    from .attention_memory_reference import AttentionMemoryReference, missing_dependencies
    from .checkpoint import MODEL, REVISION
    from .compressed_attention_validation import profiles, cleanup
    from .compression import NativeCompression
    from .compression_reference import QUANTIZATION_CUDA
    from .indexer import NativeIndexer
    from .indexer_reference import INDEX_QUANT_CUDA, INDEX_QUERY_TRACE_CUDA
    from .long_context_reference import compact_controls
    from .normalization import NativeNormRoPE
    from .projections import NativeProjections
    from .resources import MemoryBudget, MiB
    from .topology_reference import SharingOracle
    missing = missing_dependencies()
    if missing: return {"probe": "native_attention_topology_parity", "status": "blocked", "missing_dependencies": missing}
    import torch
    import tilelang
    budget = MemoryBudget(cap=2048 * MiB)
    budget.acquire(2048 * MiB)
    previous = torch.get_default_dtype()
    try:
        arrays, fixture = load_group(fixtures, layers)
        torch.set_default_dtype(torch.bfloat16)
        names, controls, control_names, seed_controls, negatives = registry(layers), [], [], [], []
        saved = None
        with torch.inference_mode(), torch.cuda.device(ordinal), torch.device(f"cuda:{ordinal}"), ExitStack() as stack:
            memory = stack.enter_context(AttentionMemoryReference(directory, ordinal, profile="long_context"))
            qualification = memory.qualify()
            if qualification["status"] != "pass": raise RuntimeError("Original sparse-Attention memory qualification failed")
            reference = SharingOracle(directory, arrays, layers, memory, ordinal)
            stack.callback(reference.close)
            ps = {l: stack.enter_context(NativeProjections(arrays, ordinal, layer=l)) for l in layers}
            norm = stack.enter_context(NativeNormRoPE(ordinal))
            comp = stack.enter_context(NativeCompression(arrays, ordinal, layers=tuple(l for l in layers if role(l).owns_kv)))
            index = stack.enter_context(NativeIndexer(arrays, ordinal, layers=tuple(l for l in layers if role(l).owns_index)))
            attn = stack.enter_context(NativeAttention(bundle, ordinal, profile="long_context"))
            ops = ps, norm, comp, index, attn
            if any(p.converted_sha256 != reference.p[l].converted_sha256 for l, p in ps.items()): raise RuntimeError("Original grouped weight conversion differs")
            def seed_reference(identity, actual, m):
                refops = reference.p, reference.norm, reference.comp_adapter, reference.index, None
                expected, em = make_seed(arrays, refops, layers, m["batch"], m["position"], m["seed"])
                right = flatten(actual)
                checks = [check_arrays(a, right[n], n) for n, a in flatten(expected).items()]
                if m != em or any(c["status"] != "pass" for c in checks): raise RuntimeError("Original/native diagnostic seed differs")
                seed_controls.append({"id": identity, "meta": em, "checks": checks})
                reference.prime_diagnostic(expected, m["position"])
            def checker(m, xs, actual):
                nonlocal control_names, negatives
                expected = reference.step(xs, m["position"])
                if set(actual) != boundary_names(layers, m): raise RuntimeError("Sharing boundary coverage differs from registered roles")
                full = reference.controls.pop()
                cn = sorted(c["check"] for c in full)
                if not control_names: control_names = cn
                if cn != control_names: raise RuntimeError("Original full sharing control names differ")
                controls.append(compact_controls(full, control_names))
                if m["id"] == "primed/b1/0": negatives = connection_negatives(reference)
                return compact_comparison(expected, actual, names)
            def lifecycle(event):
                nonlocal saved
                if event == "snapshot": saved = reference.snapshot()
                else: reference.rollback(saved)
            rows, faults, events, seeds, minimum = exercise(layers, arrays, ops, checker, reference.reset, seed_reference, lifecycle, progress)
            iq = {str(f): hashlib.sha256(reference.index.kernel.fp4_quant_kernel(128, 32, inplace=f).get_kernel_source().encode()).hexdigest() for f in (False, True)}
            cq = {str(f): hashlib.sha256(reference.comp.kernel.fp4_quant_kernel(512, 16, scale_dtype="float8_e4m3fn", inplace=f).get_kernel_source().encode()).hexdigest() for f in (False, True)}
            trace = hashlib.sha256(reference.index.trace_kernel.get_kernel_source().encode()).hexdigest()
            if iq != INDEX_QUANT_CUDA or cq != QUANTIZATION_CUDA or trace != INDEX_QUERY_TRACE_CUDA: raise RuntimeError("Original quantizer/query CUDA identity differs")
            counters = {"projection": {str(l): p.fp8_controls for l, p in reference.p.items()},
                "index_projection": len(reference.index.projection_controls), "index_sum": len(reference.index.sum_controls),
                "index_packed_vs_inplace": len(reference.index.quant_controls), "compressed_packed_vs_inplace": len(reference.comp.quantization_controls)}
            native_profiles = profiles(ops)
        cleaned = cleanup(ops)
        passed = (all(r["status"] == "pass" for r in rows + faults + events + controls)
            and len(negatives) == 3 and all(n["mismatches"] > 0 for n in negatives)
            and all(c["device_owned_bytes"] == 0 and c["host_reserved_bytes"] == 0 and not c["owned_context_active"] for c in cleaned.values()))
        return {"schema_version": 1, "probe": "native_attention_topology_parity", "status": "pass" if passed else "fail",
            "model": MODEL, "revision": REVISION, "layers": list(layers), "fixtures": fixture,
            "reference": reference.p[layers[0]].source, "blas_reference": reference.p[layers[0]].blas_source,
            "topk_reference": reference.index.topk_source, "reduction_reference": reference.index.reduction_source,
            "index_reference": reference.index.index_source, "softmax_reference": reference.comp.softmax_source,
            "original_methods_ast_sha256": reference.methods, "original_wo_a_conversion_ast_sha256": reference.p[layers[0]].methods["wo_a_conversion"],
            "original_index_quant_cuda_sha256": iq, "original_compression_quant_cuda_sha256": cq, "original_index_query_trace_cuda_sha256": trace,
            "attention_reference_qualification": qualification, "native_profiles": native_profiles,
            "boundary_names": names, "boundary_columns": ["name_id", "storage_elements", "mismatches", "expected_sha256"],
            "frame_cases": rows, "original_control_names": control_names, "unchanged_complete_Attention_controls": controls,
            "instrumentation_counts": counters, "diagnostic_seeds": seeds, "original_seed_controls": seed_controls,
            "faults": faults, "snapshot_events": events, "negative_controls": negatives, "torch_version": torch.__version__, "torch_git_version": torch.version.git_version,
            "torch_cuda_version": torch.version.cuda, "tilelang_version": tilelang.__version__, "reference_runtime_workarounds": reference.index.workarounds,
            "memory": {"cleanup": cleaned, "combined_native_host_allowance_bytes": (512 * len(layers) + 3872) * MiB,
                "combined_native_and_reference_host_allowance_bytes": (512 * len(layers) + 4960) * MiB,
                "retained_fixture_host_allowance_bytes": 2048 * MiB,
                "minimum_sampled_host_available_bytes": minimum, "all_frame_reservations_released": True,
                "state_storage": "bounded_host_diagnostic_frame", "reservation_is_RSS_limit": False},
            "weights_fetched_during_comparison_bytes": 0, "scope": "Named original Attention sharing roles; generated independent layer inputs and explicit synthetic history; no complete Transformer blocks",
            "maximum_unprimed_continuous_context": 260, "maximum_seeded_context": 1028, "capacity": 1152,
            "untouched_full_Attention_kernel": "blocked_by_shared_memory_limit", "full_model_quality": "not_run", "end_to_end_speed": "not_measured"}
    finally:
        torch.set_default_dtype(previous)
        budget.release(2048 * MiB)
