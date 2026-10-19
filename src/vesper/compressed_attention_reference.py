"""Optional full original compressed-Attention controls and diagnostic oracle."""

import ast
import copy
from functools import lru_cache
import hashlib
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np

from .attention_storage_reference import _numpy
from .attention_validation import check_arrays
from .compressed_attention import CAPACITY, LAYERS, NativeCompressedAttention
from .compression_reference import CompressionOracle
from .indexer_reference import IndexerOracle
from .projection_reference import ProjectionOracle

METHODS = {
    "linear": "269a2858eac23cd6e35d246e37b3ec5b20b06161bce0675049811e5a284146c3",
    "get_window_topk_idxs": "1149100a37d6621054ec6cf208cf61397ce176d6104e16f89f4e735960b86320",
    "select_candidate_blocks": "766de6c8ea9cd09638410fe1200f75adaf86869a961e24f4e9d96e6533bfd099",
    "Compressor.forward": "f64470bf198a1772424894240d26065a19be2d0e733217b8efa1b113d944aca9",
    "Indexer.forward": "87f5d3af5e696b381697c42b69b94c3cb5443c68c5c9c66b3e650155a7798c7a",
    "Attention._window_kv": "516354ec0fd77d1c392dcca0e8a39f6200ee699621b9cfeb5123756dd896d8cf",
    "Attention._compress_topk_idxs": "20b150bbc385158746f9cc40716000fd7a301f01d2f9e4b5276889cab0f06be4",
    "Attention._compress_kv": "0480c387247e2cdabfd3b20773eb89507b5733c83d5082ed70b65f10d5a50dd9",
    "Attention.forward": "b1951e7bbc27c0223dab068d5a82228fb403c7e1bf51b9c8dc5bcf75e06bf465",
}


class NormBridge:
    """Use original learned modules, including explicit group-first positions."""
    def __init__(self, arrays, projections, compression, ordinal):
        self.arrays, self.projections, self.compression, self.ordinal, self.opened = arrays, projections, compression, ordinal, True

    def norm(self, values, weight):
        name = next(n for n, a in self.arrays.items() if a is weight)
        layer = int(name.split('.')[1])
        math = self.compression.math if '.compressor.' in name else self.projections[layer].math
        return math.norm(values, name)

    def rope(self, values, start, profile="window", inverse=False, *, positions=None):
        if positions is not None:
            if inverse or profile != "compressed-yarn": raise ValueError("Unexpected explicit reference rotation")
            return self.compression.rope(values, positions)
        return next(iter(self.projections.values())).math.rope(values, start, profile, inverse)


class CompressedAttentionOracle:
    def __init__(self, directory, arrays, compression_arrays, index_arrays, memory, ordinal=0):
        self.arrays, self.memory, self.ordinal = arrays, memory, ordinal
        self.p = {layer: ProjectionOracle(directory, arrays, ordinal, layer=layer) for layer in LAYERS}
        self.comp = CompressionOracle(directory, compression_arrays, ordinal)
        self.index = IndexerOracle(directory, index_arrays, ordinal)
        self.torch = self.index.torch
        self.norm = NormBridge({**arrays, **compression_arrays}, self.p, self.comp, ordinal)
        self.comp_adapter = SimpleNamespace(opened=True, ordinal=ordinal, arrays=compression_arrays,
            project=self.comp.project, pool=self.comp.pool, round_kv=self.comp.round_kv, norm=self.norm)
        self.index.opened = True
        for p in self.p.values(): p.opened = True
        torch, kernel = self.torch, self.index.kernel
        self.scope = {"torch": torch, "F": torch.nn.functional, "lru_cache": lru_cache, "world_size": 1,
            "fp8_block_size": 32, "fp4_block_size": 32, "scale_fmt": "ue8m0", "scale_dtype": torch.float8_e8m0fnu,
            "act_quant": kernel.act_quant, "fp4_act_quant": kernel.fp4_act_quant,
            "fp8_gemm": kernel.fp8_gemm, "fp4_gemm": kernel.fp4_gemm,
            "apply_rotary_emb": self.p[2].math.scope["apply_rotary_emb"], "sparse_attn": self.sparse}
        tree = ast.parse((Path(directory) / "model.py").read_bytes())
        self.functions, self.methods = {}, {}
        for label, digest in METHODS.items():
            parts = label.split('.')
            parent = tree if len(parts) == 1 else next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == parts[0])
            node = next(n for n in parent.body if isinstance(n, ast.FunctionDef) and n.name == parts[-1])
            actual = hashlib.sha256(ast.dump(node).encode()).hexdigest()
            if actual != digest: raise ValueError("Unregistered complete compressed Attention method AST")
            self.methods[label] = actual
            exec(compile(ast.Module(body=[node], type_ignores=[]), "<pinned-compressed-Attention>", "exec"), self.scope)
            self.functions[label] = self.scope[parts[-1]]
        self.controls = []

    def sparse(self, q, kv, sink, idx, scale):
        out = self.memory(q, kv, sink, idx, scale)
        self.captured = {"query_bf16": _numpy(q), "KV_bf16": _numpy(kv), "indices_i32": _numpy(idx), "output_bf16": _numpy(out)}
        return out

    def reset(self, batch, *, profile="baseline", capacity=None):
        if hasattr(self, "diagnostic"): self.diagnostic.__exit__()
        self.batch = batch
        self.diagnostic = NativeCompressedAttention(self.arrays, self.p, self.norm, self.comp_adapter, self.index,
            SimpleNamespace(opened=True, ordinal=self.ordinal, __call__=None), batch, profile=profile, capacity=capacity)
        # Native frame calls this callable, while __enter__ also checks identity.
        class AttentionCall:
            opened, ordinal = True, self.ordinal
            def __call__(stub, q, kv, sink, idx, *, diagnostics=False):
                from .attention_storage import STAGES
                values = (self.index.tensor(q), self.index.tensor(kv), self.torch.from_numpy(sink).to(f"cuda:{self.ordinal}"), self.torch.from_numpy(idx).to(f"cuda:{self.ordinal}"))
                out, traces, _ = self.memory._run(64, 512, diagnostics, values)
                return dict(zip(("output_bf16", *STAGES), map(_numpy, (out, *traces))))
        self.diagnostic.attention = AttentionCall()
        self.diagnostic.attention.profile = profile
        self.diagnostic.__enter__()
        torch, functions = self.torch, self.functions
        self.scope["shared_attn"] = SimpleNamespace()
        self.capacity = self.diagnostic.capacity
        self.index.reset(batch, self.capacity)
        class IndexCall(SimpleNamespace):
            def __call__(stub, *args): return functions["Indexer.forward"](stub, *args)
        class CompressorCall(SimpleNamespace):
            def __call__(stub, *args): return functions["Compressor.forward"](stub, *args)
        self.originals = {}
        for layer in LAYERS:
            p, ratio, own = self.p[layer], 2 if layer == 2 else 1, layer != 24
            stub = SimpleNamespace(n_local_heads=64, n_local_groups=8, head_dim=512, rope_head_dim=64, o_lora_rank=1024,
                window_size=128, compress_ratio=ratio, softmax_scale=512**-0.5, is_kv_source=own, is_index_source=True,
                freqs_cis=p.math.tables["compressed-yarn"], q_norm=p.math.modules[f"layers.{layer}.attn.q_norm.weight"],
                kv_norm=p.math.modules[f"layers.{layer}.attn.kv_norm.weight"], attn_sink=p.sink,
                wo_a=SimpleNamespace(weight=p.weights["wo_a"]))
            for name in ("wq_a", "wq_b", "wkv", "wo_b"):
                setattr(stub, name, lambda x, n=name, p=p: functions["linear"](x, p.weights[n]))
            stub.window_kv_cache = torch.zeros((batch, 128, 512), dtype=torch.bfloat16, device=f"cuda:{self.ordinal}")
            index = IndexCall(**vars(self.index.states[layer]))
            stub.indexer = index
            if own:
                comp = CompressorCall(compress_ratio=ratio, head_dim=512,
                    norm=self.comp.math.modules[f"layers.{layer}.attn.compressor.norm.weight"],
                    wkv=lambda x, l=layer: functions["linear"](x, self.comp.weights[l, "wkv"]))
                if ratio == 2:
                    comp.wgate = lambda x: functions["linear"](x, self.comp.weights[2, "wgate"])
                    comp.kv_state = torch.zeros((batch, 2, 512), dtype=torch.float32, device=f"cuda:{self.ordinal}")
                    comp.score_state = torch.full_like(comp.kv_state, -torch.inf)
                stub.compressor = comp
                stub.compress_kv_cache = torch.zeros((batch, self.capacity // ratio, 512), dtype=torch.bfloat16, device=f"cuda:{self.ordinal}")
            for name in ("_window_kv", "_compress_kv", "_compress_topk_idxs"):
                setattr(stub, name, MethodType(functions["Attention." + name], stub))
            self.originals[layer] = stub

    def step(self, xs, position):
        stages = self.diagnostic.step(xs, start_pos=position)
        controls = []
        for layer in LAYERS:
            original = self.originals[layer]
            out = self.functions["Attention.forward"](original, self.index.tensor(xs[layer]), position)
            prefix = f"layer{layer}_"
            for name, expected, actual in (
                ("output", stages[prefix + "wo_b_output_bf16"], _numpy(out)),
                ("window", stages[prefix + "window_cache_bf16"], _numpy(original.window_kv_cache)),
                ("query", stages[prefix + "query_rope_output_bf16"], self.captured["query_bf16"]),
                ("combined_KV", stages[prefix + "combined_KV_bf16"], self.captured["KV_bf16"]),
                ("combined_indices", stages[prefix + "combined_indices_i32"], self.captured["indices_i32"]),
                ("Attention_output", stages[prefix + "attention_output_bf16"], self.captured["output_bf16"])):
                controls.append(check_arrays(expected, actual, f"unchanged_layer{layer}_{name}"))
            if layer in (2, 20):
                controls.append(check_arrays(stages[prefix + "compressor_cache_bf16"], _numpy(original.compress_kv_cache), f"unchanged_layer{layer}_compressed_cache"))
                controls.append(check_arrays(stages[f"index_cache{layer}_bf16"], _numpy(original.indexer.k_cache), f"unchanged_layer{layer}_index_cache"))
            if layer == 2:
                for name in ("kv_state", "score_state"):
                    controls.append(check_arrays(stages[prefix + "compressor_" + name + "_fp32"], _numpy(getattr(original.compressor, name)), "unchanged_compressor_" + name))
        shared = self.scope["shared_attn"]
        for name, expected, actual in (
            ("shared_compressed_KV", stages["layer20_compressor_cache_bf16"], _numpy(shared.compress_kv)),
            ("shared_index_K", stages["index_cache20_bf16"], _numpy(shared.index_k)),
            ("shared_candidates", stages["index_shared20_to24_candidates_bool"], _numpy(shared.candidates)),
            ("shared_indices", stages["index_shared24_to25_indices_i32"], _numpy(shared.topk_idxs))):
            controls.append(check_arrays(expected, actual, "unchanged_" + name))
        if any(c["status"] != "pass" for c in controls): raise RuntimeError("Diagnostic frame differs from unchanged complete compressed Attention")
        self.controls.append(controls)
        self.last_stages, self.last_position = stages, position
        return stages

    def snapshot(self):
        # Torch clones preserve the original callable objects and weight bindings.
        return (self.diagnostic.snapshot(), {l: {n: getattr(o, n).clone() for n in ("window_kv_cache", "compress_kv_cache") if hasattr(o, n)}
                for l, o in self.originals.items()}, {l: o.indexer.k_cache.clone() for l, o in self.originals.items() if l != 24},
                self.originals[2].compressor.kv_state.clone(), self.originals[2].compressor.score_state.clone(),
                {n: v.clone() for n, v in vars(self.scope["shared_attn"]).items()})

    def prime_diagnostic(self, state, position):
        self.diagnostic.prime_diagnostic(state, position)
        for layer, original in self.originals.items():
            original.window_kv_cache.copy_(self.index.tensor(state["windows"][layer]))
            if layer in (2, 20):
                original.compress_kv_cache.copy_(self.index.tensor(state["compressors"][layer][2]))
                original.indexer.k_cache.copy_(self.index.tensor(state["index_keys"][layer]))
        torch = self.torch
        for name, value in zip(("kv_state", "score_state"), state["compressors"][2][:2]):
            getattr(self.originals[2].compressor, name).copy_(torch.from_numpy(value).to(f"cuda:{self.ordinal}"))
        self.scope["shared_attn"] = SimpleNamespace(compress_kv=self.originals[20].compress_kv_cache, index_k=self.originals[20].indexer.k_cache)

    def rollback(self, saved):
        handle, layers, indices, kv, scores, shared = saved
        self.diagnostic.rollback(handle)
        for layer, values in layers.items():
            for n, v in values.items(): getattr(self.originals[layer], n).copy_(v)
        for l, v in indices.items(): self.originals[l].indexer.k_cache.copy_(v)
        self.originals[2].compressor.kv_state.copy_(kv)
        self.originals[2].compressor.score_state.copy_(scores)
        self.scope["shared_attn"] = SimpleNamespace(**shared)

    def close(self):
        if hasattr(self, "diagnostic"): self.diagnostic.__exit__()


def negative_controls(reference):
    torch, stages, capture = reference.torch, reference.last_stages, reference.captured
    q, kv, idx = (reference.index.tensor(capture[n]) for n in ("query_bf16", "KV_bf16", "indices_i32"))
    sink = reference.p[24].sink
    original = capture["output_bf16"]
    wrong_idx = idx.clone()
    wrong_idx[..., 128:] = -1
    omitted = _numpy(reference.memory(q, kv, sink, wrong_idx, 512**-0.5))
    # A wrong source cache retains legal positions but supplies another layer's values.
    wrong_kv = kv.clone()
    donor = reference.index.tensor(stages["layer2_compressed_visible_bf16"])
    n = kv.shape[1] - 128
    wrong_kv[:, 128:] = donor.repeat(1, (n + donor.shape[1] - 1) // donor.shape[1], 1)[:, :n]
    sharing = _numpy(reference.memory(q, wrong_kv, sink, idx, 512**-0.5))
    unrotated = stages["layer24_wq_b_output_bf16"].reshape(q.shape)
    wrong_q = reference.norm.rope(unrotated, reference.last_position, "window")["output_bf16"]
    frequency = _numpy(reference.memory(reference.index.tensor(wrong_q), kv, sink, idx, 512**-0.5))
    latent = stages["layer2_compressor_latent_pre_rope_bf16"]
    positions = stages["layer2_compressor_latent_positions_i32"]
    good_key = reference.index.key(latent, 2, positions)["quant_rounded_bf16"]
    wrong_key = reference.index.key(stages["layer2_compressor_quant_rounded_bf16"], 2, positions)["quant_rounded_bf16"]
    return [check_arrays(original, omitted, "omit_compressed_positions_from_Attention"),
            check_arrays(original, sharing, "read_layer2_KV_instead_of_layer20_for_consumer24"),
            check_arrays(original, frequency, "use_window_frequencies_for_compressed_query"),
            check_arrays(good_key, wrong_key, "index_key_from_post_RoPE_rounded_latent")]


def compare_compressed_attention(directory, fixture_path, compression_path, index_path, bundle, ordinal=0):
    from contextlib import ExitStack
    from .attention import NativeAttention
    from .attention_memory_reference import AttentionMemoryReference, missing_dependencies
    from .checkpoint import MODEL, REVISION
    from .compressed_attention_fixture import load_compressed_attention_fixture
    from .compressed_attention_validation import exercise, profiles, cleanup, boundary_names
    from .compression import NativeCompression
    from .compression_fixture import load_compression_fixture
    from .indexer import NativeIndexer
    from .indexer_fixture import load_indexer_fixture
    from .normalization import NativeNormRoPE
    from .projections import NativeProjections
    from .resources import MemoryBudget, MiB
    missing = missing_dependencies()
    if missing: return {"probe": "native_compressed_attention_parity", "status": "blocked", "missing_dependencies": missing}
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
        goldens, saved = [], None
        with torch.inference_mode(), torch.cuda.device(ordinal), torch.device(f"cuda:{ordinal}"), ExitStack() as stack:
            memory = stack.enter_context(AttentionMemoryReference(directory, ordinal))
            qualification = memory.qualify()
            if qualification["status"] != "pass": raise RuntimeError("Full Attention oracle memory qualification failed")
            reference = CompressedAttentionOracle(directory, arrays, comp_arrays, index_arrays, memory, ordinal)
            stack.callback(reference.close)
            ps = {l: stack.enter_context(NativeProjections(arrays, ordinal, layer=l)) for l in LAYERS}
            norm = stack.enter_context(NativeNormRoPE(ordinal))
            comp = stack.enter_context(NativeCompression(comp_arrays, ordinal))
            index = stack.enter_context(NativeIndexer(index_arrays, ordinal))
            attn = stack.enter_context(NativeAttention(bundle, ordinal))
            ops = ps, norm, comp, index, attn
            if any(ps[l].converted_sha256 != reference.p[l].converted_sha256 for l in LAYERS):
                raise RuntimeError("Original/native per-layer wo_a conversion differs")
            def checker(meta, xs, actual):
                expected = reference.step(xs, meta["position"])
                if set(expected) != set(actual) or set(actual) != boundary_names(meta): raise ValueError("Connected frame boundary set differs")
                goldens.append({"meta": meta, "stage_sha256": {n: hashlib.sha256(a.tobytes()).hexdigest() for n, a in expected.items()}})
                return [check_arrays(a, actual[n], n) for n, a in expected.items()]
            def lifecycle(event):
                nonlocal saved
                if event == "snapshot": saved = reference.snapshot()
                elif event == "rollback": reference.rollback(saved)
            rows, faults, events, minimum = exercise(arrays, ops, checker, reference.reset, lifecycle)
            projection_controls = {str(l): p.fp8_controls for l, p in reference.p.items()}
            index_controls = {"projection": list(reference.index.projection_controls), "sum": list(reference.index.sum_controls),
                              "packed_vs_inplace": list(reference.index.quant_controls)}
            compression_controls = list(reference.comp.quantization_controls)
            negatives = negative_controls(reference)
            from .compression_reference import QUANTIZATION_CUDA
            from .indexer_reference import INDEX_QUANT_CUDA, INDEX_QUERY_TRACE_CUDA
            index_quant = {str(flag): hashlib.sha256(reference.index.kernel.fp4_quant_kernel(128, 32, inplace=flag).get_kernel_source().encode()).hexdigest() for flag in (False, True)}
            compression_quant = {str(flag): hashlib.sha256(reference.comp.kernel.fp4_quant_kernel(512, 16, inplace=flag, scale_dtype="float8_e4m3fn").get_kernel_source().encode()).hexdigest() for flag in (False, True)}
            query_trace = hashlib.sha256(reference.index.trace_kernel.get_kernel_source().encode()).hexdigest()
            if index_quant != INDEX_QUANT_CUDA or compression_quant != QUANTIZATION_CUDA or query_trace != INDEX_QUERY_TRACE_CUDA:
                raise ValueError("Unregistered generated original compressed/index quantizer or query diagnostic")
            profile = profiles(ops)
        cleaned = cleanup(ops)
        passed = (all(r["status"] == "pass" for r in rows + faults + events) and all(n["mismatches"] > 0 for n in negatives)
                  and all(not c["device_owned_bytes"] and not c["host_reserved_bytes"] and not c["owned_context_active"] for c in cleaned.values()))
        return {"schema_version": 1, "probe": "native_compressed_attention_parity", "status": "pass" if passed else "fail",
            "model": MODEL, "revision": REVISION, "reference": reference.p[2].source, "blas_reference": reference.p[2].blas_source,
            "topk_reference": reference.index.topk_source, "reduction_reference": reference.index.reduction_source,
            "index_reference": reference.index.index_source, "softmax_reference": reference.comp.softmax_source,
            "fixtures": {"attention": fixture, "compression": comp_fixture, "indexer": index_fixture}, "native_profiles": profile,
            "original_methods_ast_sha256": reference.methods, "original_wo_a_conversion_ast_sha256": reference.p[2].methods["wo_a_conversion"],
            "original_index_quant_cuda_sha256": index_quant, "original_compression_quant_cuda_sha256": compression_quant, "original_index_query_trace_cuda_sha256": query_trace,
            "attention_reference_qualification": qualification, "frame_cases": rows, "faults": faults, "snapshot_events": events,
            "reference_goldens": goldens, "unchanged_complete_Attention_controls": reference.controls,
            "projection_instrumentation_controls": projection_controls, "indexer_instrumentation_controls": index_controls,
            "compression_packed_vs_inplace_controls": compression_controls, "negative_controls": negatives,
            "torch_version": torch.__version__, "torch_git_version": torch.version.git_version, "torch_cuda_version": torch.version.cuda,
            "tilelang_version": tilelang.__version__, "reference_runtime_workarounds": reference.index.workarounds,
            "memory": {"cleanup": cleaned, "combined_native_host_allowance_bytes": 2624 * MiB,
                "combined_native_and_reference_host_allowance_bytes": 4096 * MiB, "minimum_sampled_host_available_bytes": minimum,
                "all_frame_reservations_released": True, "state_storage": "bounded_host_diagnostic_frame", "reservation_is_RSS_limit": False},
            "weights_fetched_during_comparison_bytes": 0, "new_fixture_original_bytes": fixture["weights_fetched_bytes"],
            "scope": "Complete Attention profiles 2/20/24; original compressed-KV/key/candidate sharing 20-to-24; generated independent layer inputs; one rank, batch 1/2, prefill <=3, context <=64",
            "untouched_full_Attention_kernel": "blocked_by_shared_memory_limit", "other_layers_and_larger_context": "not_qualified",
            "complete_Transformer_blocks": "not_implemented_for_these_layers", "full_model_quality": "not_run", "end_to_end_speed": "not_measured"}
    finally:
        if budget.reserved: budget.release(budget.reserved)
        torch.set_default_dtype(previous)
