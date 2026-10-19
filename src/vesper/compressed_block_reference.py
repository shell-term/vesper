"""Original complete source-2 Block.forward, Attention and actual MoE controls."""

from contextlib import ExitStack
import copy
from pathlib import Path
from types import MethodType, SimpleNamespace

from .attention_storage_reference import _numpy
from .attention_validation import check_arrays
from .compressed_block_validation import frame_values, state_digest
from .formats import bf16_values


class CompressedBlockOracle:
    def __init__(self, arrays, hc, attention, moe, norm):
        self.arrays, self.hc, self.attention, self.moe, self.norm = arrays, hc, attention, moe, norm
        self.controls, self.attention_controls = [], []
        self.forward = attention.p[2].method("Block", "forward")

    def reset(self):
        self.attention.reset(1)
        self.original_attention = copy.deepcopy(self.attention.originals[2])
        block = SimpleNamespace(norm_eps=1e-20, hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6,
            attn_norm=self.norm.modules["layers.2.attn_norm.weight"], ffn_norm=self.norm.modules["layers.2.ffn_norm.weight"])
        for part in ("attn", "ffn"):
            for suffix in ("fn", "scale", "base"): setattr(block, f"hc_{part}_{suffix}", self.hc.weights[part + "_" + suffix])
        for name in ("hc_mixes", "hc_pre", "hc_post"): setattr(block, name, MethodType(self.hc.functions[name], block))
        block.attn = lambda x, pos: self.attention.functions["Attention.forward"](self.original_attention, x, pos)
        self.original_block = block

    def step(self, hidden, pre, image, position):
        stages = {}
        def add(prefix, values):
            stages.update((prefix + n, a) for n, a in values.items())
            return values.get("output_bf16")
        mixes = self.hc.mixes(hidden, "attn")
        add("hc_attn_", mixes)
        x = add("attn_pre_", self.hc.pre(hidden, pre))
        x = add("attn_norm_", self.norm.norm(x, "layers.2.attn_norm.weight"))
        a = self.attention.step({2: x}, position)
        self.attention_controls.append(self.attention.controls.pop())
        add("attention_", a)
        after = add("attn_post_", self.hc.post(a["layer2_wo_b_output_bf16"], hidden, mixes["post_fp32"], mixes["comb_fp32"]))
        ffn = self.hc.mixes(after, "ffn")
        add("hc_ffn_", ffn)
        x = add("ffn_pre_", self.hc.pre(after, mixes["pre_fp32"]))
        x = add("ffn_norm_", self.norm.norm(x, "layers.2.ffn_norm.weight"))
        moe = self.moe.forward(bf16_values(x.reshape(5120)), image)
        add("moe_", moe)
        out = add("ffn_post_", self.hc.post(moe["aggregation.output_bf16"].reshape(1, 1, 5120), after, ffn["post_fp32"], ffn["comb_fp32"]))
        stages["output_bf16"], stages["next_pre_mix_fp32"] = out, ffn["pre_fp32"]
        calls = []
        def original_moe(tx, mask):
            if mask.shape != (1, 1) or bool(mask[0, 0].item()) != image: raise ValueError("Original block image mask differs")
            calls.append(1)
            values = self.moe.forward(bf16_values(_numpy(tx).reshape(5120)), image)
            return self.hc.tensor(values["aggregation.output_bf16"].reshape(1, 1, 5120))
        self.original_block.ffn = original_moe
        torch = self.hc.torch
        mask = torch.full((1, 1), image, dtype=torch.bool, device=f"cuda:{self.hc.ordinal}")
        full, next_pre = self.forward(self.original_block, self.hc.tensor(hidden), position, self.hc.tensor(pre), mask)
        o, diagnostic = self.original_attention, self.attention.diagnostic.inspect()["state"]
        controls = [check_arrays(out, _numpy(full), "unchanged_Block_forward_output_BF16"),
            check_arrays(ffn["pre_fp32"], _numpy(next_pre), "unchanged_Block_forward_next_pre_FP32"),
            check_arrays(diagnostic["windows"][2], _numpy(o.window_kv_cache), "unchanged_Block_forward_window_KV"),
            check_arrays(diagnostic["compressors"][2][2], _numpy(o.compress_kv_cache), "unchanged_Block_forward_compressed_KV"),
            check_arrays(diagnostic["index_keys"][2], _numpy(o.indexer.k_cache[:, :576]), "unchanged_Block_forward_index_keys"),
            check_arrays(diagnostic["sharing"]["indices"], _numpy(self.attention.scope["shared_attn"].topk_idxs), "unchanged_Block_forward_selected_indices")]
        controls.extend(check_arrays(diagnostic["compressors"][2][i], _numpy(getattr(o.compressor, n)), "unchanged_Block_forward_" + n)
            for i, n in enumerate(("kv_state", "score_state")))
        if len(calls) != 1 or any(c["status"] != "pass" for c in controls): raise RuntimeError("Diagnostic differs from unchanged complete original Block.forward")
        self.controls.append(controls)
        self.last_stages = stages
        return frame_values(stages, diagnostic), diagnostic

    def snapshot(self):
        return self.attention.snapshot(), copy.deepcopy(self.original_attention)

    def rollback(self, saved):
        sharing, original = saved
        self.attention.rollback(sharing)
        self.original_attention = copy.deepcopy(original)

    def negatives(self):
        stages = self.last_stages
        wrong = self.hc.pre(stages["attn_post_output_bf16"], stages["hc_ffn_pre_fp32"])["output_bf16"]
        first = check_arrays(stages["ffn_pre_output_bf16"], wrong, "use_FFN_own_pre_for_current_FFN")
        captured = self.attention.captured
        q, kv, idx = (self.attention.index.tensor(captured[n]) for n in ("query", "combined_KV", "combined_indices"))
        wrong_idx = idx.clone(); wrong_idx[..., 128:] = -1
        wrong = self.attention.memory(q, kv, self.attention.p[2].sink, wrong_idx, 512**-0.5)
        second = check_arrays(captured["Attention_output"], _numpy(wrong), "omit_compressed_KV_in_block_Attention")
        return [first, second]


def compare_block(directory, topology, fixtures, bundle, plan_file, store, experts, ordinal=0, progress=None):
    from .attention_memory_reference import AttentionMemoryReference, missing_dependencies
    from .checkpoint import MODEL, REVISION
    from .compressed_attention_validation import profiles, cleanup
    from .compressed_block_fixture import load_plan
    from .compressed_block_validation import load_arrays, open_operators, open_block, exercise, registry, compact_comparison, identity, coverage
    from .connected_reference import OfficialConnected
    from .hc_reference import HCOracle
    from .normalization_reference import ReferenceMath
    from .resources import MemoryBudget, MiB
    from .topology_reference import SharingOracle
    if missing_dependencies(): return {"probe": "native_compressed_block_parity", "status": "blocked", "missing_dependencies": missing_dependencies()}
    import torch
    import tilelang
    budgets = MemoryBudget(cap=768 * MiB), MemoryBudget(cap=2048 * MiB)
    previous, root = torch.get_default_dtype(), Path(fixtures)
    torch.set_default_dtype(torch.bfloat16)
    try:
        with ExitStack() as stack, torch.inference_mode(), torch.cuda.device(ordinal), torch.device(f"cuda:{ordinal}"):
            for b in budgets:
                b.acquire(b.cap); stack.callback(b.release, b.cap)
            arrays, manifests = load_arrays(topology, fixtures)
            plan = load_plan(plan_file)
            if plan["fixtures"] != manifests: raise ValueError("Original block plan fixtures differ")
            memory = stack.enter_context(AttentionMemoryReference(directory, ordinal, profile="long_context"))
            qualification = memory.qualify()
            if qualification["status"] != "pass": raise RuntimeError("Original Attention oracle is unqualified")
            sharing = SharingOracle(directory, arrays, (2,), memory, ordinal)
            stack.callback(sharing.close)
            hc_ref = HCOracle(directory, arrays, ordinal, layer=2)
            norm_ref = ReferenceMath(directory, {n: arrays[n] for n in ("layers.2.attn_norm.weight", "layers.2.ffn_norm.weight")}, ordinal)
            moe_ref = OfficialConnected(root / "gate/native-gate.npz", root / "router/native-router.npz", root / "shared/native-shared.npz",
                experts, directory, ordinal, layer=2)
            reference = CompressedBlockOracle(arrays, hc_ref, sharing, moe_ref, norm_ref)
            hc, ops = open_operators(stack, arrays, bundle, ordinal)
            block = open_block(stack, arrays, hc, ops, store, fixtures, ordinal)
            if ops[0][2].converted_sha256 != sharing.p[2].converted_sha256: raise RuntimeError("Original layer-2 grouped weight conversion differs")
            names, saved = registry(plan), None
            state_controls = []
            def checker(m, hidden, pre, actual, state):
                expected, expected_state = reference.step(hidden, pre, m["image_span"], m["position"])
                expected_hash = state_digest(expected_state)
                if state_digest(state) != expected_hash: raise RuntimeError("Complete compressed block committed metadata/state differs")
                state_controls.append({"case": m["id"], "status": "pass", "expected_sha256": expected_hash, "actual_sha256": state_digest(state)})
                return compact_comparison(expected, actual, names)
            def lifecycle(event):
                nonlocal saved
                if event == "snapshot": saved = reference.snapshot()
                else: reference.rollback(saved)
            rows, faults, events, minimum = exercise(block, ops, plan, checker, reference.reset, lifecycle, progress)
            negatives = reference.negatives()
            native_profiles = profiles(ops)
            hc_profile = {"compiler": hc.compiler, "cubin_sha256": hc.cubin_sha256, "cublas_version": hc.blas.version, "cublas_math_mode": hc.blas.math_mode}
            block_stats, moe_stats = block.stats(), block.moe.snapshot()
            reference_identity = {"source": sharing.p[2].source, "blas_source": sharing.p[2].blas_source,
                "hc_reduction_source": hc_ref.reduce_source, "topk_source": sharing.index.topk_source,
                "index_source": sharing.index.index_source, "index_reduction_source": sharing.index.reduction_source,
                "softmax_source": sharing.comp.softmax_source, "torch_version": torch.__version__, "torch_git_version": torch.version.git_version,
                "torch_cuda_version": torch.version.cuda, "tilelang_version": tilelang.__version__}
        cleaned = cleanup(ops)
        passed = all(r["status"] == "pass" for r in rows + faults + events) and all(n["mismatches"] > 0 for n in negatives)
        result = {"schema_version": 1, "probe": "native_compressed_block_parity", "status": "pass" if passed else "fail", "model": MODEL, "revision": REVISION, "layer": 2,
            "fixtures": manifests, "moe_fixtures": {"gate": moe_ref.gate_manifest, "router": moe_ref.router_manifest, "shared": moe_ref.shared_manifest,
                "experts": list(moe_ref.provenance.values())}, "plan": plan, "native_profiles": native_profiles, "hc_profile": hc_profile,
            "reference_identity": reference_identity, "original_methods": {"Attention": sharing.methods, "HC": hc_ref.methods,
                "Block": sharing.p[2].methods["Block.forward"], "wo_a_conversion": sharing.p[2].methods["wo_a_conversion"]},
            "hc_instrumentation": hc_ref.instrumentation, "attention_reference_qualification": qualification,
            "boundary_names": names, "boundary_columns": ["name_id", "storage_elements", "mismatches", "expected_sha256"], "block_cases": rows,
            "unchanged_Block_forward_controls": reference.controls, "unchanged_Attention_forward_controls": reference.attention_controls,
            "whole_committed_state_controls": state_controls, "faults": faults, "snapshot_events": events, "negative_controls": negatives,
            "memory": {"operator_cleanup": cleaned, "HC_cleanup": hc.stats(), "moe_cleanup": block.moe.cleanup, "block_cleanup": block.stats(),
                "block_active_stats": block_stats, "moe_active_stats": moe_stats, "combined_native_host_allowance_bytes": 3584 * MiB,
                "retained_fixture_host_allowance_bytes": 768 * MiB, "additional_reference_host_allowance_bytes": 2048 * MiB,
                "minimum_sampled_host_available_bytes": minimum, "all_frame_reservations_released": block.attention.budget.reserved == 0,
                "reservation_is_RSS_limit": False, "state_storage": "bounded_host_diagnostic_frame"},
            "reference_runtime_workarounds": hc_ref.workarounds + sharing.index.workarounds,
            "weights_fetched_during_comparison_bytes": 0, "scope": "Complete original layer-2 single-token block; four-position generated HC streams, text/image-span routing, exact offloaded experts; no upstream blocks or Engram",
            "maximum_unprimed_continuous_context": 4, "full_model_quality": "not_run", "end_to_end_speed": "not_measured",
            "untouched_full_Attention_kernel": "blocked_by_shared_memory_limit"}
        result["qualification_identity_sha256"], result["coverage_sha256"] = identity(result), coverage(result)
        return result
    finally: torch.set_default_dtype(previous)
