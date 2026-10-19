"""Independent unchanged Block.forward calls with real source/consumer handoff."""

from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace, MethodType

from .attention_storage_reference import _numpy
from .attention_validation import check_arrays
from .block_pair import LAYERS
from .formats import bf16_values


class BlockPairOracle:
    def __init__(self, arrays, hc, sharing, norm, moe):
        self.arrays, self.hc, self.sharing, self.norm, self.moe = arrays, hc, sharing, norm, moe
        self.forward = sharing.p[2].method("Block", "forward")
        self.block_controls, self.attention_controls = [], []

    def reset(self):
        self.sharing.reset(1)
        self.blocks = {}
        for l in LAYERS:
            block = SimpleNamespace(norm_eps=1e-20, hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6,
                attn_norm=self.norm.modules[f"layers.{l}.attn_norm.weight"], ffn_norm=self.norm.modules[f"layers.{l}.ffn_norm.weight"])
            for part in ("attn", "ffn"):
                for suffix in ("fn", "scale", "base"): setattr(block, f"hc_{part}_{suffix}", self.hc[l].weights[part + "_" + suffix])
            for name in ("hc_mixes", "hc_pre", "hc_post"): setattr(block, name, MethodType(self.hc[l].functions[name], block))
            self.blocks[l] = block

    def step(self, hidden, pre, image, position):
        phases, carry, result = {}, [hidden, pre], {}
        def prepare(l):
            h, p = carry
            stages = {"input_hidden_bf16": h.copy(), "incoming_pre_fp32": p.copy()}
            def add(prefix, values):
                stages.update((prefix + n, a) for n, a in values.items())
                return values.get("output_bf16")
            mixes = self.hc[l].mixes(h, "attn"); add("hc_attn_", mixes)
            x = add("attn_pre_", self.hc[l].pre(h, p))
            x = add("attn_norm_", self.norm.norm(x, f"layers.{l}.attn_norm.weight"))
            phases[l] = stages, mixes, add, h
            return x
        def finish(l, attention):
            stages, mixes, add, h = phases[l]
            after = add("attn_post_", self.hc[l].post(attention["wo_b_output_bf16"], h, mixes["post_fp32"], mixes["comb_fp32"]))
            ffn = self.hc[l].mixes(after, "ffn"); add("hc_ffn_", ffn)
            x = add("ffn_pre_", self.hc[l].pre(after, mixes["pre_fp32"]))
            x = add("ffn_norm_", self.norm.norm(x, f"layers.{l}.ffn_norm.weight"))
            moe = self.moe[l].forward(bf16_values(x.reshape(5120)), image); add("moe_", moe)
            out = add("ffn_post_", self.hc[l].post(moe["aggregation.output_bf16"].reshape(1, 1, 5120), after, ffn["post_fp32"], ffn["comb_fp32"]))
            stages["output_bf16"], stages["next_pre_mix_fp32"] = out, ffn["pre_fp32"]
            carry[:] = out, ffn["pre_fp32"]
        def finalize(attention):
            result.update({f"layer{l}": phases[l][0] for l in LAYERS})
            result.update(attention=attention, handoff_hidden_bf16=phases[3][3].copy(),
                handoff_pre_mix_fp32=phases[3][0]["incoming_pre_fp32"].copy(), output_bf16=carry[0], next_pre_mix_fp32=carry[1])
            return result
        self.sharing.diagnostic.step_connected(prepare, finish, start_pos=position, finalize=finalize)
        state = self.sharing.diagnostic.inspect()["state"]
        self._full_controls(hidden, pre, image, position, result, state)
        self.last_result, self.last_initial_pre = result, pre
        return result, state

    def _full_controls(self, hidden, pre, image, position, result, state):
        torch = self.hc[2].torch
        h, p = self.hc[2].tensor(hidden), self.hc[2].tensor(pre)
        mask = torch.full((1, 1), image, dtype=torch.bool, device=f"cuda:{self.hc[2].ordinal}")
        bc, ac = [], []
        attention = result["attention"]
        for l in LAYERS:
            expected, block, original = result[f"layer{l}"], self.blocks[l], self.sharing.originals[l]
            prefix = f"unchanged_layer{l}_Block_"
            bc.extend((check_arrays(expected["input_hidden_bf16"], _numpy(h), prefix + "input_hidden"),
                check_arrays(expected["incoming_pre_fp32"], _numpy(p), prefix + "incoming_pre")))
            called = []
            def attn(tx, pos, l=l, expected=expected, original=original, prefix=prefix):
                called.append("attention")
                bc.append(check_arrays(expected["attn_norm_output_bf16"], _numpy(tx), prefix + "attention_input"))
                out = self.sharing.functions["Attention.forward"](original, tx, pos)
                pfx, captured, shared = f"layer{l}_", self.sharing.captured, self.sharing.scope["shared_attn"]
                pairs = {"output": (attention[pfx + "wo_b_output_bf16"], _numpy(out)),
                    "window": (state["windows"][l], _numpy(original.window_kv_cache)),
                    "query": (attention[pfx + "query_rope_output_bf16"], captured["query"]),
                    "combined_KV": (attention[pfx + "combined_KV_bf16"], captured["combined_KV"]),
                    "combined_indices": (attention[pfx + "combined_indices_i32"], captured["combined_indices"]),
                    "Attention_output": (attention[pfx + "attention_output_bf16"], captured["Attention_output"]),
                    "selected": (attention[pfx + "selected_indices_i32"], _numpy(shared.topk_idxs)),
                    "compressed_visible": (attention[pfx + "compressed_visible_bf16"], _numpy(shared.compress_kv[:, :(pos + 1) // 2]))}
                if l == 2:
                    pairs.update(compressed_cache=(state["compressors"][2][2], _numpy(original.compress_kv_cache)),
                        key_cache=(state["index_keys"][2], _numpy(original.indexer.k_cache[:, :576])))
                    pairs.update({n: (state["compressors"][2][i], _numpy(getattr(original.compressor, n))) for i, n in enumerate(("kv_state", "score_state"))})
                ac.extend(check_arrays(a, b, f"unchanged_layer{l}_" + name) for name, (a, b) in pairs.items())
                return out
            def moe(tx, actual_mask, l=l, expected=expected, prefix=prefix):
                called.append("moe")
                if actual_mask.shape != (1, 1) or bool(actual_mask[0, 0].item()) != image: raise RuntimeError("Original pair mask differs")
                bc.append(check_arrays(expected["ffn_norm_output_bf16"], _numpy(tx), prefix + "MoE_input"))
                values = self.moe[l].forward(bf16_values(_numpy(tx).reshape(5120)), image)
                for n in ("logits", "indices", "weights"):
                    bc.append(check_arrays(expected["moe_gate." + n], values["gate." + n], prefix + "gate_" + n))
                return self.hc[l].tensor(values["aggregation.output_bf16"].reshape(1, 1, 5120))
            block.attn, block.ffn = attn, moe
            h, p = self.forward(block, h, position, p, mask)
            bc.extend((check_arrays(expected["output_bf16"], _numpy(h), prefix + "output"),
                check_arrays(expected["next_pre_mix_fp32"], _numpy(p), prefix + "next_pre")))
            if called != ["attention", "moe"]: raise RuntimeError("Original pair block invocation order differs")
        if any(c["status"] != "pass" for c in bc + ac): raise RuntimeError("Pair diagnostics differ from consecutive unchanged original blocks")
        self.block_controls.append(bc); self.attention_controls.append(ac)

    def snapshot(self): return self.sharing.snapshot()

    def rollback(self, saved): self.sharing.rollback(saved)

    def negatives(self):
        r, hc = self.last_result, self.hc[3]
        s = r["layer3"]
        wrong_pre = hc.pre(s["input_hidden_bf16"], self.last_initial_pre)["output_bf16"]
        wrong_hidden = hc.pre(r["layer2"]["input_hidden_bf16"], s["incoming_pre_fp32"])["output_bf16"]
        controls = [check_arrays(s["attn_pre_output_bf16"], wrong_pre, "replace_transferred_pre_with_initial_pre"),
            check_arrays(s["attn_pre_output_bf16"], wrong_hidden, "bypass_source_block_hidden_output")]
        c, index = self.sharing.captured, self.sharing.index
        q, kv, idx = (index.tensor(c[n]) for n in ("query", "combined_KV", "combined_indices"))
        wrong_kv, wrong_idx = kv.clone(), idx.clone()
        wrong_kv[:, 128:] = kv[:, :kv.shape[1] - 128]
        wrong_idx[..., 128:] = -1
        for label, keys, ids in (("replace_shared_KV_with_consumer_window", wrong_kv, idx), ("drop_reused_source_selected_indices", kv, wrong_idx)):
            out = self.sharing.memory(q, keys, self.sharing.p[3].sink, ids, 512**-0.5)
            controls.append(check_arrays(c["Attention_output"], _numpy(out), label))
        return controls


def compare_pair(directory, topology, source, consumer, bundle, plan_file, store2, store3, experts2, experts3, ordinal=0, progress=None):
    from .attention_memory_reference import AttentionMemoryReference, missing_dependencies
    from .checkpoint import MODEL, REVISION
    from .compressed_attention_validation import profiles, cleanup
    from .block_pair_fixture import load_plan
    from .block_pair_validation import load_arrays, open_operators, open_pair, exercise, registry, values, state_digest, compact_comparison, identity, golden_digest
    from .block_pair_validation import NATIVE_HOST_MIB, REFERENCE_HOST_MIB, FIXTURE_HOST_MIB
    from .connected_reference import OfficialConnected
    from .hc_reference import HCOracle
    from .normalization_reference import ReferenceMath
    from .resources import MemoryBudget, MiB
    from .topology_reference import SharingOracle
    if missing_dependencies(): return {"probe": "consecutive_complete_block_parity", "status": "blocked", "missing_dependencies": missing_dependencies()}
    import torch
    import tilelang
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with ExitStack() as stack, torch.inference_mode(), torch.cuda.device(ordinal), torch.device(f"cuda:{ordinal}"):
            for allowance in (FIXTURE_HOST_MIB, REFERENCE_HOST_MIB):
                b = MemoryBudget(cap=allowance * MiB); b.acquire(b.cap); stack.callback(b.release, b.cap)
            arrays, fixtures = load_arrays(topology, source, consumer)
            plan = load_plan(plan_file)
            if plan["fixtures"] != fixtures: raise ValueError("Pair plan original fixtures differ")
            memory = stack.enter_context(AttentionMemoryReference(directory, ordinal, profile="long_context"))
            qualification = memory.qualify()
            if qualification["status"] != "pass": raise RuntimeError("Original pair Attention oracle is unqualified")
            sharing = SharingOracle(directory, arrays, LAYERS, memory, ordinal); stack.callback(sharing.close)
            hc_ref = {l: HCOracle(directory, arrays, ordinal, layer=l) for l in LAYERS}
            norm = ReferenceMath(directory, {n: arrays[n] for l in LAYERS for n in (f"layers.{l}.attn_norm.weight", f"layers.{l}.ffn_norm.weight")}, ordinal)
            moe_ref = {l: OfficialConnected(Path(f) / "gate/native-gate.npz", Path(f) / "router/native-router.npz", Path(f) / "shared/native-shared.npz",
                e, directory, ordinal, layer=l) for l, f, e in ((2, source, experts2), (3, consumer, experts3))}
            reference = BlockPairOracle(arrays, hc_ref, sharing, norm, moe_ref)
            hc, ops = open_operators(stack, arrays, bundle, ordinal)
            pair = open_pair(stack, arrays, hc, ops, source, consumer, store2, store3, ordinal)
            if any(ops[0][l].converted_sha256 != sharing.p[l].converted_sha256 for l in LAYERS): raise RuntimeError("Pair original grouped output conversion differs")
            names, saved, state_controls = registry(plan), None, []
            def checker(m, h, p, actual, state):
                expected, expected_state = reference.step(h, p, m["image_span"], m["position"])
                expected_hash = state_digest(expected_state)
                if state_digest(state) != expected_hash: raise RuntimeError("Consecutive original and native committed states differ")
                state_controls.append({"case": m["id"], "status": "pass", "expected_sha256": expected_hash, "actual_sha256": state_digest(state)})
                return compact_comparison(values(expected, expected_state), actual, names)
            def lifecycle(event):
                nonlocal saved
                if event == "snapshot": saved = reference.snapshot()
                else: reference.rollback(saved)
            rows, faults, events, minimum = exercise(pair, ops, plan, checker, reference.reset, lifecycle, progress)
            negatives = reference.negatives()
            native_profiles = profiles(ops)
            hc_profiles = {str(l): {"compiler": h.compiler, "cubin_sha256": h.cubin_sha256, "cublas_version": h.blas.version, "cublas_math_mode": h.blas.math_mode} for l, h in hc.items()}
            pair_stats, moe_stats = pair.stats(), {str(l): m.snapshot() for l, m in pair.moe.items()}
            reference_identity = {"source": sharing.p[2].source, "blas_source": sharing.p[2].blas_source,
                "hc_reduction_source": hc_ref[2].reduce_source, "topk_source": sharing.index.topk_source,
                "index_source": sharing.index.index_source, "index_reduction_source": sharing.index.reduction_source,
                "softmax_source": sharing.comp.softmax_source, "torch_version": torch.__version__, "torch_git_version": torch.version.git_version,
                "torch_cuda_version": torch.version.cuda, "tilelang_version": tilelang.__version__}
        cleaned = cleanup(ops)
        passed = all(r["status"] == "pass" for r in rows + faults + events) and all(c["mismatches"] > 0 for c in negatives)
        result = {"schema_version": 1, "probe": "consecutive_complete_block_parity", "status": "pass" if passed else "fail",
            "model": MODEL, "revision": REVISION, "layers": [2, 3], "fixtures": fixtures,
            "moe_fixtures": {str(l): {"gate": m.gate_manifest, "router": m.router_manifest, "shared": m.shared_manifest,
                "experts": list(m.provenance.values())} for l, m in moe_ref.items()}, "plan": plan,
            "native_profiles": native_profiles, "hc_profiles": hc_profiles, "reference_identity": reference_identity,
            "original_methods": {"Attention": sharing.methods, "HC": hc_ref[2].methods, "Block": sharing.p[2].methods["Block.forward"],
                "wo_a_conversion": sharing.p[2].methods["wo_a_conversion"]}, "hc_instrumentation": {str(l): h.instrumentation for l, h in hc_ref.items()},
            "attention_reference_qualification": qualification, "boundary_names": names,
            "boundary_columns": ["name_id", "storage_elements", "mismatches", "expected_sha256"], "pair_cases": rows,
            "unchanged_Block_forward_controls": reference.block_controls, "unchanged_Attention_forward_controls": reference.attention_controls,
            "whole_pair_state_controls": state_controls, "faults": faults, "snapshot_events": events, "negative_controls": negatives,
            "memory": {"operator_cleanup": cleaned, "HC_cleanup": {str(l): h.stats() for l, h in hc.items()},
                "moe_cleanup": {str(l): m.cleanup for l, m in pair.moe.items()}, "pair_cleanup": pair.stats(),
                "pair_active_stats": pair_stats, "moe_active_stats": moe_stats,
                "combined_native_host_allowance_bytes": NATIVE_HOST_MIB * MiB, "additional_reference_host_allowance_bytes": REFERENCE_HOST_MIB * MiB,
                "retained_fixture_host_allowance_bytes": FIXTURE_HOST_MIB * MiB, "minimum_sampled_host_available_bytes": minimum,
                "all_frame_reservations_released": pair.attention.budget.reserved == 0, "reservation_is_RSS_limit": False,
                "state_storage": "bounded_host_diagnostic_frame"}, "reference_runtime_workarounds": hc_ref[2].workarounds + sharing.index.workarounds,
            "weights_fetched_during_comparison_bytes": 0, "maximum_unprimed_continuous_context": 4,
            "scope": "Consecutive complete original blocks 2/3; generated initial layer-2 HC streams, actual hidden/coefficient handoff and source KV/index reuse; no upstream blocks or Engram",
            "untouched_full_Attention_kernel": "blocked_by_shared_memory_limit", "full_model_quality": "not_run", "end_to_end_speed": "not_measured"}
        result["source_fixture_compiler_identity_sha256"], result["complete_goldens_sha256"] = identity(result), golden_digest(result)
        return result
    finally: torch.set_default_dtype(previous)
