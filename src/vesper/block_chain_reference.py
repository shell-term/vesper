"""Original consecutive Block.forward controls through real index refresh at 24."""

from contextlib import ExitStack
from pathlib import Path

from .attention_topology import REFRESH_CHAIN, validate_block_chain
from .attention_validation import check_arrays
from .block_pair_reference import BlockPairOracle
from .block_chain_fixture import folder, load_plan


class BlockChainOracle(BlockPairOracle):
    _validate_layers = staticmethod(validate_block_chain)

    def _result(self, phases, attention, carry):
        result = {f"layer{l}": phases[l][0] for l in self.layers}
        result.update(attention=attention,
            handoffs={l: {"hidden_bf16": phases[l][3].copy(), "pre_mix_fp32": phases[l][0]["incoming_pre_fp32"].copy()} for l in self.layers[1:]},
            output_bf16=carry[0], next_pre_mix_fp32=carry[1])
        return result

    def negatives(self):
        controls = super().negatives()
        controls[1]["check"] = "bypass_preceding_complete_blocks"
        a = self.last_result["attention"]
        for name, stage in (("reuse_source20_index_query_at24", "index_query_projection_bf16"),
                            ("reuse_source20_index_scores_at24", "index_score_score_bf16")):
            controls.append(check_arrays(a[f"layer24_{stage}"], a[f"layer20_{stage}"], name))
        return controls


def compare_chain(directory, topology, root, bundle, plan_directory, ordinal=0, progress=None):
    from .attention_memory_reference import AttentionMemoryReference, missing_dependencies
    from .block_chain_validation import load_arrays, open_operators, open_chain, exercise, registry, values, state_digest, compact_comparison, identity, golden_digest
    from .block_chain_validation import FIXTURE_HOST_MIB, REFERENCE_HOST_MIB, NATIVE_HOST_MIB
    from .compressed_attention_validation import profiles, cleanup
    from .checkpoint import MODEL, REVISION
    from .connected_reference import OfficialConnected
    from .hc_reference import HCOracle
    from .normalization_reference import ReferenceMath
    from .resources import MemoryBudget, MiB
    from .topology_reference import SharingOracle
    if missing_dependencies(): return {"probe": "complete_refresh_chain_parity", "status": "blocked", "missing_dependencies": missing_dependencies()}
    import torch
    import tilelang
    previous = torch.get_default_dtype(); torch.set_default_dtype(torch.bfloat16)
    try:
        with ExitStack() as stack, torch.inference_mode(), torch.cuda.device(ordinal), torch.device(f"cuda:{ordinal}"):
            for allowance in (FIXTURE_HOST_MIB, REFERENCE_HOST_MIB):
                b = MemoryBudget(cap=allowance * MiB); b.acquire(b.cap); stack.callback(b.release, b.cap)
            arrays, fixtures = load_arrays(topology, root)
            plans = [load_plan(Path(plan_directory) / f"layer{l}.json") for l in (22, 23, 24, 25)]
            plan = plans[-1]
            if plan["fixtures"] != fixtures: raise ValueError("Chain plan and execution fixtures differ")
            memory = stack.enter_context(AttentionMemoryReference(directory, ordinal, profile="long_context"))
            qualification = memory.qualify()
            if qualification["status"] != "pass": raise RuntimeError("Original chain Attention oracle is unqualified")
            sharing = SharingOracle(directory, arrays, REFRESH_CHAIN, memory, ordinal); stack.callback(sharing.close)
            hc_ref = {l: HCOracle(directory, arrays, ordinal, layer=l) for l in REFRESH_CHAIN}
            norm = ReferenceMath(directory, {n: arrays[n] for l in REFRESH_CHAIN for n in (f"layers.{l}.attn_norm.weight", f"layers.{l}.ffn_norm.weight")}, ordinal)
            moe_ref = {l: OfficialConnected(folder(root, l) / "gate/native-gate.npz", folder(root, l) / "router/native-router.npz",
                folder(root, l) / "shared/native-shared.npz", folder(root, l, "experts"), directory, ordinal, layer=l) for l in REFRESH_CHAIN}
            reference = BlockChainOracle(arrays, hc_ref, sharing, norm, moe_ref)
            hc, ops = open_operators(stack, arrays, bundle, ordinal)
            chain = open_chain(stack, arrays, hc, ops, root, ordinal)
            if any(ops[0][l].converted_sha256 != sharing.p[l].converted_sha256 for l in REFRESH_CHAIN): raise ValueError("Original chain output-weight conversion differs")
            native_allowance = (FIXTURE_HOST_MIB * MiB + chain.budget.cap + chain.attention.budget.cap
                + sum(h.budget.cap for h in hc.values()) + sum(p.budget.cap for p in ops[0].values())
                + sum(o.budget.cap for o in (*ops[1:], ops[2].norm, ops[3].norm))
                + sum(m.budget.cap for m in chain.moe.values()))
            if native_allowance != NATIVE_HOST_MIB * MiB: raise ValueError("Complete-chain host allowance omits a native ledger")
            names, saved, state_controls = registry(plan), None, []
            def checker(m, h, p, actual, state):
                expected, expected_state = reference.step(h, p, m["image_span"], m["position"])
                digest = state_digest(expected_state)
                if state_digest(state) != digest: raise RuntimeError("Original and native complete-chain states differ")
                state_controls.append({"case": m["id"], "status": "pass", "expected_sha256": digest, "actual_sha256": state_digest(state)})
                return compact_comparison(values(expected, expected_state), actual, names)
            def lifecycle(event):
                nonlocal saved
                if event == "snapshot": saved = reference.snapshot()
                else: reference.rollback(saved)
            rows, faults, events, minimum = exercise(chain, ops, plan, checker, reference.reset, lifecycle, progress)
            negatives = reference.negatives()
            native_profiles = profiles(ops)
            hc_profiles = {str(l): {"compiler": h.compiler, "cubin_sha256": h.cubin_sha256, "cublas_version": h.blas.version, "cublas_math_mode": h.blas.math_mode} for l, h in hc.items()}
            chain_stats, moe_stats = chain.stats(), {str(l): m.snapshot() for l, m in chain.moe.items()}
            reference_identity = {"source": sharing.p[20].source, "blas_source": sharing.p[20].blas_source,
                "hc_reduction_source": hc_ref[20].reduce_source, "topk_source": sharing.index.topk_source,
                "index_source": sharing.index.index_source, "index_reduction_source": sharing.index.reduction_source,
                "softmax_source": sharing.comp.softmax_source, "torch_version": torch.__version__, "torch_git_version": torch.version.git_version,
                "torch_cuda_version": torch.version.cuda, "tilelang_version": tilelang.__version__}
        result = {"schema_version": 1, "probe": "complete_refresh_chain_parity",
            "status": "pass" if all(r["status"] == "pass" for r in rows + faults + events) and all(c["mismatches"] > 0 for c in negatives) else "fail",
            "model": MODEL, "revision": REVISION, "layers": list(REFRESH_CHAIN), "fixtures": fixtures,
            "moe_fixtures": {str(l): {"gate": m.gate_manifest, "router": m.router_manifest, "shared": m.shared_manifest, "experts": list(m.provenance.values())} for l, m in moe_ref.items()},
            "plan": plan, "expert_plans": plans, "native_profiles": native_profiles, "hc_profiles": hc_profiles, "reference_identity": reference_identity,
            "original_methods": {"Attention": sharing.methods, "HC": hc_ref[20].methods, "Block": sharing.p[20].methods["Block.forward"], "wo_a_conversion": sharing.p[20].methods["wo_a_conversion"]},
            "hc_instrumentation": {str(l): h.instrumentation for l, h in hc_ref.items()}, "attention_reference_qualification": qualification,
            "boundary_names": names, "boundary_columns": ["name_id", "storage_elements", "mismatches", "expected_sha256"], "chain_cases": rows,
            "unchanged_Block_forward_controls": reference.block_controls, "unchanged_Attention_forward_controls": reference.attention_controls,
            "whole_chain_state_controls": state_controls, "faults": faults, "snapshot_events": events, "negative_controls": negatives,
            "memory": {"operator_cleanup": cleanup(ops), "HC_cleanup": {str(l): h.stats() for l, h in hc.items()}, "moe_cleanup": {str(l): m.cleanup for l, m in chain.moe.items()},
                "chain_cleanup": chain.stats(), "chain_active_stats": chain_stats, "moe_active_stats": moe_stats,
                "combined_native_host_allowance_bytes": native_allowance, "additional_reference_host_allowance_bytes": REFERENCE_HOST_MIB * MiB,
                "retained_fixture_host_allowance_bytes": FIXTURE_HOST_MIB * MiB, "minimum_sampled_host_available_bytes": minimum,
                "all_frame_reservations_released": chain.attention.budget.reserved == 0, "reservation_is_RSS_limit": False, "state_storage": "bounded_host_diagnostic_frame"},
            "reference_runtime_workarounds": hc_ref[20].workarounds + sharing.index.workarounds,
            "weights_fetched_during_comparison_bytes": 0, "maximum_unprimed_continuous_context": 4,
            "scope": "Complete consecutive original blocks 20..25; generated layer-20 entry, all actual hidden/pre handoffs, source-20 KV/candidates and index refresh24/consumer25; no upstream backbone or Engram",
            "untouched_full_Attention_kernel": "blocked_by_shared_memory_limit", "full_model_quality": "not_run", "end_to_end_speed": "not_measured"}
        result["source_fixture_compiler_identity_sha256"], result["complete_goldens_sha256"] = identity(result), golden_digest(result)
        return result
    finally: torch.set_default_dtype(previous)
