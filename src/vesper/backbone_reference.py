"""Unchanged original Transformer.forward over prepared, original-address weights."""

from contextlib import ExitStack
import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np

from .attention_memory_reference import AttentionMemoryReference
from .attention_storage_reference import _numpy
from .attention_topology import BACKBONE_COMPRESSED_LAYERS, KV_SOURCES, role
from .attention_validation import check_arrays
from .backbone import LAYERS
from .backbone_assets import load as load_bindings
from .backbone_prepare import TOKENS
from .backbone_runtime import embeddings, moe_files, validate_moe_manifests
from .connected_reference import OfficialConnected
from .engram_reference import EngramOracle
from .engram_reference_storage import VirtualTables
from .formats import bf16_values
from .hc_reference import HCOracle
from .memory import MemoryPolicy, RuntimeMemory
from .model_artifacts import canonical, digest, read_checked
from .normalization_reference import ReferenceMath
from .output_assets import load as load_head
from .output_reference import OutputOracle
from .projection_reference import ProjectionOracle
from .resources import GiB, MemoryBudget
from .sampling_reference import original_sample
from .shared_attention import CAPACITY
from .token_prefix_reference import original_hash
from .topology_reference import SharingOracle
from .topology_validation import state_digest


class BackboneOracle:
    def __init__(self, stack, root, arrays, head_arrays, progress=None, *, token_rows=None, rows_root=None,
            experts_root=None, max_seq_len=4, temperature=.7, max_rows=1024, paged_rows=False, expert_loader=None):
        import torch
        self.torch, self.root, self.progress = torch, Path(root), progress
        directory = self.root / "reference"
        self.memory = stack.enter_context(AttentionMemoryReference(directory, 0, profile="long_context"))
        self.qualification = self.memory.qualify()
        if self.qualification["status"] != "pass": raise RuntimeError("Unqualified original Attention adapter")
        self.hc = {l: HCOracle(directory, arrays, 0, layer=l, profile="backbone_probe") for l in LAYERS}
        self.p = {l: ProjectionOracle(directory, arrays, 0, layer=l) for l in (0, 1)}
        for p in self.p.values(): p.reset(1, self.memory)
        self.sharing = SharingOracle(directory, arrays, BACKBONE_COMPRESSED_LAYERS, self.memory, 0, binding_profile="backbone_probe")
        self.sharing.reset(1); stack.callback(self.sharing.close)
        self.norm = ReferenceMath(directory, {n: arrays[n] for l in LAYERS
            for n in (f"layers.{l}.attn_norm.weight", f"layers.{l}.ffn_norm.weight")}, 0)
        self.head = OutputOracle(directory, head_arrays)
        self.hash = original_hash(self.root / "import-source", max_seq_len=max_seq_len).to("cuda:0")
        if paged_rows:
            from .paged_tables import PagedVirtualTables
        virtual = stack.enter_context((PagedVirtualTables if paged_rows else VirtualTables)(0,max_rows=max_rows))
        self.engram = EngramOracle(directory, self.root / "import-source", self.root / "engram-fixture", virtual, rows_root=rows_root,max_rows=max_rows,paged_rows=paged_rows)
        self.rows = embeddings(root) if token_rows is None else token_rows
        self.experts_root = self.root / "backbone-experts" if experts_root is None else Path(experts_root)
        self.temperature = temperature
        self.expert_loader = expert_loader
        raw = (virtual.paged_table if paged_rows else virtual.table)((129280, 10240), {i: a.tobytes() for i, a in self.rows.items()})
        self.embedding = SimpleNamespace(weight=raw.view(torch.bfloat16), vocab_start_idx=0, vocab_end_idx=129280)
        self.embedding_forward = self.p[0].method("ParallelEmbedding", "forward")
        self.p[0].scope["sample"] = self.head.sample
        self.forward = self.p[0].method("Transformer", "forward")
        self.block_forward = self.p[0].method("Block", "forward")
        # The AST function owns these globals. Binding its original sample does
        # not replace any Transformer, Block, norm, head or sampler arithmetic.
        self.blocks, self.trace, self.position, self.expert_provenance = {}, {}, 0, {}
        owner = self
        class BlockCall(SimpleNamespace):
            def __call__(block, hidden, pos, pre, mask):
                row = {"input_hidden_bf16": _numpy(hidden), "incoming_pre_fp32": _numpy(pre)}
                owner.trace[f"layer{block.layer}"] = row
                result, next_pre = owner.block_forward(block, hidden, pos, pre, mask)
                row["output_bf16"], row["next_pre_mix_fp32"] = _numpy(result), _numpy(next_pre)
                if owner.progress: owner.progress({"original_layer": block.layer, "position": pos})
                return result, next_pre
        class EngramCall:
            def __init__(self, layer): self.layer, self.layer_hash_index = layer, (1, 14).index(layer)
            def __call__(self, hidden, hashes, mask):
                result = owner.engram.step(self.layer, _numpy(hidden), _numpy(hashes), None if mask is None else _numpy(mask))
                owner.trace[f"engram{self.layer}"] = result
                return owner.hc[self.layer].tensor(result["output_bf16"])
        for l in LAYERS:
            block = BlockCall(layer=l, norm_eps=1e-20, hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6,
                engram=EngramCall(l) if l in (1, 14) else None,
                attn_norm=self.norm.modules[f"layers.{l}.attn_norm.weight"], ffn_norm=self.norm.modules[f"layers.{l}.ffn_norm.weight"])
            for part in ("attn", "ffn"):
                for suffix in ("fn", "scale", "base"):
                    setattr(block, f"hc_{part}_{suffix}", self.hc[l].weights[part + "_" + suffix])
            def pre_call(b, x, pre, l=l):
                out = self.hc[l].functions["hc_pre"](b, x, pre)
                row = self.trace[f"layer{l}"]
                key = next((n for n in ("attn_pre_output_bf16", "ffn_pre_output_bf16") if n not in row), None)
                if key is not None: row[key] = _numpy(out)
                return out
            def post_call(b, x, residual, post, comb, l=l):
                out = self.hc[l].functions["hc_post"](b, x, residual, post, comb)
                row = self.trace[f"layer{l}"]
                part = "ffn" if "hc_ffn_pre_fp32" in row else "attn"
                row[part + "_post_output_bf16"] = _numpy(out)
                return out
            block.hc_pre = MethodType(pre_call, block)
            block.hc_post = MethodType(post_call, block)
            def mixes(b, x, fn, scale, base, l=l):
                result = self.hc[l].functions["hc_mixes"](b, x, fn, scale, base)
                part = "attn" if fn is b.hc_attn_fn else "ffn"
                self.trace[f"layer{l}"][f"hc_{part}_pre_fp32"] = _numpy(result[0])
                return result
            block.hc_mixes = MethodType(mixes, block)
            block.attn = lambda x, pos, l=l: self.attention(l, x, pos)
            block.ffn = lambda x, mask, l=l: self.moe(l, x, mask)
            self.blocks[l] = block
        def embed(ids):
            if any(int(i) not in self.rows for i in _numpy(ids).reshape(-1)):
                raise ValueError("Refuse unbacked original embedding rows before GPU access")
            if paged_rows:virtual.activate(self.embedding.weight,_numpy(ids))
            result = self.embedding_forward(self.embedding, ids)
            self.trace["embedding_bf16"] = _numpy(result.unsqueeze(2).repeat(1, 1, 4, 1))
            return result
        self.target_layers = tuple(json.loads((directory / "config.json").read_bytes())["dspark_target_layer_ids"])
        self.model = SimpleNamespace(engram_hash=self.hash, embed=embed, hc_mult=4, target_layer_ids=self.target_layers,
            layers=list(self.blocks.values()), temperature=temperature,
            norm=lambda x: self.head.RMSNorm_forward(SimpleNamespace(weight=self.head.norm, eps=1e-20), x),
            head=lambda x: self.head.ParallelHead_forward(SimpleNamespace(weight=self.head.weight), x))
        self.methods = {"Transformer": self.p[0].methods["Transformer.forward"], "Block": self.p[0].methods["Block.forward"],
            "embedding": self.p[0].methods["ParallelEmbedding.forward"], "sharing": self.sharing.methods,
            "HC": {str(l): o.methods for l, o in self.hc.items()}, "Engram": self.engram.methods, "output": self.head.methods}
        self.virtual = virtual
        self.minimum_gpu_free = torch.cuda.mem_get_info(0)[0]
        torch.cuda.manual_seed(2026)

    def attention(self, layer, x, pos):
        row = self.trace[f"layer{layer}"]
        row["attn_norm_output_bf16"] = _numpy(x)
        if layer < 2:
            p = self.p[layer]; out = p.full_forward(p.original, x, pos)
        else:
            original = self.sharing.originals[layer]
            out = self.sharing.functions["Attention.forward"](original, x, pos)
            captured, shared = self.sharing.captured, self.sharing.scope["shared_attn"]
            prefix = f"layer{layer}_"
            values = {"query_rope_output_bf16": captured["query"], "combined_KV_bf16": captured["combined_KV"],
                "combined_indices_i32": captured["combined_indices"], "attention_output_bf16": captured["Attention_output"],
                "selected_indices_i32": _numpy(shared.topk_idxs), "window_cache_bf16": _numpy(original.window_kv_cache),
                "compressed_visible_bf16": _numpy(shared.compress_kv[:, :(pos + x.shape[1]) // role(layer).ratio])}
            self.trace.setdefault("sharing_attention", {}).update((prefix + n, a) for n, a in values.items())
        row["attention_output_bf16"] = _numpy(out)
        return out

    def moe(self, layer, x, mask):
        if mask is not None: raise ValueError("This full original-forward corpus is text only")
        self.trace[f"layer{layer}"]["ffn_norm_output_bf16"] = _numpy(x)
        oracle = OfficialConnected(*moe_files(self.root, self.bindings, layer), self.experts_root / f"layer{layer}",
            self.root / "reference", layer=layer, profile="backbone_probe", expert_loader=self.expert_loader)
        try:
            validate_moe_manifests(oracle, self.bindings, layer)
            stages = oracle.forward(bf16_values(_numpy(x).reshape(5120)), False)
            for key, manifest in oracle.provenance.items():
                if key in self.expert_provenance and self.expert_provenance[key] != manifest:
                    raise ValueError("An original expert changed between token calls")
                self.expert_provenance[key] = manifest
            self.minimum_gpu_free = min(self.minimum_gpu_free, self.torch.cuda.mem_get_info(0)[0])
            self.trace[f"layer{layer}"].update(("moe_" + n, a) for n, a in stages.items())
            return self.hc[layer].tensor(stages["aggregation.output_bf16"].reshape(1, 1, 5120))
        finally:
            oracle.experts.clear()
            # Original projection closures retain their owner. Collect those
            # cycles before opening another layer's temporary expert weights.
            del oracle
            import gc
            gc.collect()

    def state(self):
        torch, shared = self.torch, self.sharing.scope["shared_attn"]
        compressors, keys = {}, {}
        for l in KV_SOURCES:
            o = self.sharing.originals[l]
            partial = (_numpy(o.compressor.kv_state), _numpy(o.compressor.score_state)) if role(l).ratio == 2 else (
                np.empty((1, 0, 512), np.float32), np.empty((1, 0, 512), np.float32))
            compressors[l] = (*partial, _numpy(o.compress_kv_cache))
            keys[l] = _numpy(o.indexer.k_cache[:, :CAPACITY // role(l).ratio])
        pos = self.position
        sharing = {"kv_owner": 20, "key_owner": 20, "index_owner": 36, "candidate_owner": 20,
            "candidates": _numpy(shared.candidates), "indices": _numpy(shared.topk_idxs), "position": pos - 1, "sequence": 1}
        return {"window0": _numpy(self.p[0].original.window_kv_cache), "window1": _numpy(self.p[1].original.window_kv_cache),
            "sharing": {"windows": {l: _numpy(o.window_kv_cache) for l, o in self.sharing.originals.items()},
                "compressors": compressors, "index_keys": keys, "sharing": sharing},
            "history": _numpy(self.hash.cache[:1, max(0, pos - 3):pos]), "position": pos,
            "rng_offset": torch.cuda.default_generators[0].get_offset()}

    def step(self, ids, *, sample=True):
        torch, pos = self.torch, self.position
        if type(sample) is not bool: raise ValueError("Expected an explicit sampling decision")
        self.model.temperature = self.temperature if sample else 0.
        self.trace = {}
        before = torch.cuda.get_rng_state()
        selected, logits, hidden = self.forward(self.model, torch.from_numpy(ids.copy()).to("cuda:0"), pos)
        after = torch.cuda.get_rng_state()
        if hidden is None or tuple(hidden.shape) != (1, 1, 5120 * len(self.target_layers)):
            raise RuntimeError("Original auxiliary target-layer collection differs")
        self.last_auxiliary = {"dtype": str(hidden.dtype), "shape": list(hidden.shape), "sha256": digest(_numpy(hidden).tobytes())}
        result = self.trace
        result["output_bf16"], result["next_pre_mix_fp32"] = result["layer39"]["output_bf16"], result["layer39"]["next_pre_mix_fp32"]
        result["head"] = self.head.evaluate(result["output_bf16"], result["next_pre_mix_fp32"])
        if not np.array_equal(result["head"]["logits_fp32"].view(np.uint32), _numpy(logits).view(np.uint32)):
            raise RuntimeError("Unchanged full Transformer logits differ from independent original tail")
        torch.cuda.set_rng_state(before)
        sampling, start, end = original_sample(torch, self.head.sample, _numpy(logits), self.model.temperature)
        if not np.array_equal(sampling["ids_i64"], _numpy(selected)) or not torch.equal(torch.cuda.get_rng_state(), after):
            raise RuntimeError("Unchanged Transformer sampler or RNG progression differs")
        if not sample and start != end: raise RuntimeError("Prompt ingestion consumed original RNG")
        result["sampling"] = {**(sampling if sample else {}), "offset_before": start, "offset_after": end}
        result["initial_pre_fp32"] = result["layer0"]["incoming_pre_fp32"].copy()
        result["hash_ids_i64"] = _numpy(self.hash(torch.from_numpy(ids.copy()).to("cuda:0"), pos))
        result["compressed_i64"] = _numpy(self.hash.cache[:1, pos:pos + 1])
        result["lookbacks_i64"] = np.array([[[int(self.hash.cache[0, pos - shift].item()) if pos >= shift else int(self.hash.pad_id)
            for shift in range(4)]]], np.int64)
        self.position += 1
        return result, self.state()

    def close(self):
        """Release framework owners, including original functions' LRU tables.

        Extracted original functions can outlive their wrapper through compiler
        caches. Clearing an owner's Python reference alone does not release the
        CUDA tensors retained by its original frequency/index caches.
        """
        projections = [*self.p.values(), *self.sharing.p.values()]
        maths = [self.norm, self.sharing.comp.math, self.sharing.index.math, *(p.math for p in projections)]
        scopes = [self.sharing.scope, *(p.scope for p in projections), *(m.scope for m in maths)]
        for scope in scopes:
            for function in tuple(scope.values()):
                if callable(getattr(function, "cache_clear", None)): function.cache_clear()
        for block in self.blocks.values(): vars(block).clear()
        vars(self.model).clear(); vars(self.embedding).clear()
        for original in self.sharing.originals.values():
            for name in ("indexer", "compressor"):
                if getattr(original, name, None) is not None: vars(getattr(original, name)).clear()
            vars(original).clear()
        vars(self.sharing.scope["shared_attn"]).clear()
        for state in self.sharing.index.states.values(): vars(state).clear()
        self.sharing.index.states.clear(); self.sharing.index.weights.clear(); self.sharing.comp.weights.clear()
        for p in projections:
            p.weights.clear(); p.sink = None
            if hasattr(p, "original"): vars(p.original).clear()
        for m in maths: m.modules.clear(); m.tables.clear(); m.inverses.clear()
        for h in self.hc.values(): h.weights.clear()
        self.head.weight = self.head.norm = None
        self.engram.weights.clear(); self.engram.embeds.clear()
        self.hash.to("cpu")


def capture_original(root, progress=None):
    import torch
    from .backbone_validation import capture, identity, scope
    root = Path(root); bindings = json.loads((root / "backbone-controls.json").read_bytes())
    plan = json.loads((root / "backbone-plan.json").read_bytes())
    if (plan.get("status"), plan.get("layers"), plan.get("tokens"), plan.get("bindings_sha256")) != (
            "pass", list(LAYERS), list(TOKENS), bindings["bindings_sha256"]): raise ValueError("Unpinned backbone artifact plan")
    previous = torch.get_default_dtype(); torch.set_default_dtype(torch.bfloat16)
    try:
        torch.cuda.init()
        baseline_gpu = torch.cuda.memory_allocated(0)
        torch.cuda.reset_peak_memory_stats(0)
        with RuntimeMemory(MemoryPolicy(host_cap_bytes=32 * GiB)) as policy:
            with ExitStack() as stack, torch.inference_mode(), torch.cuda.device(0), torch.device("cuda:0"):
                budget = MemoryBudget(cap=20 * GiB, owner="backbone_original_fixture_and_framework")
                budget.acquire(budget.cap); stack.callback(budget.release, budget.cap)
                arrays = load_bindings(root, bindings); head_arrays, head_manifest = load_head(root / "output-fixture")
                stack.callback(arrays.clear); stack.callback(head_arrays.clear)
                oracle = BackboneOracle(stack, root, arrays, head_arrays, progress); oracle.bindings = bindings
                if digest(_numpy(oracle.hash.token_map).astype(np.int64).tobytes()) != plan["tokenizer"]["compressed_map_sha256"]:
                    raise ValueError("Independent original compressed tokenizer differs")
                cases = []
                for pos, token in enumerate(TOKENS):
                    result, state = oracle.step(np.array([[token]], np.int64))
                    cases.append({"meta": {"position": pos, "token_id": token}, "boundaries": capture(result, state),
                        "original_auxiliary_main_hidden": oracle.last_auxiliary,
                        "state_sha256": state_digest(state), "sampled_ids": result["sampling"]["ids_i64"].tolist(),
                        "ordered_routes": {str(l): {"indices": result[f"layer{l}"]["moe_gate.indices"].tolist(),
                            "weight_bits": result[f"layer{l}"]["moe_gate.weights"].view(np.uint32).tolist()} for l in LAYERS}})
                    if progress: progress({"original_position": pos, "boundaries": len(cases[-1]["boundaries"]), "status": "pass"})
                negatives = [check_arrays(result["engram14"]["output_bf16"], result["layer13"]["output_bf16"], "skip_original_Engram14"),
                    check_arrays(result["layer20"]["incoming_pre_fp32"], result["layer19"]["incoming_pre_fp32"], "stale_pre_at_ratio_transition")]
                if any(c["status"] != "fail" or not c["mismatches"] for c in negatives): raise RuntimeError("Undetected altered model semantics")
                report = {"schema_version": 1, "probe": "original_all40_backbone_goldens", "status": "pass", "scope": scope(),
                    "bindings_sha256": bindings["bindings_sha256"], "output_fixture": head_manifest, "expert_plan": plan["experts"],
                    "expert_provenance": oracle.expert_provenance,
                    "tokenizer": plan["tokenizer"], "cases": cases, "negative_controls": negatives,
                    "reference_methods": oracle.methods, "original_attention_adapter": oracle.qualification,
                    "original_source": oracle.p[0].source,
                    "original_ngram_source_sha256": digest(read_checked(root / "import-source", "inference/engram.py")),
                    "Engram_controls": oracle.engram.controls, "Engram_fixtures": oracle.engram.manifests,
                    "original_embedding_transport": oracle.virtual.stats(),
                    "minimum_sampled_reference_device_free_bytes": oracle.minimum_gpu_free,
                    "original_grouped_conversion": {str(l): (oracle.p[l] if l < 2 else oracle.sharing.p[l]).converted_sha256 for l in LAYERS},
                    "weight_payloads_fetched_in_reference": 0}
                report["identity_sha256"], report["complete_goldens_sha256"] = identity(report), canonical(cases)
                oracle.close()
                del result, state, oracle
            import gc
            gc.collect(); torch.cuda.synchronize()
            report["reference_torch_device_allocated_before_workspace_cleanup_bytes"] = torch.cuda.memory_allocated(0)
            # The qualified PyTorch build retains private cuBLAS/cuBLASLt
            # workspaces after tensor owners die. Release only this process's
            # framework workspaces; do not reset the GPU or its other clients.
            torch._C._cuda_clearCublasWorkspaces()
            gc.collect(); torch.cuda.empty_cache()
            report["reference_torch_device_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
            report["reference_torch_device_allocated_baseline_bytes"] = baseline_gpu
            report["reference_torch_device_allocated_after_cleanup_bytes"] = torch.cuda.memory_allocated(0)
            if report["reference_torch_device_allocated_after_cleanup_bytes"] != baseline_gpu:
                live = [{"shape": list(t.shape), "dtype": str(t.dtype), "bytes": t.numel() * t.element_size()}
                    for t in gc.get_objects() if type(t) in (torch.Tensor, torch.nn.Parameter) and t.device.type == "cuda"]
                if progress: progress({"reference_live_tensor_cleanup_diagnostic": live})
                raise RuntimeError("Original framework retained GPU allocations after all-layer cleanup: "
                    + str(report["reference_torch_device_allocated_after_cleanup_bytes"]) + " versus " + str(baseline_gpu))
            report["memory_before_policy_close"] = policy.snapshot()
        report["memory_after_policy_close"] = policy.snapshot()
        return report
    finally: torch.set_default_dtype(previous)
