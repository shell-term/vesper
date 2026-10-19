"""Independent original full-forward controls for every registered sharing role."""

import ast
from functools import lru_cache
import hashlib
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np

from .attention_storage import STAGES
from .attention_storage_reference import _numpy
from .attention_topology import role, validate_config
from .attention_validation import check_arrays
from .prefix_reference import PREFIX_SOURCE_HASHES
from .compressed_attention_reference import METHODS, NormBridge
from .compression_reference import CompressionOracle
from .indexer_reference import IndexerOracle
from .projection_reference import ProjectionOracle
from .shared_attention import CAPACITY, SharedAttentionFrame


class SharingOracle:
    def __init__(self, directory, arrays, layers, memory, ordinal=0, *, binding_profile="qualified"):
        import json
        raw = (Path(directory) / "config.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != PREFIX_SOURCE_HASHES["config.json"]: raise ValueError("Pinned topology config bytes differ")
        self.config = json.loads(raw)
        validate_config(self.config)
        self.arrays, self.layers, self.memory, self.ordinal = arrays, layers, memory, ordinal
        self.binding_profile = binding_profile
        self.p = {l: ProjectionOracle(directory, arrays, ordinal, layer=l) for l in layers}
        owners, sources = tuple(l for l in layers if role(l).owns_kv), tuple(l for l in layers if role(l).owns_index)
        self.comp = CompressionOracle(directory, arrays, ordinal, layers=owners)
        self.index = IndexerOracle(directory, arrays, ordinal, layers=sources)
        self.torch = self.index.torch
        self.norm = NormBridge(arrays, self.p, self.comp, ordinal)
        self.comp_adapter = SimpleNamespace(opened=True, ordinal=ordinal, arrays=arrays, layers=self.comp.layers,
            project=self.comp.project, pool=self.comp.pool, round_kv=self.comp.round_kv, norm=self.norm)
        self.index.opened = True
        for p in self.p.values(): p.opened = True
        torch, kernel = self.torch, self.index.kernel
        self.scope = {"torch": torch, "F": torch.nn.functional, "lru_cache": lru_cache, "world_size": 1,
            "fp8_block_size": 32, "fp4_block_size": 32, "scale_fmt": "ue8m0", "scale_dtype": torch.float8_e8m0fnu,
            "act_quant": kernel.act_quant, "fp4_act_quant": kernel.fp4_act_quant,
            "fp8_gemm": kernel.fp8_gemm, "fp4_gemm": kernel.fp4_gemm,
            "apply_rotary_emb": self.p[layers[0]].math.scope["apply_rotary_emb"], "sparse_attn": self.sparse}
        tree = ast.parse((Path(directory) / "model.py").read_bytes())
        self.functions, self.methods, self.controls = {}, {}, []
        for label, expected in METHODS.items():
            parts = label.split('.')
            parent = tree if len(parts) == 1 else next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == parts[0])
            node = next(n for n in parent.body if isinstance(n, ast.FunctionDef) and n.name == parts[-1])
            actual = hashlib.sha256(ast.dump(node).encode()).hexdigest()
            if actual != expected: raise ValueError("Original sharing method AST differs")
            self.methods[label] = actual
            exec(compile(ast.Module(body=[node], type_ignores=[]), "<pinned-sharing-Attention>", "exec"), self.scope)
            self.functions[label] = self.scope[parts[-1]]

    def sparse(self, q, kv, sink, idx, scale):
        out = self.memory(q, kv, sink, idx, scale)
        self.captured = {"query": _numpy(q), "combined_KV": _numpy(kv), "combined_indices": _numpy(idx), "Attention_output": _numpy(out)}
        return out

    def reset(self, batch):
        if hasattr(self, "diagnostic"): self.diagnostic.__exit__()
        reference = self
        class AttentionCall:
            opened, ordinal, profile = True, self.ordinal, "long_context"
            def __call__(stub, q, kv, sink, idx, *, diagnostics=False):
                vals = reference.index.tensor(q), reference.index.tensor(kv), reference.torch.from_numpy(sink).to(f"cuda:{reference.ordinal}"), reference.index.tensor(idx)
                out, traces, _ = reference.memory._run(64, 512, diagnostics, vals)
                return dict(zip(("output_bf16", *STAGES), map(_numpy, (out, *traces))))
        self.diagnostic = SharedAttentionFrame(self.arrays, self.p, self.norm, self.comp_adapter, self.index, AttentionCall(), self.layers, batch, binding_profile=self.binding_profile)
        self.diagnostic.__enter__()
        self.index.reset(batch, CAPACITY)
        self.scope["shared_attn"] = SimpleNamespace()
        torch, functions = self.torch, self.functions
        class IndexCall(SimpleNamespace):
            def __call__(stub, *args): return functions["Indexer.forward"](stub, *args)
        class CompressorCall(SimpleNamespace):
            def __call__(stub, *args): return functions["Compressor.forward"](stub, *args)
        self.originals = {}
        for l in self.layers:
            p, r = self.p[l], role(l)
            own_kv, own_index, ratio = l in self.config["kv_source_layers"], l in self.config["index_source_layers"], self.config["compress_ratios"][l]
            o = SimpleNamespace(n_local_heads=64, n_local_groups=8, head_dim=512, rope_head_dim=64, o_lora_rank=1024,
                window_size=128, compress_ratio=ratio, softmax_scale=512**-0.5, is_kv_source=own_kv, is_index_source=own_index,
                freqs_cis=p.math.tables["compressed-yarn"], q_norm=p.math.modules[f"layers.{l}.attn.q_norm.weight"],
                kv_norm=p.math.modules[f"layers.{l}.attn.kv_norm.weight"], attn_sink=p.sink, wo_a=SimpleNamespace(weight=p.weights["wo_a"]))
            for name in ("wq_a", "wq_b", "wkv", "wo_b"):
                setattr(o, name, lambda x, n=name, p=p: functions["linear"](x, p.weights[n]))
            o.window_kv_cache = torch.zeros((batch, 128, 512), dtype=torch.bfloat16, device=f"cuda:{self.ordinal}")
            o.indexer = IndexCall(**vars(self.index.states[l])) if own_index else None
            if own_index:
                o.indexer.owns_k, o.indexer.compress_ratio = own_kv, ratio
                o.indexer.is_candidate_source = l == self.config["candidate_source_layer"]
                o.indexer.uses_candidates = 0 <= self.config["candidate_source_layer"] < l
            if own_kv:
                c = CompressorCall(compress_ratio=ratio, head_dim=512,
                    norm=self.comp.math.modules[f"layers.{l}.attn.compressor.norm.weight"],
                    wkv=lambda x, l=l: functions["linear"](x, self.comp.weights[l, "wkv"]))
                if ratio == 2:
                    c.wgate = lambda x, l=l: functions["linear"](x, self.comp.weights[l, "wgate"])
                    c.kv_state = torch.zeros((batch, 2, 512), dtype=torch.float32, device=f"cuda:{self.ordinal}")
                    c.score_state = torch.full_like(c.kv_state, -torch.inf)
                o.compressor = c
                o.compress_kv_cache = torch.zeros((batch, CAPACITY // ratio, 512), dtype=torch.bfloat16, device=f"cuda:{self.ordinal}")
            for name in ("_window_kv", "_compress_kv", "_compress_topk_idxs"):
                setattr(o, name, MethodType(functions["Attention." + name], o))
            self.originals[l] = o

    def step(self, xs, position):
        stages = self.diagnostic.step(xs, start_pos=position)
        checks = []
        for l in self.layers:
            o, p = self.originals[l], f"layer{l}_"
            out = self.functions["Attention.forward"](o, self.index.tensor(xs[l]), position)
            values = {**self.captured, "output": _numpy(out), "window": _numpy(o.window_kv_cache),
                      "selected": _numpy(self.scope["shared_attn"].topk_idxs),
                      "compressed_visible": _numpy(self.scope["shared_attn"].compress_kv[:, :(position + xs[l].shape[1]) // role(l).ratio])}
            expected = {"output": p + "wo_b_output_bf16", "window": p + "window_cache_bf16", "query": p + "query_rope_output_bf16",
                        "combined_KV": p + "combined_KV_bf16", "combined_indices": p + "combined_indices_i32", "Attention_output": p + "attention_output_bf16",
                        "selected": p + "selected_indices_i32", "compressed_visible": p + "compressed_visible_bf16"}
            checks.extend(check_arrays(stages[expected[n]], a, f"unchanged_layer{l}_" + n) for n, a in values.items())
            if role(l).owns_kv:
                checks.append(check_arrays(stages[p + "compressor_cache_bf16"], _numpy(o.compress_kv_cache), f"unchanged_layer{l}_compressed_cache"))
                checks.append(check_arrays(stages[f"owner{l}_index_cache_bf16"], _numpy(o.indexer.k_cache[:, :CAPACITY // role(l).ratio]), f"unchanged_layer{l}_key_cache"))
                if role(l).ratio == 2:
                    for name in ("kv_state", "score_state"):
                        checks.append(check_arrays(stages[p + "compressor_" + name + "_fp32"], _numpy(getattr(o.compressor, name)), f"unchanged_layer{l}_" + name))
        if 20 in self.layers:
            checks.append(check_arrays(stages["layer20_index_selection_candidates_bool"], _numpy(self.scope["shared_attn"].candidates), "unchanged_source20_candidates"))
        if any(c["status"] != "pass" for c in checks): raise RuntimeError("Sharing diagnostic differs from unchanged original full Attention")
        self.controls.append(checks)
        self.last_stages = stages
        return stages

    def prime_diagnostic(self, state, position):
        self.diagnostic.prime_diagnostic(state, position)
        for l, o in self.originals.items():
            o.window_kv_cache.copy_(self.index.tensor(state["windows"][l]))
            if role(l).owns_kv:
                o.compress_kv_cache.copy_(self.index.tensor(state["compressors"][l][2]))
                o.indexer.k_cache[:, :CAPACITY // role(l).ratio].copy_(self.index.tensor(state["index_keys"][l]))
                if role(l).ratio == 2:
                    for name, value in zip(("kv_state", "score_state"), state["compressors"][l][:2]):
                        getattr(o.compressor, name).copy_(self.torch.from_numpy(value).to(f"cuda:{self.ordinal}"))
        self.scope["shared_attn"] = SimpleNamespace()

    def snapshot(self):
        state = {l: {n: getattr(o, n).clone() for n in ("window_kv_cache", "compress_kv_cache") if hasattr(o, n)} for l, o in self.originals.items()}
        keys = {l: o.indexer.k_cache.clone() for l, o in self.originals.items() if role(l).owns_kv}
        partial = {l: {n: getattr(o.compressor, n).clone() for n in ("kv_state", "score_state")} for l, o in self.originals.items() if role(l).owns_kv and role(l).ratio == 2}
        return self.diagnostic.snapshot(), state, keys, partial, {n: v.clone() for n, v in vars(self.scope["shared_attn"]).items()}

    def rollback(self, saved):
        handle, state, keys, partial, shared = saved
        self.diagnostic.rollback(handle)
        for l, values in state.items():
            for n, v in values.items(): getattr(self.originals[l], n).copy_(v)
        for l, v in keys.items(): self.originals[l].indexer.k_cache.copy_(v)
        for l, values in partial.items():
            for n, v in values.items(): getattr(self.originals[l].compressor, n).copy_(v)
        self.scope["shared_attn"] = SimpleNamespace(**shared)

    def close(self):
        if hasattr(self, "diagnostic"): self.diagnostic.__exit__()
