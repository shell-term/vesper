"""Atomic host diagnostic frames for pinned KV/index source and reuse roles."""

import copy
import numpy as np

from .attention_topology import role, validate_group
from .compressed_attention import NativeCompressedAttention
from .compression import stage_compression
from .kv import window_indices
from .projections import validate_values

CAPACITY = 1152


def initial_state(layers, batch):
    validate_group(layers)
    owners = [l for l in layers if role(l).owns_kv]
    return {"windows": {l: np.zeros((batch, 128, 512), np.uint16) for l in layers},
        "compressors": {l: (np.zeros((batch, 2 if role(l).ratio == 2 else 0, 512), np.float32),
            np.full((batch, 2 if role(l).ratio == 2 else 0, 512), -np.inf, np.float32),
            np.zeros((batch, CAPACITY // role(l).ratio, 512), np.uint16)) for l in owners},
        "index_keys": {l: np.zeros((batch, CAPACITY // role(l).ratio, 128), np.uint16) for l in owners}, "sharing": None}


def validate_frame(xs, layers, batch, position):
    validate_group(layers)
    if (type(position) is not int or not 0 <= position < CAPACITY or not isinstance(xs, dict) or set(xs) != set(layers)):
        raise ValueError("Invalid exact sharing-frame position/owners")
    shape = None
    for x in xs.values():
        validate_values(x, 5120)
        if (x.ndim != 3 or x.shape[0] != batch or not 1 <= x.shape[1] <= 3 or (position and x.shape[1] != 1)
                or (shape is not None and shape != x.shape) or position + x.shape[1] > CAPACITY):
            raise ValueError("Expected bounded same-layout prefill or single-query decode")
        shape = x.shape
    return shape[1]


def validate_seed(state, layers, batch, position):
    if (type(position) is not int or not 0 < position < CAPACITY or position % 2 or not isinstance(state, dict)
            or set(state) != {"windows", "compressors", "index_keys", "sharing"} or state["sharing"] is not None):
        raise ValueError("Expected explicit even-position diagnostic history")
    owners = {l for l in layers if role(l).owns_kv}
    if (not isinstance(state["windows"], dict) or set(state["windows"]) != set(layers)
            or not isinstance(state["compressors"], dict) or set(state["compressors"]) != owners
            or not isinstance(state["index_keys"], dict) or set(state["index_keys"]) != owners):
        raise ValueError("Diagnostic history owner identities differ")
    def array(a, shape, dtype, scores=False, compressed=False):
        if not isinstance(a, np.ndarray) or a.shape != shape or a.dtype != dtype or not a.flags.c_contiguous:
            raise ValueError("Diagnostic history dtype/layout differs")
        if dtype == np.uint16:
            if np.any((a & 0x7f80) == 0x7f80) or (compressed and np.any((a & 0x7fff) > 0x4528)):
                raise ValueError("Diagnostic history is outside finite original scope")
        elif np.any(~(np.isfinite(a) | (np.isneginf(a) if scores else np.zeros(a.shape, np.bool_)))):
            raise ValueError("Diagnostic partial state is nonfinite")
    for l in layers: array(state["windows"][l], (batch, 128, 512), np.uint16)
    for l in owners:
        r, c = role(l).ratio, state["compressors"][l]
        if not isinstance(c, tuple) or len(c) != 3: raise ValueError("Malformed diagnostic compressor state")
        array(c[0], (batch, 2 if r == 2 else 0, 512), np.float32)
        array(c[1], c[0].shape, np.float32, scores=True)
        array(c[2], (batch, CAPACITY // r, 512), np.uint16, compressed=True)
        array(state["index_keys"][l], (batch, CAPACITY // r, 128), np.uint16)
        if np.any(c[2][:, position // r:]) or np.any(state["index_keys"][l][:, position // r:]):
            raise ValueError("Diagnostic future cache positions must be zero")


def stage_group(arrays, projections, norm, compressor, indexer, attention, layers, batch, pos, source, xs):
    sequence = validate_frame(xs, layers, batch, pos)
    state, stages = copy.deepcopy(source), {}
    shared = {}
    def add(prefix, values):
        stages.update((prefix + n, a) for n, a in values.items())
        return values.get("output_bf16")
    for l in layers:
        r, p, prefix = role(l), projections[l], f"layer{l}_"
        qr = add(prefix + "wq_a_", p.fp8("wq_a", xs[l]))
        qr = add(prefix + "q_norm_", norm.norm(qr, arrays[f"layers.{l}.attn.q_norm.weight"]))
        q = add(prefix + "wq_b_", p.fp8("wq_b", qr)).reshape(batch, sequence, 64, 512)
        q = add(prefix + "query_rope_", norm.rope(q, pos, "compressed-yarn"))
        kv = add(prefix + "wkv_", p.fp8("wkv", xs[l]))
        kv = add(prefix + "kv_norm_", norm.norm(kv, arrays[f"layers.{l}.attn.kv_norm.weight"]))
        kv = add(prefix + "kv_rope_", norm.rope(kv, pos, "compressed-yarn"))
        rounded = p.round_window(kv)
        add(prefix + "window_", rounded)
        cache = state["windows"][l]
        if pos == 0: cache[:, :sequence] = rounded["rounded_bf16"]
        else: cache[:, pos % 128] = rounded["rounded_bf16"][:, 0]
        window = rounded["rounded_bf16"] if pos == 0 else cache.copy()
        window_idx = window_indices(batch, sequence, pos)
        stages[prefix + "window_cache_bf16"], stages[prefix + "window_visible_bf16"] = cache.copy(), window.copy()
        stages[prefix + "window_indices_i32"] = window_idx
        n, latent = (pos + sequence) // r.ratio, None
        if r.owns_kv:
            cs, state["compressors"][l] = stage_compression(compressor, l, batch, pos, source["compressors"][l], xs[l], max_position=CAPACITY)
            add(prefix + "compressor_", cs)
            if cs["latent_pre_rope_bf16"].shape[1]: latent = cs["latent_pre_rope_bf16"]
            if n and latent is not None:
                keys = indexer.key(latent, l, cs["latent_positions_i32"])
                add(prefix + "index_key_", keys)
                begin = pos // r.ratio
                state["index_keys"][l][:, begin:begin + latent.shape[1]] = keys["quant_rounded_bf16"]
            shared["kv_owner"] = l
            # An empty first group skips Indexer.forward, including its globals.
            if n: shared["key_owner"] = l
        if shared.get("kv_owner") != r.kv_owner: raise ValueError("Wrong or missing shared compressed-KV owner")
        compressed = state["compressors"][r.kv_owner][2][:, :n].copy(order="C")
        stages[prefix + "compressed_visible_bf16"] = compressed.copy()
        if r.owns_index:
            if n:
                if shared.get("key_owner") != r.kv_owner: raise ValueError("Wrong or missing shared index-key owner")
                keys = state["index_keys"][r.kv_owner][:, :n].copy(order="C")
                stages[prefix + "index_keys_bf16"] = keys.copy()
                query = indexer.queries(xs[l], qr, l, pos)
                add(prefix + "index_query_", query)
                score = indexer.score(query["quant_rounded_bf16"], keys, query["weights_scaled_bf16"])
                add(prefix + "index_score_", score)
                lengths = (np.broadcast_to(np.arange(1, sequence + 1, dtype=np.int32)[None, :] // r.ratio, (batch, sequence)).copy()
                           if pos == 0 else np.full((batch, sequence), n, np.int32))
                selection = indexer.select(score["score_bf16"], lengths, window.shape[1], candidate_source=l == 20,
                                           candidates=shared.get("candidates") if l > 20 else None)
                if l > 20 and shared.get("candidate_owner") != 20: raise ValueError("Wrong or missing source-20 candidates")
                add(prefix + "index_selection_", selection)
                idx = selection["indices_i32"]
                if l == 20: shared["candidate_owner"], shared["candidates"] = 20, selection["candidates_bool"].copy()
            else: idx = np.empty((batch, sequence, 0), np.int32)
            shared["index_owner"], shared["indices"] = l, idx.copy()
        else:
            if shared.get("index_owner") != r.index_owner: raise ValueError("Wrong or missing reused selected-index owner")
            idx = shared["indices"].copy()
        stages[prefix + "selected_indices_i32"] = idx.copy()
        combined, indices = np.concatenate((window, compressed), axis=1), np.concatenate((window_idx, idx), axis=-1)
        stages[prefix + "combined_KV_bf16"], stages[prefix + "combined_indices_i32"] = combined, indices
        o = add(prefix + "attention_", attention(q, combined, arrays[f"layers.{l}.attn.attn_sink"], indices, diagnostics=True))
        o = add(prefix + "inverse_rope_", norm.rope(o, pos, "compressed-yarn", inverse=True))
        o = add(prefix + "wo_a_", p.grouped(o.reshape(batch, sequence, 8, 4096)))
        add(prefix + "wo_b_", p.fp8("wo_b", o.reshape(batch, sequence, 8192)))
    for l, value in state["index_keys"].items(): stages[f"owner{l}_index_cache_bf16"] = value.copy()
    shared["position"], shared["sequence"] = pos, sequence
    state["sharing"] = shared
    return stages, state


class SharedAttentionFrame(NativeCompressedAttention):
    def __init__(self, arrays, projections, norm, compressor, indexer, attention, layers, batch=1):
        self.layers = validate_group(layers)
        super().__init__(arrays, projections, norm, compressor, indexer, attention, batch, profile="long_context", capacity=CAPACITY)

    def __enter__(self):
        with self.lock:
            if self.opened or set(self.projections) != set(self.layers): raise ValueError("Sharing-frame projection ownership differs")
            ordinal = self.projections[self.layers[0]].ordinal
            if (any(not op.opened or op.ordinal != ordinal for op in (*self.projections.values(), self.norm, self.compressor, self.indexer, self.attention))
                    or any(self.projections[l].layer != l for l in self.layers)
                    or getattr(self.attention, "profile", None) != "long_context"
                    or set(self.compressor.layers) != {l for l in self.layers if role(l).owns_kv}
                    or set(self.indexer.layers) != {l for l in self.layers if role(l).owns_index}):
                raise ValueError("Sharing frame requires matching original operator identities")
            self.budget.acquire(self.host_allowance)
            try:
                self._clear()
                self.opened = True
                return self
            except BaseException:
                self.budget.release(self.host_allowance)
                raise

    def _clear(self):
        state = initial_state(self.layers, self.batch)
        self.state, self.position, self.handle, self.saved = state, 0, None, None

    def step(self, xs, *, start_pos=None, finalize=None):
        with self.lock:
            self._ready()
            pos = self.position if start_pos is None else start_pos
            if type(pos) is not int or pos != self.position or (finalize is not None and not callable(finalize)):
                raise ValueError("Sharing frame requires its next exact position and optional callable finalizer")
            stages, state = stage_group(self.arrays, self.projections, self.norm, self.compressor, self.indexer, self.attention,
                                       self.layers, self.batch, pos, self.state, xs)
            result = stages if finalize is None else finalize(stages)
            self.budget.acquire(0)
            self.state, self.position = state, pos + xs[self.layers[0]].shape[1]
            return result

    def prime_diagnostic(self, state, position):
        with self.lock:
            self._ready()
            if self.position or self.handle is not None: raise ValueError("Only a fresh sharing frame accepts diagnostic history")
            validate_seed(state, self.layers, self.batch, position)
            candidate = copy.deepcopy(state)
            self.budget.acquire(0)
            self.state, self.position = candidate, position
