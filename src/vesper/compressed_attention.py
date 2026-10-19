"""Atomic diagnostic frames connecting original compressed Attention 2/20/24.

All committed state is bounded host storage. GPU operators return diagnostic
copies; this is not a GPU-resident production cache or a full model executor.
"""

import copy
from dataclasses import dataclass
import threading

import numpy as np

from .compression import stage_compression
from .indexer_state import NativeIndexSession
from .kv import window_indices
from .projections import validate_values
from .resources import MemoryBudget, MiB

LAYERS, CAPACITY, MAX_POSITION = (2, 20, 24), 128, 64


def validate_frame(xs, batch, position):
    if not isinstance(xs, dict) or set(xs) != set(LAYERS): raise ValueError("A compressed Attention frame needs all three profiles")
    if type(position) is not int or not 0 <= position < MAX_POSITION: raise ValueError("Unsupported bounded frame position")
    shape = None
    for x in xs.values():
        validate_values(x, 5120)
        if x.ndim != 3 or x.shape[0] != batch or not 1 <= x.shape[1] <= 3 or (position and x.shape[1] != 1):
            raise ValueError("Expected bounded frame batch, prefill or single-query decode")
        if shape is not None and x.shape != shape: raise ValueError("Frame inputs disagree in layout")
        shape = x.shape
    if position + shape[1] > MAX_POSITION: raise ValueError("Compressed Attention frame exceeds its context qualification")
    return shape[1]


def initial_state(batch):
    compressors = {}
    for layer, ratio in ((2, 2), (20, 1)):
        partial = np.zeros((batch, 2 if ratio == 2 else 0, 512), np.float32)
        compressors[layer] = (partial, np.full_like(partial, -np.inf), np.zeros((batch, CAPACITY // ratio, 512), np.uint16))
    return {"windows": {l: np.zeros((batch, 128, 512), np.uint16) for l in LAYERS},
            "compressors": compressors, "index_keys": {l: np.zeros((batch, CAPACITY, 128), np.uint16) for l in (2, 20)}, "sharing": None}


def stage_frame(arrays, projections, norm, compressor, indexer, attention, batch, pos, source, xs):
    """Compose original operators into a wholly unpublished candidate state."""
    sequence = validate_frame(xs, batch, pos)
    stages, qs, qrs, windows, cs, latents = {}, {}, {}, {}, {}, {}
    state = {"windows": {}, "compressors": {}, "index_keys": {}, "sharing": None}
    def add(prefix, values):
        stages.update((prefix + n, a) for n, a in values.items())
        return values.get("output_bf16")
    for layer in LAYERS:
        prefix, p = f"layer{layer}_", projections[layer]
        qr = add(prefix + "wq_a_", p.fp8("wq_a", xs[layer]))
        qr = add(prefix + "q_norm_", norm.norm(qr, arrays[f"layers.{layer}.attn.q_norm.weight"]))
        qrs[layer] = qr
        q = add(prefix + "wq_b_", p.fp8("wq_b", qr)).reshape(batch, sequence, 64, 512)
        qs[layer] = add(prefix + "query_rope_", norm.rope(q, pos, "compressed-yarn"))
        kv = add(prefix + "wkv_", p.fp8("wkv", xs[layer]))
        kv = add(prefix + "kv_norm_", norm.norm(kv, arrays[f"layers.{layer}.attn.kv_norm.weight"]))
        kv = add(prefix + "kv_rope_", norm.rope(kv, pos, "compressed-yarn"))
        rounded = p.round_window(kv)
        add(prefix + "window_", rounded)
        cache = source["windows"][layer].copy()
        if pos == 0: cache[:, :sequence] = rounded["rounded_bf16"]
        else: cache[:, pos % 128] = rounded["rounded_bf16"][:, 0]
        state["windows"][layer] = cache
        windows[layer] = rounded["rounded_bf16"] if pos == 0 else cache.copy()
        stages[prefix + "window_cache_bf16"] = cache.copy()
        stages[prefix + "window_visible_bf16"] = windows[layer].copy()
        stages[prefix + "window_indices_i32"] = window_indices(batch, sequence, pos)
        if layer in (2, 20):
            cs[layer], state["compressors"][layer] = stage_compression(compressor, layer, batch, pos, source["compressors"][layer], xs[layer])
            add(prefix + "compressor_", cs[layer])
            latent = cs[layer]["latent_pre_rope_bf16"]
            latents[layer] = latent.copy() if latent.shape[1] else None
    # Index keys always consume the original pre-RoPE latent copy. The original
    # cache rotates/rounds its own storage only after that consumer has read it.
    with NativeIndexSession(indexer, batch, CAPACITY) as index:
        if pos:
            index.prime(source["index_keys"][2][:, :pos // 2].copy(), source["index_keys"][20][:, :pos].copy(), pos)
        indexed = index.step(xs, qrs, latents, start_pos=pos)
        snapshot = index.inspect()
        state["index_keys"] = {2: snapshot["cache2_bf16"], 20: snapshot["cache20_bf16"]}
        state["sharing"] = snapshot["shared"]
    add("index_", indexed)
    for layer in LAYERS:
        prefix, p = f"layer{layer}_", projections[layer]
        owner = 2 if layer == 2 else 20
        compressed = cs[owner]["visible_bf16"].copy()
        stages[prefix + "compressed_visible_bf16"] = compressed.copy()
        if layer == 24: stages["shared20_to24_compressed_KV_bf16"] = compressed.copy()
        kv = np.concatenate((windows[layer], compressed), axis=1)
        idx = np.concatenate((stages[prefix + "window_indices_i32"], indexed[f"layer{layer}_selection_indices_i32"]), axis=-1)
        stages[prefix + "combined_KV_bf16"], stages[prefix + "combined_indices_i32"] = kv, idx
        o = add(prefix + "attention_", attention(qs[layer], kv, arrays[f"layers.{layer}.attn.attn_sink"], idx, diagnostics=True))
        o = add(prefix + "inverse_rope_", norm.rope(o, pos, "compressed-yarn", inverse=True))
        o = add(prefix + "wo_a_", p.grouped(o.reshape(batch, sequence, 8, 4096)))
        add(prefix + "wo_b_", p.fp8("wo_b", o.reshape(batch, sequence, 8192)))
    return stages, state


@dataclass(frozen=True, eq=False)
class CompressedAttentionSnapshot:
    position: int


class NativeCompressedAttention:
    """One serialized frame owns all source/consumer state and one snapshot."""
    def __init__(self, arrays, projections, norm, compressor, indexer, attention, batch=1):
        if type(batch) is not int or batch not in (1, 2): raise ValueError("Unsupported compressed Attention batch")
        self.arrays, self.projections, self.norm, self.compressor, self.indexer, self.attention = arrays, projections, norm, compressor, indexer, attention
        self.batch, self.opened = batch, False
        self.lock, self.budget = threading.Lock(), MemoryBudget(cap=256 * MiB)

    def __enter__(self):
        with self.lock:
            if self.opened or set(self.projections) != set(LAYERS): raise ValueError("Invalid compressed Attention ownership")
            ordinal = self.projections[2].ordinal
            if any(not op.opened or op.ordinal != ordinal for op in (*self.projections.values(), self.norm, self.compressor, self.indexer, self.attention)):
                raise ValueError("All qualified frame operators must be open on the same device")
            if any(self.projections[l].layer != l for l in LAYERS): raise ValueError("Original layer projection identity differs")
            self.budget.acquire(256 * MiB)
            try:
                self._clear()
                self.opened = True
                return self
            except BaseException:
                self.budget.release(256 * MiB)
                raise

    def _clear(self):
        state = initial_state(self.batch)
        self.state, self.position, self.handle, self.saved = state, 0, None, None

    def _ready(self):
        if not self.opened: raise RuntimeError("Compressed Attention frame is not open")
        self.budget.acquire(0)

    def step(self, xs, *, start_pos=None, finalize=None):
        with self.lock:
            self._ready()
            pos = self.position if start_pos is None else start_pos
            if type(pos) is not int or pos != self.position or (finalize is not None and not callable(finalize)):
                raise ValueError("Compressed Attention needs the next exact frame and optional callable finalizer")
            stages, state = stage_frame(self.arrays, self.projections, self.norm, self.compressor, self.indexer, self.attention, self.batch, pos, self.state, xs)
            result = stages if finalize is None else finalize(stages)
            self.budget.acquire(0)
            self.state, self.position = state, pos + xs[2].shape[1]
            return result

    def inspect(self):
        with self.lock:
            self._ready()
            return {"position": self.position, "state": copy.deepcopy(self.state)}

    def snapshot(self):
        with self.lock:
            self._ready()
            if self.handle is not None: raise RuntimeError("Only one compressed Attention snapshot is available")
            saved = copy.deepcopy(self.state)
            self.saved, self.handle = saved, CompressedAttentionSnapshot(self.position)
            return self.handle

    def _check(self, handle):
        if self.handle is None or handle is not self.handle: raise ValueError("Foreign, consumed or expired compressed Attention snapshot")

    def rollback(self, handle):
        with self.lock:
            self._ready()
            self._check(handle)
            state = copy.deepcopy(self.saved)
            result = {"position": handle.position, "state": copy.deepcopy(state)}
            self.budget.acquire(0)
            self.state, self.position, self.handle, self.saved = state, handle.position, None, None
            return result

    def discard(self, handle):
        with self.lock:
            self._ready()
            self._check(handle)
            self.handle, self.saved = None, None

    def reset(self):
        with self.lock:
            self._ready()
            self._clear()

    def __exit__(self, *_):
        with self.lock:
            if not self.opened: return
            self.opened, self.state, self.handle, self.saved = False, None, None, None
            self.budget.release(256 * MiB)
