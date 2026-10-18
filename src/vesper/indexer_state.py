"""Transactional diagnostic index frames for owners 2/20 and consumer 24.

This owns bounded host key-cache copies and one snapshot. It qualifies index
sharing within a frame, not compressed KV or full Attention-layer execution.
"""

import copy
from dataclasses import dataclass
import threading

import numpy as np

from .indexer import MAX_INDEX_POSITIONS, finite_bf16, validate_query
from .resources import MemoryBudget, MiB


@dataclass(frozen=True, eq=False)
class IndexSnapshot:
    position: int


class NativeIndexSession:
    def __init__(self, runtime, batch=1, capacity=128):
        if type(batch) is not int or batch not in (1, 2) or type(capacity) is not int or not 1 <= capacity <= MAX_INDEX_POSITIONS:
            raise ValueError("Unsupported bounded index-session profile")
        self.runtime, self.batch, self.capacity = runtime, batch, capacity
        self.lock, self.budget, self.opened = threading.Lock(), MemoryBudget(cap=128 * MiB), False

    def __enter__(self):
        with self.lock:
            if self.opened or not self.runtime.opened:
                raise RuntimeError("Index session needs an opened runtime and single ownership")
            self.budget.acquire(128 * MiB)
            try:
                self._clear()
                self.opened = True
                return self
            except BaseException:
                self.budget.release(128 * MiB)
                raise

    def _clear(self):
        caches = {layer: np.zeros((self.batch, self.capacity, 128), np.uint16) for layer in (2, 20)}
        self._caches, self.position, self._last = caches, 0, None
        self.handle, self.saved = None, None

    def _ready(self):
        if not self.opened or not self.runtime.opened:
            raise RuntimeError("Index session is not open")
        self.budget.acquire(0)

    def inspect(self):
        with self.lock:
            self._ready()
            return {"position": self.position, "cache2_bf16": self._caches[2].copy(),
                    "cache20_bf16": self._caches[20].copy(), "shared": copy.deepcopy(self._last)}

    def prime(self, keys2, keys20, position):
        """Explicit diagnostic initial state; never an implicit cache/model resume."""
        with self.lock:
            self._ready()
            if self.position or self.handle is not None or type(position) is not int or not 0 < position < self.capacity:
                raise ValueError("Only a fresh session accepts a bounded diagnostic initial state")
            for keys, n in ((keys2, position // 2), (keys20, position)):
                if (not isinstance(keys, np.ndarray) or keys.dtype != np.uint16 or keys.shape != (self.batch, n, 128)
                        or not keys.flags.c_contiguous or np.any((keys & 0x7f80) == 0x7f80)):
                    raise ValueError("Invalid diagnostic original-key initial state")
            caches = {layer: a.copy() for layer, a in self._caches.items()}
            caches[2][:, :keys2.shape[1]], caches[20][:, :keys20.shape[1]] = keys2, keys20
            self._caches, self.position = caches, position

    def _validate(self, xs, qrs, latents, position):
        if (any(not isinstance(a, dict) for a in (xs, qrs, latents)) or set(xs) != {2, 20, 24} or set(qrs) != {2, 20, 24} or set(latents) != {2, 20}
                or type(position) is not int or position != self.position):
            raise ValueError("Index frame must supply all source/consumer inputs at the next exact position")
        for layer in (2, 20, 24):
            validate_query(xs[layer], qrs[layer], layer, position)
        shape = xs[2].shape[:2]
        if (shape[0] != self.batch or any(a.shape[:2] != shape for a in (*xs.values(), *qrs.values()))
                or (position and shape[1] != 1) or position + shape[1] > self.capacity):
            raise ValueError("Index frame layout or bounded decode/prefill position differs")
        for layer, ratio in ((2, 2), (20, 1)):
            count = shape[1] // ratio if position == 0 else (position + 1) // ratio - position // ratio
            latent = latents[layer]
            if count:
                finite_bf16(latent, 512, 16)
                if latent.shape != (self.batch, count, 512):
                    raise ValueError("Pre-RoPE latent count does not complete the original groups")
            elif latent is not None:
                raise ValueError("Incomplete compression groups must not publish an index latent")
        return shape[1]

    def step(self, xs, qrs, latents, *, start_pos=None):
        with self.lock:
            self._ready()
            pos = self.position if start_pos is None else start_pos
            sequence = self._validate(xs, qrs, latents, pos)
            caches, stages, shared = {layer: a.copy() for layer, a in self._caches.items()}, {}, {}
            offset = sequence if pos == 0 else 128
            for layer in (2, 20, 24):
                ratio = 2 if layer == 2 else 1
                n = (pos + sequence) // ratio
                def add(prefix, values):
                    stages.update((f"layer{layer}_" + prefix + name, a) for name, a in values.items())
                if layer != 24:
                    latent = latents[layer]
                    if latent is not None:
                        positions = (np.arange(latent.shape[1], dtype=np.int32) * ratio if pos == 0
                                     else np.array([pos + 1 - ratio], np.int32))
                        key = self.runtime.key(latent, layer, positions)
                        add("key_", key)
                        caches[layer][:, pos // ratio:pos // ratio + latent.shape[1]] = key["quant_rounded_bf16"]
                    keys = caches[layer][:, :n].copy(order="C")
                else:
                    # Key/candidate owners are fixed by the pinned model configuration.
                    keys = shared["keys20"].copy()
                    stages["shared20_to24_keys_bf16"] = keys.copy()
                    stages["shared20_to24_candidates_bool"] = shared["candidates20"].copy()
                stages[f"layer{layer}_index_keys_bf16"] = keys.copy()
                query = self.runtime.queries(xs[layer], qrs[layer], layer, pos)
                add("query_", query)
                score = self.runtime.score(query["quant_rounded_bf16"], keys, query["weights_scaled_bf16"])
                add("score_", score)
                lengths = (np.broadcast_to(np.arange(1, sequence + 1, dtype=np.int32)[None, :] // ratio, (self.batch, sequence)).copy()
                           if pos == 0 else np.full((self.batch, sequence), n, np.int32))
                selection = self.runtime.select(score["score_bf16"], lengths, offset, candidate_source=layer == 20,
                                                candidates=shared["candidates20"] if layer == 24 else None)
                add("selection_", selection)
                if layer == 2:
                    shared["indices3"] = selection["indices_i32"].copy()
                    stages["shared2_to3_indices_i32"] = selection["indices_i32"].copy()
                elif layer == 20:
                    shared["keys20"], shared["candidates20"] = keys.copy(), selection["candidates_bool"].copy()
                else:
                    shared["indices25"] = selection["indices_i32"].copy()
                    stages["shared24_to25_indices_i32"] = selection["indices_i32"].copy()
            for layer in (2, 20): stages[f"cache{layer}_bf16"] = caches[layer].copy()
            shared["position"], shared["sequence"] = pos, sequence
            self.budget.acquire(0)
            # Commit all owners and sharing metadata only after the complete frame succeeds.
            self._caches, self.position, self._last = caches, pos + sequence, shared
            return stages

    def shared_indices(self, consumer, position, sequence):
        with self.lock:
            self._ready()
            if (type(consumer) is not int or consumer not in (3, 25) or self._last is None
                    or type(position) is not int or type(sequence) is not int
                    or self._last["position"] != position or self._last["sequence"] != sequence):
                raise ValueError("Missing, stale or foreign original index-sharing frame")
            return self._last["indices3" if consumer == 3 else "indices25"].copy()

    def snapshot(self):
        with self.lock:
            self._ready()
            if self.handle is not None:
                raise RuntimeError("Only one bounded index snapshot is available")
            saved = ({layer: a.copy() for layer, a in self._caches.items()}, copy.deepcopy(self._last))
            self.handle, self.saved = IndexSnapshot(self.position), saved
            return self.handle

    def _check_handle(self, handle):
        if self.handle is None or handle is not self.handle:
            raise ValueError("Foreign, expired or consumed index snapshot")

    def rollback(self, handle):
        with self.lock:
            self._ready()
            self._check_handle(handle)
            caches, last = copy.deepcopy(self.saved)
            result = {"position": handle.position, "cache2_bf16": caches[2].copy(), "cache20_bf16": caches[20].copy(), "shared": copy.deepcopy(last)}
            self._caches, self.position, self._last = caches, handle.position, last
            self.handle, self.saved = None, None
            return result

    def discard(self, handle):
        with self.lock:
            self._ready()
            self._check_handle(handle)
            self.handle, self.saved = None, None

    def reset(self):
        with self.lock:
            self._ready()
            self._clear()

    def __exit__(self, *_):
        with self.lock:
            if not self.opened: return
            self.opened, self.handle, self.saved, self._last, self._caches = False, None, None, None, None
            self.budget.release(128 * MiB)
