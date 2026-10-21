"""Original layer-0 window Attention, with transactional KV publication."""

from contextlib import ExitStack
import threading

import numpy as np

from .memory import runtime_operation
from .kv import NativeWindowKV, window_indices
from .projections import validate_values


def validate_layer_input(values, batch, position):
    if (not isinstance(values, np.ndarray) or values.ndim != 3 or values.shape[0] != batch
            or not 1 <= values.shape[1] <= 3):
        raise ValueError("Window Attention supports one/two batches and one to three queries")
    validate_values(values, 5120)
    window_indices(batch, values.shape[1], position)


class NativeWindowAttention:
    """Own one KV sequence; reuse opened native projection/norm/Attention tools.

    Input is the original Attention.forward boundary, after the block's norm.
    Diagnostic arrays cross the host API; this is not a resident speed path.
    """

    def __init__(self, arrays, projections, normalization, attention, batch=1, *, profile="qualified", execution_target=None):
        from .prefix_profile import validate_layer
        if type(projections.layer) is not int or projections.layer not in (0, 1):
            raise ValueError("Window Attention belongs only to original layers zero and one")
        self.layer = validate_layer(projections.layer, (0,), profile)
        self.arrays, self.projections, self.norm, self.attention = arrays, projections, normalization, attention
        self.kv = (NativeWindowKV(batch, projections.ordinal) if execution_target is None else
                   NativeWindowKV(batch, projections.ordinal, execution_target=execution_target))
        self.batch, self.position, self.pending = batch, 0, None
        self.opened, self.stack, self.lock = False, ExitStack(), threading.Lock()

    def __enter__(self):
        with self.lock:
            if self.opened:
                raise RuntimeError("Window Attention is already open")
            operators = (self.projections, self.norm, self.attention)
            if any(not op.opened or op.ordinal != self.projections.ordinal for op in operators):
                raise ValueError("Layer operators must be open on the same qualified device")
            self.stack.enter_context(self.kv)
            self.position, self.pending = 0, None
            self.opened = True
            return self

    def _ready(self, recovering=False):
        if not self.opened:
            raise RuntimeError("Window Attention is not open")
        if self.pending is not None and not recovering:
            raise RuntimeError("Recover the pending KV snapshot before another layer call")

    def _recover(self):
        restored = self.kv.rollback(self.pending)
        if restored["position"] != self.position:
            raise RuntimeError("Restored layer position differs from the last committed call")
        self.pending = None
        return restored

    def recover(self):
        with self.lock:
            self._ready(recovering=True)
            if self.pending is None:
                raise RuntimeError("No layer recovery is pending")
            return self._recover()

    def reset(self):
        with self.lock:
            self._ready(recovering=True)
            self.kv.reset()
            self.pending, self.position = None, 0

    def snapshot(self):
        with self.lock:
            self._ready()
            if self.kv.position != self.position:
                raise RuntimeError("Window layer and KV positions differ")
            return self.kv.checkpoint()

    def rollback(self, handle):
        with self.lock, self.kv.budget.restoration(), self.kv.budget.publication():
            self._ready(recovering=True)
            restored = self.kv.restore(handle)
            self.position, self.pending = restored["position"], None
            return restored

    def discard(self, handle):
        with self.lock:
            self._ready()
            self.kv.release_checkpoint(handle)

    @runtime_operation
    def step(self, values, *, start_pos=None, finalize=None):
        with self.lock:
            self._ready()
            if finalize is not None and not callable(finalize):
                raise ValueError("Attention finalizer must be callable")
            position = self.position if start_pos is None else start_pos
            if type(position) is not int or position != self.position:
                raise ValueError("Window Attention requires the next exact committed position")
            validate_layer_input(values, self.batch, position)
            sequence = values.shape[1]
            stages = {}
            def add(prefix, result):
                stages.update((prefix + name, a) for name, a in result.items())
                return result["output_bf16"]
            q = add("wq_a_", self.projections.fp8("wq_a", values))
            q = add("q_norm_", self.norm.norm(q, self.arrays[f"layers.{self.layer}.attn.q_norm.weight"]))
            q = add("wq_b_", self.projections.fp8("wq_b", q)).reshape(self.batch, sequence, 64, 512)
            q = add("query_rope_", self.norm.rope(q, position))
            kv = add("wkv_", self.projections.fp8("wkv", values))
            kv = add("kv_norm_", self.norm.norm(kv, self.arrays[f"layers.{self.layer}.attn.kv_norm.weight"]))
            kv = add("kv_rope_", self.norm.rope(kv, position))
            self.pending = self.kv.snapshot()
            try:
                cached = self.kv.append(kv, start_pos=position)
                stages.update(("kv_" + name, a) for name, a in cached.items())
                o = add("attention_", self.attention(q, cached["window_bf16"], self.arrays[f"layers.{self.layer}.attn.attn_sink"], cached["indices_i32"], diagnostics=True))
                o = add("inverse_rope_", self.norm.rope(o, position, inverse=True))
                o = add("wo_a_", self.projections.grouped(o.reshape(self.batch, sequence, 8, 4096)))
                add("wo_b_", self.projections.fp8("wo_b", o.reshape(self.batch, sequence, 8192)))
                result = stages if finalize is None else finalize(stages)
                with self.kv.budget.publication():
                    self.kv.discard(self.pending)
                    self.pending, self.position = None, self.kv.position
                return result
            except BaseException:
                try:
                    self._recover()
                except BaseException as recovery_error:
                    raise RuntimeError("Layer failed; KV recovery remains pending and further calls are blocked") from recovery_error
                raise

    def __exit__(self, *_):
        with self.lock:
            self.opened, self.pending = False, None
            self.stack.close()
