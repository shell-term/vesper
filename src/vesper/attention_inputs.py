"""Deterministic bounded Attention inputs, without a tensor framework."""

import hashlib

import numpy as np

from .attention_storage import trace_shapes
from .formats import bf16_bits

CASES = ((1, 1, 1, "normal", .125), (1, 3, 32, "normal", 1.), (1, 2, 64, "masked", 8.),
         (2, 2, 65, "masked", .5), (1, 2, 128, "empty", 1.), (1, 2, 193, "duplicates", 1.),
         (1, 1, 128, "zeros", 1.), (1, 2, 65, "signed_zeros", 1.), (1, 1, 127, "normal", 1e-6),
         (1, 2, 128, "ties", 1.), (1, 1, 129, "sink_positive", .125), (1, 1, 129, "sink_negative", .125))
PROFILES = ((64, 256), (16, 512), (64, 512))


def attention_inputs(heads, dimension, case):
    batch, sequence, topk, mode, scale = case
    trace_shapes(batch, sequence, heads, dimension, topk)
    if mode not in {c[3] for c in CASES} or scale not in {c[4] for c in CASES}:
        raise ValueError("Unknown bounded Attention input condition")
    rng = np.random.default_rng(20261018 + topk + batch * 1000 + sequence * 100)
    q = bf16_bits(rng.standard_normal((batch, sequence, heads, dimension)).astype(np.float32) * np.float32(scale))
    kv = bf16_bits(rng.standard_normal((batch, 257, dimension)).astype(np.float32) * np.float32(scale))
    sink = rng.standard_normal(heads).astype(np.float32)
    idx = rng.integers(0, 257, (batch, sequence, topk), dtype=np.int32)
    if mode == "masked":
        idx[:, :, ::3] = -1
    elif mode == "empty":
        idx.fill(-1)
    elif mode == "duplicates":
        idx[:, :, 1::2] = idx[:, :, ::2][:, :, :topk // 2]
    elif mode == "zeros":
        q.fill(0)
        kv.fill(0)
    elif mode == "signed_zeros":
        q.fill(0x8000)
        kv[..., ::2] = 0
        kv[..., 1::2] = 0x8000
    elif mode == "ties":
        q.fill(0x3F00)
        kv.fill(0x3F00)
        sink.fill(0)
    elif mode == "sink_positive":
        sink.fill(300)
    elif mode == "sink_negative":
        sink.fill(-300)
    arrays = q, kv, sink, idx
    provenance = {name: hashlib.sha256(a.tobytes()).hexdigest()
                  for name, a in zip(("q", "kv", "sink", "indices"), arrays)}
    return arrays, provenance
