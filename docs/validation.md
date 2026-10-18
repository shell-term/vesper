# Validation gates

These gates describe evidence, not a list of interchangeable accuracy scores.

| Gate | Evidence required | Initial state |
| --- | --- | --- |
| Released artifacts | Pinned checkpoint, exact weight/scale bytes, matching tokenizer and Engram | Small W1 fixture only |
| Format specification | All FP4 codes, scale encodings, FP8/BF16 rounding boundaries, signed zero | Pass on Ada and Ampere |
| Projection specification | Native weight slice against an independent block arithmetic oracle | Pass on Ada and Ampere |
| Official kernels | Per-operation exact comparisons with the pinned TileLang code | W1 slice: 4090 passes 8 inputs; 3080 Ti candidate fails, 14/1,024 output mismatches |
| Full expert/layer | Activation quantization, clamps, weighted SwiGLU, routing and accumulation | Not implemented |
| Full model | Router sets/weights, attention/indexing/KV, logits and sampling | Not implemented |
| Workloads | Reproducible reasoning, coding and tool-use evaluations | Not run |
| Resource policy | All owned memory plus pressure behavior, concurrency with other work | Reservation checks only |
| End-to-end speed | Prefill, TTFT, decode, actual storage traffic and memory on named hardware | Not run |

## Reference

The initial source is `deepseek-ai/DeepSeek-V4.1-Flash-1017`, revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`. The official `inference/kernel.py`
quantizes BF16 activations to E4M3 with per-32 E8M0 scales, multiplies per-32
FP8/FP4 blocks, applies scales, accumulates in FP32 and writes BF16.

Vesper's scalar projection uses a deliberate, explicit FP32 accumulation order.
Its NumPy oracle computes each inner block dot in float64 before FP32 rounding.
Agreement is useful evidence, but the official Tensor Core reduction order may
differ. Do not loosen a tolerance or call a high correlation "exact" to clear a
failed gate. Record the discrepancy and its cause.

The initial scalar candidate passed its specification oracle but differed from
the official kernel. The Ada FP8 MMA candidate then matched all eight tested
inputs. The experimental Ampere BF16 MMA candidate still differed. This supports
retaining reference arithmetic as a gate; it does not establish universal Ada
parity or identify every cause of the Ampere differences. See the
[first measurements](first-probe.md) for scope, negative results and raw data.

Preserving bytes is necessary but insufficient. In particular, do not replace
the original FP8 activation behavior with a supposedly more precise path and
assume model behavior is unchanged. Near-tied router scores can select different
experts after a small numerical change.

## Measurement rules

- Report the source revision, hardware, compile options and raw mismatch count.
- Compare GPU paths with the same inputs; keep NaN classification separate from finite bit identity.
- Report cold/first-content and resident/cache-reuse performance separately.
- Mark derived bandwidth ceilings as assumptions, never as measured token rates.
- Measure actual cache misses and physical I/O before building a predictor.
- Validate speculative sampling against the unchanged target. Batch shape and rollback are part of the check.
- Keep weights, local paths and identifiers out of committed results.

## Initial resource scope

Unit tests cover memory reservation refusal, checkpoint range refusal and fixture
tamper detection. These are boundary checks, not proof of the future full-engine
memory governor. All probes must retain their explicit, small allocation limits.
