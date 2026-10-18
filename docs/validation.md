# Validation gates

These gates describe evidence, not a list of interchangeable accuracy scores.

| Gate | Evidence required | Current evidence |
| --- | --- | --- |
| Released artifacts | Pinned checkpoint, exact weight/scale bytes, matching tokenizer and Engram | W1 slice and three complete experts; tokenizer/Engram pending |
| Format specification | All FP4 codes, scale encodings, FP8/BF16 rounding boundaries, signed zero | Pass on Ada and Ampere |
| Projection specification | Native weight slice against an independent block arithmetic oracle | Pass on Ada and Ampere |
| Official kernels | Per-operation exact comparisons with the pinned TileLang code | W1 slice: 4090 passes 8 inputs; 3080 Ti candidate fails, 14/1,024 output mismatches |
| Full expert | Activation quantization, clamps, weighted SwiGLU and three projections | One native expert on 4090 passes 128 single-token conditions, including intermediate bits |
| Tiered expert storage | Identity, original bytes, eviction/reload and unchanged expert computation | Three native experts pass 232 calls across SSD/RAM/two GPUs, including intermediate bits |
| Device integrity audits | Original-byte digest anchors, padding/chunk boundaries, corruption refusal and unchanged computation | Both GPUs pass 96 payload cases and 10 fault injections; optional path passes another 232 official comparisons |
| Routing postprocessing | Scores, trained bias, ordered expert IDs/weights and tie handling from supplied FP32 logits | 244 official gate conditions and 232 selection conditions pass all checked bits; pinned single-row CUDA top-k profile only |
| Original gate projection and routing | Original BF16 weights promoted to FP32, actual linear projection and connected expert selection | 307 generated inputs across both masks give 614 exact-bit passes against unchanged linear/Gate.forward; one row, named cuBLAS/PyTorch builds |
| Connected MoE sublayer | Gate projection, routing, accumulation order and shared expert | Not implemented |
| Full Transformer block | Connected attention, MoE, Engram, Hyper-Connections and residual boundaries | Not implemented |
| Full model | Router sets/weights, attention/indexing/KV, logits and sampling | Not implemented |
| Workloads | Reproducible reasoning, coding and tool-use evaluations | Not run |
| Resource policy | All owned memory plus pressure behavior, concurrency with other work | Bounded cache buffers and injected pressure/refusal/recovery pass; full-engine/OS policy pending |
| End-to-end speed | Prefill, TTFT, decode, actual storage traffic and memory on named hardware | Not run |
| Warm cache timing | Equal capacities/traces, alternating measurement order, complete audits | Six pairs: median call-time ratio 5.02; no model-token-rate claim |

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

The [complete-expert follow-up](expert-validation.md) passes the 4090 pipeline
within its stated scope. Its repeated-shape checks also exposed shared launch
handles in the optional TileLang 0.1.12 NVRTC adapter. The harness isolates those
handles while preserving the official model and CUDA kernel source/arithmetic.
The workaround is explicit in the result, not a tolerance or an oracle rewrite.

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

The [cache follow-up](cache-validation.md) adds exclusive-tier identity checks,
corrupted/failed transfer injection, cap refusal, RAM release and retry. Its
real-weight run compares all thirteen expert stages with the unchanged official
expert while switching source tiers. Available-memory pressure is injected into
the observation function; no real machine memory exhaustion is attempted. The
actual observed minimum available memory is reported separately. Buffer cleanup
counters reach zero; that is not proof that CUDA contexts or Python RSS vanish.

The [GPU-audit follow-up](cache-performance.md) measures CPU SHA-256 as the main
baseline cost and moves repeated verification to GPU admission. Its digests are
derived only after the original full-block and six tensor hashes pass. All
uploaded chunks must match before a cache slot becomes visible. Intermediate
RAM copies can be unverified until this point. Unit tests explicitly corrupt
RAM and transfers and require refusal before compute-slot publication. Device
tests compare standard SHA-256 digests and inject single-bit faults on both GPUs.
These tests are scoped engineering checks, not a cryptographic certification.

The [router probe](router-validation.md) isolates postprocessing by replacing
only the reference projection with supplied FP32 logits. It now reproduces the
pinned CUDA top-k gather order and 32-cell bitonic network, including boundary
ties. Ordered tuples and normalization match exactly, so no permutation-based
equivalence is assumed. The original partial report is retained as historical
evidence. Qualification covers one row, 384 finite FP32 logits and top-six
selection against the named PyTorch build; other shapes/builds and connected
MoE/model execution remain separate gates.

The [gate probe](gate-validation.md) now uses the original gate matrix and
unchanged `linear` plus `Gate.forward` bodies without an identity substitution.
Native cuBLAS with explicit FP32/default-math settings reproduces the observed
reference projection and routing bits. Inputs are generated boundary vectors,
not recorded full-model activations. This does not establish expert/shared-
expert accumulation, connected MoE execution, whole-model quality or speed.
