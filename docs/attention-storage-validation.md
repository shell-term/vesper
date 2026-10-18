# Attention Query storage experiment

This initial result is retained as historical evidence. The subsequent
[PTX memory adapter](attention-memory-validation.md) qualifies an explicit
official-arithmetic reference and supplies [real MoE inputs](real-moe-validation.md).
The untouched full-kernel launch limit below still applies.

The full **64-head, 512-dimensional** sparse Attention candidate now executes on
the RTX 4090 with **75,776 bytes (74 KiB)** of dynamic shared memory. The original
specialization requests **141,312 bytes (138 KiB)**, above this device's
**101,376-byte (99 KiB)** opt-in limit. Candidate launch feasibility passes;
the original full-profile reference gate remains **blocked**.

[Issue #7](https://github.com/shell-term/vesper/issues/7) remains open. No candidate
was substituted into prefix capture or the native runtime by this experiment.
At this point real MoE inputs had not been captured and M1 remained open.
This is an optional TileLang experiment, not native Attention or
whole-model qualification.

## Storage change

The harness verifies the pinned `kernel.py` hash before generating executable
code. The uninstrumented candidate changes exactly one AST attribute:

```python
q_shared = T.alloc_shared((h, d), BF16)
# Explicit candidate:
q_shared = T.alloc_fragment((h, d), BF16)
```

It checks that reversing that change restores the complete original function
AST. Head count, dimension, block size 64, 256 threads, two pipeline stages,
GEMM policy, softmax expressions, sink, accumulation and BF16 boundaries remain
in the AST. A fragment distributes Query values across threads; compiler spills
remain possible. The name `register-query` does not guarantee exclusive use of
physical registers or imply a speedup.

Unchanged arithmetic AST is useful evidence, not proof of bit-identical
generated full-profile arithmetic. Generated Python/CUDA hashes are recorded;
downloaded/generated source stays in ignored artifacts. Per-specialization
NVRTC handles are isolated to avoid the TileLang 0.1.12 adapter collision.

## Comparisons

Each profile uses 12 generated conditions: batch 1/2, sequence 1/2/3, top-k sizes
1/32/64/65/127/128/129/193, partial and empty masks, duplicate indices, ties,
signed zeros, small/large inputs and positive/negative sinks. These are bounded
attention inputs, not recorded model activations or a prefill/decode KV test.

| Heads × dimension | Original shared bytes | Candidate shared bytes | Evidence |
| --- | ---: | ---: | --- |
| 64 × 256 | 75,776 | 43,008 | Original and candidate execute; all checked bits match |
| 16 × 512 | 86,016 | 69,632 | Original and candidate execute; all checked bits match |
| 64 × 512 | 141,312; cannot launch | 75,776 | Candidate executes; full original comparison unavailable |

The first two profiles provide **24 cases, 192 comparisons, 3,712,192 checked
elements and zero mismatches**. Diagnostic stores expose QK FP32 values,
probabilities before BF16 conversion, five softmax state vectors, value
accumulators after every block and FP32 outputs before final rounding. Their
BF16 outputs also match the uninstrumented kernels. FP32 comparisons preserve
signed zeros, infinities and NaN payload bits; no tolerance is used.

The full candidate provides **12 cases and 96 checks over 4,159,232 elements**
with zero mismatches. These check instrumentation, deterministic repeats and QK
components against four unchanged 16-head original kernels. Matching QK does
**not** qualify the subsequent 64-head softmax/value computation against the
unavailable full original. All full-candidate BF16 outputs are finite.

The combined **288 comparisons / 7,871,424 elements** therefore contain different
kinds of evidence. They are not 288 full-original reference passes. The report
remains `partial`, with `candidate_scoped_checks: pass` and
`full_profile_original_reference_gate: blocked`.

## Rejected head-partition control

Four 16-head launches preserve mathematical head independence but change the
floating-point reduction layout. Against the unchanged **64 × 256** original,
four conditions show **39,775 FP32 output differences**, although their BF16
outputs match. Against the **64 × 512 register-query candidate**, the control
has **79,632 FP32 differences and one BF16 difference**. That second comparison
is between candidate profiles, not against the full original. All QK component
comparisons match. Head partitioning is not adopted.

## Resources and reproduction

The final run used PyTorch 2.14.1+cu130, TileLang 0.1.12 and NumPy 2.5.2. It fetched
**zero model-weight bytes**. The host reservation is 32 MiB, with a 32 GiB
available-memory floor. The minimum sampled host availability is in the raw
report. The reservation is not a process-RSS or OS memory limit. Torch peak
allocations are **10,027,008 bytes**, with **10,485,760 bytes** reserved; these
figures exclude driver/compiler memory and other applications.

With the optional reference dependencies and their CUDA/NVRTC environment:

```sh
vesper compare-attention-storage --device 0 \
  --reference-directory artifacts/reference \
  --output artifacts/attention-storage.json
```

The command returns exit code 2 for a partial/blocked gate. Missing dependencies
produce a blocked result. Source/cache tampering and excessive diagnostic shapes
are rejected. The CPU suite passes **95 tests**.

Complete results: [raw report](../results/2026-10-18-attention-storage/attention-storage.json)
and [summary with raw-file SHA-256](../results/2026-10-18-attention-storage/summary.json).

The follow-up [PTX memory adapter](attention-memory-validation.md) now provides
an explicitly qualified official-arithmetic profile and real prefix activations.
Untouched full-kernel launch remains blocked. Native Attention, RoPE/indexing,
Hyper-Connections, KV quantization/state and full-model quality remain separate
work under Issue #7 and subsequent milestones.
