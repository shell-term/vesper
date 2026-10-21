# PC improvement worklist

This worklist belongs to the RTX 4090 + RTX 3080 Ti PC and supports the
[repeated quality/speed loop](quality-speed.md). It does not create a new
pass-once milestone sequence. [Issue #10](https://github.com/shell-term/vesper/issues/10)
owns current execution/evaluation work; the closed M2 experiment remains intact.

The objective is **20 accepted decode tokens/s for one conversation**, with
measured quality differences, a 96 GiB engine host cap and at least 32 GiB
system-available memory. Startup, prefill, request TTFT and decode are separate
measurements. No quality degradation threshold or deadline is invented here.

## Evidence and candidate mechanisms

| Problem | Evidence and limits | Improvement hypothesis | Check before adoption |
| --- | --- | --- | --- |
| The selected benchmark cannot execute through the current native path | [Fixed GPQA 3-question inputs](gpqa-subset.md) and scoring exist; actual answers do not. The serial request admits 512 calls, attention state 1,152 positions, and tokenizer decoding 4,096 IDs. Dynamic benchmark assets/backend are missing. | Connect bounded execution/persistence to a native backend that resolves actual input assets; implement explicit capacity/resource handling. Raising request constants alone is insufficient. | Preserve full questions, sampling and generation allowance. Save real outputs, genuine EOS or interruption, quality, timing and resources. Never qualify synthetic backend tests as model results. |
| CPU integrity work is expensive | [M2](m2-placement.md) records 24.5..24.8 seconds of hashing/chunk preparation across five calls, with existing GPU upload audits. This is scoped host time, not a complete profile. | Reuse immutable, source-bound digest metadata and reduce repeated preparation/copies, retaining payload/corruption checks. Inspect `storage.NativeStore.read_into` and its indexed-store use before selecting a concrete change. | Short released-precision output/state and corruption checks, actual stage timings, bounded metadata, then return to the selected benchmark. No unverified cache entry becomes trusted. |
| Too many expert bytes cross the storage/host/device tiers | M2 still reads 843 selected payloads / 14.7605 GiB across five calls. Host-retained avoids helper transfers but keeps only a small expert cache. Actual RSS and conservative reservations differ. | Share staging ownership and use available cache capacity more effectively; measure locality and eviction before changing placement. | Actual bytes/read counts, transfer time, setup, TTFT, accepted decode rate, RSS/private commit/pinned/device peaks and cleanup under the same reserve policy. Cache-hit rate alone is insufficient. |
| Diagnostic execution is used for timing | Current operators return diagnostic intermediate arrays and use synchronous calls. The measured 0.10972 tps is a short prototype result. | Separate an inference return path from detailed diagnostics; retain detailed tracing for mismatches and selected qualification. Consider fused/resident GPU work where measured copies/launches dominate. | Declare changed math/rounding when applicable. Compare actual outputs and state/quality under the appropriate numerical profile, including wall time between accepted tokens. |
| Prompt processing is serial | The selected prompts contain 286/123/256 tokens. Existing short timings cannot establish their actual prefill cost. | Batch or chunk causal prefill rather than repeat full serial setup for each prompt token. | Full prompt preserved, causal/state checks, explicit chunk-prefill equivalence or declared approximate behavior, actual prefill/TTFT and memory. Shortening a question is not an optimization. |
| Original expert bytes remain large | Experts are already FP4. A conditional all-selected-experts transfer is about 4.51 GB/token; at 20 tps it would require about 90.2 GB/s, versus the [historical measured host-to-GPU rate](first-probe.md#transfers-and-storage) of about 12.35 GB/s. Cache reuse, compute placement, quantization and speculation change this arithmetic. | Consider separately named mixed-precision/quantized candidates and existing fast-engine approaches when transfer reduction alone is insufficient. | Same fixed task outputs and protocol, numerical profile identity, measured quality differences and total accepted-token performance. Do not treat compressed size as a speed or quality result. |

The transfer arithmetic is a conditional traffic estimate, **not a universal
hardware speed ceiling**. Adding SSD capacity does not by itself fix bandwidth.
The helper GPU is not an automatically faster shared 36 GB GPU. Its compute
path and speculative decoding remain optional hypotheses, not prerequisites.

[Xolotl's 1017 model card](https://huggingface.co/IndrasMirror/DeepSeek-V4.1-Flash-1017-XQ-Uncensored)
reports roughly 22..26 tps on a 4090/90 GB RAM Linux system using an additionally
quantized, edited derivative. That motivates a bounded reproduction/comparison;
it does not establish original-model quality, Windows performance or Vesper's
rate. No derivative or engine is adopted by this worklist, and no large weight
download is authorized as a small probe.

## Current bounded implementation slice

Keep the [same fixed GPQA subset](gpqa-subset.md) as the selected loop input.
The attempt controller now saves separate immutable attempts, journals successful
model calls, skips completed questions including wrong answers, and refuses
resumption with changed inputs/candidate/settings/resource policy. Interrupted
questions restart from their original prompt; KV/RNG state restoration is not
implemented. Candidate execution context and actual resources must be supplied
and enforced by the backend; this controller is not an OS memory governor or a
model-execution attestation.

The CPU-only readiness recheck preserves the original plan and computes exact
prompt embedding/Engram requirements without running model layers or fetching
weights. The full native benchmark path remains **not ready**. It still needs
dynamic source-asset resolution, sufficient attention/decode capacity and a
backend connected to the controller. A large declared generation allowance is
a ceiling, not a required response length; silently reducing it is not a fix.

Complete that executable path within bounded work sessions, then select the
next performance experiment from the measured costs above. Backend integration
must not turn into another exhaustive reference-capture prerequisite. Existing
raw-text/component tests are supplementary checks. If native capacity or asset
preparation would make the next session impractical, record the cost and compare
existing backend integration options before committing to a long implementation
or capture. External reports alone cannot qualify a candidate.

Each iteration records its candidate, current coverage, expected benefit,
maximum elapsed work/model budget and stop condition. The current working limit
is 15..20 minutes including implementation, preparation, checks and publication;
save incomplete work rather than automatically retry or extend. Report scoped
failures and unresolved checks. Expand coverage only through the accepted loop.
Full benchmark execution still requires the user's decision on a concrete plan.
