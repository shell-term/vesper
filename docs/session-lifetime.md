# PC session-lifetime experiment

On 2026-10-21, retaining bounded MoE contexts across calls completed the same
five-call raw-text continuation as the ephemeral comparator. Observed decode
rate rose from **0.0472 to 0.0762 tokens/s (1.615x)**. This is one exploratory
comparison of a diagnostic prototype, not practical production throughput or
a repeated performance conclusion. Resource admission ceilings differ.

[Complete raw reports and publication manifest](../results/pc-heterogeneous/2026-10-21-session-lifetime/summary.json)
retain actual execution identities, timing, routes, memory, cache activity and
cleanup. This is PC evidence only.

## Input and numerical check

Both native processes reuse the qualified [short-generation input](text-generation-validation.md):
`Once upon a`, three prompt tokens, temperature 0.7, seed 2026 and three
generated IDs `[1014, 260, 223]`. Each makes five whole-model calls through all
forty layers. The continuation is `" time a "`; it has no task-quality score.
No new independent original execution is claimed.

Both match all 646,400 FP32 logits, 200 ordered top-six routes and weight bits,
selected-ID feedback and RNG progression against the existing original
reference. Final complete model state and the qualified fourteen compiled
images also match. Routine full intermediate fingerprinting is omitted;
diagnostic operator tensor materialization remains and is timed.

Released FP4 expert bytes, scales, activations, routing, sampling and model
arithmetic are unchanged. No additional quantization, pruning or retraining is
introduced. This does not qualify an approximate profile or broad task quality.

## Observations

| Measurement | Ephemeral | Retained |
| --- | ---: | ---: |
| Engine setup, seconds | 57.48 | 59.46 |
| Three prompt forwards, seconds | 74.49 | 44.73 |
| Engine start to first token, seconds | 131.97 | 104.19 |
| Request start to first token, seconds | 74.49 | 44.73 |
| Two subsequent decode forwards, seconds | 42.41 | 26.26 |
| Decode tokens/s | 0.0472 | 0.0762 |
| All five forwards, seconds | 116.90 | 70.99 |
| Engine wall including checks/cleanup, seconds | 176.57 | 132.82 |
| MoE context opens / closes | 200 / 200 | 40 / 40 |
| Direct expert reads | 1,200 | 843 |
| Direct expert payload bytes, GiB | 21.01 | 14.76 |
| Sampled process resident peak, GiB | 7.95 | 15.80 |
| Sampled private commit peak, GiB | 19.91 | 32.99 |
| Minimum sampled system-available RAM, GiB | 125.18 | 117.74 |
| Explicit GPU allocation peak, 4090 / 3080 Ti, GiB | 9.73 / 0.04 | 12.99 / 1.41 |

Decode rate uses two post-prefill calls, not three divided by two call times.
Engine TTFT includes setup; request TTFT starts after setup and includes the
small between-call output checks. Python imports and reference validation are
excluded. Forward times exclude those output checks. Total wall time includes
them and cleanup. Each process had a 300-second cooperative budget and completed.

The 3080 Ti stores bytes only; model arithmetic remains on the 4090. Cache
capacities are identical: six RAM experts, two primary slots and two helper
slots per layer. All upload audits remain enabled. Retention yields 340 RAM
hits and 17 helper hits; 843 accesses read the original store. Direct-read
counters cover selected expert payloads, not all process I/O or measured
physical SSD traffic.

The actual trace has 1,200 expert selections and 786 distinct layer/expert
pairs, with 13 to 26 distinct experts per layer over this request. Retention
therefore still rereads 57 previously seen payloads after eviction. These
working sets describe this input rather than an assumed general locality rate.

## Resource policy and failed attempts

Ephemeral execution at clean `a920fa3` used 64 GiB host / 1 GiB pinned admission
caps. Retained setup first hit the pinned ceiling: forty owners need about
1.40 GiB of staging. A retry at `c048996` with 2 GiB pinned then hit the 64 GiB
host admission ceiling while opening the output head. Neither attempt completed
a model call or produced a completed execution receipt. The two published
refusal records are operator observations, not stamped model execution or
quality passes.

Successful retention at clean `68bbde5` uses the provisional engine's 96 GiB
host cap and benchmark-only 2 GiB pinned cap. Peak host charges are 69.16 GiB
versus 46.64 GiB for ephemeral. Conservative legacy component charges overlap;
they are not RSS or OS quotas. Actual process memory is reported above. Pinned
peaks are 1.41 GiB versus 0.035 GiB. Both keep the 32 GiB available-memory floor
and close with zero owned host, pinned/device leases, reclaimers and in-flight
operations. There are no monitor errors, HTTP calls or weight fetches. The normal
runtime pinned default remains 1 GiB. No other workload, page-file setting or
GPU state was changed.

Input indexes were created at `a920fa3` with zero expert payload copies/fetches.
Raw receipts keep their own source contexts; publication does not restamp their
executions. This is not a fully matched resource-policy experiment. Ephemeral
caps were not reached.

## Limits and next decisions

Order is ephemeral then retained, one run each in separate processes. OS and
driver caches were not reset. Retention is opt-in; historical validation
providers remain unchanged. Longer requests, arbitrary expert working sets,
reasoning/tool scores and production throughput remain open.

Recorded MoE kernel sections total about 1.87 seconds in the retained run. CPU
payload hashing/chunk preparation total about 24.24 seconds, and helper H2D/D2H
sections about 17.72 seconds. These scoped host wall-time sections are not a
complete model GPU profile. They motivate testing staging/integrity costs and
helper placement before assuming extra weight compression is the main speed
opportunity. A shared serial staging pool could reduce pinned memory. No
unmeasured optimization is adopted by this report.

Continue the [quality/speed workflow](quality-speed.md) under
[#11](https://github.com/shell-term/vesper/issues/11). Repeated matched trials
and approximate quality candidates remain open. Broader comparator evaluations
in #10 and M2 remain open; the interrupted reasoning capture is unscored.

Local regression ran all 485 CPU tests: 482 passed and three Windows privilege
checks skipped. The wheel contains both new modules and all five required
license/notice files, with license bytes matching their sources.
