# M2 bounded helper-placement cycle

The PC operator targets 20 accepted decode tokens/s for one conversation,
while limiting development/measurement time and measuring output quality.
That remains a stretch target, not a measured result or this cycle's completion
condition. [M2 / #11](https://github.com/shell-term/vesper/issues/11) ends with
one controlled comparison and an adoption/rejection decision. M3's complete
reasoning/tool cases remain pending in #10.

## Measured result and decision

On 2026-10-21 all four planned processes completed at clean execution source
[`78a9cc0`](https://github.com/shell-term/vesper/commit/78a9cc00348ddde16afe9998c8dde2a490e85770).
Their total timed engine wall was **504.10 seconds (8.40 minutes)**, within the
planned model-run budget, with no additional model attempts. This excludes
implementation, preparation, Python imports/reference validation and CPU checks.
The [raw receipts and summary](../results/pc-heterogeneous/2026-10-21-m2-placement/summary.json)
retain the original execution binding/registry and source identities.

| Trial, in actual order | Decode tokens/s | Request TTFT, s | Engine TTFT, s | Resident peak, GiB |
| --- | ---: | ---: | ---: | ---: |
| 1: retained helper comparator | 0.07302 | 47.38 | 109.28 | 15.81 |
| 2: host-retained candidate | 0.10933 | 33.10 | 92.40 | 17.11 |
| 3: host-retained candidate | 0.11011 | 33.05 | 92.47 | 17.12 |
| 4: retained helper comparator | 0.07359 | 47.06 | 109.20 | 15.81 |

Each variant produces four post-prefill decode tokens across its two trials.
Pooled rate, computed from those tokens divided by their total decode time,
is **0.07331 versus 0.10972 tokens/s (1.4967x)**. Paired improvements are 1.4972x
and 1.4962x. Request TTFT falls from about 47.1..47.4 to 33.1 seconds.
Engine TTFT includes setup; request TTFT starts after setup.

All four runs match the existing reference's 646,400 logit elements, 200 ordered
top-six routes/weights, selected-ID feedback, RNG and final complete state per
trial. Across four runs that is 2,585,600 logit elements and 800 ordered route
sets. There is no new independent original execution and no broad task score.
The comparator's fourteen images and candidate's thirteen primary-only images
match their expected qualified sets. Original weights/arithmetic are unchanged.

Selected-payload direct reads remain **843 / 14.7605 GiB per trial** in both
profiles. The candidate replaces 17 helper hits with RAM hits (357 versus 340),
and helper H2D/D2H sections fall from about 18.0..18.1 seconds to zero. These
are scoped host wall-time sections, not a complete GPU profile or physical SSD
traffic measurement. CPU hashing/chunk preparation still takes about 24.5..24.8
seconds across the five calls. Its cost remains a next design question.

Actual candidate residency peaks at 17.12 GiB, private commit at 32.68 GiB,
with at least 86.62 GiB system-available RAM sampled. The common 4090 explicit
peak is 12.99 GiB. The comparator owns 1.406 GiB on the helper; the candidate
allocates no helper device. Host admission peaks are 69.16 / 70.56 GiB, distinct
from process residency. All resource owners close with zero live leases,
reclaimers, in-flight operations and owned host/pinned/device charges; HTTP and
weight fetch counts are zero.

**Decision:** adopt `host-retained` as an opt-in storage profile for the measured
short PC path. Both ordered pairs improve decode and request TTFT with matching
outputs and preserved headroom. The historical default providers remain intact.
**20 tokens/s is not achieved**: the pooled candidate is still about 182x below
that target. This cycle establishes a useful storage decision, not a production
engine or a general throughput prediction. Longer tasks, image input and the
remaining #10 evaluations need their own evidence.

Development checks passed all 31 targeted benchmark/cache/session tests,
including corrupt RAM refusal/recovery without the helper. The publication
checkout also ran all 492 CPU tests: 489 passed and three Windows privilege
checks skipped. The published-corpus checks bind raw file hashes, execution
identities, exact outputs/images, matched capacity, memory cleanup and measured
rate/decision calculations. CPU checks are distinct from the actual GPU receipts.

## Hypothesis and selected candidate

The [earlier retained-session run](session-lifetime.md) records 17.72 seconds
of helper H2D/D2H sections across five whole-model calls, versus 1.87 seconds
in recorded expert-kernel sections. These are scoped host wall-time sections,
not a complete GPU profile. Its cache has only 17 helper hits among 1,200 expert
selections. Test whether storing those helper slots in RAM saves transfer/audit
time without increasing selected-payload store reads enough to offset the gain.

`host-retained` retains all forty MoE contexts like the `retained` comparator.
Per layer, both keep two primary-GPU slots and ten resident expert payloads
in total. The comparator has two helper-GPU slots plus six RAM slots; the
candidate has zero helper slots plus eight RAM slots. Original SSD bytes,
model arithmetic, trained routing, selected expert count, activation/KV
representation, accumulation and sampling are unchanged (`released-fp4`).
All original source checks and primary-GPU upload audits remain enabled.
The candidate compiles the same thirteen qualified Ada images; the comparator
also needs the qualified Ampere SHA-256 storage-audit image. No helper model
compute is introduced. This is an opt-in benchmark storage profile.

Both configurations keep the 96 GiB engine host admission cap, 32 GiB
available-memory floor, 2 GiB pinned cap and the same component reservation
ceilings. Actual residency/private commit and both GPUs are measured; moving
two slots per layer to RAM intentionally changes the allocation by tier.
Admission counters are not OS quotas.

## Fixed measurement plan

- Reuse the already qualified raw-text five-call request and cached original
  comparator; no new original execution or full intermediate trace.
- Four fresh native processes in order `retained`, `host-retained`,
  `host-retained`, `retained`, with two independently prepared index directories.
  Preparation copies/fetches zero expert payload bytes.
- Each timed process receives a 300-second cooperative budget. Stop at four
  attempts; no automatic extra retries, sweep or long quality capture. The
  planned 20-minute model-run budget excludes implementation/preparation/CPU
  checks, and setup/cleanup can overrun a checkpoint.
- Check all 646,400 logit elements, 200 ordered top-six routes/weight bits,
  selected-ID feedback/RNG, completion and final complete model state against
  the existing short reference. Verify the expected compiled-image set.
- Record setup, request/engine TTFT, prefill and subsequent decode separately,
  plus actual process/GPU memory, store reads, tier transfers and cleanup.
  OS/driver caches are not reset; alternating order reduces but does not remove
  their influence. Diagnostic tensor materialization remains timed.

The request has only two post-prefill decode calls per process and no task
score. Repeated timing applies to this input only. It cannot establish general
or production throughput, broad quality, 20 tps, or complete #10. Required
comparisons must complete; failures/timeouts remain incomplete. Adopt the
storage profile only within the measured scope if its benefit is consistent
and its exact-output/resource checks pass. A negative completed result can
reject it and still finish the finite investigation.
