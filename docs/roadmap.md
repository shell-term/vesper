# PC roadmap and progress tracking

This M0..M3 roadmap belongs to the consumer-PC track. Select the target through
[workspace rules](../platforms/README.md). Spark has its own entry and gates;
shared code does not transfer qualification.

[PC Issues](https://github.com/shell-term/vesper/issues?q=is%3Aissue+label%3Aplatform%3Apc-heterogeneous)
are the source of task state. This page defines responsibilities and acceptance,
not a second mutable status log or a completion estimate from issue counts.
The [quality/speed policy](quality-speed.md) is authoritative for current PC
numerical experiments. The [2026-10-21 policy review](policy-review-2026-10-21.md)
records why this plan changed.

| Milestone | Outcome and scope | Issues |
| --- | --- | --- |
| [M0: Verified component baseline](https://github.com/shell-term/vesper/milestone/1) | Historical released-precision expert, bounded byte cache and integrity baseline. Original exact gates remain scoped evidence. | [#1](https://github.com/shell-term/vesper/issues/1), [#2](https://github.com/shell-term/vesper/issues/2) |
| [M1: Exact routed MoE sublayer](https://github.com/shell-term/vesper/milestone/2) | Historical original gate/routing/accumulation/shared-expert baseline. A MoE sublayer is not a whole Transformer model. | [#3](https://github.com/shell-term/vesper/issues/3)..[#6](https://github.com/shell-term/vesper/issues/6) |
| [M2: Bounded generation and task quality](https://github.com/shell-term/vesper/milestone/3) | Integrated model/assets/memory, state recovery and actual representative task outputs/scores. Reuse valid comparator evidence; full intermediate tracing is optional diagnosis. | [#7](https://github.com/shell-term/vesper/issues/7)..[#10](https://github.com/shell-term/vesper/issues/10) |
| [M3: Measured quality/speed optimization](https://github.com/shell-term/vesper/milestone/4) | Compare selected candidates on actual quality, startup/prefill/TTFT/decode, memory and I/O. Quality changes are permitted when declared and measured. | [#11](https://github.com/shell-term/vesper/issues/11) |

M0/M1's completed exact-comparison scopes remain unchanged. They do not require
every new candidate to preserve original precision or routing. M2 and M3 can
progress in parallel: a qualified short input can support bounded timing while
broader task quality is still open. No performance claim is inferred from an
unrelated component or synthetic cache result.

[#10](https://github.com/shell-term/vesper/issues/10) separates already measured
generation/recovery and the coding case from unfinished reasoning/tool scoring.
Complete each named task with its declared prompt/budget; a shortened smoke case
cannot close it. Interrupted captures remain unscored. Do not restart the long
104-call reference capture automatically, or require its exhaustive per-boundary
replay before every optimization. Reuse recorded comparator outputs only when
their input/settings, provenance and scope support the intended comparison.

[#11](https://github.com/shell-term/vesper/issues/11) owns the next measured
optimization. Select mechanisms from actual bottlenecks rather than making
prefetch, overlap, speculation or extra compression mandatory deliverables.
Repeated controlled trials are needed before a general speed conclusion.
The [first session measurement](session-lifetime.md) is a short exploratory
baseline with different admission ceilings, not completion of M3.

[#12](https://github.com/shell-term/vesper/issues/12) remains a deferred optional
3080 Ti compute investigation, without a milestone dependency. A reference-exact
compute path must pass its numerical gate; an approximate path needs a separately
declared quality/speed evaluation. Its historical failed candidate remains
unqualified, and the current helper continues to store/audit bytes.

## Updating work

1. State the outcome, platform, numerical profile, acceptance and resource limits.
2. Distinguish required work from optional mechanisms and historical evidence.
3. Use relevant local checks; choose exact short checks or task-quality measures
   according to the candidate. Use detailed tracing for a concrete uncertainty.
4. Publish sanitized source/hardware/settings, outputs, scores, timing, memory,
   failures and comparison limits. Preserve original raw receipts and assets.
5. Update through the guarded workspace workflow. Mark implemented work with an
   open required check `status:needs-validation`; use `status:in-progress` for
   selected active work and `status:deferred` for optional parked investigations.
6. Close only the explicitly stated acceptance scope. Deferring or removing an
   unnecessary old procedure is not passing its former test. A numerical mismatch
   cannot become an exact pass by renaming it or relaxing a tolerance.

`priority:P0` covers runtime/resource correctness and required quality evidence;
exact fidelity is mandatory when that profile claims it. `priority:P1` covers
measured quality/speed optimization and need not wait for every broader M2 task.
`priority:P2` covers optional investigation. Do not invent deadlines, universal
quality tolerances or whole-engine completion percentages.

Use the [current validation workflow](validation.md) and [historical evidence
index](reference-evidence.md), rather than interpreting an old report's next
steps as today's queue.
