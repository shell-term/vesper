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
| [M2: Bounded quality/speed iteration](https://github.com/shell-term/vesper/milestone/3) | Keep the integrated model/assets/memory foundation from #7..#9; finish one measured optimization cycle and an evidence-backed adoption/rejection decision. This is not a practical-throughput promise. | [#7](https://github.com/shell-term/vesper/issues/7)..[#9](https://github.com/shell-term/vesper/issues/9), [#11](https://github.com/shell-term/vesper/issues/11) |
| [M3: Representative task quality](https://github.com/shell-term/vesper/milestone/4) | Complete and score the declared reasoning/tool cases; preserve the already measured generation, recovery and coding evidence. Reuse valid comparator outputs. | [#10](https://github.com/shell-term/vesper/issues/10) |

M0/M1's completed exact-comparison scopes remain unchanged. They do not require
every new candidate to preserve original precision or routing. The 2026-10-21
order revision puts bounded optimization before the remaining long quality
evaluations. Milestone IDs/URLs stay stable: #10 moves from milestone 3 to 4,
#11 from 4 to 3, and completed #7..#9 stay in 3. No check is passed by moving it.
Earlier reports mentioning #10/M2 or #11/M3 describe the former assignment.

The first retained-session result still measures only 0.0762 decode tokens/s
on its short input. Running the long tasks first would make iteration expensive.
M2 therefore has a finite comparison/decision scope; it does not mean optimizing
until an unspecified ideal speed. M3 preparation and scoring code can proceed
alongside M2. Required quality evidence is never waived for a promoted numerical
profile. No performance claim is inferred from an unrelated component or
synthetic cache result.

## Next execution order and time limits

The operator's stretch target is **20 accepted decode tokens/s for one
conversation (batch 1)**, with output-quality differences reported alongside
speed. Startup and input prefill are measured separately. This target is not
yet demonstrated and does not turn M2 into an unlimited run-until-20 search.
The [selected first cycle](m2-placement.md) tests helper-versus-RAM placement.

1. **M2 / #11:** choose one candidate from a measured bottleneck, declare its
   numerical profile and limits, then compare with a valid short reference.
   Staging/integrity work and helper transfers are measured candidates for
   investigation; prefetch, speculation and extra compression are not mandatory.
2. **Finish that cycle:** the default timing plan is two baseline/candidate pairs
   on the qualified five-call input, alternating order and matching resource
   policies where possible. At most four timed processes each get a 300-second
   cooperative budget: 20 minutes of planned model-run budget. This excludes
   development, preparation and CPU checks; setup/cleanup can overrun a checkpoint.
   It is not a wall-clock completion guarantee. Record cache state and uncertainty;
   four short runs support only their recorded input, not general throughput.
3. **Make a decision:** publish quality, timing, memory, failures and adoption or
   rejection. Required comparisons must actually complete; timeouts stay incomplete.
   Stop at the declared budget rather than automatically adding retries, candidates
   or a larger suite. A negative completed comparison is useful evidence and does
   not require finding a faster replacement to finish this finite cycle.
4. **M3 / #10:** schedule each remaining complete task on the selected, qualified
   execution path with its own estimated cost and explicit budget. Reuse valid
   comparator outputs and omit routine exhaustive intermediate tracing. Do not
   automatically restart the old long reference capture.

For an unchanged-math candidate, the short output/state check supplies the
appropriate comparator gate. An approximate candidate additionally needs actual
named, scored task outputs against a valid comparator before adoption; declare
that evaluation's separate time scope before running it. A raw-text smoke result
can screen a candidate but cannot qualify its task quality. Broader #10 tasks
stay open. Missing required evidence leaves the relevant issue/milestone open;
a budget stop is not an automatic pass. Further optimization cycles are chosen
from the results, not accumulated as mandatory work in this one.

## Remaining quality and optional mechanisms

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
baseline with different admission ceilings, not completion of the new M2 cycle.

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

`priority:P0` covers immediate runtime/resource blockers and the next bounded
M2 optimization cycle. `priority:P1` schedules the remaining M3 task-quality
evaluation; this priority does not waive a profile's quality gate.
`priority:P2` covers optional investigation. Exact fidelity remains mandatory
for a profile claiming it. Do not invent deadlines, universal quality tolerances
or whole-engine completion percentages.

Use the [current validation workflow](validation.md) and [historical evidence
index](reference-evidence.md), rather than interpreting an old report's next
steps as today's queue.
