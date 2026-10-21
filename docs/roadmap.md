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
| [M3: Representative task quality](https://github.com/shell-term/vesper/milestone/4) | Track pending task-quality coverage, fixed development subsets and final full evaluations of selected benchmarks within the quality/speed loop. Preserve existing evidence; changed candidates return to earlier coverage. | [#10](https://github.com/shell-term/vesper/issues/10) |

M0/M1's completed exact-comparison scopes remain unchanged. They do not require
every new candidate to preserve original precision or routing. The 2026-10-21
order revision exchanged the open issue assignments: #10 moved from milestone
3 to 4, #11 from 4 to 3, and completed #7..#9 stayed in 3. IDs/URLs remain stable.
The later loop clarification supersedes the one-way execution order, without
reopening a completed experiment or passing a pending check. Earlier reports
mentioning #10/M2 or #11/M3 describe the former assignment.

Milestones organize recorded outcomes and pending coverage. They do not impose
a pass-once sequence of speed work followed by quality work. Quality and speed
remain joint criteria throughout development; each changed candidate returns
to the existing evaluation coverage. No performance claim is inferred from an
unrelated component or synthetic cache result.

## Improvement loop and time limits

<a id="next-execution-order-and-time-limits"></a>

The operator's stretch target is **20 accepted decode tokens/s for one
conversation (batch 1)**, with output-quality differences reported alongside
speed. Startup and input prefill are measured separately. This target is not
yet demonstrated. It remains an objective across iterations, not a claim that
one local experiment or milestone closure achieves it. The [completed placement
cycle](m2-placement.md) is historical evidence for a small improvement.

The control loop is **measure quality and speed -> identify unmet criteria ->
improve -> return to existing coverage -> measure again**. When the current
scope meets its criteria, extend coverage deliberately. If the extension exposes
a problem, the improved candidate returns through earlier coverage as well.
Earlier passes remain historical evidence; they are not permanent passes for
changed code, profiles or settings.

Declare each iteration's candidate/profile, active evaluation coverage, quality
criteria, timing conditions and budget before execution. Choose actual cases and
their order separately; no fixed ladder of input types or quality thresholds is
adopted from the operator's illustrative example. The existing 20 tps objective
does not define new per-case latency or prefill requirements.

Start with short checks that can reject an unsuitable candidate promptly. Reuse
valid comparator outputs and then complete the required checks for the declared
coverage. An unmet criterion sends work back into the loop. Do not automatically
launch a long task merely because a finite optimization experiment closed.
Required quality evidence is never waived for a promoted numerical profile.

Agree the elapsed-work budget separately from any model-process budget and
save incomplete work at that boundary. Setup, preparation, evaluation and
publication count toward elapsed work. Cooperative process checks can overrun
a checkpoint; report overruns and
avoid starting work known not to fit. At a stop, preserve results, candidate
identity, outstanding coverage and the next action. Do not automatically add
retries, candidates, sweeps or larger reference captures.

The completed M2 experiment used four alternating short trials with 300-second
cooperative model budgets. That was its measurement design, not a mandatory
template for every new iteration or a 20-minute end-to-end guarantee. New timing
claims choose controls/repetition appropriate to their scope.

For an unchanged-math candidate, the short output/state check supplies the
appropriate comparator gate. An approximate candidate additionally needs actual
named, scored task outputs against a valid comparator before adoption; declare
that evaluation's separate time scope before running it. A raw-text smoke result
can screen a candidate but cannot qualify its task quality. Broader #10 tasks
stay open. Missing required evidence leaves the relevant issue/milestone open;
a budget stop is not an automatic pass. Closing a scoped issue records its
declared outcome; later changes still return to the improvement/validation loop.

Use fixed subsets of selected public benchmarks during development. After
optimization, prepare the complete declared evaluation split of each selected
benchmark for the user's execution decision, with matched released-precision reference
outputs and per-benchmark degradation metrics. Select the suite and quality
tolerances separately and estimate its cost before running it; it does not run
automatically. Present scope, subset results, time estimates, reference-cache
coverage, resource use and resume options, then wait for explicit authorization
before any full comparator/candidate run. A final failure returns work through
earlier coverage; additional full revalidation outside the approved plan also
needs the user's decision. Deferral keeps full-suite qualification pending.
See [benchmark validation](validation.md#public-benchmarks-and-final-degradation-measurement).

The selected loop entry is [GPQA Diamond's fixed three-question development
subset](gpqa-subset.md). Complete its native execution readiness and collect
provenance-bound outputs before claiming benchmark quality. Existing short
independent inputs remain supplementary checks; their timings do not replace
this subset. The [improvement worklist](pc-improvement-plan.md) separates measured
bottlenecks from unproven mechanisms. Input readiness exposes request, attention,
decode and asset-coverage limits. The native backend now connects question
persistence/restart to actual model execution. Its first attempt completes four
input calls before a budget interruption; no native subset answer/score exists.
The selected next improvement is context/operator reuse in that same path.

## Remaining quality and optional mechanisms

[#10](https://github.com/shell-term/vesper/issues/10) separates already measured
generation/recovery and the coding case from unfinished reasoning/tool scoring.
Complete each named task with its declared prompt/budget; a shortened smoke case
cannot close it. Interrupted captures remain unscored. Do not restart the long
104-call reference capture automatically, or require its exhaustive per-boundary
replay before every optimization. Reuse recorded comparator outputs only when
their input/settings, provenance and scope support the intended comparison.

[#11](https://github.com/shell-term/vesper/issues/11) records the completed finite
optimization experiment. Select each new change from measured problems rather than making
prefetch, overlap, speculation or extra compression mandatory deliverables.
Repeated controlled trials are needed before a general speed conclusion.
The [first session measurement](session-lifetime.md) is a short exploratory
baseline with different admission ceilings; the later M2 result has its own scope.

[#12](https://github.com/shell-term/vesper/issues/12) remains a deferred optional
3080 Ti compute investigation, without a milestone dependency. A reference-exact
compute path must pass its numerical gate; an approximate path needs a separately
declared quality/speed evaluation. Its historical failed candidate remains
unqualified, and the current helper continues to store/audit bytes.

Image input is an accepted product requirement tracked separately in
[#26](https://github.com/shell-term/vesper/issues/26), currently planned/deferred
without a milestone assignment. It needs actual preprocessing, vision-encoder
execution and vision-to-text integration, with named visual task outputs and
separate image-prefill/TTFT/decode measurements. The current message adapter
rejects images; text-format or text-only speed results do not qualify vision.
This future feature does not add image runs to the four-process M2 cycle.

The [completed M2 placement measurement](m2-placement.md) adopts an opt-in host
storage profile for its short input: 0.07331 to 0.10972 pooled decode tokens/s
with matching outputs/state. It does not achieve the 20 tps target or complete
M3/image coverage. GitHub Issues/Milestones remain the source of closure state.

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

`priority:P0` covers immediate runtime/resource blockers and a selected bounded
improvement. `priority:P1` covers pending evaluation coverage; priority alone
does not prescribe a case order or waive a profile's quality gate.
`priority:P2` covers optional investigation. Exact fidelity remains mandatory
for a profile claiming it. Do not invent deadlines, universal quality tolerances
or whole-engine completion percentages.

Use the [current validation workflow](validation.md) and [historical evidence
index](reference-evidence.md), rather than interpreting an old report's next
steps as today's queue.
