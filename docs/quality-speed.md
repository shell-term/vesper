# PC quality and speed workflow

The PC objective is useful generation speed with as much output quality as
possible, within the 96 GiB engine host cap and 32 GiB available-memory floor.
Quality variation is permitted when the measured speed/memory tradeoff is
worthwhile. Exact intermediate equality is a reference tool rather than the
acceptance rule for every candidate. This direction does not change Spark's
qualification gates or retrospectively broaden existing results.

The current PC stretch target is 20 accepted decode tokens/s for one conversation
(batch 1), with measured quality differences and startup/prefill reported
separately. It is an unachieved target. Each experiment has a finite budget
and an evidence-backed adoption/rejection decision. The [completed placement
cycle](m2-placement.md) is one recorded experiment, not completion of this target.

This is the canonical PC numerical policy. [Roadmap](roadmap.md) defines the
milestones, [validation](validation.md) chooses checks, and GitHub PC Issues own
live acceptance/state. Historical exact reports are comparator evidence, not
universal restrictions on new candidates. See the [policy alignment record](policy-review-2026-10-21.md).

## Repeated improvement and validation

Quality and speed are evaluated together in a repeating loop: measure the
current candidate, identify unmet criteria, improve it, and return to the existing
evaluation coverage. A pass is evidence for that candidate and scope. It does
not permanently exempt an earlier check after the implementation changes.

Expand coverage only after the current scope meets its declared criteria. If
new coverage exposes a problem, improve the candidate and return to earlier
coverage as well; passing only the newly failing case is insufficient. Choose
the concrete cases, their order, quality criteria and measurement conditions
explicitly. This policy does not prescribe a fixed sequence of input categories
or derive acceptance thresholds from an illustrative example.

Bound each iteration's elapsed work and model execution before starting. Use
short checks first and reuse valid comparator evidence; expand checks within the
declared coverage as the candidate warrants. A budget stop saves the candidate,
results, outstanding checks and next action. It neither passes those checks nor
automatically starts a longer run. A completed local experiment is progress in
the loop, not proof that the product objective is satisfied.

## Benchmark coverage during development and final testing

Use selected public benchmarks with their supported scoring protocols. Routine
iterations use a small, fixed subset to detect quality changes while keeping
the loop short. Fix the question identities and comparison settings before
measuring a candidate; preserve each question's prompt and generation budget.
Subset scores describe only that subset, not the benchmark's full score.

Once optimization produces a final candidate, prepare the full evaluation for
the user's execution decision. An authorized final evaluation runs **all items
in each selected benchmark's declared evaluation split**. Compare
with the same model release/revision under released precision and matched
evaluation conditions. Reuse valid reference outputs when their provenance and
coverage support the entire comparison. Record score degradation together with
speed and memory, following the [benchmark reporting rules](validation.md#public-benchmarks-and-final-degradation-measurement).

Select the benchmarks, versions/splits and acceptable quality/speed criteria
before final testing. Earlier benchmark suggestions remain candidates, not an
adopted suite or invented tolerance. Estimate the complete workload and give
final testing its own budget; interrupted work retains pending item identities
and cannot be reported as a completed full evaluation. A failed final criterion
returns the candidate to the improvement loop, including earlier checks and a
new complete final evaluation after changes. Full question coverage does not
require routine per-layer/per-token intermediate tracing.

Before running either path on a full benchmark, present the selected scope, recent
subset results, expected time range and its assumptions, reference-cache
coverage, resource impact and pause/resume capability. The user can choose to
run the plan, narrow its selected scope, or defer it. Start only after explicit
authorization for that plan; reaching a milestone or completing preparation
does not authorize execution. Routine preparation and previously authorized
bounded subset work can continue. Reuse an existing authorization for its
unchanged scope/budget, including supported resumptions; bring material plan
changes or additional full reruns back to the user. Deferral leaves full-suite
qualification pending and does not manufacture a pass.

The benchmark runner must save completed question/trial results and resume the
remaining items under the same candidate and evaluation settings. An interrupted
question may need to restart; make that limitation clear in the proposed plan.

## Comparator and candidate identity

Keep original tensor bytes immutable and retain a released-precision comparator.
Existing exact expert, block, short-generation and coding results qualify only
their recorded implementations and inputs.

Each experiment records a numerical profile independently of the workspace
execution profile `native`. Declare changed weight/scale precision, activation
or KV representation, expert count, routing and sampling. An approximate
candidate cannot reuse a released-precision qualification merely because its
decoded answer happens to match. No lossy candidate is implemented or promoted
by this workflow document.

The expert tensors are already packed **FP4 with E8M0 scales**. The current FP8
compute path unpacks those original tensors for arithmetic. An additional
compression experiment therefore starts from an already compressed source.
Measure decompression, staging and device execution as well as stored bytes.
Lossless compression may cost more CPU time than it saves in I/O; lossy
compression may affect outputs. Both require measurements.

## Development checks

| Situation | Check |
| --- | --- |
| Local edit | Relevant CPU/resource/lifecycle tests |
| Documentation/progress edit | Links, platform scope and preservation of recorded facts; no model replay solely for prose |
| Exact execution optimization | A short qualified reference with final logits, selected IDs, routing/RNG and final state; investigate mismatches |
| Approximate numerical candidate | Named task outputs/scores against the comparator, numerical differences and failures; exact equality is not required |
| Timing run | Bounded real generation, explicit startup/prefill/TTFT/decode timing, memory and actual I/O; omit routine full intermediate fingerprinting |
| Unexplained mismatch or major release gate | Opt-in detailed tracing of the affected operators/positions; broader checks when the result justifies them |

Preserve the full prompt and generation budget of a named quality task. A
shortened smoke test is a different scope and cannot complete an earlier
reasoning/coding/tool-use evaluation. Do not rescore interrupted runs as passes.
The earlier full coding comparison remains a comparator. The interrupted
67-call reasoning reference is partial, unscored and has no completed reference
execution receipt; its preparation and private records are retained.

Reuse recorded comparator outputs when their provenance, input/settings and
coverage support the intended comparison. A new full original capture is not
required by default. Keep task completion/scoring distinct from independent
bit-fidelity qualification. Optional speculation, extra quantization and helper
compute become deliverables only if selected for a concrete experiment.

## Measurement and selection

First measure the execution path rather than infer token speed from cache-hit
ratios or synthetic transfers. The historical default provider recreates each
layer's MoE execution/cache context every call. The opt-in retained provider now
has a [first bounded measurement](session-lifetime.md); its single comparison
does not establish a general speedup. Choose the next mechanism from the measured
bottleneck. Weight integrity, cleanup and host/device reserve checks remain active.

Report setup separately from request TTFT and decode. State whether Python
startup, reference checks, diagnostic tensor materialization and output
fingerprinting are included. Control requests/settings, budgets and hardware
where possible. Declare intentional numerical/sampling/resource-policy changes
and compare their quality/speed/memory tradeoff. Report cache state and order;
one short pair or a comparison with different admission ceilings is exploratory,
rather than a repeated controlled performance conclusion.

Compare output quality, TTFT, decode throughput, memory and storage/transfer
traffic together. Retain the useful alternatives rather than selecting solely
on byte reduction. Introduce lossy transformations only as named experiments
with their actual changes and measured effects. Small task scores do not
establish broad reasoning/coding/tool quality.

Track generation/recovery and open task evaluations in [#10](https://github.com/shell-term/vesper/issues/10).
[#11](https://github.com/shell-term/vesper/issues/11) records the completed M2
experiment. Its closure does not automatically schedule long M3 captures.
Follow the [roadmap's loop and time limits](roadmap.md#next-execution-order-and-time-limits):
select a bounded change from measured problems, return to existing coverage,
and assess quality/speed/memory together. Declare any additional approximate-quality
evaluation's time scope before adoption. Qualified short inputs support timing
while broader #10 evaluations remain open; they do not establish broad task quality.

## Initial session-lifetime experiment

`python -m vesper.session_benchmark prepare` creates only source indexes for
the already qualified five-call raw-text input. It reuses the legacy input
explicitly and copies/fetches zero expert payload bytes. New output is confined
to the bound PC workspace's `artifacts/pc-heterogeneous/performance/` tree.

Run `python -m vesper.session_benchmark baseline` and then
`python -m vesper.session_benchmark retained` in separate native processes.
Both use the same request, six-expert RAM allowance per layer, two primary and
two helper slots, GPU upload audits and the 96 GiB engine admission cap / 32 GiB floor.
The benchmark explicitly admits up to 2 GiB of pinned staging memory: forty
retained layer owners currently have two buffers each (about 1.40 GiB total).
These buffers count toward the host cap. Retaining all forty owners also keeps
their conservative legacy component allowances live; actual RSS and private
commit are measured separately. A future serial staging pool could
share them; the default runtime's 1 GiB pinned ceiling remains unchanged.
Only context lifetime changes. The experimental retained path holds those
bounded contexts across tokens; it is not the default trajectory provider.

Each process has a default **300-second cooperative budget**, checked before
each MoE layer and model call. Setup/cleanup operations can overrun a checkpoint;
this is not an OS execution deadline. An exhausted budget is reported as such,
with no complete-reference match or completed quality claim.
Resource admission refusals are likewise incomplete results with cleanup and
memory observations, rather than completed output comparisons.

The quick released-precision check compares every full-vocabulary logit hash,
ordered routes/weights, selected-ID feedback and RNG offset, plus final complete
model state. It omits full intermediate fingerprinting. Diagnostic operators
still materialize intermediate tensors, and that cost is included; these
numbers are measurements of this prototype, not a finished production engine.
The raw continuation has no reasoning/coding/tool task score. No new lossy
numerical profile is implemented by this context-lifetime experiment.

The first [actual measurement and resource tradeoff](session-lifetime.md) is
published separately, including setup refusals and differing admission ceilings.
