# PC policy alignment, 2026-10-21

This records the initial policy cleanup. The later [milestone-order revision](#milestone-order-revision)
below supersedes its M2/M3 assignment; use the [roadmap](roadmap.md) for current
order and acceptance. Earlier planning rows are retained as the decision history.

The operator now permits quality variation for useful speed/memory gains while
preserving as much output quality as practical. Earlier work prioritized exact
original-model arithmetic. That work remains the released-precision comparator;
it is no longer a universal requirement for every PC optimization.

## Planning changes

| Surface | Alignment |
| --- | --- |
| M0/M1 and closed component issues | Retain original completed acceptance/evidence; historical next-step wording is not the active queue |
| M2 / #10 | Bounded generation, recovery and representative actual task quality; reuse valid comparator outputs, trace in detail only when needed |
| M3 / #11 | Measured quality/speed/memory tradeoffs; choose mechanisms from bottlenecks, with repeated controlled measurements before general speed claims |
| Deferred #12 | May investigate exact or explicitly approximate 3080 Ti compute; old failed candidate remains unqualified, current storage helper unchanged |
| Validation and issue template | Match acceptance to documentation, runtime, exact-comparator, approximate-quality or timing claims; optional mechanisms are conditional |
| Entry documents | README and roadmap describe current policy; evidence/reproduction have separate indexes and historical scope notices |

The former combined open #10 item mixes deterministic recovery, an already
scored coding task and uncompleted reasoning/tool tasks. The revised acceptance
separates those facts; it does not mark the unfinished tasks passed. Likewise,
conditional speculation or an unselected lossy design is not mandatory work
merely because it appeared as a future idea.

## Evidence preserved

Original model assets, raw result JSON, source contexts, exact comparisons,
failed Ampere candidate, interrupted reasoning attempts and scores are unchanged.
M0/M1's completed scopes remain historical. M2/M3 are not completed by this
administrative review. No model run, numerical optimization or new quality
qualification is performed to edit policy documents.

The previous entry/roadmap can be inspected at
[source 4ec25bf](https://github.com/shell-term/vesper/tree/4ec25bf9269f6c60aedbcdf466061d389bade465).
The [evidence index](reference-evidence.md) and each report retain the recorded
scopes. Numerical profiles do not rebind a platform workspace or transfer
qualification to another platform. Spark-specific files/gates are outside this
PC policy revision.

## Document responsibilities

- [Quality/speed](quality-speed.md): current PC objective and numerical selection.
- [Roadmap](roadmap.md): milestone/issue responsibilities, dependencies and closure.
- [Validation](validation.md): smallest relevant check and reporting requirements.
- [Architecture](architecture.md): current mechanisms and unproven hypotheses.
- [Evidence index](reference-evidence.md): recorded comparator experiments.
- [Probe procedures](probes.md): explicit opt-in component reproduction.
- GitHub PC Issues/Milestones: live task state and acceptance.

Documentation checks validate local links, platform/issue scope and privacy.
The bodies of 36 recorded PC reports are preserved beneath their scope notices;
all nine closed issues retain their original acceptance/evidence. This revision
changes Markdown only, with no runtime, raw-result or model-execution change.

## Milestone-order revision

The operator subsequently asked to reduce elapsed development time and reconsider
the milestone order. The initial cleanup allowed M3 optimization alongside M2,
but still numbered long task-quality evaluation before speed work. That ordering
did not reflect the recommended execution sequence. The measured retained path
still generates only 0.0762 decode tokens/s on the short input, with substantial
integrity/staging and helper-transfer costs; running long quality cases first
would make the feedback loop expensive.

- Milestone 3 becomes **M2 - Bounded quality/speed iteration**. Completed #7..#9
  stay there, and active #11 moves in from milestone 4 at priority P0.
- Milestone 4 becomes **M3 - Representative task quality**. #10 moves there at
  priority P1 with its six checked items and two pending reasoning/tool cases.
- #11 now bounds the next work to one candidate, two short baseline/candidate
  pairs with 300-second cooperative budgets per timed process, and a documented
  adoption/rejection decision. It is not an indefinite search for ideal speed.
  The planned 20-minute model-run budget excludes implementation/preparation/CPU
  checks and is not a hard deadline; setup/cleanup can overrun a checkpoint.
- A promoted approximate profile still needs actual named task scores against a
  valid comparator, with that evaluation budget declared separately. Short smoke
  screening does not qualify broad quality. Timeouts do not pass required checks.
- Original task prompts/budgets, prior raw evidence, stopped captures, completed
  M0/M1, closed issue acceptance and deferred #12 stay intact. M2/M3 remain open.
  No runtime change, new model execution or new numerical qualification is made
  by this ordering revision.
