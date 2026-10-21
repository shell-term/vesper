# Consumer NVIDIA PC entry

Target: **pc-heterogeneous**, RTX 4090 + RTX 3080 Ti. Current qualified model
arithmetic is on the 4090; the helper's model-compute path remains unqualified.

Use [workspace rules](../README.md) in a separate PC checkout. A legacy active
checkout is migrated only by its own task; Spark work never changes its binding,
branch, artifacts or local settings. The registered PC branch is `main`; add an
explicit reviewed branch rule before using a different PC development branch.

[PC issues](https://github.com/shell-term/vesper/issues?q=is%3Aissue+label%3Aplatform%3Apc-heterogeneous)
and M0..M3 (milestones 1..4) own progress. Setup originally preserved Issues
1..12 and added the PC label. Current accepted policy revisions may update
planning/acceptance through the guarded workflow; preserve historical evidence
and do not mark deferred checks passed.
Deferred issue 12 explicitly has no milestone. Read allowed IDs through the
helper; Spark S0..S3 are separate. No shared issue reads are currently allowlisted.

The current order is M2's bounded optimization cycle (#11, alongside the completed
#7..#9 foundation), then M3's remaining task-quality evaluations (#10). Milestone
IDs 3/4 stay stable; their open issue assignments were exchanged on 2026-10-21.
Use the roadmap's finite cycle and time limits rather than running long quality
captures before every speed experiment.

The [quality/speed policy](../../docs/quality-speed.md),
[validation workflow](../../docs/validation.md) and [PC roadmap](../../docs/roadmap.md)
define current planning. [Architecture](../../docs/architecture.md) contains
mechanism hypotheses; the [evidence index](../../docs/reference-evidence.md)
records scoped historical results. Quality variation is permitted for useful
measured gains; exact comparator gates do not apply to every numerical candidate.
Preserve historical result paths; new bound runs use `results/pc-heterogeneous/`.

For a bounded working set that outgrows the system SSD, use
[explicit expert-bank SSD roots](../../docs/external-expert-banks.md). Logical
index names and original payload bytes remain intact; private physical paths
and PC qualification stay within this track.
