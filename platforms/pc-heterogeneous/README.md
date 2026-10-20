# Consumer NVIDIA PC entry

Target: **pc-heterogeneous**, RTX 4090 + RTX 3080 Ti. Current qualified model
arithmetic is on the 4090; the helper's model-compute path remains unqualified.

Use [workspace rules](../README.md) in a separate PC checkout. A legacy active
checkout is migrated only by its own task; Spark work never changes its binding,
branch, artifacts or local settings. The registered PC branch is `main`; add an
explicit reviewed branch rule before using a different PC development branch.

[PC issues](https://github.com/shell-term/vesper/issues?q=is%3Aissue+label%3Aplatform%3Apc-heterogeneous)
and M0..M3 (milestones 1..4) own progress. Issues 1..12 keep their original bodies,
states, acceptance and milestones; only the PC label was added during setup.
Deferred issue 12 explicitly has no milestone. Read allowed IDs through the
helper; Spark S0..S3 are separate. No shared issue reads are currently allowlisted.

[Architecture](../../docs/architecture.md), [validation](../../docs/validation.md)
and the [historical PC roadmap](../../docs/roadmap.md) remain reference entries.
Preserve historical result paths; new bound runs use `results/pc-heterogeneous/`.

For a bounded working set that outgrows the system SSD, use
[explicit expert-bank SSD roots](../../docs/external-expert-banks.md). Logical
index names and original payload bytes remain intact; private physical paths
and PC qualification stay within this track.
