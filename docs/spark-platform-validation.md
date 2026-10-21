# DGX Spark S0: isolated management and a bounded platform probe

Observed 2026-10-20. This report covers management isolation and a tiny CUDA
compatibility check. It does **not** qualify DeepSeek model arithmetic, model
quality, full-checkpoint loading, long contexts, or generation speed.
The [Spark roadmap](spark-roadmap.md) defines subsequent work. This is the
historical S0 report; later S1 execution evidence is recorded separately in
the [native integration report](spark-native-integration.md).

## Management before hardware execution

Management code was recorded first in `2582243bde0887197d34a386febed638206fb5f5`,
followed by the probe in `0e329db4e398f16e634f521017ab56d46cc99f80`.
At that S0 setup, the [registry](../platforms/registry.json) separated PC issues
1..12 / milestones 1..4 from Spark issues 13..14 / milestones 5..8. Existing PC
issue bodies, states and milestones were compared before/after setup and did
not change; only a platform label was appended. The active PC checkout was not
modified.

The local binding checks real checkout/Git directory, origin, branch, platform,
profile and registry identity. Guarded CLI operations also verify the imported
code's location, rejecting a different clone or installation before progress
reads or GPU work. Issue reads/updates reject overlapping ownership, mixed
platform labels and mismatched milestones. API host is fixed to github.com.
Result paths reject traversal/reparse escapes, and source context is checked at
the beginning and end. Live checks accepted Spark issue 13 and refused PC issue
10, a foreign checkout, PC result roots and path traversal.

An initial complete CPU regression run passed **399 tests, one skipped**. After
independent review, final focused checks passed **34 workspace cases** (32 pass,
two local Windows link-creation skips) and **25 Spark probe cases**. The reparse
attribute boundary has a separate permission-independent test. Ordinary CI runs
the complete current suite on Windows/Linux and Python 3.11/3.13; those are CPU
checks, not additional Spark hardware runs.

Checks apply to helper-mediated operations, not arbitrary tools or OS access.
The current result checker proves compatibility with the current binding/source;
historical archives retain their original remote contexts and are not restamped
as results from the publishing checkout. See the
[management evidence](../results/dgx-spark/2026-10-20-platform/management-summary.json).

## Actual Spark observations

The bounded probe ran on Linux AArch64, NVIDIA GB10 (compute capability 12.1),
driver API 13.0 and NVRTC 13.0. The compiler library was found in the ARM SBSA
toolkit layout. The source and compiler options are recorded with the result.
No reference framework installation or model checkpoint was needed.

| Run | Outcome | Scope |
| --- | --- | --- |
| [r01](../results/dgx-spark/2026-10-20-platform/gb10-probe-r01.json) | Safely refused before explicit device/pinned allocation | Physical reserve was incorrectly demanded a second time from CUDA's free-memory view |
| [Diagnostic](../results/dgx-spark/2026-10-20-platform/gb10-memory-diagnostic-r01.json) | Physical available about 46.05 GB, CUDA free about 24.01 GB | Observation only; no explicit buffers |
| [r02](../results/dgx-spark/2026-10-20-platform/gb10-probe-r02.json) | Bounded integer round trip passed | 262,144 uint32 elements, zero mismatches |

Correction `521a841439f3915cdfe016e932ccd3c1aa85d91d` retains the **32 GiB
physical available-memory floor**, pending-buffer checks, 256 MiB admission
headroom and default/max buffer caps. CUDA free memory is a separate allocator
view: it now must accommodate the owned device buffer, without demanding the
physical floor a second time. Query failures, invalid figures and insufficient
CUDA buffer capacity still refuse allocation and retain diagnostic values.

R02 used one **1 MiB device buffer and one 1 MiB pinned buffer**. All observed
physical available-memory samples remained above 46.04 GB. CUDA context/compiler/
module memory is outside those explicit buffer figures. The owned cleanup calls
succeeded and the previous context was restored. No GPU reset, existing service
stop/load/unload, swap change, model inference, or large disk benchmark occurred.
The tiny integer-kernel duration is not a model-performance estimate.

## Subsequent work

After this S0 probe, Spark S1 still required original FP4 expert execution,
bounded expert/Engram stores, one physical-memory ledger for CPU/GPU state and
complete-model integration. S1's later ordinary response and cleanup are
recorded in its separate report; they are not inferred from this tiny probe or
from PC numerical evidence.

The adopted next order is S2 native-baseline setup/I/O improvement and matched
latency measurements, followed by S3 benchmark connection and practical-quality
evaluation. Future expansion or compression candidates need their own scope
and evaluation; this historical report adopts none. The existing 32 GiB reserve
and service-preservation rules continue to apply.
