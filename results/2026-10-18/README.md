# Initial evidence — 2026-10-18

These are complete JSON probe outputs selected for publication, not model
weights or compiler logs. Private paths and machine identifiers are excluded.
Read the [measurement report](../../docs/first-probe.md) for scope and limitations.

| Record | Result |
| --- | --- |
| [summary.json](summary.json) | Eight-input totals and SHA-256 hashes of the records below |
| [official-matrix.json](official-matrix.json) | Original TileLang oracle: Ada 0/1,024 output mismatches; experimental Ampere 14/1,024 |
| [scalar-reference-negative.json](scalar-reference-negative.json) | Earlier scalar candidate: 2/128 official-output mismatches |
| [components.json](components.json) | Independent specification: six passing checks per GPU |
| [fixture-provenance.json](fixture-provenance.json) | Source revision, tensor ranges and hashes; 348,160 model bytes fetched |
| [transfers.json](transfers.json) | Verified host/GPU and explicit host-staged inter-GPU transfers |
| [storage-model-volume.json](storage-model-volume.json) | Direct reads of an existing model file opened read-only |
| [storage-workspace.json](storage-workspace.json) | Direct reads of a unique temporary scratch file |

The official matrix was saved by the experiment harness after each case. It
contains the input seed and complete per-check counts; it has no per-case wall
clock timestamps. The date comes from the experiment session. Other CLI records
include their original UTC timestamps.

Passing the scalar specification and failing the official comparison are
compatible results: the two oracles check different arithmetic. The failed
Ampere candidate is retained as evidence and an explicit experiment, not as an
approved inference path. Full expert/model execution and generation throughput
are not implemented.
