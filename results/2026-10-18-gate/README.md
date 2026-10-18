# Gate projection and connected routing evidence — 2026-10-18

[Report](../../docs/gate-validation.md) · [Issue #4](https://github.com/shell-term/vesper/issues/4)

307 generated inputs across both text/image masks give 614 exact-bit passes
against unchanged original `linear` and `Gate.forward` using the real original
BF16 matrix and trained biases. A separate environment with no PyTorch installed
reproduces all 614 published logit/ID/weight goldens. Both commands return `pass`.

| Record | Contents |
| --- | --- |
| [summary.json](summary.json) | Scope, counts, resource exclusions and record hashes |
| [gate-projection.json](gate-projection.json) | All 614 full-gate comparison conditions, stages, exact mismatches, ordered IDs/weight bits, input/logit hashes, and source/library profiles |
| [native-only-recheck.json](native-only-recheck.json) | All 614 native-only rechecks, input metadata, logits/IDs/weight bits and cleanup; no PyTorch installed/imported |
| [gate-fixture-provenance.json](gate-fixture-provenance.json) | Original BF16 matrix identity, range and hash; no matrix values |
| [router-fixture-provenance.json](router-fixture-provenance.json) | Original text/image biases and pinned configuration; no bias values |
| [pytorch-gate-provenance.json](pytorch-gate-provenance.json) | Three ATen source hashes/URLs; top-k provenance is also embedded in the full report |

Published record hashes refer to LF bytes. The native recheck names the exact
reference-report hash. Original model arrays, downloaded reference sources,
NVIDIA libraries, private paths and machine identifiers are excluded.

This is a single-row gate/projection check on the named builds/hardware with
generated inputs, not a running MoE layer/model or a generation-speed result.
Expert/shared-expert accumulation, actual model activations, full-model quality
and other shapes/backends remain separate gates.
