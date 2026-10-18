# Routing tie-resolution evidence — 2026-10-18

[Report](../../docs/router-validation.md) · [Issue #3](https://github.com/shell-term/vesper/issues/3)

All 244 supplied-logit gate conditions and 232 synthetic selection conditions
pass exact-bit comparisons with the pinned reference. Ordered expert IDs,
normalization and ID-associated weights match, including the previously
unsupported 29 gate tie conditions. The command returns `pass` and exit code 0.

| Record | Contents |
| --- | --- |
| [summary.json](summary.json) | Counts, qualification scope, resource exclusions and record hashes |
| [router-postprocessing.json](router-postprocessing.json) | All 244 gate and 232 selection conditions, comparisons, ordered IDs/weight bits, admissions and source profile |
| [router-fixture-provenance.json](router-fixture-provenance.json) | Original text/image bias identities, ranges/hashes; no bias values |
| [pytorch-topk-provenance.json](pytorch-topk-provenance.json) | Exact PyTorch source revision, URLs, sizes and seven source/license hashes |

The [historical partial results](../2026-10-18-routing/README.md) are retained
unchanged. This qualification covers one row of finite FP32 logits, 384 experts,
top-six and the named CUDA PyTorch path. Gate projection, actual expert/shared-
expert aggregation, connected MoE/model execution, other shapes/builds and
whole-model quality remain untested. No original model arrays, downloaded
reference source code, private paths or machine identifiers are included.
