# Routing evidence — 2026-10-18

The [report](../../docs/router-validation.md) explains the isolated projection,
tie refusal, ordered tuples versus expert-ID/weight maps and remaining gates.
[Issue #3](https://github.com/shell-term/vesper/issues/3) remains open. These
records contain no model bytes or private paths.

| Record | Contents |
| --- | --- |
| [summary.json](summary.json) | Counts, open qualification state, allocation scope and record hashes |
| [router-postprocessing.json](router-postprocessing.json) | All 116 conditions, exact comparisons, complete mismatch details, ordered diagnostic IDs/weights, semantic map checks and admission outcomes |
| [router-fixture-provenance.json](router-fixture-provenance.json) | Original text/image bias identities, ranges/hashes and pinned routing configuration; no bias values |

All tested expert sets and ID-associated FP32 weight maps match. All 87 non-tied
conditions pass the ordered-output checks. The candidate refuses 29 tied
conditions and remains unqualified, even though their diagnostic ID-weight maps
match in this test set. Ordered tuple permutations are not, alone, model-quality
failures. Gate projection, full MoE/block execution and full-model quality remain
untested. The comparison deliberately returns `partial` and exit code 2.
