# Complete-expert evidence — 2026-10-18

Read the [measurement report](../../docs/expert-validation.md) for scope,
reference-runtime correction, rejected hypotheses and reproduction commands.
These are complete JSON records without model bytes, compiler logs or private
paths. The initial [transfer/storage measurements](../2026-10-18/README.md) are
preserved separately.

| Record | Result |
| --- | --- |
| [summary.json](summary.json) | Aggregate counts and SHA-256 hashes of the records below |
| [full-expert.json](full-expert.json) | One complete 4090 expert: 128/128 conditions pass, including intermediate bits; 655,360 final BF16 outputs match |
| [expert-fixture-provenance.json](expert-fixture-provenance.json) | Six original tensors, exact ranges/hashes; 18,800,640 model bytes fetched |
| [reduction-diagnostics.json](reduction-diagnostics.json) | Eight inputs: native FP8 and BF16 block dots differ in 19,019/163,840 positions; trace-integrity checks pass |
| [projection-regression.json](projection-regression.json) | Original W1 slice after runtime isolation: Ada 0/1,024 final differences; Ampere 14/1,024 |

The diagnostic native trace is separate from the original official oracle. The
float64 dot oracle also is not a substitute for official arithmetic. The Ampere
path remains unqualified even in a case whose final BF16 output matches.

Full expert records contain an experiment completion timestamp; projection
regression records contain seeds and complete check counts without per-case
timestamps. All measurements here were collected on the stated experiment date.
No layer/model fidelity or token-rate claim is made.
