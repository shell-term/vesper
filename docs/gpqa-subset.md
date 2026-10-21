# GPQA Diamond development subset

The first selected public-benchmark input for the PC improvement loop is a
fixed three-question GPQA Diamond subset. GPQA Diamond appears in DeepSeek's
[V4.1 evaluation](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash#evaluation-results).
The 1017 model is evaluated locally under the protocol below; the parent
release's published score is not a comparator score for this subset.

The [upstream GPQA dataset](https://github.com/idavidrein/gpqa) is by Irving David
Rein and collaborators, licensed CC BY 4.0. Pin upstream revision
`56686c06f5e19865c153de0fdb11be3890014df7`, archive SHA-256
`461ae7329f15a3e35f8184d2dac24b990f34fdf12f366ca4062d8e6638cd08dc`,
and the 198-row Diamond CSV SHA-256
`41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305`.
Questions, choices, answers and the archive stay in ignored PC artifacts.

## Fixed input and local protocol

Selection seed `20261021` ranks record IDs by SHA-256 and selects three before
any output is observed. Choice order is independently fixed by the same seed
and item identity. Do not select shorter/easier questions, rewrite the question,
include its explanation in the prompt, or lower its declared generation budget
to manufacture a completed run. Source anomalies are retained and flagged.

The local protocol is `vesper-gpqa-zero-shot-thinking-v1`: full question and
four choices, zero-shot thinking effort 75, temperature/top-p 1.0, and an explicit
262,144-token maximum. This is a ceiling, not a required output length. It
retains the [1017 card's recommended large generation allowance](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017#minimal-inference),
but the local prompt, selection and strict final-letter grader are not an
official score reproduction. Record genuine EOS and grade the final channel;
incomplete execution has no correctness pass. Unit-test answers are synthetic
grader checks, never model-quality evidence.

Prepare inputs from the pinned local archive without starting inference:

```sh
python -m vesper.gpqa_subset --archive artifacts/pc-heterogeneous/benchmarks/gpqa-source/dataset.zip --run gpqa-diamond-dev-v1 --count 3 --seed 20261021 --max-new-tokens 262144
```

The command requires the bound PC workspace, refuses an existing result and
rejects a full-population request. It does not call a model API or fetch weights.
It creates a private, provenance-bound plan and reports readiness explicitly.

## Current executable scope

Input preparation and the local scorer are implemented. The three actual
encoded prompts contain 286, 123 and 256 tokens in the fixed selection order.
All are within the text adapter's character bound. None has a model answer or
quality score from this work.

The [CPU preflight record](../results/pc-heterogeneous/2026-10-21-gpqa-subset/preflight.json)
retains the source identities and explicit not-ready reasons, without
redistributing question text or claiming model execution.

The current native serial request is capped at 512 model calls, including both
input and generation. It cannot admit the declared benchmark budget. A native
benchmark inference adapter is also still required. Preflight reports both
limits without truncating the prompt/budget or pretending preparation was
inference. No new GPU execution, speedup or full-suite qualification is claimed.

The next selected implementation work is an executable, bounded native path
for these same inputs and time accounting, including honest interruption and
question/trial persistence. Compare released precision and each candidate on
the fixed benchmark subset. Existing independent raw-text/component checks are
supplementary implementation checks, not substitutes for benchmark quality.
Full evaluation remains subject to the [user's execution decision](quality-speed.md#benchmark-coverage-during-development-and-final-testing).
