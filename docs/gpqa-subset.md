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

The historical serial request is capped at 512 model calls, including both input
and generation. Attention state has 1,152 positions and tokenizer decoding admits
4,096 IDs. These historical constants were not increased. The separate native
benchmark backend below preserves the complete declared allowance and explicitly
reports partial-capacity attempts; full-allowance execution remains unqualified.
Request admission alone cannot extend attention state or qualify long context.

`python -m vesper.gpqa_readiness --record NEW-RECORD-NAME` rechecks the immutable
prepared plan and pinned source selection. It computes prompt embedding/Engram
requirements on CPU, writes a new bound private record, and never fetches
payloads or runs a model. The fixed prompts need 192/72/125 distinct embedding
rows and 6,768/2,720/4,712 Engram rows **per Engram layer**, respectively. Combined
logical prompt embedding/Engram bytes are about 5.28/2.07/3.59 MiB. These are
not physical reads, future expert requirements, generation requirements or proof
that all needed assets are cached. Other historical input caches are not audited.
The [new CPU readiness record](../results/pc-heterogeneous/2026-10-21-gpqa-readiness/native-readiness.json)
preserves its clean execution source and unchanged prepared-plan identity.

`benchmark_attempt.run_subset` now implements cooperative elapsed budgets,
exclusive attempt writing, successful-call journals, completed-question reuse
and immutable incomplete attempts. Its backend interface receives the deadline
and must enforce it inside long operations. Wrong completed answers are retained
and not rerun until correct. A context-capacity refusal never clips the prompt
or declared allowance. An interrupted question restarts from its original prompt
with a new attempt; KV/RNG state is not restored. Candidate/settings/resources
must match to resume. Forward-only and wall decode timing (including journal
writes) are distinct; accepted decode counts explicitly include a selected EOS.

An opt-in native backend now connects this controller to the actual forty-layer
engine. It computes its own routes and resolves original cached expert/input
assets as needed, without a saved answer or full trajectory. It preserves the
declared generation allowance while explicitly permitting partial-capacity
attempts. The current attention, distinct-embedding and Engram-row bounds remain
active. If EOS occurs within them, grade the actual answer; otherwise report
`backend_scope_exhausted`, time/missing-asset/error interruption, never an
invented completion. This improves on rejecting every request solely because its
maximum allowance exceeds current capacity. It does not qualify full-allowance
or long-context execution.

```sh
python -m vesper.gpqa_native --run gpqa-native-v1 --budget-seconds 180 --maximum-fetch-bytes 0
```

The command runs the same fixed three questions in their selected order, within
one cooperative budget, saving each actual attempt. A zero import allowance is
explicit offline mode: verified existing embedding/Engram ranges and original
expert objects are reused; unavailable assets stop the attempt. An explicit
nonzero allowance permits bounded original-range imports (at most 1 GiB per
question). Preparation/import time is included. Existing transport retries and
in-flight operations can overrun cooperative checks; this is not an OS deadline.
No complete checkpoint download is part of this path. The source-root lease
prevents cooperating storage migration during execution. Original math runs on
the 4090, with temperature/top-p 1.0 and seed 2026; no approximation is promoted.

Synthetic backend regression tests establish lifecycle behavior only. Actual
execution receipts, answers and timing must be reported separately. The backend
does not inherit the older task's exact qualification or establish broad quality.

## First actual native attempt

The [actual execution record](../results/pc-heterogeneous/2026-10-21-gpqa-native/summary.json)
at clean source `2220715` runs the selected first question through the native
engine with zero payload fetches. It completes **4 of 286 input calls** and
interrupts the fifth during layer work at the 180-second cooperative deadline.
Observed elapsed time is 181.93 seconds including cleanup; setup takes 56.30
seconds and the four committed input calls total 102.15 seconds. There are no
generated IDs, EOS, answer, decode-rate measurement or scored questions. All
three questions remain pending. This is an execution-path result, not GPQA quality.

It reuses 104,320 verified input bytes, inventories 7,251 existing original
experts and imports zero new experts. Thirteen compiled images match the
previously qualified primary-GPU set. The first BOS logit hash matches the
previous coding original's same first BOS call; that single shared input does
not establish full GPQA parity. Original tensor precision/router logic are
retained; temperature/top-p 1.0 follows this GPQA protocol and must not be
confused with older tasks' sampling settings.

Sampled resident RAM peaks at 7.79 GiB, private commit at 19.63 GiB, and sampled
available RAM stays at least 124.98 GiB. All charged resources/leases/reclaimers
and in-flight operations reach zero after close; the monitor stops without an
error. These observations are not OS quotas. Execution records also count
1,464 driver/context initializations and 1,468 module loads. Repeated per-layer
setup is a concrete next optimization target; this one interrupted prefix does
not establish a general prefill rate or a generation-speed prediction.

The next selected implementation work is bounded context/operator reuse in
this same native benchmark path, with unchanged full inputs and explicit
resource/time limits. Avoid launching a long evaluation of the current slow
path. Complete actual answers and compare the released-precision path and each
candidate on the fixed subset before claiming benchmark quality. Existing
independent raw-text/component checks remain supplementary implementation checks.
Full evaluation remains subject to the [user's execution decision](quality-speed.md#benchmark-coverage-during-development-and-final-testing).
