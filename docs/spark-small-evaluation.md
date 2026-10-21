# Spark small official-benchmark evaluation

This is the operational **development protocol proposal** for
[#22](https://github.com/shell-term/vesper/issues/22) and
[#23](https://github.com/shell-term/vesper/issues/23). The user adopted the
small-subset/improvement/held-out loop and a jointly decided final full test.
The exact subset counts and finite budgets below make that policy reviewable;
they are not evidence that these numeric conditions or an approximate quality
tolerance have already been adopted. No model, external judge or full-suite
execution is performed by `vesper.spark_quality`.

## Selected first protocol

Use the author's GPQA Diamond CSV, all **198** input questions, pinned to
author repository commit `56686c06f5e19865c153de0fdb11be3890014df7` and CSV
SHA256 `41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305`.
The [author release](https://github.com/idavidrein/gpqa) and
[paper, dataset splits](https://arxiv.org/html/2311.12022v1#S2.SS3)
identify Diamond as the 198-question subset. Retain downloaded data, question
text, correct answers and completions only under ignored private Spark artifacts;
do not add them to public source/results or external scoring requests.

| Condition | Concrete development proposal |
| --- | --- |
| Fixed development set | 12 questions |
| Separate held-out set | 6 questions, disjoint from development; do not tune against its answers |
| Selection | Seed 2026; at least one per high-level domain, then proportional largest remainders; rank content IDs by SHA256, not observed correctness or runtime fit |
| Choices | Per-question seed-bound permutation, identical for original and improved runtime; no correct-position bias |
| Model | Original `DeepSeek-V4.1-Flash-1017`, revision `1bae0e1088993b9b82de08f59de5be5a34e42170`, native profile |
| Inference | Thinking ON, effort 100, greedy/temperature 0, top-p 1, seed 2026, batch 1/concurrency 1, one pass |
| Tools and grading | No tools or external judge; strict final-channel answer-choice grading |
| Proposed finite bounds | 1,152 context positions, at most 768 generated tokens including reasoning/answer/EOS, 1,152 model calls, 600 seconds per question; 3,600 seconds for development or 1,800 seconds for held-out |
| Execution status | Preparation only; no quality inference or all-question execution authorized by this document |

The present prompt includes only `Question`, `Correct Answer`,
`Incorrect Answer 1` through `3`, rendered as four permuted unlabelled choices.
It never includes the author's explanation, validator metadata or previous
question/answer revisions. `Record ID` and canonical content hashes bind the
private manifest to source rows. The evaluator receives the gold key; the
model's request list excludes that key.

Private preparation on 2026-10-21 fixed development domain counts at
Biology 2 / Chemistry 5 / Physics 5, and held-out counts at
Biology 1 / Chemistry 3 / Physics 2. Manifest SHA256 is
`122c59b13a582a3b358063d96786868a66442f6d5c8fa15e5cf832e040db3c95`;
reversing CSV row order reproduces it, and the two sets do not overlap.
The author's CSV contains two rows with duplicate incorrect choices; those
choices are retained exactly, not repaired or dropped. No correct answer is
duplicated among its incorrect choices.

The pinned Spark tokenizer preflight made **zero model calls and no GPU work**.
All 18 selected requests fit the character bounds; input lengths were 162–475
tokens. With the proposed 768-token continuation, 17 fit the 1,152-position
admission and one is unsupported (475 + 768 - 1 = 1,242). Keep that ID and
unsupported status; do not lower the global continuation budget or resample
to hide it. This is request-admission evidence, **not inference, numerical
qualification or a quality result**. The private overview is
`artifacts/dgx-spark/quality/development-proposal-r03/sanitized-summary.json`;
its adjacent admission report retains all fixed IDs. Model-facing requests,
gold manifest and draft identity-bound plans are separate private files.
The draft plans retain their original uncommitted source identity and require
new clean-source plans before actual execution; they are not restamped.

This is **not official-score reproduction**. The initial V4.1 instruct card
reports GPQA Diamond and uses effort 100, temperature 1.0 and top-p 0.95;
our deterministic development comparison deliberately uses different sampling.
The [1017 card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
does not publish a GPQA result. Its HLE/agent/image entries do not establish an
official GPQA setup for 1017. See the
[initial V4.1 instruct conditions](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash#instruct-model).
Thinking OFF is a separate functional check, not a speed substitute for this
reasoning protocol.

## Admission, stopping and grading

Keep every preselected ID. Before execution, serialize with the pinned message
adapter and tokenize with the pinned tokenizer. A request above the current
4,000 user-character / 4,096 serialized-character bound, or whose input plus
the declared continuation exceeds the bounded context, is **unsupported**;
do not replace it with an easier/shorter question or silently clip its output
budget. The current native request admits `input_tokens + max_generated_tokens
- 1 <= context_tokens`, because the final selected EOS needs no further model
call. With a 768-token continuation and 1,152 positions, the input fits at most
385 tokens. The extension is a technical admission cap, not evidence that
long reasoning or all later positions are hardware/quality-qualified. Keep
the existing 32 GiB available-memory floor and existing services.

The 768-token bound may truncate useful reasoning. A bounded stop is not an
ordinary EOS answer and cannot qualify reasoning quality. Do not reinterpret
length/time stops as wrong knowledge or relax the budget after seeing scores.
If this protocol is not viable, revise it explicitly, retain the prior trial,
and rerun matched conditions. Existing deferred long-quality checks remain
deferred. A cooperative between-call duration check may overrun while one
model call is in flight; preserve actual elapsed time and the limit violation.

Only an explicit normal EOS completion with one terminal line such as
`Answer: (B)` earns an answer-choice score. Parse **final-channel content only**;
do not extract a convenient letter from reasoning. Multiple terminal answer
markers, missing/malformed answers and EOS omissions remain separate invalid
states. Timeouts, truncation, cancellation, runtime errors, unsupported inputs
and not-run questions retain `score: null` and their distinct status; they are
ungraded, not evidence of incorrect model knowledge. Report planned/graded
coverage and all status counts. `completed_accuracy` describes only valid
graded answers; `correct_over_planned` records coverage-aware earned credit
and must not be presented as a full quality score. `full_subset_score` stays
null unless every selected answer is gradable and the protocol budgets hold.
No automatic retries or response repair are part of this protocol.

Every observation names its content-bound run-plan SHA256, question ID, raw
record hash, normal-EOS state and final-channel text. Record input, generated,
thinking, answer and special-token counts; generated equals thinking + answer
+ special tokens. Record actual model calls, per-question wall time and
exclusive startup/input/thinking/answer/image/tool/grading/cleanup/other phase
times when measured. An absent phase measurement stays unknown; known
inapplicable image/tool work can be explicitly zero. Per-question and total
budget overruns are reported even if a syntactically correct answer appears.

The paired comparison requires identical dataset/subset/choice order/settings/
budgets/runtime/hardware/toolchain/diagnostics. Source commits/dirty digests
may differ for an optimization comparison and remain recorded. List each
correct-to-incorrect and incorrect-to-correct transition among graded pairs,
and completion losses/gains separately. Native/component
exactness remains its own contract. These small task results are descriptive
regression evidence, not a new approximation tolerance, statistical proof of
broad non-degradation or a complete S2/S3 pass. Revised data, conditions or
affected code require fresh records; historical runs are never restamped.

## Offline commands and records

All input/output paths below are Spark artifact-relative, checked against the
bound workspace. Outputs are exclusive new files. `prepare` reads local bytes;
there is no downloader or generation command. A private conditions JSON must
declare the target identity and **every** budget:

```json
{
  "identity": {
    "model": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
    "model_revision": "1bae0e1088993b9b82de08f59de5be5a34e42170",
    "profile": "native",
    "code_commit": "REPLACE_WITH_ACTUAL_40_HEX_RUNTIME_COMMIT",
    "dirty_diff_sha256": null,
    "runtime": "vesper.spark_text_probe",
    "hardware": "REPLACE_WITH_ACTUAL_OBSERVED_HARDWARE",
    "toolchain": "REPLACE_WITH_ACTUAL_OBSERVED_TOOLCHAIN",
    "diagnostic_mode": "REPLACE_WITH_DECLARED_DIAGNOSTIC_MODE"
  },
  "budget": {
    "context_tokens": 1152,
    "max_generated_tokens": 768,
    "max_model_calls": 1152,
    "per_question_seconds": 600,
    "total_seconds": 3600
  }
}
```

```console
python -m vesper.spark_quality --root . prepare --dataset quality/SOURCE/gpqa_diamond.csv --revision 56686c06f5e19865c153de0fdb11be3890014df7 --sha256 41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305 --conditions quality/TRIAL/conditions.json --output quality/TRIAL/prepared.json
python -m vesper.spark_quality --root . grade --prepared quality/TRIAL/prepared.json --records quality/TRIAL/observations.json --output quality/TRIAL/graded.json
python -m vesper.spark_quality --root . compare --baseline quality/BASE/graded.json --candidate quality/CANDIDATE/graded.json --output quality/PAIR/paired.json
python -m vesper.spark_quality --root . estimate --prepared quality/TRIAL/prepared.json --reports quality/BASE/graded.json quality/CANDIDATE/graded.json --repetitions 1 --fixed-overhead-seconds 0 --excluded-cost "Unmeasured queue delay" --output quality/PAIR/full-duration-estimate.json
```

For the held-out plan use `--subset held_out`, its own conditions/budget file,
and a new output path. Preparing the held-out IDs does not execute or authorize
examining their answers for tuning. The prepared JSON is private: its manifest
contains gold, while `requests` is the model-facing no-gold projection.
Separate these for execution so the generation process only receives requests.
The run plan describes the intended runtime identity; the runner must verify
it against the **actual** executed source before attaching observations.

`observations.json` is a JSON array. Each entry has `question_id`,
`plan_sha256`, `status`, `eos_complete`, `final_text`, `input_tokens`,
`generated_tokens`, `thinking_tokens`, `answer_tokens`, `special_tokens`,
`model_calls`, `wall_seconds`, and `phase_seconds`. Valid input statuses are
`completed`, `timeout`, `truncated`, `cancelled`, `error`, `unsupported`,
`not_run`. Missing entries are retained as `not_run`; duplicates, foreign
question IDs and plan mismatches fail closed. Preserve raw observations beside
derived grading; hashes bind them without copying completions into summaries.

## Final coverage decision

After optimization, estimate the entire selected benchmark/version/split from
measured small trials of each intended configuration. `estimate` stratifies
per-question **actual wall time** by domain and expands it to the complete
198-question population, multiplied by declared repetitions. It adds all
provided baseline/candidate configurations and a declared fixed overhead;
report startup, input, thinking/output, image/tool, grading and cleanup costs,
and explicitly list unmeasured or excluded costs. Different budgets, context,
settings, source, cache or hardware require a new representative estimate.

The report provides domain-weighted mean and observed min/max scenarios.
These are **not confidence intervals or guaranteed bounds**: a small sample
can miss long tails. Missing domains prevent a total forecast. Timeouts,
length stops, unsupported or missing results prevent a finite complete-answer
mean/upper forecast, as do invalid protocol budgets; they cannot be projected as if the timeout were the time
to a correct complete answer. An incomplete answer parser result with normal
EOS is timed completed work but remains zero quality credit.

Present the estimate, sample counts, variability, censoring and remaining
costs to the user. Obtain an explicit joint decision before any all-question
run. The offline helper always emits `full_run_authorized: false` and has no
full-run execution path. Final coverage stays unqualified until actually run
under the selected conditions; findings return to the improvement loop.
