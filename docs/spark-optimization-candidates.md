# Spark S2 candidate and target policy

日本語案内: S2に圧縮・削減候補の評価を含めます。精度を第一に、品質差を測って構成を選びます。検証・改善・再検証はS2/S3をまたいで繰り返し、変更の影響を受ける既合格項目も再確認します。20 tpsは挑戦目標であり、達成済み・達成見込み・必須の完了条件を意味しません。

The user explicitly expanded S2 on 2026-10-21 and adopted 20 tokens/s as an
aspirational target. This supersedes the earlier native-only S2 scope and
withdraws the unadopted 900s ordinary-response / 192 calls / 3 h proposal. Follow
the [roadmap](spark-roadmap.md) and own-platform Issues; PC policy is unchanged.

## Candidate classes

| Class | Required distinction | Evaluation |
| --- | --- | --- |
| Native preparation/I/O reuse | Reuse authenticated catalogs, bounded backends, transfer or exact prepared data without changing model semantics | Exact native boundaries, routing/state, identity, ownership and matched timings; #18/#19 |
| Capability-specific data | Image input is required: retain vision, aligner, learned markers and trained image-routing bias. Prove dependencies before excluding any other capability, such as unused MTP execution data | Bytes/capability manifest, supported behavior and reversible packaging; a short unselected expert is not unused data |
| Lossless representation and residency | Layout/packing, authenticated caching/prefetch and bounded residency may reduce traffic without modifying trained routes | Actual compression ratio, miss/hit/read costs, decode overhead, pressure and unchanged outputs |
| Additional lossy compression | Original expert weights are already FP4; evaluate selected tensor/precision changes rather than assuming another low-bit label saves space | Named profile, authenticated transform, declared quality differences and numerical/practical comparisons |
| Expert/tensor reduction or changed routing | These can change the model's behavior and supported capabilities | Named approximate profile, held-out quality screen, severe-regression stops and explicit adoption/rejection |

These are hypotheses, not adopted implementations. #23 inventories bytes/use,
selects candidate(s), and fixes acceptable quality differences and comparison
methods before approximate experiments. #24 implements the selected bounded
experiments and records the choice. Native remains the original contract;
approximate results must never inherit its receipts or be labeled native passes.
Image input is a required S3 gate [#25](https://github.com/shell-term/vesper/issues/25).
S2 text evidence leaves image functionality and quality unqualified. Lossy
changes to shared experts/backbone require image-quality evaluation before
the chosen configuration is accepted as an image-enabled product profile,
even if its vision weights remain unchanged. Current Spark has no vision
encoder/aligner/image-feature insertion path; retaining weights alone is insufficient.
Keep source originals intact; derive reversible manifests/artifacts with explicit
SSD quotas instead of destructively editing the pinned checkpoint. No specific
extra quantization, expert deletion, routing change or derivative profile is
adopted by this plan alone.

## Quality and iteration

Accuracy is first, speed second. Validation, improvement and revalidation
repeat across S2/S3 work as findings require; milestones group required outcomes,
not a mandatory task sequence. Native improvements use bounded numerical checks;
selected approximate candidates require fixed short direct-runtime comparisons
under the agreed quality protocol. Include representative supported Japanese,
instruction following, reasoning and code behavior beyond simple arithmetic;
record reference answers/scoring, held-out selection, critical regressions,
numeric tolerances, repeats and budgets before the qualifying screen. The
specific tolerances are not yet agreed. A generic one-percent rule or a correct
arithmetic answer is not a substitute.

On 2026-10-21 the user chose further short checks and acceleration, deferring
long quality checks. Neither the proposed 1,200-second ordinary EOS budget nor
the four-case, 192-call / three-hour quality-screen proposal was adopted. Their
draft inputs/rubrics are not an accepted quality protocol. S2 remains open;
finite numerical checks do not waive its selection/quality/readiness gates.
The iterative methodology does not adopt those budgets or authorize deferred
long quality checks. Every run still needs its actual technical prerequisites,
current source-bound receipts, resource bounds and declared finite budget.

Bit equality remains applicable to the native and lossless boundaries, not to
every approximate product candidate. A recorded rejection and retention of native
is a valid selection result if supported by evidence. Broader harness, tools and
practical task quality remain S3 #21/#22. Their prototypes and bounded checks
are not blocked by blanket S2 closure; each result qualifies only its declared
capability and conditions. Final integrated text/image acceptance still requires
the relevant connection and required image evidence, not a text-only pass.

Candidate or harness findings can return work to an affected implementation,
protocol or readiness issue. Rerun affected previously passed checks and any
freshness checks required by the runtime's source-identity gates. Preserve old
passes for their exact source/profile/hardware/input scope and mark affected
current claims as needing validation; do not restamp receipts or inherit a
blanket pass. A revised protocol or candidate requires a separately declared
trial, retaining earlier results rather than changing their passing criteria
after inspection. This feedback is not a cyclic GitHub blocking dependency.

### Official-benchmark subsets and final coverage

On 2026-10-21 the user adopted fixed small subsets of official-benchmark
questions for the development loop, with periodic checks on held-out questions.
Compare the pinned original model revision
`1bae0e1088993b9b82de08f59de5be5a34e42170` and each improved profile under
matched inputs, inference, tool and grading conditions. Keep subset membership
and scoring fixed before inspecting results; do not tune against the held-out
questions. These checks detect regressions within their declared scope; a small
sample cannot establish small differences in broad model quality.

The final post-optimization test covers every question of each selected
benchmark, version and split. Selecting a text-only split means its entire
pinned text-only split, not the benchmark's excluded multimodal questions.
This does not select every benchmark discussed as a candidate. The final
benchmark list, dataset versions/splits, repeats, output/reasoning settings,
tools, judges, quality criteria and execution budgets must be fixed before
execution. A complete single pass is not automatically a reproduction of an
official score: the official sampling, tool, grading and aggregation conditions
also matter. Use the [1017 model's official evaluation table](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
and the [official evaluation instructions](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/evaluation/README.md)
to establish those conditions rather than assuming all candidate evaluations
share one protocol.

Before a final full-suite run, estimate total duration from measured small
development trials of the relevant configuration. Use observed generation
speed and thinking/output lengths, and include input/image processing, tool
execution, grading, startup, question counts, repetitions and each comparison
profile. State the assumptions, remaining unmeasured costs and uncertainty;
an aspirational token rate is not a measured duration forecast. Present this
estimate to the user and decide together whether to execute. An explicit user
decision to proceed is required; optimization readiness and a declared plan
alone must not trigger the full run. If it is deferred or its scope is revised,
record that decision and leave unperformed final coverage unqualified.

Long full-suite runs remain deferred pending that decision and the actual
execution prerequisites. This policy starts no benchmark run or dataset
download and does not waive existing quality, image, source-identity or resource gates.
Final-test findings return to improvement and affected revalidation; final
coverage is an acceptance scope, not a one-way development sequence.

## The 20 tokens/s target

Measure batch 1, concurrency 1, accepted continuation tokens, including active
SSD reads/cache misses, transfers and computation. Use declared prompts,
context length, output length, thinking mode, warmup and repeats. Report the
number of tokens/timed intervals and sustained rate, not a selected fast token.
Speculative proposals do not count as accepted tokens. First-output latency,
prompt processing, setup and cleanup are reported separately. Serving and
full-diagnostic modes must disclose different work; compare like modes.

The S1 arithmetic run selected only `4` and EOS after 18 prompt tokens. It is an
ordinary completion regression, not evidence of sustained 20 tokens/s. Its last
model call read 4,606,590,976 physical expert-data bytes, plus 154,877,952 integrity
index bytes. At unchanged traffic, 20 calls/s would require 92.13 GB/s of expert
data reads alone. This is a calculated traffic requirement, not a storage
benchmark or a prediction that compression/cache residency will suffice.
The [raw baseline](../results/dgx-spark/2026-10-21-s1-optimized/generate-r04.json)
preserves the original conditions. Actual sustained throughput and cache locality
remain to be measured.

S2 must publish target attainment or the remaining barriers honestly. 20 tokens/s
is not mandatory for closure; measured quality-first selection, ordinary EOS,
owned cleanup, resource preservation and a declared viable next evaluation plan
are. Numeric execution budgets are agreed before final qualification, never
relaxed after inspecting a result.

## Resource and evidence boundaries

Keep the 32GiB physical-available floor and existing services. CPU/device/pinned
buffers share one physical pool. Declare retained and temporary memory, SSD
quota, source/model/toolchain identity, per-run/total timeouts and cancellation
before each candidate run. Do not assume a full expanded checkpoint or another
model fits. Preserve failed measurements and unchanged S0/S1 raw evidence;
separate CPU/mock failure tests from actual Spark qualification.
