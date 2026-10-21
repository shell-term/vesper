# Spark S2 candidate and target policy

日本語案内: S2に圧縮・削減候補の評価を含めます。精度を第一に、品質差を測って構成を選びます。20 tpsは挑戦目標であり、達成済み・達成見込み・必須の完了条件を意味しません。

The user explicitly expanded S2 on 2026-10-21 and adopted 20tokens/s as an
aspirational target. This supersedes the earlier native-only S2 scope and
withdraws the unadopted 900s ordinary-response / 192 calls / 3 h proposal. Follow
the [roadmap](spark-roadmap.md) and own-platform Issues; PC policy is unchanged.

## Candidate classes

| Class | Required distinction | Evaluation |
| --- | --- | --- |
| Native preparation/I/O reuse | Reuse authenticated catalogs, bounded backends, transfer or exact prepared data without changing model semantics | Exact native boundaries, routing/state, identity, ownership and matched timings; #18/#19 |
| Text-unused capability data | Prove the actual text execution dependency graph before excluding any vision/MTP or other data | Bytes/capability manifest, text invariance and reversible packaging; a short unselected expert is not unused data |
| Lossless representation and residency | Layout/packing, authenticated caching/prefetch and bounded residency may reduce traffic without modifying trained routes | Actual compression ratio, miss/hit/read costs, decode overhead, pressure and unchanged outputs |
| Additional lossy compression | Original expert weights are already FP4; evaluate selected tensor/precision changes rather than assuming another low-bit label saves space | Named profile, authenticated transform, declared quality differences and numerical/practical comparisons |
| Expert/tensor reduction or changed routing | These can change the model's behavior and supported capabilities | Named approximate profile, held-out quality screen, severe-regression stops and explicit adoption/rejection |

These are hypotheses, not adopted implementations. #23 inventories bytes/use,
selects candidate(s), and fixes acceptable quality differences and comparison
methods before approximate experiments. #24 implements the selected bounded
experiments and records the choice. Native remains the original contract;
approximate results must never inherit its receipts or be labeled native passes.
Keep source originals intact; derive reversible manifests/artifacts with explicit
SSD quotas instead of destructively editing the pinned checkpoint. No specific
extra quantization, expert deletion, routing change or derivative profile is
adopted by this plan alone.

## Quality and order

Accuracy is first, speed second. Accelerate native execution using bounded
numerical checks, then evaluate selected approximate candidates on fixed short
direct-runtime comparisons. Include representative supported Japanese,
instruction following, reasoning and code behavior beyond simple arithmetic;
record reference answers/scoring, held-out selection, critical regressions,
numeric tolerances, repeats and budgets before the qualifying screen. The
specific tolerances are not yet agreed. A generic one-percent rule or a correct
arithmetic answer is not a substitute.

Bit equality remains applicable to the native and lossless boundaries, not to
every approximate product candidate. A recorded rejection and retention of native
is a valid selection result if supported by evidence. Broader harness, tools and
practical task quality remain S3 #21/#22, after S2 readiness in #20.

## The 20tokens/s target

Measure batch 1, concurrency 1, accepted continuation tokens, including active
SSD reads/cache misses, transfers and computation. Use declared prompts,
context length, output length, thinking mode, warmup and repeats. Report the
number of tokens/timed intervals and sustained rate, not a selected fast token.
Speculative proposals do not count as accepted tokens. First-output latency,
prompt processing, setup and cleanup are reported separately. Serving and
full-diagnostic modes must disclose different work; compare like modes.

The S1 arithmetic run selected only `4` and EOS after 18 prompt tokens. It is an
ordinary completion regression, not evidence of sustained20tokens/s. Its last
model call read 4,606,590,976 physical expert-data bytes, plus 154,877,952 integrity
index bytes. At unchanged traffic, 20 calls/s would require 92.13 GB/s of expert
data reads alone. This is a calculated traffic requirement, not a storage
benchmark or a prediction that compression/cache residency will suffice.
The [raw baseline](../results/dgx-spark/2026-10-21-s1-optimized/generate-r04.json)
preserves the original conditions. Actual sustained throughput and cache locality
remain to be measured.

S2 must publish target attainment or the remaining barriers honestly. 20tokens/s
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
