# DGX Spark roadmap

日本語案内: S0/S1の実績と原版基準を保持し、S2で準備・I/O改善に加え、圧縮・削減候補の品質差と速度・資源を比較して構成を選びます。20 tpsは挑戦目標であり必須の完了条件ではありません。S3でハーネス接続と広い実用精度を評価します。

This is the Spark plan expanded by explicit user decision on 2026-10-21.
It supersedes that day's earlier native-optimization-only S2 scope. GitHub Issues own task state and
acceptance decisions; this document defines responsibility, dependencies and
policy. Select the [Spark workspace](../platforms/dgx-spark/README.md) before
reading or changing progress. PC progress and qualification remain separate.

## Milestone responsibilities

| Milestone | Required outcome | Evidence or work entry |
| --- | --- | --- |
| [S0: Isolated management and platform baseline](https://github.com/shell-term/vesper/milestone/5) | Separate checkout/progress/result identities, then bounded ARM64/GB10 and unified-memory checks | [Management #13](https://github.com/shell-term/vesper/issues/13), [platform #14](https://github.com/shell-term/vesper/issues/14), [S0 report](spark-platform-validation.md) |
| [S1: Native model integration](https://github.com/shell-term/vesper/milestone/6) | All 40 blocks, both Engrams, original routed experts, real head and generated-token feedback complete an ordinary EOS response with owned cleanup | [Integration #16](https://github.com/shell-term/vesper/issues/16), [S1 report](spark-native-integration.md) |
| [S2: Optimization candidates and measured selection](https://github.com/shell-term/vesper/milestone/7) | Improve native setup/I/O, evaluate compression/reduction against agreed quality gates, select a configuration and establish measured readiness | [Metadata #18](https://github.com/shell-term/vesper/issues/18), [backends #19](https://github.com/shell-term/vesper/issues/19), [candidate protocol #23](https://github.com/shell-term/vesper/issues/23), [experiments #24](https://github.com/shell-term/vesper/issues/24), [readiness #20](https://github.com/shell-term/vesper/issues/20) |
| [S3: Benchmark connection and practical quality](https://github.com/shell-term/vesper/milestone/8) | Connect the selected benchmark/harness and evaluate practical quality within an explicit supported scope and matched conditions | [Connection #21](https://github.com/shell-term/vesper/issues/21), [quality #22](https://github.com/shell-term/vesper/issues/22) |

S0 and S1 retain their completed evidence and original acceptance scope. S1's
ordinary answer establishes functional integration, not useful interactive
speed or broad task accuracy. The previously unused milestone IDs 7 and 8 are
retained with responsibilities matching the agreed execution order. Heavy
practical evaluation follows acceleration and the bounded S2 quality screen;
accuracy remains the first product priority throughout. Including a candidate
authorizes evaluation, not adoption as the default model.

## Work breakdown and dependencies

| Work | Dependency and outcome | Required evidence |
| --- | --- | --- |
| [#18: Measure opening phases and reuse authenticated model metadata/indexes](https://github.com/shell-term/vesper/issues/18) | First S2 work, after completed #16. Separate catalog parsing, fixture checks, backend setup and uploads; reuse authenticated metadata within an explicitly owned runtime lifetime | Bounded retained bytes and lifetime; changed/corrupt input handling; unchanged relevant native numerical boundaries; measured setup breakdown |
| [#19: Reuse bounded execution, transfer and projection preparation](https://github.com/shell-term/vesper/issues/19) | Follows #18. Retain only the preparation demonstrated useful by measurement, including backend handles or verified conversion state where appropriate | Native numerical boundaries, failed-switch/retry behavior, pressure refusal/eviction, cleanup and actual retained memory; no claim that every proposed cache must be built |
| [#23: Inventory compression/reduction candidates and fix quality gates](https://github.com/shell-term/vesper/issues/23) | After completed #16; design can run alongside native improvements. Separate unused capability storage, lossless representation/cache work and lossy weight/expert changes | Source-derived bytes/use inventory, named profiles and agreed numerical/practical quality limits fixed before approximate experiments |
| [#24: Compare candidates and select a configuration](https://github.com/shell-term/vesper/issues/24) | Follows #19 and #23. Implement bounded candidates and run short matched quality/latency/resource comparisons after acceleration | Quality differences, sustained accepted-token speed, I/O, resource bounds, failure cleanup and adoption/rejection; retaining native is valid if alternatives fail |
| [#20: Verify selected end-to-end behavior and evaluation readiness](https://github.com/shell-term/vesper/issues/20) | Follows #18, #19 and #24. Fix inputs, selected profile, conditions and budgets before measurement; reconfirm ordinary EOS and report the 20tps target's status | Prompt processing, first output, sustained continuation, I/O/memory/cleanup, comparable diagnostic modes and a bounded S3 execution plan |
| [#21: Connect and check the existing benchmark/harness](https://github.com/shell-term/vesper/issues/21) | Follows S2 measurement/readiness in #20. Choose the actual connection method and targets in the issue | Input/output, tools, cancellation, usage accounting and setting correspondence; supported and unsupported behavior recorded explicitly |
| [#22: Evaluate practical Japanese, reasoning, coding and tool behavior](https://github.com/shell-term/vesper/issues/22) | Follows the connection check in #21, using the recorded S2 execution budget | Fixed common tasks, grading and settings; baseline comparisons; accuracy, time and resource results reported separately; timeout, unsupported scope and task failure distinguished |

Implementation methods are hypotheses to test, not a commitment to build every
cache or derivative. A phase is not accepted merely because code exists or a
component passes. If the intended evaluation cannot fit its declared budget,
record that unresolved readiness condition and revise the plan; do not invent
an after-the-fact speed threshold or promote an unfinished run to a pass.

## Baseline and measurement policy

The [S1 baseline](../results/dgx-spark/2026-10-21-s1-optimized/generate-r04.json)
uses runtime source `4b7e816ff7725b7b23b37ee65867231d85af64af` and original
model revision `1bae0e1088993b9b82de08f59de5be5a34e42170`. Its ordinary short
request completed 19 model calls in 1,268.294 seconds, including initialization,
diagnostics and cleanup. Opening MoE operators took 567.756 seconds and opening
projections 257.692 seconds. These are aggregated host-wall observations:
MoE opening includes catalog/fixture validation, setup and uploads; its time is
not a catalog-only measurement. This diagnostic request is not a general model
throughput benchmark.

S2 first changes repeated reads, setup and reuse while preserving the original
weights, scales, activation/KV rounding, trained routing and sampling baseline.
Changes need relevant exact-boundary checks against that baseline before long
evaluation. Short probes establish numerical/ownership evidence; a complete
ordinary response checks the integrated path after improvements. Component
parity, full-model functional execution, practical quality and speed remain
separate claims. A compiler-cache hit never grants numerical qualification.

After acceleration, #23/#24 evaluate selected compression/reduction candidates
under separate named profiles. Their bounded direct-runtime quality screen
does not require the full S3 harness/tool suite to run first and does not
qualify broad task quality.

Compare equivalent cache/initialization conditions or state the limitation.
Fix tasks, model/profile/source identity, harness settings, tool scope, output
and context limits, cancellation behavior, grading and time budgets before
interpreting a comparison. Preserve failed results and historical raw records.
Do not restamp old evidence as execution of a new source or profile.

## Precision and resource boundaries

The original native profile and its existing evidence remain the reference.
S2 includes evaluating compression and reduction, with acceptable quality
differences and comparison methods fixed before experiments. It adopts no
specific extra quantization, expert removal or changed routing yet, and does
not require approximate candidates to be bit-identical to native. Preserve
originals and keep candidates reversible. An expert unselected in a short
probe is not unused model data. Text-unused capability removal may save storage
without reducing active-token work; establish the actual dependency graph.

See [candidate and target policy](spark-optimization-candidates.md). The
aspirational target is at least 20 accepted continuation tokens/s for batch 1,
concurrency 1 on a sustained declared workload with repeats. It is not a
mandatory S2 completion threshold or a claim of likely attainment. Quality,
measured selection, cleanup and declared evaluation readiness are completion
gates. First-output latency is separate. The earlier 900s ordinary-response
and 192 calls / 3 h proposal was not adopted and is withdrawn; numeric quality and
execution-budget gates must be fixed before qualifying measurements.

Keep at least 32 GiB of physical memory available and preserve existing
services. Spark has one physical CPU/GPU pool; reported VRAM is not additional
RAM. Account for retained metadata, CPU/device weights, pinned staging, scratch
and diagnostics, including cache lifetime and pressure cleanup. The S1
composition's 12 GiB shared ledger, 1 GiB asset cap and 5 GiB device cap are the
measured starting limits, not permission to retain every layer. Admission
counters and sampled RSS are not an OS hard-memory guarantee.

Images, longer context/reasoning and MTP execution are separate expansion candidates.
Confirm model and implementation support and define their own acceptance
scope before scheduling them; implementing every candidate is not required to
finish this seven-issue plan. Inventorying capability-specific data does not
imply implementing that capability. No scheduled task or concrete approximate
model profile is introduced by this management update.
