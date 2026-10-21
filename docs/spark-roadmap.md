# DGX Spark roadmap

日本語案内: S0/S1の実績を保持し、S2で原版基準を変えずに実行時間を改善・計測した後、S3でベンチマーク接続と実用精度を評価します。最新の進捗はSparkのGitHub Issueを参照してください。

This is the Spark plan adopted on 2026-10-21. GitHub Issues own task state and
acceptance decisions; this document defines responsibility, dependencies and
policy. Select the [Spark workspace](../platforms/dgx-spark/README.md) before
reading or changing progress. PC progress and qualification remain separate.

## Milestone responsibilities

| Milestone | Required outcome | Evidence or work entry |
| --- | --- | --- |
| [S0: Isolated management and platform baseline](https://github.com/shell-term/vesper/milestone/5) | Separate checkout/progress/result identities, then bounded ARM64/GB10 and unified-memory checks | [Management #13](https://github.com/shell-term/vesper/issues/13), [platform #14](https://github.com/shell-term/vesper/issues/14), [S0 report](spark-platform-validation.md) |
| [S1: Native model integration](https://github.com/shell-term/vesper/milestone/6) | All 40 blocks, both Engrams, original routed experts, real head and generated-token feedback complete an ordinary EOS response with owned cleanup | [Integration #16](https://github.com/shell-term/vesper/issues/16), [S1 report](spark-native-integration.md) |
| [S2: Execution latency and measurement](https://github.com/shell-term/vesper/milestone/7) | Improve setup/I/O on the original native baseline, measure matched end-to-end behavior, and establish the next evaluation budget | [Metadata #18](https://github.com/shell-term/vesper/issues/18), [backends #19](https://github.com/shell-term/vesper/issues/19), [measurement #20](https://github.com/shell-term/vesper/issues/20) |
| [S3: Benchmark connection and practical quality](https://github.com/shell-term/vesper/milestone/8) | Connect the selected benchmark/harness and evaluate practical quality within an explicit supported scope and matched conditions | [Connection #21](https://github.com/shell-term/vesper/issues/21), [quality #22](https://github.com/shell-term/vesper/issues/22) |

S0 and S1 retain their completed evidence and original acceptance scope. S1's
ordinary answer establishes functional integration, not useful interactive
speed or broad task accuracy. The previously unused milestone IDs 7 and 8 are
retained with responsibilities matching the agreed execution order. Heavy
practical evaluation follows latency work; accuracy remains the first product
priority throughout.

## Work breakdown and dependencies

| Work | Dependency and outcome | Required evidence |
| --- | --- | --- |
| [#18: Measure opening phases and reuse authenticated model metadata/indexes](https://github.com/shell-term/vesper/issues/18) | First S2 work, after completed #16. Separate catalog parsing, fixture checks, backend setup and uploads; reuse authenticated metadata within an explicitly owned runtime lifetime | Bounded retained bytes and lifetime; changed/corrupt input handling; unchanged relevant native numerical boundaries; measured setup breakdown |
| [#19: Reuse bounded execution, transfer and projection preparation](https://github.com/shell-term/vesper/issues/19) | Follows #18. Retain only the preparation demonstrated useful by measurement, including backend handles or verified conversion state where appropriate | Native numerical boundaries, failed-switch/retry behavior, pressure refusal/eviction, cleanup and actual retained memory; no claim that every proposed cache must be built |
| [#20: Measure matched end-to-end latency/I/O/memory and establish the evaluation budget](https://github.com/shell-term/vesper/issues/20) | Follows #18 and #19. Fix comparison inputs, settings and execution budgets before measurement; use short numerical probes first and reconfirm an ordinary response after integrating improvements | Input processing, time to first output, output generation, SSD reads, retained memory and cleanup; separate initialization/diagnostics; an explicit budget for S3 workloads |
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

Compare equivalent cache/initialization conditions or state the limitation.
Fix tasks, model/profile/source identity, harness settings, tool scope, output
and context limits, cancellation behavior, grading and time budgets before
interpreting a comparison. Preserve failed results and historical raw records.
Do not restamp old evidence as execution of a new source or profile.

## Precision and resource boundaries

The original native profile and its existing evidence remain the reference.
The immediate work favors setup/I/O improvements that preserve its numerical
behavior. Future compression or approximation is a separate candidate profile:
define acceptable quality differences and comparison methods first, then
measure accuracy, speed and resources before adoption. This plan does not adopt
additional quantization, expert removal, changed routing or another lossy
configuration, and does not assert that every future profile must be bit-identical.

Keep at least 32 GiB of physical memory available and preserve existing
services. Spark has one physical CPU/GPU pool; reported VRAM is not additional
RAM. Account for retained metadata, CPU/device weights, pinned staging, scratch
and diagnostics, including cache lifetime and pressure cleanup. The S1
composition's 12 GiB shared ledger, 1 GiB asset cap and 5 GiB device cap are the
measured starting limits, not permission to retain every layer. Admission
counters and sampled RSS are not an OS hard-memory guarantee.

Images, longer context/reasoning and MTP are separate expansion candidates.
Confirm model and implementation support and define their own acceptance
scope before scheduling them; implementing every candidate is not required to
finish this five-issue plan. No scheduled task, new runtime experiment or model
profile is introduced by this management update.
