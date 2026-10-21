# DGX Spark entry

Target: **dgx-spark**, Linux ARM64 / NVIDIA GB10 / one unified physical RAM pool.
Read [workspace rules](../README.md). Use a separate checkout on a
`feature/dgx-spark-` branch and bind `--platform dgx-spark --profile native`.
Do not infer Spark progress from the PC roadmap.

Current plan: [Spark roadmap](../../docs/spark-roadmap.md). It keeps the S0/S1
evidence, includes native latency work and compression/reduction candidate
evaluation in S2, and benchmark connection and broader practical quality in S3.
These milestones group outcomes. Development loops through validation,
improvement and revalidation, including affected previously passed checks;
bounded work may cross milestone boundaries when its actual prerequisites are
met. Long quality checks remain deferred. Accuracy is first; 20 tokens/s is an
aspirational continuation target, not a mandatory S2 completion threshold.
GitHub owns live task state.

Evidence: [S0 management and bounded hardware report](../../docs/spark-platform-validation.md).
S1 execution evidence: [native integration report](../../docs/spark-native-integration.md).
The report separates short-text execution, component parity and practical-quality gates.
S2 evidence: [measured preparation and candidate progress](../../docs/spark-s2-preparation.md).
It records bounded component/all-layer partial checks and unadopted codec/audit
experiments; S2, ordinary EOS requalification and image input remain open.

## Progress entry

- [Management isolation #13](https://github.com/shell-term/vesper/issues/13).
- [Bounded platform baseline #14](https://github.com/shell-term/vesper/issues/14).
- [S1 native model integration #16](https://github.com/shell-term/vesper/issues/16).
- [Opening phases and authenticated metadata reuse #18](https://github.com/shell-term/vesper/issues/18).
- [Bounded execution and projection preparation reuse #19](https://github.com/shell-term/vesper/issues/19).
- [Matched end-to-end measurements and evaluation budget #20](https://github.com/shell-term/vesper/issues/20).
- [Compression/reduction protocol and quality gates #23](https://github.com/shell-term/vesper/issues/23).
- [Candidate comparisons and configuration selection #24](https://github.com/shell-term/vesper/issues/24).
- [Benchmark/harness connection checks #21](https://github.com/shell-term/vesper/issues/21).
- [Practical-quality evaluation #22](https://github.com/shell-term/vesper/issues/22).
- [Required bounded image input #25](https://github.com/shell-term/vesper/issues/25).
- [Spark issues](https://github.com/shell-term/vesper/issues?q=is%3Aissue+label%3Aplatform%3Adgx-spark).
- [S0: Management and platform](https://github.com/shell-term/vesper/milestone/5).
- [S1: Native model integration](https://github.com/shell-term/vesper/milestone/6).
- [S2: Optimization candidates and measured selection](https://github.com/shell-term/vesper/milestone/7).
- [S3: Benchmark connection and practical quality](https://github.com/shell-term/vesper/milestone/8).

These are task links, not pass claims. Read current state using
`python -m vesper.workspace read-progress --issue 13` or `--issue 14`.
Use `--issue 16` for S1 implementation and validation progress.
Native preparation evidence is under `--issue 18` and `--issue 19`;
candidate inventory/protocol is `--issue 23`. Select each iteration from the
roadmap and live own-platform issue state. Its dependencies describe final
acceptance; they do not prohibit earlier bounded work with the required inputs.
Registry changes are required before additional task IDs become valid. Shared
model source may be inspected, but PC progress/results do not qualify Spark.

The completed S0 baseline (#13/#14) checked ARM64, GB10, CUDA discovery and a tiny
bounded GPU round trip. That baseline downloaded no checkpoint, loaded/unloaded
no existing service, reset no GPU, changed no swap and measured no model tokens/s.
Runtime hardware must match Spark. Keep the existing 32 GiB physical
available-memory floor; never add reported VRAM to host RAM. Context/compiler
overhead is separate from explicit buffer caps. Store sanitized results under
`results/dgx-spark/`, private
data under `artifacts/dgx-spark/`. The reference profile uses original native
weights. S2 preserves that baseline while reducing repeated setup and reads,
then evaluates named compression/reduction profiles with quality tolerances
and comparisons fixed before experiments. This plan adopts no specific extra
quantization, expert removal or changed routing. See the
[candidate and target policy](../../docs/spark-optimization-candidates.md).
Image input is required under #25. Preserve its vision/aligner/marker/routing
weights in S2; current Spark execution remains text-only and candidate image
quality is unqualified until the required image gate passes. The 20 tokens/s
aspiration is text continuation; image encoding/prefill has a separate budget.

The completed S1 integration connected original expert and Engram storage with
all 40 model blocks, the real vocabulary head and generated-token feedback.
Its ordinary EOS response and owned cleanup are recorded in the S1 report;
component tests and prepared route replays alone did not complete that gate.
The S1 SSD acquisition was separate from the small S0 probe: the pinned
510.30 GB checkpoint was acquired and verified without loading it into RAM.
Keep the existing services running and budget CPU, GPU and pinned allocations
against the same physical pool, including dense-weight staging and scratch.
Spark Attention and head candidates require their own toolchain and numerical
qualification; PC binary receipts cannot be reused as Spark qualification.
