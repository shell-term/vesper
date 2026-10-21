# DGX Spark entry

Target: **dgx-spark**, Linux ARM64 / NVIDIA GB10 / one unified physical RAM pool.
Read [workspace rules](../README.md). Use a separate checkout on a
`feature/dgx-spark-` branch and bind `--platform dgx-spark --profile native`.
Do not infer Spark progress from the PC roadmap.

Current plan: [Spark roadmap](../../docs/spark-roadmap.md). It keeps the S0/S1
evidence, puts native-baseline latency work in S2, and follows with benchmark
connection and practical-quality evaluation in S3. GitHub owns live task state.

Evidence: [S0 management and bounded hardware report](../../docs/spark-platform-validation.md).
S1 execution evidence: [native integration report](../../docs/spark-native-integration.md).
The report separates short-text execution, component parity and practical-quality gates.

## Progress entry

- [Management isolation #13](https://github.com/shell-term/vesper/issues/13).
- [Bounded platform baseline #14](https://github.com/shell-term/vesper/issues/14).
- [S1 native model integration #16](https://github.com/shell-term/vesper/issues/16).
- [Opening phases and authenticated metadata reuse #18](https://github.com/shell-term/vesper/issues/18).
- [Bounded execution and projection preparation reuse #19](https://github.com/shell-term/vesper/issues/19).
- [Matched end-to-end measurements and evaluation budget #20](https://github.com/shell-term/vesper/issues/20).
- [Benchmark/harness connection checks #21](https://github.com/shell-term/vesper/issues/21).
- [Practical-quality evaluation #22](https://github.com/shell-term/vesper/issues/22).
- [Spark issues](https://github.com/shell-term/vesper/issues?q=is%3Aissue+label%3Aplatform%3Adgx-spark).
- [S0: Management and platform](https://github.com/shell-term/vesper/milestone/5).
- [S1: Native model integration](https://github.com/shell-term/vesper/milestone/6).
- [S2: Execution latency and measurement](https://github.com/shell-term/vesper/milestone/7).
- [S3: Benchmark connection and practical quality](https://github.com/shell-term/vesper/milestone/8).

These are task links, not pass claims. Read current state using
`python -m vesper.workspace read-progress --issue 13` or `--issue 14`.
Use `--issue 16` for S1 implementation and validation progress.
Start S2 with `--issue 18`; follow the roadmap's dependencies
for the remaining work.
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
weights. S2 preserves that baseline while reducing repeated setup and reads.
Future compression/approximation requires a separately evaluated profile with
quality tolerances and comparison methods defined before adoption; this plan
adopts no additional quantization, expert removal or changed routing.

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
