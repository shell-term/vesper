# DGX Spark entry

Target: **dgx-spark**, Linux ARM64 / NVIDIA GB10 / one unified physical RAM pool.
Read [workspace rules](../README.md). Use a separate checkout on a
`feature/dgx-spark-` branch and bind `--platform dgx-spark --profile native`.
Do not infer Spark progress from the PC roadmap.

Evidence: [S0 management and bounded hardware report](../../docs/spark-platform-validation.md).
This report is scoped platform evidence; model integration/quality remain open.

## Progress entry

- [Management isolation #13](https://github.com/shell-term/vesper/issues/13).
- [Bounded platform baseline #14](https://github.com/shell-term/vesper/issues/14).
- [Spark issues](https://github.com/shell-term/vesper/issues?q=is%3Aissue+label%3Aplatform%3Adgx-spark).
- [S0: Management and platform](https://github.com/shell-term/vesper/milestone/5).
- [S1: Native model integration](https://github.com/shell-term/vesper/milestone/6).
- [S2: Practical quality](https://github.com/shell-term/vesper/milestone/7).
- [S3: Measured optimization](https://github.com/shell-term/vesper/milestone/8).

These are task links, not pass claims. Read current state using
`python -m vesper.workspace read-progress --issue 13` or `--issue 14`.
Registry changes are required before additional task IDs become valid. Shared
model source may be inspected, but PC progress/results do not qualify Spark.

After #13 passes, #14 checks ARM64, GB10, CUDA discovery and a tiny bounded GPU
round trip. It downloads no checkpoint, loads/unloads no existing service,
resets no GPU, changes no swap and measures no model tokens/s. Runtime hardware
must match Spark. Keep the existing 32 GiB physical available-memory floor;
never add reported VRAM to host RAM. Context/compiler overhead is separate from
explicit buffer caps. Store sanitized results under `results/dgx-spark/`, private
data under `artifacts/dgx-spark/`. Initial profile: original native weights;
lossy requantization and approximate routing are not adopted.
