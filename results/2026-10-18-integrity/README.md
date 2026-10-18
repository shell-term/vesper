# GPU-audit and profiling evidence — 2026-10-18

The [measurement report](../../docs/cache-performance.md) explains the trust
anchor, deferred host verification, exact-output gates, paired timing and scope.
These are complete records with no model bytes or private paths. Previous
[cache evidence](../2026-10-18-cache/README.md) remains unchanged.

| Record | Contents |
| --- | --- |
| [summary.json](summary.json) | Counts, paired ratios, buffer cost and hashes of all records |
| [readback-official.json](readback-official.json) | Baseline: 232 official comparisons with thirteen exact checks per call, complete stage timing, traffic and pressure recovery |
| [gpu-audit-official.json](gpu-audit-official.json) | GPU audit mode: same 232 calls and checks, original source digests, stage timing, traffic and resource/cleanup records |
| [gpu-sha256-validation.json](gpu-sha256-validation.json) | 96 payload cases across both GPUs, known-message checks and ten single-bit fault/refusal/restoration checks |
| [paired-cache-benchmark.json](paired-cache-benchmark.json) | Six alternating pairs, 672 timed calls, every per-call timing, output fingerprints, startup costs and cleanup |

The modes repeat the same official comparison conditions; their combined
2,375,680 final values do not represent independent input coverage. The paired
benchmark has equal cache capacities and no timed SSD reads. Its median ratio
5.02 is not a full-model or token-generation speedup. Setup and cold-source
digest preparation are separate costs. The helper GPU still does no model
arithmetic. GPU hashing tests are not a cryptographic certification.
