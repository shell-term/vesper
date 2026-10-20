# DGX Spark S1 native integration

Active task: [Spark #16](https://github.com/shell-term/vesper/issues/16),
[S1 milestone](https://github.com/shell-term/vesper/milestone/6).
This report records bounded component evidence. Complete-model text generation,
practical quality and speed are not yet qualified on Spark.

## Original expert: measured on 2026-10-20

Source: `af0112331ace08e44079184f7221fc1ea64a0cea`.
Model revision: `1bae0e1088993b9b82de08f59de5be5a34e42170`.
Hardware/toolchain: one GB10, compute capability 12.1, Linux AArch64,
CUDA driver API 13000, NVRTC 13.0.88. Reference environment: PyTorch
2.14.1+cu130 and TileLang 0.1.12, isolated from the native NumPy/Driver process.

One original layer-0/expert-0 contains 18,800,640 bytes of unchanged FP4 codes
and original E8M0 scales. Two input seeds, two BF16 input magnitudes and four
route-weight conditions produce 16 cases. All 144 comparisons (nine stage or
output arrays per case) passed, including original activation quantization,
intermediate projections and final BF16 output. Each process also repeated its
first case successfully. No PC numerical receipt was transferred to Spark.

The native executor's peak explicit device allocation was 18,903,528 bytes.
Minimum sampled physical available memory was 45,947,031,552 bytes in native
execution and 44,253,736,960 bytes in reference execution, above the 32 GiB
floor. Compiler/context/framework overhead is separate from explicit buffers;
sampled observations and cooperative admissions are not an OS memory guarantee.
Diagnostic wall times are not a model-throughput benchmark.

Full records, preserved byte-for-byte with their original remote workspace
identity: [fetch](../results/dgx-spark/2026-10-20-s1-expert/fetch.json),
[native](../results/dgx-spark/2026-10-20-s1-expert/native.json),
[reference](../results/dgx-spark/2026-10-20-s1-expert/reference.json),
[all comparisons](../results/dgx-spark/2026-10-20-s1-expert/compare.json),
[archive hashes](../results/dgx-spark/2026-10-20-s1-expert/transport.json).
The comparison is scoped to one expert and these inputs, not every expert or
full-model answer quality. No full checkpoint was downloaded for this probe.

## Integration candidates

The following generated diagnostic corpora passed in separate native/reference
GB10 processes at `4de28456167aafe1440145f362883ab9140e629c`:

| Component | Cases | Stage/output checks | Result |
| --- | ---: | ---: | --- |
| RMS / both RoPE profiles | 14 | 74 | Exact match |
| Window KV quantization / wrap | 12 | 72 | Exact match |
| Full-vocabulary sampling / RNG offsets | 10 | 52 | Exact match |

[Full component records and hashes](../results/dgx-spark/2026-10-20-s1-components/transport.json)
include the initial failures. The first RoPE run matched final BF16 outputs but
differed in four imaginary FP32 intermediate arrays. A causal GPU comparison
identified the original SM121 complex-multiply FMA order. The Spark source
profile now matches that order; PC source bytes stay unchanged. The sampler's
first native build failed on a missing CUDA 13 CCCL include; adding that include
for SM121 sampling alone fixed compilation. All cases were regenerated under
the fixed source. These are component checks with generated coefficients or
logits, not learned full-model quality measurements.

Fresh Spark Attention binaries also passed 36 plain/diagnostic consistency and
two pressure-refusal/healthy-retry cases. Cleanup left no owned buffers, modules
or active context; neither Torch nor TileLang was imported in native execution.
That result is not the independent original Attention numerical gate. The full
original shared-memory kernel requests 141,312 bytes, above this GB10's measured
101,376-byte limit. A separately pinned Spark reference adapter preserves its
floating-point/control instructions while changing integer Query storage. It
must pass fresh data-movement, negative-control and intermediate comparisons;
the PC adapter and receipt are not substitutes.

- A new explicit `spark-sm121-candidate` target compiles the shared arithmetic
  on Spark and records the actual source, compiler, options and binary. Existing
  PC target checks remain the default. A candidate never issues a qualification
  receipt merely because compilation succeeds.
- Spark Attention builds separate GB10 bundles from the pinned Query-storage
  transformation. Generated source, launch contract and CUBIN identities are
  checked; the foreign SM89 bundle remains rejected. Numerical qualification
  and complete-model integration remain separate gates.
- The output candidate projects every vocabulary row in FP32 while staging
  bounded original BF16 rows. It requires its own same-device numerical gate.
- An opt-in unified-memory policy charges device allocations to the same host
  ceiling as CPU, pinned and retained state. Discrete-GPU accounting keeps its
  existing default behavior. The physical available floor is not a VRAM floor.
- Full acquisition is a distinct, explicit integration operation: 48 pinned
  original shards (510,296,708,312 bytes), bounded transfer buffers, resumable
  checked prefixes, whole-file SHA-256, SSD quota and exclusive publication.
  Verified local span readers avoid a second 288.78 GB expert pack and do not
  fetch from the network during forward execution.

CPU/mock tests check bounds, failure/refusal, cleanup, corruption rejection and
control flow. They do not qualify GB10 arithmetic. Current acceptance and pending
gates belong to #16; component evidence alone cannot close S1.
