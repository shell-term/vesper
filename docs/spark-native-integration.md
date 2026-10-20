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
floating-point/control instructions while changing integer Query storage.

At `e0edf283a995fa8bb8ad6e060208a55ba3059f2a`, that independent Spark reference
gate passed all 36 intermediate/output cases, all four load/store bit patterns,
and the wrong-column negative control. Its minimum sampled physical available
memory was 45,253,332,992 bytes, above the reserve. The full untouched kernel
remains blocked by its shared-memory requirement. The
[complete oracle report](../results/dgx-spark/2026-10-20-s1-components/attention/reference-r01.json)
and [byte-preserving archive hash](../results/dgx-spark/2026-10-20-s1-components/attention/oracle-transport-r01.json)
qualify this reference adapter and corpus. A fresh native Driver comparison
against those original-computation boundaries is a separate gate; plain/trace
self-consistency and the PC receipt cannot replace it.

- A new explicit `spark-sm121-candidate` target compiles the shared arithmetic
  on Spark and records the actual source, compiler, options and binary. Existing
  PC target checks remain the default. A candidate never issues a qualification
  receipt merely because compilation succeeds.
- Spark Attention builds separate GB10 bundles from the pinned Query-storage
  transformation. Generated source, launch contract and CUBIN identities are
  checked; the foreign SM89 bundle remains rejected. Numerical qualification
  and complete-model integration remain separate gates.
- The output candidate projects every vocabulary row in FP32 while staging
  bounded original BF16 rows. Its separate-process component probe authenticates
  the complete released head shard and the original tensor hash, then compares
  all 129,280 logits against the unchanged official `ParallelHead.forward`.
  That probe is implemented; same-device numerical execution remains pending.
  Its first native attempt stopped before computation because the initial
  Spark candidate incorrectly assumed cuBLAS followed NVRTC's 13.0 version.
  Separate actual observations found cuBLAS version `130101`, math mode `0`,
  in both environments. The candidate now pins that exact observed version;
  it does not widen an unmeasured version range or change FP32/TF32 policy.
- An opt-in unified-memory policy charges device allocations to the same host
  ceiling as CPU, pinned and retained state. Discrete-GPU accounting keeps its
  existing default behavior. The physical available floor is not a VRAM floor.
- Full acquisition is a distinct, explicit integration operation: 48 pinned
  original shards (510,296,708,312 bytes), bounded transfer buffers, resumable
  checked prefixes, whole-file SHA-256, SSD quota and exclusive publication.
  Verified local span readers avoid a second 288.78 GB expert pack and do not
  fetch from the network during forward execution.
- Acquisition supports one, two or four bounded streams, with immutable shard
  priorities, checked resume, and all workers joined before a completion or
  failure result. Four streams reserve 256 MiB for transfer work. The active
  acquisition uses its own fixed-source Spark checkout, so integration edits
  cannot silently change the code identity of an in-flight import.
- All 40 blocks, both original Engram tables, the streamed head and generated
  token feedback are connected through a lazy runtime. The explicit text probe
  prepares small local dense derivatives, then requires a complete EOS response
  and owned-resource cleanup. Its code and CPU control-flow tests do not count
  as successful full-model text generation.
  Before generation, current source-bound Attention oracle/native receipts
  must match the immutable bundle, its actually loaded binaries, the observed
  device/driver and both window/sharing Attention bindings. Missing, changed
  or foreign evidence refuses execution before the first model forward.

CPU/mock tests check bounds, failure/refusal, cleanup, corruption rejection and
control flow. They do not qualify GB10 arithmetic. Current acceptance and pending
gates belong to #16; component evidence alone cannot close S1.
