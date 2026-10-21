# DGX Spark S1 native integration

Completed integration task: [Spark #16](https://github.com/shell-term/vesper/issues/16),
[S1 milestone](https://github.com/shell-term/vesper/milestone/6).
This report records component evidence and a completed ordinary short text
response on Spark. S1 functional integration passes; practical quality and
general speed remain unqualified.
The [current Spark roadmap](spark-roadmap.md) assigns subsequent native-baseline
latency work to S2, then benchmark connection and practical-quality evaluation
to S3. GitHub Issues own their live status; this report preserves S1 evidence.

## Completed ordinary text response: 2026-10-21

Runtime source: `4b7e816ff7725b7b23b37ee65867231d85af64af`.
Pinned original model revision: `1bae0e1088993b9b82de08f59de5be5a34e42170`.
The unchanged chat-mode request was:

> What is 2 + 2? Answer with the number only.

The real model returned **`4`**, then its original EOS. Eighteen original
prompt tokens and one generated-token feedback call produced 19 complete
model calls. Selected IDs were `[22, 1]`; token 22 was actually fed back before
EOS 1 was selected. Every call traversed all 40 blocks, all six trained expert
routes per block, both Engrams, the full 129,280-row head and the requested
sampler schedule. There was no prerecorded route replay or checkpoint fetch
in forward execution. The bounded one-call and full-run first-call semantic
hashes also agree exactly.

The full request took **1,268.294 seconds (21 minutes 8 seconds)**, including
initialization, diagnostics and cleanup. This establishes functional execution,
not practical throughput. Shared admission peak was 8,882,970,212 bytes within
the 12 GiB cap; explicit device peak was 3,355,523,836 bytes within 5 GiB.
Minimum sampled physical available memory was 40,924,119,040 bytes (38.11 GiB),
above the 32 GiB floor. Final owned host/device charges, leases, reclaimers and
in-flight operations were zero, and the memory monitor stopped. These are
owned-buffer/cooperative admission observations, not an OS RSS cap or a
continuous physical-memory guarantee.

The [full generation record](../results/dgx-spark/2026-10-21-s1-optimized/generate-r04.json),
[complete byte-preserving transport](../results/dgx-spark/2026-10-21-s1-optimized/transport-r02.json)
and [derived acceptance audit](../results/dgx-spark/2026-10-21-s1-optimized/closure-audit.json)
retain the actual input/output, every route and Engram/logit hash, original
runtime context, memory/I/O observations and exact source/model identity.
The audit is a derived record, not a new runtime or quality qualification.
Separate read-only agent reviews checked upload/storage ownership, compiler
identity/caching, cancellation handling, the published numerical/timing scope,
and the final eight raw reports. The final review independently checked all
760 layer-route records, 38 Engram records, selected-token feedback and cleanup,
and found no S1 closure blocker. This review is not external model grading.

S1 is the integration gate. This one arithmetic answer does not qualify
practical task accuracy, long context/reasoning, images or MTP. Those remain
later work, along with useful interactive speed. Native fault evidence covers
the real Attention pressure/allocation failures and healthy retries; cache
corruption/short-read and compiler-failure cases are CPU/mock regressions.
No whole-model hardware corruption injection is claimed. The original weights,
scales, activation/KV rounding, trained routing and sampling algorithm were
preserved; no lossy profile was introduced.

## S1 integration memory correction: 2026-10-21

Acquisition and integrity registration completed for all 48 pinned originals
(510,296,708,312 bytes). The first full-model attempt at `ca352867` stopped
before completing its first input token: opening the first original expert
store requested a 512 MiB catalog admission while the internal ledger had
reached 8,327,851,284 bytes against an 8 GiB cap. Recorded physical available
memory stayed above the 32 GiB reserve; the cap short-circuit did not observe
physical availability at the exact failing allocation. Owned cleanup passed.
No response or full-model quality pass was produced.
The [custody, preparation and failed generation reports](../results/dgx-spark/2026-10-21-s1-text/transport-r01.json)
are preserved byte-for-byte with their original source/checkout identities.

The ledger intentionally adds conservative legacy component admissions and
actual unified GPU allocations. Removing either without tracking its real
ownership would undercount memory. The short text composition now budgets
12 GiB: 6 GiB for component/control/diagnostic admissions and transient metadata
(about 5.3 GiB of enumerated allowances), a separately enforced 1 GiB original
asset budget, and the existing 5 GiB device cap. The allowances can overlap in
their coverage of payloads; this is a conservative admission ledger, not RSS
or an OS memory limit. Every allocation still passes the shared cap, pinned
cap and 32 GiB physical available-memory reserve. A bounded refusal record
distinguishes cap and physical-pressure failures and survives owned cleanup.
The change does not cache an unauthenticated catalog, alter model arithmetic,
or qualify a complete response by itself. At that point fresh source-bound
checks and generation remained required; the later completed response above
records their separate integrated outcome.

At the fixed corrected runtime source `0fcb7fdbb9ec41aed6de7d831d803f20279b29e7`,
the Spark CPU suite passed 660 tests (one environment skip). Fresh original
Attention reference and native executions again passed all 36 cases and the
two refusal/retry checks. The complete-vocabulary head again passed all 24
comparisons across eight inputs, with owned cleanup. The
[full fresh numerical records](../results/dgx-spark/2026-10-21-s1-text/transport-r02.json)
retain the original remote context and exact byte hashes. The ordinary text
retry was explicitly stopped by the user before it completed; its private
worker recorded cancellation and no immutable generation result was produced.
A successful answer was still unconfirmed for that attempt. A subsequent
test-only correction made CPU regression observers independent of CI host RAM;
it did not alter that attempt's runtime.

## Bounded S1 latency correction: 2026-10-21

Inspection of the cancelled run found repeated validation/upload reads of the
same immutable dense tensors, 4 KiB storage operations, and repeated compilation
when opening successive layers. The Spark-only composition now reuses tensor
arrays during one operator's initialization, authenticates bounded reads in
8 MiB aligned batches, and retains at most 64 MiB / 32 entries of exact compiled
kernel inputs and binaries inside one charged runtime scope. It retains no
extra model-wide tensor cache, changes no weight or arithmetic precision, and
does not issue numerical qualification from a compilation-cache hit.

The layer-zero probe compares exact returned-array bits with and without upload
reuse, including original six-expert routing. Both modes use the bulk reader;
their fixed cold/warm order means their timings cannot isolate each change's
speedup. Fresh Attention and full-vocabulary head gates, a one-call all-40-layer
probe, and the unchanged ordinary EOS text request were then run on hardware.
The local CPU suite passed 679 tests with two environment-dependent skips.
Private progress snapshots record completed calls and current layer without
substituting partial progress for immutable acceptance evidence. The completed
response and owned-cleanup evidence are recorded above.

### Measured optimized checks

At runtime source `4b7e816ff7725b7b23b37ee65867231d85af64af`, the new
[byte-preserved records](../results/dgx-spark/2026-10-21-s1-optimized/transport-r01.json)
establish the following scopes:

| Check | Measured result |
| --- | --- |
| Spark CPU regressions | 688 tests, one environment skip |
| Fresh Attention reference/native | 36 exact cases; two native fault/healthy-retry cases |
| Fresh full-vocabulary head | 24 exact checks across eight inputs in separate processes |
| Layer-zero upload modes | 104 diagnostic arrays match exactly |
| One complete native model call | All 40 layers, both Engrams and 129,280 logits; 60.79 seconds for the call, 71.91 seconds including initialization/cleanup |

The layer comparison's dense-tensor logical reads fell from 54 to 27, and
logical bytes from 261,343,152 to 130,671,576. The two modes took 1.925 and
1.261 seconds, respectively, with the explicit cold/warm ordering limitation
above. Both use the bulk reader; this is not a measured speedup against the
previous 4 KiB reader. The scoped compiler cache produced reuse without
retaining modules or a qualification receipt, and released all entries at exit.

The one-call probe deliberately stopped before output sampling. Its shared
ledger peak was 8,882,961,892 bytes; minimum sampled available memory was
41,313,849,344 bytes, above the 32 GiB floor. Owned bytes, leases, reclaimers
and device charges reached zero after runtime shutdown. Its all-layer/Engram
hashes are functional intermediate evidence, not a complete response.
The Attention comparison qualifies the pinned original-arithmetic memory
adapter. The untouched full reference kernel remains blocked by its shared
memory requirement; that limitation is not promoted to a pass.

A test-only follow-up, `8fff4f9`, mocks both imported memory observers and
checks refusal below the physical reserve. It changes no runtime source. All
four Ubuntu/Windows, Python 3.11/3.13
[CI environments passed](../results/dgx-spark/2026-10-21-s1-optimized/ci-test-followup.json).
The prior user-cancelled attempt has a separate
[derived controller summary](../results/dgx-spark/2026-10-21-s1-optimized/prior-user-stop.json),
not an invented runtime or cleanup receipt.

These timings identify further optimization work after S1. In the single call,
opening MoE operators took 27.36 seconds and opening projections 10.69 seconds;
HC and projection arithmetic together took 0.79 seconds. MoE opening includes
catalog/fixture validation, setup and uploads, so 27.36 seconds cannot be
attributed to catalog parsing alone. The 15.23-second MoE forward measurement
includes storage, verification, transfers and cleanup. Qualified immutable
metadata reuse, bounded operator/backend reuse, and verified projection
derivatives are hypotheses for the subsequent S2 latency phase. They are not
implemented or measured speedup claims. Practical quality, long context,
images and MTP remain unqualified.

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
qualify this reference adapter and corpus. At
`334ffd035daf8eecd523b1e3a5b9ca8c535d4bcf`, a new reference process and the
native Driver comparison both passed all 36 cases: final BF16 storage and all
five FP32 boundaries match the original-computation hashes exactly. Both
pressure/refusal/healthy-retry cases use the same golden boundaries. Native
cleanup reports zero owned device bytes, zero modules and no active owned
context. The immutable candidate bundle remains separately identified; its
external [native parity receipt](../results/dgx-spark/2026-10-20-s1-components/attention/native-validation-r01.json)
and [fresh reference](../results/dgx-spark/2026-10-20-s1-components/attention/reference-r02.json)
are source- and binary-bound component evidence. Plain/trace self-consistency
and the PC receipt cannot replace this gate.

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
  After fixing the library-version assumption, the first streamed-GEMM
  comparison passed all eight greedy IDs and every batch-two logit, but three
  batch-one logit arrays differed in FP32 storage. The largest absolute
  difference was `1.52587890625e-5`. The
  [complete failed comparison](../results/dgx-spark/2026-10-20-s1-components/head/compare-r02.json)
  and [saved-array diagnosis](../results/dgx-spark/2026-10-20-s1-components/head/difference-r02.json)
  remain visible; no tolerance was relaxed.
  The replacement candidate builds the 2,647,654,400-byte FP32 head once on
  device using 512-row host chunks, then uses the original full `M=129280`
  SGEMM shape. Forward execution rereads no head rows. The component ceiling
  is 3 GiB, within the full backbone's 5 GiB device ceiling, which in turn is
  inside the then-current 8 GiB shared host/device pool; the physical 32 GiB
  reserve still applied. The later S1 composition used the 12 GiB ledger
  described above. At `ca352867b8e7c511016a35fad4c665b977d3686e`, its fresh
  [24 exact comparisons](../results/dgx-spark/2026-10-20-s1-components/head/compare-r03.json)
  all passed, across eight generated inputs and the complete vocabulary.
  Native peak owned device memory was 2,657,249,280 bytes. Minimum sampled
  physical available memory was 42,901,000,192 bytes in native execution and
  42,650,394,624 bytes in reference execution; both processes cleaned up their
  owned resources. Device-head construction took 43.06 seconds. After the
  first 207 ms call, the remaining synthetic head calls took 11–18 ms each;
  those component timings do not measure model token throughput. Full
  [native](../results/dgx-spark/2026-10-20-s1-components/head/native-r03.json),
  [reference](../results/dgx-spark/2026-10-20-s1-components/head/reference-r03.json)
  and [archive hashes](../results/dgx-spark/2026-10-20-s1-components/transport-r04.json)
  preserve the original source and observed toolchain identities.
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
  failure result. Four streams reserve 256 MiB for transfer work. The completed
  acquisition used its own fixed-source Spark checkout, so integration edits
  could not silently change the code identity of that in-flight import.
- All 40 blocks, both original Engram tables, the streamed head and generated
  token feedback are connected through a lazy runtime. The explicit text probe
  prepares small local dense derivatives, then requires a complete EOS response
  and owned-resource cleanup. Its code and CPU control-flow tests do not count
  as successful full-model text generation.
  Before generation, current source-bound Attention oracle/native receipts
  must match the immutable bundle, its actually loaded binaries, the observed
  device/driver and both window/sharing Attention bindings. Missing, changed
  or foreign evidence refuses execution before the first model forward.
  Both Attention reference and native numerical gates were rerun successfully
  at the same `ca352867b8e7c511016a35fad4c665b977d3686e` source used by the
  passing cached-head comparison. The
  [native receipt for that source](../results/dgx-spark/2026-10-20-s1-components/attention/native-validation-r02.json)
  binds the [reference for that source](../results/dgx-spark/2026-10-20-s1-components/attention/reference-r03.json)
  and all six actual loaded binaries. That earlier source had not yet completed
  full-model generation; the optimized S1 response above is separate evidence.

CPU/mock tests check bounds, failure/refusal, cleanup, corruption rejection and
control flow. At `ca352867b8e7c511016a35fad4c665b977d3686e`, all four
[CI environments](https://github.com/shell-term/vesper/actions/runs/39369617214)
passed 681 tests: Ubuntu/Windows with Python 3.11/3.13, with one environment
dependent skip on Linux. Those tests did not qualify GB10 arithmetic or close
S1 by themselves. The later complete-response evidence and accepted integration
scope are recorded above and in #16. Subsequent work belongs to the separate
S2/S3 issues in the [Spark roadmap](spark-roadmap.md).
