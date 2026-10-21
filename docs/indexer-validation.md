# Native indexers and atomic index sharing

> **Recorded PC evidence.** Results, requirements and next-step wording below
> describe this report's experiment/profile. They do not set the current
> development queue or require exact equality from every new candidate. See
> [current policy](quality-speed.md), [validation](validation.md) and
> [PC roadmap](roadmap.md). Raw evidence and original scope are unchanged.

The original-weight indexers for layers 2, 20 and 24 now pass their bounded
component and sharing-frame reference gates on the RTX 4090. All **261 component
cases and 99 successful frames** match at **11,189 boundaries / 105,972,523 storage
elements**, with zero differences. An independent process with neither Torch nor
TileLang installed reproduces every recorded boundary hash.

This qualifies index arithmetic and the named source/consumer relationships.
The subsequent [connected compressed-Attention probe](compressed-attention-validation.md)
now joins profiles 2/20/24 and compressed-KV sharing 20-to-24 at context <=64.
Larger-context integration, other layer profiles, Engram, full-model quality
and generation speed remain unfinished. Issue #7 was still open at this component gate; see the later
[block-operation acceptance audit](block-ops-qualification.md). M2 remains open. No retraining or new lossy quantization is
part of this implementation: the fixtures retain the released weights/scales.

## Original data and execution profile

The checkpoint and original inference sources are pinned to revision
`1bae0e1088993b9b82de08f59de5be5a34e42170` of
[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170/inference).
Only 13 indexer tensors are fetched, totalling **16,989,696 original bytes**:

- Layers 2/20/24: FP8 E4M3 `wq_b.weight`, per-32 E8M0 scales and BF16
  `weights_proj.weight`.
- Key owners 2/20: BF16 `wk.weight` and learned BF16 `k_norm.weight`.
- Layer 24 computes its own queries/weights and consumes layer 20's keys and
  candidate mask. It has no separately fetched key weights.

Range requests are at most 8 MiB; no full shard or checkpoint is downloaded.
Fixture headers, dimensions, dtype, byte counts and finite encodings are checked
before array allocation, followed by per-tensor SHA-256 checks. Original weights
and downloaded source files remain ignored local artifacts.

The registered native profile uses SM89, NVRTC 12.4, cuBLAS 120402, default math
and the registered default workspace policy. Its source SHA-256 is
`aaeade922ba2f4efb56e23f3320d8f45716e3d72aab0cefef908463b7f0d63e5`;
its compiled binary SHA-256 is
`411f05580290d098d830bac502f9c6b1c7c98eafc12f65628823a9fdbc3d277f`.
The existing normalization binary is unchanged; width-128 normalization and
rotation are newly admitted and compared here. Other compiler/library profiles
are refused pending their own qualification.

The optional reference uses Torch `2.14.1+cu130`, pinned git revision
`5c4886908584029761b579af026dcfb627c84070`, and TileLang 0.1.12. It requires the
original BF16 default dtype, cuBLAS backend and original reduction settings.
The reports record source/AST hashes, generated quantizer/query diagnostic CUDA,
and the existing TileLang launch-handle isolation. Downloaded ATen sources cover
BLAS layout, ReLU, reduction and CUDA top-k membership.

## Preserved arithmetic and selection

Key construction retains BF16 GEMM, learned RMS normalization, the original
compressed/YaRN rotation and **per-32 FP4 E2M1 / E8M0** quantize/dequantize.
Ratio-2 keys rotate at the first token of each completed group. These index
formats differ from the compressed-KV path's per-16 FP4/E4M3 scales; they must
not be interchanged.

Queries retain the original FP8 projection, per-32 activation/weight scales,
FP32 accumulation, BF16 boundary, rotation and FP4/E8M0 rounding. Head weights
retain their BF16 projection and BF16 scaling by 1/64. Score computation retains
BF16 dot output, BF16 ReLU and BF16 weighted products before the original FP32
head reduction and final BF16 score. The head reduction follows the pinned
ATen tree for one key and its four-accumulator order for multiple keys.

Selection preserves causal masking and the ordered output indices, with at
most 512 compressed positions per query. BF16 radix ordering distinguishes
positive and negative zero; replacing it with stable floating-point sorting
changes the selected set. The final position sort permits a simpler native
radix threshold/gather without changing that set in this qualified profile.

Layer 20 forms candidates by taking maxima of eight-position blocks, pinning
the latest reachable block to positive infinity, selecting at most 2,048 blocks
and discarding unreachable negative-infinity blocks. It exports this mask while
selecting its own indices from its unfiltered scores. Layer 24 applies the
exported mask to its own scores. The frame also checks selected-index handoff
from layer 2 to 3 and from layer 24 to 25.

The bounded API admits batch 1/2, initial prefill of at most eight queries,
single-query subsequent decode, and at most 16,392 compressed index positions.
This keeps selection within the pinned CUDA single-block gather profile, with
at most 16 top-k slices. It does not qualify the multiblock top-k path, every
possible admitted input or a full 131,072-token context. Key/query components
include rotary positions near 131,071 independently of the bounded cache.

## Coverage and independent controls

| Operation | Component cases |
| --- | ---: |
| FP4/E8M0 code, scale and BF16 rounding | 23 |
| Width-128 RMS, both original learned norms | 24 |
| Original key projection, normalization and rotation | 30 |
| Original query and head-weight projections | 45 |
| BF16 score boundaries and FP32 head reduction | 40 |
| Causal, candidate and ordered top-512 selection | 99 |
| Total | 261 |

Rounding covers all 65,280 finite BF16 encodings, FP4 ties, signed zero,
subnormals, scale extremes and mixed scales. Extreme finite rounding inputs can
produce infinite BF16 outputs exactly as the original quantizer does; the
subsequent key/query score APIs still require finite inputs. Selection cases
include unreachable rows, ties and signed zero at 511/512/513 and
16,384/16,385/16,392 positions. At 16,385 positions there are 2,049 candidate
blocks, so retaining 2,048 actually excludes a block.

The 85 core frames cover 22 streams: batch 1/2, odd/even initial prefill lengths,
partial ratio-2 groups, zero/signed-zero/small/mixed-scale inputs, a continuous
32-token stream and three explicitly primed longer histories. The primed
histories repeat original first-key component outputs. They test addressing,
large-N arithmetic and candidate handoff, **not real autoregressive histories
or later-layer model activations**. All current frame inputs are generated.

Every frame compares 86 or 100 boundaries, depending on whether a new ratio-2
key is completed. The oracle separately executes unchanged `Indexer.forward`
and `Attention._compress_topk_idxs`, including the actual shared-state globals
and downstream consumers. Its nine controls per frame compare the three index
outputs, two downstream handoffs, two owner caches and shared keys/candidates.
Additional unchanged controls check 342 projections, 337 score reductions,
563 packed-versus-in-place quantizations and three original-key priming pairs.
These controls are additional to the 11,189 native/oracle comparisons.

Six deliberately altered controls all produce differences:

| Alteration | Changed storage elements |
| --- | ---: |
| Replace index FP4 with FP8 | 307 / 384 |
| Remove BF16 weighted-product rounding | 269 / 1,026 |
| Replace ATen head reduction with a sequential sum | 3 / 3 |
| Replace radix signed-zero membership with stable float ties | 511 / 512 |
| Omit newest-candidate-block pinning | 9 / 16,385 |
| Rotate a ratio-2 key at the last group token | 36 / 384 |

## State publication and resource scope

`NativeIndexSession` uses bounded host diagnostic key caches. It stages both
owners and the layer-24 consumer, then publishes position, keys, candidates and
shared indices only after the complete frame and a final headroom check succeed.
Returned diagnostics and downstream index arrays are copies. This serialized
diagnostic implementation does not yet provide a production GPU-resident cache
or a generation-throughput claim.

Five injected failures cover early host pressure, key readback, consumer-query
failure after both owners have staged, failure after consumer selection, and
late publication pressure. Each preserves both committed caches, the position
and all shared data; every exact retry passes. Their five initial frames and
five retries, plus four snapshot-stream frames, bring the total to 99.
Snapshot rollback restores both owners and all sharing metadata together.
Four stale, expired or duplicate accesses are refused.

The native components and one session use combined **480 MiB host allowances**;
the optional paired reference brings this to **992 MiB**. Both retain the
**32 GiB system-available floor**. The paired run's minimum sampled available
RAM was 129,428,426,752 bytes (about 120.5 GiB). These allowances are admission
bookkeeping, not process RSS or an OS memory limit. Caller-retained diagnostics,
Python, compiler/driver/module/context memory and optional framework overhead
are outside the counters. The full-engine memory governor remains separate work.

Owned GPU allocations are capped at 128 MiB for the indexer and 64 MiB for
normalization, with a 2 GiB free-memory floor at allocation. Their observed
individual peaks were 53,424,800 and 1,059,096 bytes; these are not a simultaneous
whole-process VRAM measurement. All owned buffers and context retains are
released after both runs. No other workload is stopped or GPU reset.

## Reproduce

Fetch the bounded original tensors once:

```sh
vesper fetch-indexer-fixture --directory artifacts/indexer-fixture
```

In the optional pinned Torch/TileLang reference environment with its existing
CUDA setup:

```sh
vesper compare-indexer --reference-directory artifacts/reference \
  --fixture artifacts/indexer-fixture/native-indexer.npz --device 0 \
  --output artifacts/indexer.json
```

Then in the ordinary NumPy-only environment with the registered CUDA Toolkit:

```sh
python examples/recheck_indexer.py \
  --reference-report artifacts/indexer.json \
  --fixture artifacts/indexer-fixture/native-indexer.npz \
  --output artifacts/indexer-native-only.json
```

Replay binds exact report bytes, original fixture identity, source/compiler
profiles, all 360 input/state metadata records and every boundary hash. CPU
checks cover bounded layouts, malformed fixture headers before allocation,
pre-kernel refusal, atomic owner/consumer publication, snapshot identity,
allocation failure and tampered/incomplete evidence. The previous compressor
and complete layer-0 block native-only matrices also continue to pass.
All 168 local CPU tests pass; the wheel includes the five indexer modules,
native kernel and four license/notice files. All nine previous native binary
identities reproduce unchanged, alongside the newly pinned indexer binary.

- [Paired comparisons, complete goldens and unchanged-method controls](../results/2026-10-18-indexer/indexer.json)
- [Independent native-only replay](../results/2026-10-18-indexer/native-only-recheck.json)
- [Summary, raw-file hashes and resource observations](../results/2026-10-18-indexer/summary.json)

The [connected follow-up](compressed-attention-validation.md) now qualifies
complete Attention output/cache state for profiles 2/20/24, including
compressed-KV/key/candidate sharing 20-to-24, in a generated-input context <=64.
The subsequent [long-context report](long-context-validation.md) covers window
wraps, top-512 truncation and candidate pruning through complete Attention output.
Next: other indexer sources (8/14/28/32/36) and complete-block integration. Larger batching, distributed
execution, Engram and full-model logits/sampling/quality/performance remain
separate gates.
