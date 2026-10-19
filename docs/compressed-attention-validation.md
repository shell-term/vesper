# Connected original compressed Attention

Original Attention profiles **2, 20 and 24** now connect query/KV/output
projections, learned normalization, compressed/YaRN rotation, the sliding window,
original compressors, indexers and sparse Attention. Layer 24 consumes layer 20's
actual compressed KV, index keys and candidate mask. All **103 frames / 309
complete Attention-layer calls** match at **31,841 boundaries / 282,154,978
storage elements**, with zero differences. Independent replay passes with
neither Torch nor TileLang installed or imported.

This is a bounded fidelity gate: one rank, batch 1/2, initial prefill at most
three queries, subsequent one-query decode and at most **64 tokens**. Inputs are
generated independently at each layer's `Attention.forward` boundary. These are
three Attention profiles, not three connected Transformer blocks or captured
later-layer activations. Other layers, larger contexts, Engram, full-model
quality/logits/sampling and generation speed remain unfinished. Issue #7 and M2
remain open.

The subsequent [long-context probe](long-context-validation.md) now covers
continuous 1,152/260-token generated-input streams, window wraps, actual top-512
truncation and candidate pruning with explicit synthetic 16,384-position history.
The 103-frame evidence below remains the original bounded baseline.

## Original data and source controls

The model and inference sources remain pinned to revision
`1bae0e1088993b9b82de08f59de5be5a34e42170` of
[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170/inference).
The bounded fixture adds **39 original tensors / 380,229,120 bytes** for the
three Attention profiles: five FP8 projection matrices and their E8M0 scales,
two learned BF16 norms and one FP32 sink per layer. Requests are at most 8 MiB;
no full shard or checkpoint is downloaded. The five compressor tensors and
thirteen indexer tensors are reused from the earlier probes. No weight
requantization or retraining is introduced.

The registered CUDA projection/compression/indexer/normalization binaries remain
unchanged, with NVRTC 12.4, SM89, cuBLAS 120402 and default math/workspace policy.
The three grouped `wo_a` conversions each match the unchanged original
`convert.py` conversion and have distinct pinned BF16 hashes. Native projections
now carry an explicit layer identity; the existing layer-0 default is retained.

The optional reference uses Torch `2.14.1+cu130`, git revision
`5c4886908584029761b579af026dcfb627c84070`, and TileLang 0.1.12, with the original
BF16 default dtype and registered BLAS/reduction settings. Nine original method
ASTs, the grouped conversion, quantizer/query diagnostic CUDA and pinned
ATen BLAS/reduction/top-k/softmax sources are checked. The report retains the
complete live qualification of the official-arithmetic Attention memory adapter.

Untouched full-shape sparse Attention still requests too much shared memory on
the 4090. Both control and candidate explicitly use the previously qualified
[memory adaptation](attention-memory-validation.md): the reference retains
literal official PTX arithmetic/control and changes Query bit transport; the
native candidate uses its registered register-Query binary. Original
`Attention.forward` and its compressor/indexer/window methods are unchanged.
This is not a newly executable untouched full Attention kernel.

## Connected arithmetic and sharing

The chain retains original FP8 projection/scaling and BF16 boundaries, learned
query/KV normalization, compressed/YaRN query and window-KV rotation, original
post-RoPE window FP8 rounding, sparse Attention, inverse rotation, grouped
output rounding/BF16 math and final FP8 output projection.

Source layer 2 retains ratio-2 FP32 projection, pair softmax/pooling and partial
group state; source 20 retains ratio-1 BF16 projection. Both normalize their
compressed latent. Index keys consume a copy of that **pre-RoPE latent** before
its Attention-cache rotation and per-16 FP4/E4M3 rounding can overwrite it. The
indexer retains its separate per-32 FP4/E8M0 format, BF16 score boundaries and
original reduction/selection semantics. Group positions remain the first token
of each completed group.

Layer 24 computes its own query, head weights, window KV and output, while
reading layer 20's compressed KV, index keys and candidate mask. Source 20
exports the candidate mask without applying it to its own scores. Window and
compressed payloads concatenate in the original order, and compressed indices
retain the original offset: prefill chunk length initially, 128 during decode.
Source 2-to-3 and 24-to-25 selected-index diagnostic handoffs are retained; full
Attention execution for consumers 3/25 is outside this probe.

At an empty first ratio-2 group the unchanged full forward preserves its
empty-index guard. The diagnostic chain also records unconsumed index-query
boundaries there. Numerical state/output agreement does not imply an identical
operation count or generation throughput.

At 64 tokens, selection reaches 64 compressed positions and the combined index
list reaches 192 entries. This connects the two sources across three sparse
Attention tiles. It does **not** exercise top-512 truncation or actual pruning
of 2,049 candidate blocks inside the complete Attention path. Those operations
have separate [indexer evidence](indexer-validation.md); their larger-context
integration remains required. The window does not wrap in this corpus.

## Corpus and independent checks

The 89 core frames cover six normal streams (batch 1/2, prefill 1/2/3 followed
by two decodes), three zero/signed-zero/bounded-mixed-scale streams, and one
continuous generated-input 64-token stream. Five fault-test prefills, five exact
retries and four snapshot-stream frames bring the total to 103. Each frame
compares 289 or 323 intermediate/output boundaries, depending on whether a
ratio-2 group completes.

Every frame is also controlled by **28 comparisons** from separately executed
unchanged original `Attention.forward`, `_window_kv`, `_compress_kv`,
`_compress_topk_idxs`, `Compressor.forward` and `Indexer.forward` bodies. These
check all three final outputs, window caches, actual Attention queries,
concatenated KV/indices, pre-output Attention results, both compressed and index
owner caches, partial-group state and actual shared globals. Queries and
normalized latents come from these layers' original projections; they are not
supplied independently as in the earlier isolated indexer probe.

Additional controls check 1,236 original-linear FP32-diagnostic/BF16 projection
boundaries, 309 index-query projections, 309 index reductions, 473 index
packed-versus-in-place quantizations and 164 compressed-KV packed/in-place
quantizations. These are separate from the 31,841 native/oracle comparisons.

Four intentionally incorrect connection choices all produce differences:

| Alteration | Changed storage elements |
| --- | ---: |
| Omit compressed indices from Attention | 65,405 / 65,536 BF16 Attention outputs |
| Supply source-2 KV to consumer 24 instead of source 20 | 65,369 / 65,536 BF16 Attention outputs |
| Use window frequencies for the compressed query | 59,131 / 65,536 BF16 Attention outputs |
| Build index keys from post-RoPE rounded latents | 72 / 256 BF16 key values |

These controls establish sensitivity to the changed boundaries, not full-model
quality. The last control checks index keys; at this short context all reachable
compressed positions fit in top-k, so it is not a final-output-quality claim.

## Publication, snapshots and memory

All persistent state here is **bounded host diagnostic storage**. The connected
frame stages three windows, two compressors and partial groups, both key owners
and all sharing metadata. It publishes that state and position only after every
layer output, optional finalizer and a final available-memory check succeed.
GPU scratch is scoped to each operation; no GPU cache rollback is required.
The compressor's existing arithmetic was factored into a non-publishing stage
and its earlier complete matrix continues to pass.

Five injected failures cover early host pressure, index-key readback, the layer-24
Attention consumer, its final output after native computation, and late
publication pressure. Each preserves the position plus all 17 state/sharing
checks, releases temporary GPU buffers and retries exactly. Whole-frame snapshot
rollback and three expired/duplicate-handle refusals pass. Mutating every
returned diagnostic after a successful call leaves committed state unchanged.

Conservative combined native host allowances are **2,624 MiB**; including the
optional reference and its diagnostic frame gives **4,096 MiB**. All admissions
retain the **32-GiB system-available floor**. The paired/native-only runs sampled
at least 132.54 / 134.44 GiB available RAM. These are allowances and observations,
not process RSS or OS limits. Caller-retained diagnostics, Python,
framework/compiler/driver/module/context overhead and serialized JSON are
outside the counters. Fixture loading has its own 768-MiB admission allowance.
Full-engine governance remains #9.

Each projection component caps owned GPU allocations at 256 MiB and peaks at
202,367,744 bytes. The compressor/indexer/normalization/Attention components
retain their existing separate caps and a 2-GiB allocation-time free-memory
floor. The summary lists all individual peaks; their sum is not a measured
whole-process or simultaneous VRAM peak. All owned allocations and context
retains are released. No other workload is stopped, page file changed or GPU reset.

## Reproduce and inspect

Fetch the additional bounded fixture once; the previous compressor/indexer
fixtures and qualified native Attention bundle are reused:

```sh
vesper fetch-compressed-attention-fixture --directory artifacts/compressed-attention-fixture
```

In the pinned optional reference environment:

```sh
vesper compare-compressed-attention --reference-directory artifacts/reference \
  --fixture artifacts/compressed-attention-fixture/native-compressed-attention.npz \
  --compression-fixture artifacts/compression-fixture/native-compression.npz \
  --indexer-fixture artifacts/indexer-fixture/native-indexer.npz \
  --attention-bundle artifacts/native-attention-v1 --device 0 \
  --output artifacts/compressed-attention.json
```

Then in the ordinary NumPy-only environment with the registered CUDA Toolkit:

```sh
python examples/recheck_compressed_attention.py \
  --reference-report artifacts/compressed-attention.json \
  --fixture artifacts/compressed-attention-fixture/native-compressed-attention.npz \
  --compression-fixture artifacts/compression-fixture/native-compression.npz \
  --indexer-fixture artifacts/indexer-fixture/native-indexer.npz \
  --attention-bundle artifacts/native-attention-v1 \
  --output artifacts/compressed-attention-native-only.json
```

Replay checks exact report bytes, all three original fixture identities,
registered compiler/binary/conversion profiles, complete input/state metadata
and every boundary hash. All 177 CPU tests pass, including malformed headers
before allocation, operator identity, finalizer/headroom refusal, snapshot/reset
failure and tampered evidence. All ten prior native binary identities reproduce;
the earlier compressor and layer-0 HC/block native-only matrices still pass.

- [Paired comparisons, complete goldens and unchanged full-forward controls](../results/2026-10-19-compressed-attention/compressed-attention.json)
- [Independent native-only replay](../results/2026-10-19-compressed-attention/native-only-recheck.json)
- [Summary, raw-file hashes and resource observations](../results/2026-10-19-compressed-attention/summary.json)

The [long-context follow-up](long-context-validation.md) covers window wraps and
actual top-512/candidate pruning. Next: other source/consumer profiles and their
integration into complete Transformer blocks. Engram, full-model output/quality
and measured generation performance remain separate gates.
