# Connected compressed Attention at longer contexts

Original Attention profiles **2, 20 and 24** pass **1,441 frames / 4,323 complete
Attention-layer calls**, comparing **440,997 boundaries / 13,429,385,909 storage
elements** with zero differences. Each frame also passes 28 controls from
separately executed, unchanged original full-forward methods. Independent replay
checks every boundary without Torch or TileLang installed or imported.

This extends the [64-token connected probe](compressed-attention-validation.md)
to window wraps, actual top-512 truncation in both compression ratios, and
candidate pruning inside the complete Attention path. Inputs are generated
independently at each Attention layer. These are not captured later-layer
activations, complete Transformer blocks or an autoregressive full-model run.
Issue #7 and M2 remain open.

## Continuous streams and explicit diagnostic history

| Corpus | Batch | Starting history | Calls | Final context positions |
| --- | ---: | --- | ---: | ---: |
| Continuous | 1 | Empty, prefill 3 then decode | 1,150 | 1,152 |
| Continuous | 2 | Empty, prefill 3 then decode | 258 | 260 |
| Primed | 2 | Explicit synthetic state at 1,024 | 3 | 1,027 |
| Primed | 1 | Explicit synthetic state at 16,384 | 8 | 16,392 |
| Primed | 2 | Explicit synthetic state at 16,384 | 8 | 16,392 |
| Fault/retry and snapshot | 1/2 | Explicit synthetic state at 16,384 | 14 | 16,387 at furthest advance |

The empty-start streams retain state across all calls and exercise **ten window
wrap events**. The batch-1 stream exceeds 512 compressed positions at ratio 2;
ratio 1 and consumer 24 exceed 512 earlier. Combined window/compressed indices
reach **640 entries**, traversing ten sparse Attention tiles.

The 16,384-position histories are deliberately synthetic diagnostic seeds. They
are built from banks of original-weight projections, learned normalization,
RoPE and the original two quantization formats, then repeated into bounded
caches. They are not claimed to be 16,384 preceding model executions. Each seed
records all eleven bank hashes and eleven state-array hashes; the original
component path rebuilds them independently and compares every state array.

For source 20, seed construction searches its original score bank for a pair
whose ranking reverses for consumer 24. It places the low-source/high-consumer
item in the first eight-position block. Once the newest block is inserted,
**2,049 blocks are pruned to 2,048**, excluding eight reachable positions. This
forces source-20 candidates to affect consumer-24 selection and Attention output.
The latest block stays eligible. This adversarial arrangement and its float64
pair search belong only to test construction; runtime scores, routing and
selection formulas are unchanged.

Four intentionally incorrect connections produce observable differences:

| Alteration | Changed storage elements |
| --- | ---: |
| Ignore source candidates in consumer selection | 512 / 512 selected IDs |
| Feed that unfiltered selection to Attention | 31,102 / 32,768 BF16 outputs |
| Omit the compressed-index window offset | 32,677 / 32,768 BF16 outputs |
| Reverse window index order | 8,192 / 40,960 FP32 QK diagnostics |

These establish sensitivity to the tested connections, not full-model quality.

## Arithmetic and admission contracts

The model/source revision remains `1bae0e1088993b9b82de08f59de5be5a34e42170`.
All original fixture bytes, projection conversions, method ASTs and registered
CUDA binary identities are unchanged. No weights are fetched for this extension
and no retraining or extra quantization is introduced. The same 39 Attention,
five compressor and thirteen indexer fixture tensors are reused.

Native execution retains the original projection, normalization, RoPE, KV
rounding, score and BF16 boundaries. The two separate FP4 formats remain
distinct. Layer 24 reads source 20's compressed KV, index keys and candidate
mask; its own query, head weights, window and output still use its own weights.
See the earlier connected report for the full arithmetic chain.

The optional original controls retain the qualified **official-arithmetic
Attention memory adapter** and its complete live qualification. Untouched
64-head/512-dimensional sparse Attention still exceeds 4090 shared memory.
The native candidate retains its previously registered register-Query binary;
this extension changes bounded input/allocation admission, not kernel arithmetic.

The new `long_context` profile is explicit. Default 64-token connected-frame,
128-position compressor and small sparse-Attention limits remain unchanged.
Long frame capacities are restricted to **1,152 or 16,392** positions, batch 1/2,
initial prefill at most three queries and subsequent one-query decode. Only
single-query 64-head/512-dimensional Attention admits up to 16,520 KV rows and
640 indices; larger multi-query calls and other head profiles retain their old
bounds. Diagnostic seeding is restricted to a fresh long frame, even positions,
validated finite/layout-compatible state and zero future cache entries.

Additional original diagnostic controls cover **17,337** linear projections,
4,341 index-query projections, 4,341 index sums, 6,522 index packed/in-place
quantizations and 2,181 compressed-KV packed/in-place quantizations. A failed
instrumentation control raises immediately. These checks are separate from the
440,997 native/oracle boundary comparisons.

## State, memory and complete evidence

State remains **bounded host diagnostic storage**. Five failures cover early
host pressure, index-key readback, consumer Attention, its final output after
computation, and late publication pressure. Every failure preserves position
and all 17 state/sharing checks, releases scratch and retries exactly. Snapshot
rollback restores all state at position 16,385; three stale/duplicate-handle
refusals and diagnostic-copy isolation also pass.

Combined native host allowances are **3,488 MiB**, or **5,728 MiB** including the
optional reference and its diagnostic frame. The native Attention operation
has a 128-MiB host allowance and 64-MiB owned device cap, with a measured owned
device peak of 37,665,024 bytes. The connected frame has a 1,024-MiB host
allowance. Other components retain their existing caps. All admissions retain
a **32-GiB system-available RAM floor** and device allocations retain a **2-GiB
free-memory floor**. Paired/native-only runs sampled at least **130.80 / 133.28
GiB** available RAM. All owned allocations and context retains were released.

Allowances and available-memory samples are not process RSS or OS limits.
Caller-retained diagnostics, Python, framework/compiler/driver/module/context
overhead and serialized evidence are outside the counters. Component device
peaks cannot be summed into a measured simultaneous process peak. Full-engine
memory governance remains #9; no other workload is stopped or GPU reset.

The paired JSON retains **every per-boundary golden**, with integer IDs into
the shared `boundary_names` registry. Each `boundaries` row contains
`[name_id, storage_elements, mismatches, expected_sha256]`. Per-frame expected
and actual hash roots, dtype/shape roots, observations and metadata are retained;
the original full-forward controls use a separate 28-name registry. No boundary
is truncated. Replay checks each hash, element count and dtype/shape, then emits
compact per-frame results bound to the exact reference-file SHA-256. A pinned
coverage digest rejects altered layouts, missing cases and changed counts.

All **185 CPU tests** pass, including explicit shape/context admission, malformed
or future/nonfinite seeds, copy/late-pressure refusal, snapshot identity and
tampered evidence. The old 103-frame connected matrix, 49 compressor components
and 432 state calls, and 84 HC cases/26 layer-0 block calls still pass native-only
replay. All ten prior compiled native binary identities reproduce unchanged.

## Reproduce

Reuse the earlier fixtures and qualified Attention bundle. In the pinned optional
reference environment:

```sh
vesper compare-long-context --reference-directory artifacts/reference \
  --fixture artifacts/compressed-attention-fixture/native-compressed-attention.npz \
  --compression-fixture artifacts/compression-fixture/native-compression.npz \
  --indexer-fixture artifacts/indexer-fixture/native-indexer.npz \
  --attention-bundle artifacts/native-attention-v1 --device 0 \
  --output artifacts/long-context.json
```

In the ordinary NumPy-only environment with the registered CUDA Toolkit:

```sh
python examples/recheck_long_context.py \
  --reference-report artifacts/long-context.json \
  --fixture artifacts/compressed-attention-fixture/native-compressed-attention.npz \
  --compression-fixture artifacts/compression-fixture/native-compression.npz \
  --indexer-fixture artifacts/indexer-fixture/native-indexer.npz \
  --attention-bundle artifacts/native-attention-v1 \
  --output artifacts/long-context-native-only.json
```

- [All paired goldens and unchanged original controls](../results/2026-10-19-long-context/long-context.json)
- [Independent native-only replay](../results/2026-10-19-long-context/native-only-recheck.json)
- [Summary, exact file hashes and individual resource observations](../results/2026-10-19-long-context/summary.json)

The subsequent [sharing-topology probe](attention-topology-validation.md) covers
all configured KV/index sources and eight ordinary consumers, in four independent
generated-input groups. Next: compressed Attention inside complete Transformer
blocks. Engram, full-model logits/sampling/quality and generation performance
remain separate unfinished gates.
