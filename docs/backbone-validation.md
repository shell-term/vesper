# Forty-layer original text-backbone validation

The all-layer probe connects original embedding rows through **every block
0..39**, applies Engram before blocks **1 and 14**, and uses the original final
norm, full-vocabulary FP32 output head and temperature-0.7 sampler. It preserves
actual hidden tensors and HC pre coefficients between every consecutive block.
Two serial fixed-token calls match independent original execution at **10,504
typed/shaped boundaries / 54,214,575 storage elements**, including **258,560 full
FP32 logits** and **80 ordered router decisions**. Four late-failure/cancellation
conditions restore every committed state, and the healthy second-token retry
matches all **5,252** original boundaries. This advances Issue #10; natural
language generation and quality evaluation remain open.

The complete [original goldens](../results/2026-10-19-backbone/original.json),
[native replay](../results/2026-10-19-backbone/native.json),
[control bindings](../results/2026-10-19-backbone/controls.json) and
[file-bound summary](../results/2026-10-19-backbone/summary.json) are published.
They contain provenance, metadata and hashes, not model weight payloads.

## Scope

- Model `deepseek-ai/DeepSeek-V4.1-Flash-1017`, revision
  `1bae0e1088993b9b82de08f59de5be5a34e42170`.
- Rank one, batch one, one query per call, two fixed input IDs **`[0, 2]`**:
  original BOS and PAD vocabulary entries. They exercise arithmetic and state,
  not natural-language quality or an autoregressively generated continuation.
- Independent window caches at 0/1; shared compressed-KV sources at **2, 8, 14,
  20**; index sources at **2, 8, 14, 20, 24, 28, 32, 36**. Blocks 2..19 use the
  ratio-two path; blocks 20..39 use ratio one and source-20 candidate masks.
- All original gate weights, text correction biases, ordered top-six choices,
  routing weights, six selected experts, shared experts and accumulation orders
  are retained. Initial HC mixing selects stream zero as in the original.
- The real output head follows **block 39**. The earlier
  [four-block head](token-prefix-validation.md) was a diagnostic attachment.
- Native shared-state capacity is bounded to 1,152 positions. This two-call
  corpus does not qualify long whole-model histories, batched prompt prefill,
  vision or speculative DSpark/MTP execution. The original forward retains its
  configured auxiliary target layers 37/38/39 and records their returned hidden
  hash; that auxiliary output is not a native MTP comparison.

## Independent comparison

The original side executes the **unchanged `Transformer.forward` and
`Block.forward` function bodies**, extracted from hash-checked original source.
HC, RMSNorm, Attention, compressor/indexer, Engram, embedding, head and sampling
methods retain their original arithmetic. Original MoE methods load one layer's
selected experts at a time. The already qualified sparse-Attention memory
adapter preserves original arithmetic while changing Query storage for SM89;
this is not a claim that the oversized unmodified full Attention kernel launches
on the 4090.

Embedding and Engram lookups retain full original tensor shapes and row
addresses through CUDA virtual memory. Only prepared original pages have
physical backing; an unprepared ID is rejected before GPU lookup. Embedding
rows spanning allocation pages map every touched page. No compact-ID remapping,
zero substitute row, approximate lookup or retraining is used.

The unchanged Transformer returns actual full logits and sampled IDs. A separate
execution of the original tail must reproduce those logits; replaying original
sampling from the saved RNG state must reproduce its IDs and exact generator
advancement. Original goldens are saved before independent native replay.
The original identity is
`1746fa338af812fad3d981c3e6ccdc00db94094acc6f28e07644004c3b3afcb4`;
the complete two-case golden root is
`b56cdc2062210bd0388806a3057882c84c78b3ef47bc9eaab274c95ae3f1febc`.

Native replay runs without PyTorch, TileLang, Transformers or SymPy installed or
imported. Python HTTP access is disabled during native execution. Comparisons
bind dtype, shape, element count and raw storage hash for fixed per-layer HC,
norm, Attention, gate, output and pre-coefficient boundaries; every captured
selected-expert/shared-expert/aggregation intermediate is also compared. Query,
visible KV, selected/combined indices and window state are compared at each
compressed Attention layer. Both Engrams, all output/sampling arrays and every
committed KV/index/compressor/history/RNG state are included.

This retains original BF16/FP8/FP4/E8M0 storage and activation/KV rounding. It
does not introduce weight quantization, pruning, router changes, alternate
reduction precision or cache-dependent model behavior.

## Preparation and memory

All forty layers bind **1,251** original non-routed-expert, non-Engram-table
control tensors totaling **7,199,083,968 bytes**. The separately qualified head,
final norm, prepared embedding rows and required Engram rows are reused. A
checked catalog reader reuses local pinned index/header bytes and fetches only
bounded original payload ranges, with four workers and at most 8 MiB per range.
Local metadata reuse does not claim whole-shard SHA-256 registration.

The artifact planner can resolve a route and fetch its missing original experts
between computation stages. This avoids replaying the whole preceding prefix
for each missing layer. It is explicitly **preparation**, not independent
execution evidence. Public `NativeBackbone.step` has no resolver/download path:
it reads prepared immutable stores and fails if the actual original route is
unavailable. Each two-token expert union is limited to twelve experts per layer;
an ID-derived immutable store path prevents collisions between equal-sized
unions. Existing checked expert fixtures are reused.

The native probe admits at most **64 GiB**, the original comparison **32 GiB**,
and both require at least **32 GiB observed system-available RAM**. The normal
runtime defaults remain 96 GiB / 32 GiB. These are admission reservations and
observations, not process-RSS limits or OS quotas. One native MoE layer is opened
per call; this avoids keeping forty independent expert caches and is not a
performance optimization. Four compressor and eight indexer owners receive
larger bounded device allowances; their arithmetic binaries are unchanged.

The measured native process peaks were **7.93 GiB resident RAM / 19.90 GiB
private commit**, with **120.92 GiB minimum sampled system-available RAM**.
Its peak admission was **46.39 GiB**, distinct from those process measurements.
Peak explicitly owned device payloads were **9.73 GiB on the primary GPU /
36 MiB on the helper**; the helper stores original bytes and does not execute
model experts. All host/device admissions, leases and reclaimers return to zero,
and the watchdog stops after close. This bounded probe needs no additional SSD.

The original side separately records framework allocation peaks and complete
teardown. Its original frequency/index LRU tables and tensor owners are released
explicitly, followed by its process-local cuBLAS/cuBLASLt workspaces. The final
9 MiB workspace retention is cleared using the pinned PyTorch build's cleanup
hook; measured framework device allocation returns to its **zero-byte baseline**.
This does not reset the GPU or affect other clients.

## Failure and cancellation recovery

One token transaction owns window caches 0/1, the complete shared frame 2..39,
compressed token history and sampler state. Failed attempts publish no token.
The passing all-layer matrix covers failure after Engram 14, after layer 39's
expert computation, after real output sampling, and actual runtime cancellation
after sampling. Every case reaches its intended injection, restores the
complete previously committed state and retains no pending recovery. The final
healthy second-token retry matches every original boundary and RNG state.
Cancellation requires **`RuntimeCancelled`** and an actually cancelled runtime
before explicit resume; a `TypeError` cannot satisfy the test.

The earlier four-block cancellation test had caught `TypeError` rather than
`RuntimeCancelled`. Its [correction and full replay](token-prefix-validation.md)
are published separately, preserving the historical evidence. The corrected CPU
test requires the exact cancellation exception and forbids new work until
explicit resume; the all-layer test requires the same real cancellation state.

## Reproduction

Use the pinned source, metadata, native Attention package and small original
fixtures prepared by the preceding component gates. In a native environment:

```powershell
python examples/prepare_backbone.py --artifacts artifacts --controls artifacts/backbone-controls.json --output artifacts/backbone-plan.json
```

Then capture original goldens in the qualified optional reference environment:

```powershell
python examples/capture_backbone.py --artifacts artifacts --output original.json
```

After checking and sealing that complete original report, return to the native
environment:

```powershell
python examples/recheck_backbone.py --artifacts artifacts --reference original.json --output native.json
```

The preparation command needs existing bounded Engram rows for the specified
two-token corpus. It does not silently download the full checkpoint. Complete
general prompt import, longer generated continuations, deterministic
whole-model decode tests over broader histories, reasoning/coding/tool-use
evaluation and measured end-to-end speed remain required work under
[Issue #10](https://github.com/shell-term/vesper/issues/10) and
[Issue #11](https://github.com/shell-term/vesper/issues/11). M2 remains open.

A subsequent [bounded raw-text continuation](text-generation-validation.md)
now qualifies actual generated-token feedback for one five-call natural-language
corpus, with the real final head and complete original/native boundaries. It
preserves this BOS/PAD evidence. Broader histories, chat/chunk-prefill and scoped
reasoning/coding/tool-use evaluation remain open under #10.
