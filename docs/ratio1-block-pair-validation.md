# Complete ratio-1 blocks 20 and 21

This gate extends the complete-block handoff to the original ratio-1 KV source
at layer 20 and its selected-index consumer at layer 21. Layer 20 creates
compressed KV, index keys, selected IDs and candidate blocks on every token.
Layer 21 uses that source KV/selection and its own learned projections, window,
HC, trained gate biases, shared expert and routed experts.

The implementation uses the same registered-pair executor as
[blocks 2/3](block-pair-validation.md). The original layer-specific weights and
scales, BF16/FP32 boundaries, reduction order, HC pre coefficients and ordered
expert accumulation are preserved. No retraining or extra lossy quantization is
introduced. The 3080 Ti stores original expert bytes; model arithmetic runs on
the 4090.

## Result

All **36 successful pair steps / 72 complete block calls** match the original:
**19,152 boundary comparisons / 160,709,484 storage elements, zero differences**.
Every successful step also passes eighteen independently executed unchanged
`Block.forward` controls and nineteen unchanged `Attention.forward` controls:
648 block controls, 684 Attention controls and 36 whole-state comparisons.
The native-only process reproduces all intermediate/output/state boundaries
with neither Torch nor TileLang installed/imported. Eight failure/retries and
whole-pair snapshot restoration pass. Five altered handoff/KV/index/candidate
controls change the checked bits. Full reports, exact hashes and the original
source expert plan are in the
[result summary](../results/2026-10-19-ratio1-block-pair/summary.json).

## Independent original checks

The optional oracle first executes diagnostic original operations. Separately,
unchanged original `Block.forward` bodies run consecutively with their own
original Attention state and actual original MoE callbacks. Each block's
input, Attention/MoE inputs, logits, ordered expert IDs/weights, output and
returned pre coefficients are controlled. Unchanged `Attention.forward` calls
also control both windows, queries, combined KV/indices, compressed caches,
index keys, selected IDs and the source-20 candidate mask.

The generated entry input has four independent HC lanes and explicit FP32
incoming coefficients. Consumer-21 input is the actual complete source-20
output, including its returned pre coefficients. Empty-start four-position
streams use both trained text/image-span router biases and repeat twice.
This is a generated-entry complete-pair check, not upstream model execution or
an image-encoder evaluation. The candidate mask is checked in this short
context; candidate/top-512 truncation remains separate longer-context
[Attention evidence](long-context-validation.md).

## Publication and resources

Both windows, source KV/index caches, candidate mask, selected indices, sharing
metadata and position are one unpublished candidate. Both complete MoEs and
final residual downloads must finish, and the last headroom check must pass,
before publication. A consumer failure cannot commit just the source block.
Source/consumer expert failures, consumer-entry/output/download failures,
combined primary cap refusal and early/late memory pressure are injected.
Snapshot restore includes candidate state and refuses foreign, expired or
consumed handles; inspection returns isolated diagnostic copies.

Native host allowances total 4,832 MiB, including 1,024 MiB retained fixtures;
the optional reference adds 3,072 MiB. Admissions preserve the 32-GiB system
available-memory floor. Combined call-time owned GPU caps are 768 MiB primary
and 128 MiB helper, including both MoEs and nested norm devices. These are
reservation ledgers and sampled headroom checks, not process RSS or OS limits.
Initialization, reference/compiler allocations and CUDA contexts/modules are
outside the call-time GPU ledger. The full-engine 96-GiB governor is still
[Issue #9](https://github.com/shell-term/vesper/issues/9).

Both comparisons peak at **571,013,000 primary / 75,496,320 helper owned bytes**.
Sampled available RAM remains at least **129.53 GiB paired / 131.74 GiB
native-only**. All native owned buffers, context retains and frame reservations
are released. The ten native CUDA binary identities and six sparse-Attention
bundle kernels are unchanged. Existing complete 2/3 goldens, all four Attention
groups/sixteen profiles, standalone layer 2 and layer-0/HC goldens remain exact.

## Bounded original artifacts and reproduction

Reuse the pinned [Attention fixtures](attention-topology-validation.md) for
layers 20/21 and the same qualified six-kernel Attention bundle. First fetch
the source control/gate/router/shared tensors, plan its experts from actual
HC/Attention input, and import only its selected expert union. Next execute the
complete source block and consumer prefix through the original consumer gate
before fetching consumer experts. Planning never invents a consumer FFN output
or forces a route. Missing experts are refused before expert I/O during execution.
Range requests remain at most 8 MiB; no full shard/checkpoint is downloaded.

The two-stage plans select **26 source-20 / 24 consumer-21 experts**. The gate
uses **367 original tensors / 1,291,093,168 bytes**, reusing 264,436,992 existing
Attention bytes. Only 86,624,176 control/gate/bias/shared bytes and 940,032,000
selected expert bytes are fetched. Comparison and native-only replay fetch no
weights. Source and consumer import plans reject cross-layer identities before
expert fetching. The published source plan reproduces the source routes in all
eight pair-planning cases.

```sh
python examples/prepare_compressed_block.py fixtures --layer 20 --fixtures artifacts/block20-fixture
python examples/prepare_compressed_block.py plan --layer 20 --fixtures artifacts/block20-fixture --topology-fixtures artifacts/topology-fixture --attention-bundle artifacts/native-attention-v1 --plan artifacts/block20-plan.json
python examples/prepare_compressed_block.py experts --layer 20 --fixtures artifacts/block20-fixture --plan artifacts/block20-plan.json --expert-directory artifacts/block20-experts --store artifacts/block20-store
python examples/prepare_block_pair.py fixtures --source-layer 20 --source-fixtures artifacts/block20-fixture --consumer-fixtures artifacts/block21-fixture
python examples/prepare_block_pair.py plan --source-layer 20 --source-fixtures artifacts/block20-fixture --consumer-fixtures artifacts/block21-fixture --topology-fixtures artifacts/topology-fixture --attention-bundle artifacts/native-attention-v1 --source-store artifacts/block20-store --plan artifacts/block20-21-plan.json
python examples/prepare_block_pair.py experts --source-layer 20 --source-fixtures artifacts/block20-fixture --consumer-fixtures artifacts/block21-fixture --plan artifacts/block20-21-plan.json --consumer-experts artifacts/block21-experts --consumer-store artifacts/block21-store
python examples/compare_block_pair.py --source-layer 20 --reference-directory artifacts/reference --topology-fixtures artifacts/topology-fixture --source-fixtures artifacts/block20-fixture --consumer-fixtures artifacts/block21-fixture --attention-bundle artifacts/native-attention-v1 --plan artifacts/block20-21-plan.json --source-store artifacts/block20-store --consumer-store artifacts/block21-store --source-experts artifacts/block20-experts --consumer-experts artifacts/block21-experts --output artifacts/block20-21-pair.json
python examples/recheck_block_pair.py --reference-report results/2026-10-19-ratio1-block-pair/pair.json --topology-fixtures artifacts/topology-fixture --source-fixtures artifacts/block20-fixture --consumer-fixtures artifacts/block21-fixture --attention-bundle artifacts/native-attention-v1 --source-store artifacts/block20-store --consumer-store artifacts/block21-store --output artifacts/block20-21-recheck.json
```

The recheck validates every registered original golden boundary and identity
before GPU execution, then uses NumPy/native CUDA without Torch/TileLang.
Comparison/recheck fetch no model weights. Full upstream histories, refreshed
index blocks, other complete roles, longer complete-chain context/batching,
all forty bindings, Engram, logits/sampling, model quality and generation speed
remain pending. These remain M2 integration gates. The later
[six-block chain](refreshed-block-chain-validation.md) covers index refresh
with actual intervening-layer execution. The original Issue #7 checks are
qualified separately in the [acceptance audit](block-ops-qualification.md).
