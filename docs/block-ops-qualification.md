# Issue #7 block-operation qualification

> **Recorded PC evidence.** Results, requirements and next-step wording below
> describe this report's experiment/profile. They do not set the current
> development queue or require exact equality from every new candidate. See
> [current policy](quality-speed.md), [validation](validation.md) and
> [PC roadmap](roadmap.md). Raw evidence and original scope are unchanged.

[Issue #7](https://github.com/shell-term/vesper/issues/7) concerns the original
Attention, indexing, KV, normalization, Hyper-Connections and residual operations
needed for faithful decode. Its original five acceptance statements are kept
verbatim in the sealed evidence manifest. The audit binds those statements to
the complete published original goldens and fresh independent native replays;
it does not infer completion from the number of commits or partial passes.

The [acceptance result](../results/2026-10-19-block-ops-qualification/qualification.json)
and [sealed evidence manifest](../results/2026-10-19-block-ops-qualification/evidence-manifest.json)
contain all seventeen evidence groups, exact artifact bytes/SHA-256, five checks
and scope. Every original family validator runs before the audit returns a pass.
Each fresh native replay is bound to the exact original report and records
Torch/TileLang as neither installed nor imported. CPU artifact validation and
actual GPU execution remain distinct evidence.

## Original acceptance and evidence

| Original acceptance | Evidence and practical scope |
| --- | --- |
| Preserve rotary/position behavior, compressed and sliding-window KV state, source-layer sharing and sparse index selection. | [RMS/RoPE](normalization-rope-validation.md), [window KV](window-kv-validation.md), [original projections/window Attention](attention-projection-validation.md), [compression](compression-validation.md), [indexers](indexer-validation.md), [compressed Attention](compressed-attention-validation.md), [long-context Attention](long-context-validation.md), all four [named source/reuse groups](attention-topology-validation.md), and the actual [20..25 complete chain](refreshed-block-chain-validation.md). |
| Preserve original activation/KV quantization and rollback/state semantics. | Original FP8/E8M0 activations, FP4/E4M3 compressed KV, post-RoPE window rounding, partial compressor groups, source/candidate ownership, injected refusal/cleanup/retries and snapshots, including atomic six-block publication after index refresh. |
| Qualify RMS normalization, Hyper-Connection mixes/Sinkhorn and residual boundaries against the pinned source. | Original FP32 reductions, RMS epsilon/learned weights, all 20 Sinkhorn iterations and next-sublayer/next-layer pre coefficients. The [84 HC cases and complete layer-0 block](layer0-block-validation.md), [layer 2](compressed-block-validation.md), [2/3](block-pair-validation.md), [20/21](ratio1-block-pair-validation.md) and [20..25](refreshed-block-chain-validation.md) compare original intermediates and unchanged full `Block.forward`/`Attention.forward` bodies. |
| Use bounded per-component fixtures and publish exact intermediate comparisons before connecting the full block. | Earlier component gates retain their original complete intermediate evidence and immutable source revisions. Original per-layer fixtures and actually selected experts are imported in <=8-MiB ranges; every native replay checks identities/bounds before accepting goldens. Full-block/chain gates build on those component comparisons. No full checkpoint or extra lossy quantization is used. |
| Cover chunk/prefill/decode boundaries; keep batching and distributed scope explicit. | Bounded window/compressor chunks, Attention prefill of 1/2/3 queries at position zero, single-query decode, batch 1/2 and incomplete/complete compression groups; empty-start streams through 1,152 Attention positions and explicitly synthetic histories through 16,392. Complete chains use batch one, one query per step and four positions. World size is one; distributed execution is unqualified. |

The sixteen named Attention profiles cover all four KV sources, all eight index
sources and ordinary selected-index reuse. The complete 20..25 chain additionally
propagates actual hidden states through every intervening block, refreshes
selection at 24 and reuses it at 25. The forty-layer role table is configuration
coverage, not a claim that all forty original weight bindings were numerically
executed. Longer complete-chain context and full-model input histories are also
not claimed. At 16,392 positions, actual candidate pruning is qualified only for
the separate source-20/refreshed-index-24 Attention test.

## Reproduce the completion audit

The audit loads one bounded report at a time under a 2-GiB owned host allowance
and the 32-GiB available-memory floor. It reads published metadata and goldens;
it does not load checkpoint tensors or run a GPU comparison. Changing a required
report, native replay, source/golden seal, coverage or acceptance statement is a
refusal, not an automatic new qualification. Its allowance is not a process RSS
or OS memory limit.

```sh
python examples/qualify_block_operations.py --output artifacts/block-operations-qualification.json
```

To run the native comparisons again, first prepare the bounded fixtures/stores
described in each linked gate, including the complete chain. Use a NumPy/native
CUDA environment with neither Torch nor TileLang installed:

```sh
python examples/recheck_block_operations.py --fixtures-directory artifacts --output-directory artifacts/block-operations-rechecks
```

This executes the existing per-family recheck tools in separate processes and
fetches no weights. It refuses to overwrite an existing replay; `--only` can
select a failed/missing family for explicit resumption. The acceptance audit
requires all seventeen sealed results, regardless of a partial runner selection.
Original comparisons still require the exact optional reference environment
specified by the individual gates. GPU results are local hardware evidence,
separate from the four CPU CI configurations.

## Remaining M2/M3 work

The five checks qualify block operations within their published scope. They do
not assert working generation or model-wide quality. The qualified official
Attention oracle explicitly adapts PTX memory access while retaining original
arithmetic/control; the untouched full-shape official kernel remains above the
4090 shared-memory limit. The helper 3080 Ti stores/hashes original bytes; its
experimental model-compute path remains unqualified.

Full tokenizer/artifact/Engram import remains
[Issue #8](https://github.com/shell-term/vesper/issues/8), whole-runtime owned/RSS
pressure behavior remains [Issue #9](https://github.com/shell-term/vesper/issues/9),
and full backbone/per-layer/logit/sampling integration remains
[Issue #10](https://github.com/shell-term/vesper/issues/10). Workload quality and
generation throughput remain [Issue #11](https://github.com/shell-term/vesper/issues/11).
The earlier next-step notes also mentioned this broader integration work; the
original five Issue #7 checks are not changed to absorb the whole M2 milestone.
M2 stays open after the block-operation issue is completed.
