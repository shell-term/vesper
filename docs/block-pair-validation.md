# Consecutive complete blocks 2 and 3

> **Recorded PC evidence.** Results, requirements and next-step wording below
> describe this report's experiment/profile. They do not set the current
> development queue or require exact equality from every new candidate. See
> [current policy](quality-speed.md), [validation](validation.md) and
> [PC roadmap](roadmap.md). Raw evidence and original scope are unchanged.

The complete original layer-2 block now feeds its actual BF16 hidden state and
returned FP32 pre coefficients directly into the complete original layer-3
block. Layer 3 reuses layer 2's compressed KV and selected indices, while keeping
its own window KV, learned projections, HC, gate and experts. Both blocks publish
their model state together. Progress is tracked in
[Issue #7](https://github.com/shell-term/vesper/issues/7).

## Result and input scope

All **36 successful pair steps / 72 complete block calls** match the original:
**18,447 boundary comparisons / 133,921,032 storage elements, zero differences**.
Every successful step also passes eighteen independently executed unchanged
`Block.forward` controls and twenty unchanged `Attention.forward` controls:
648 block controls, 720 Attention controls and 36 whole-state comparisons.
A separate process with neither Torch nor TileLang installed/imported reproduces
every intermediate, layout, ordered route, handoff, output and state boundary.
See the [complete reports and exact hashes](../results/2026-10-19-block-pair/summary.json).

The corpus repeats empty-start four-position streams with both text/image-span
routing biases, then exercises eight failure/retry conditions and whole-pair
snapshot restoration. Generated layer-2 inputs have four independent HC lanes
and explicit incoming coefficients, as in the
[standalone layer-2 gate](compressed-block-validation.md). Layer-3 inputs are
computed from the preceding complete block; no independent or fabricated
consumer input is supplied. This is not a captured upstream/full-model history,
image-encoder evaluation or generation benchmark. Maximum continuous context
in this complete-pair gate is four positions. The longer histories and actual
top-512/candidate truncation in earlier [Attention tests](long-context-validation.md)
remain separate component evidence.

## Connected execution and independent controls

The Attention frame requests a layer input only when that layer is ready.
After layer 2's Attention, it waits for the layer's HC/MoE/final residual before
requesting layer 3's normalized input. The same arithmetic path still supports
the earlier independent-input sharing frames. This new complete-block callback
profile explicitly admits layers 2/3, batch one and one token per step.

The native run preserves layer-specific weights/scales, 20 Sinkhorn iterations,
the preceding sublayer's pre coefficients, original RMS/RoPE and KV rounding,
trained router biases, ordered top-six decisions and ordered MoE accumulation.
Layer 3 executes no index query: it uses source 2's selected IDs. The first
incomplete compression group retains the original empty-index guard. Every
step records exact hidden/coefficient handoff hashes and identical reused
compressed KV/selected IDs.

The diagnostic oracle uses original operations. Separately, unchanged
`Block.forward` bodies run consecutively with their own original Attention
state, actual original MoE callbacks and the returned preceding-layer hidden
state/coefficient tensors. They control each block's input, Attention/MoE
inputs, logits, ordered IDs/weights, output and next pre coefficients. Unchanged
Attention controls additionally check queries, window/compressed KV, selected
IDs, sparse Attention output and source compressor/index state. Removing the
handoff coefficients, bypassing the source output, replacing shared KV with
consumer-window data, or dropping reused indices all change the checked bits.

All ten native CUDA binary identities and the six-kernel sparse-Attention bundle
remain unchanged. The explicit qualified official-arithmetic PTX memory adapter
is retained; untouched full-shape official sparse Attention still exceeds 4090
shared memory. A first integration attempt stopped before the first comparison
because helper-reserve inspection left the storage GPU selected. Reserve checks
now finish on the compute GPU, with a CPU regression check and the passing
paired/native-only runs. No model arithmetic or binary was changed to fix it.

## Atomic state and bounded resources

Both window caches, source compressor partials, compressed KV, index keys,
selected IDs, sharing metadata and position form one unpublished host candidate.
Publication follows both complete MoEs, final residual downloads and the last
headroom check. A failure returns no pair output, including no partial source
output. Expert residency can change without changing routing or model state.

Eight injected failures cover early pressure, source failure after three
experts, consumer entry after the source completed, consumer Attention output,
consumer failure after three experts, final consumer residual download, combined
primary-GPU cap refusal and late publication pressure. All preserve both blocks'
state and position and retry exactly. Snapshot rollback reproduces every
boundary; consumed, foreign and reset-expired handles are refused. Inspecting or
mutating a diagnostic copy cannot change committed state.

Combined native host allowances are **4,832 MiB**, including a 1,024-MiB retained
fixture admission, both 320-MiB MoE caps and the 1,024-MiB diagnostic frame.
The paired original comparison adds a **3,072-MiB** reference allowance. All
admissions retain the 32-GiB system-available floor. During pair calls, combined
owned GPU buffers have caps of **768 MiB primary / 128 MiB helper**, covering
both MoEs and nested compressor/index norm devices. Initialization, CUDA
contexts/modules and reference/compiler allocations are outside that GPU ledger.
Reservations and sampled available RAM are not process RSS or OS limits. The
full-engine 96-GiB governor remains
[Issue #9](https://github.com/shell-term/vesper/issues/9).

Both runs peak at **586,739,584 primary / 75,496,320 helper owned bytes**.
Sampled available RAM stays at least **130.04 GiB paired / 133.37 GiB native-only**.
All native owned buffers, frame reservations and context retains are released.
Complete counters and cleanup evidence remain in both reports.

## Original artifacts and reproduction

The pair uses **380 original tensors / 1,333,937,328 bytes** and 52 actually
selected experts: 25 from layer 2 and 27 from layer 3. The 783,007,960 bytes from
the previous source-block/consumer-Attention fixtures are reused. Only
43,312,088 new consumer HC/norm/gate/bias/shared bytes and 507,617,280 selected
consumer expert bytes are fetched, with every range at most 8 MiB. Comparison
and native-only replay fetch no weights. Execution refuses missing experts
before expert I/O; there is no forced route or pruned expert set. No full shard,
full checkpoint, retraining or additional lossy quantization is used.

First prepare the pinned sources, exact optional reference runtime, Attention
bundle, [topology fixtures](attention-topology-validation.md) and
[complete source-block artifacts](compressed-block-validation.md). The planner
executes the complete source block and consumer prefix through its actual gate
before importing consumer experts. It never invents a consumer FFN output.

```sh
python examples/prepare_block_pair.py fixtures --source-fixtures artifacts/compressed-block-fixture --consumer-fixtures artifacts/block3-fixture
python examples/prepare_block_pair.py plan --source-fixtures artifacts/compressed-block-fixture --consumer-fixtures artifacts/block3-fixture --topology-fixtures artifacts/topology-fixture --attention-bundle artifacts/native-attention-v1 --source-store artifacts/block2-store --plan artifacts/block-pair-plan.json
python examples/prepare_block_pair.py experts --source-fixtures artifacts/compressed-block-fixture --consumer-fixtures artifacts/block3-fixture --plan artifacts/block-pair-plan.json --consumer-experts artifacts/block3-experts --consumer-store artifacts/block3-store
python examples/compare_block_pair.py --reference-directory artifacts/reference --topology-fixtures artifacts/topology-fixture --source-fixtures artifacts/compressed-block-fixture --consumer-fixtures artifacts/block3-fixture --attention-bundle artifacts/native-attention-v1 --plan artifacts/block-pair-plan.json --source-store artifacts/block2-store --consumer-store artifacts/block3-store --source-experts artifacts/block2-experts --consumer-experts artifacts/block3-experts --output artifacts/block-pair.json
python examples/recheck_block_pair.py --reference-report results/2026-10-19-block-pair/pair.json --topology-fixtures artifacts/topology-fixture --source-fixtures artifacts/compressed-block-fixture --consumer-fixtures artifacts/block3-fixture --attention-bundle artifacts/native-attention-v1 --source-store artifacts/block2-store --consumer-store artifacts/block3-store --output artifacts/block-pair-recheck.json
```

The separate [ratio-1 follow-up](ratio1-block-pair-validation.md) qualifies
complete blocks 20/21 using this executor and their own original artifacts.
It does not join these two independent chains. Other complete roles,
longer complete-chain context, batching,
upstream hidden states, all forty backbone bindings, Engram, logits/sampling,
model quality and generation speed remain unfinished. Issue #7 was still open at this component gate; see the later
[block-operation acceptance audit](block-ops-qualification.md). M2 remains open. The earlier sixteen-profile Attention sharing gate, standalone layer 2
and layer-0/HC goldens all remain exact after this integration.
