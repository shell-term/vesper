# Complete blocks 20 through 25 with actual index refresh

This gate executes six consecutive original blocks without skipping layers:
20 -> 21 -> 22 -> 23 -> 24 -> 25. Each block receives the preceding complete
block's actual BF16 hidden state and returned FP32 HC pre coefficients.
Layer 20 creates compressed KV, keys and candidates. Layers 21/22/23 reuse its
selected indices. Layer 24 creates new index queries, weights, scores and
selections from its actual normalized input, using source-20 keys/candidates.
Layer 25 reuses layer 24's selection. Every layer keeps its own window, learned
projections, HC, trained routing biases, shared expert and routed experts.

The full original/native reports, native-only replay, staged expert plans,
exact file hashes and measurements are published in the
[result summary](../results/2026-10-19-refresh-block-chain/summary.json).
This is supporting evidence for the
[Issue #7 acceptance audit](block-ops-qualification.md).

## Measured result

All **40 successful chain steps / 240 complete block calls** match at
**59,600 boundaries / 415,577,840 storage elements**, with zero bit differences.
Separate unchanged original methods supply **2,160 Block controls**, **2,040
Attention controls** and **40 whole-chain state controls**. All ten failure/retry
paths and whole-chain snapshot restoration pass. Seven altered handoff/KV/index/
candidate controls are detected, including substituting source-20 queries or
scores for the refreshed layer-24 values. Independent native-only replay checks
the same complete matrix without Torch or TileLang installed or imported.

The six layers retain 26/24/28/25/24/30 selected original experts respectively.
The chain begins at a generated layer-20 HC boundary, not an original full-model
input. Passing these comparisons does not establish all-input equivalence or
model-wide output quality.

## Independent controls and corpus

The diagnostic oracle uses original operations. Separately, unchanged original
`Block.forward` bodies execute consecutively with their own original Attention
state and actual original MoE callbacks. They control each block's incoming
hidden/pre coefficients, Attention/MoE inputs, gate logits, ordered expert
IDs/weights, final output and next pre coefficients. Unchanged
`Attention.forward` controls check queries, windows, combined KV/indices, both
index sources, selected IDs, source compressed/key caches and candidate masks.
Whole-chain committed state is compared independently on every successful step.

Entry HC tensors are generated four-lane BF16 values with explicit FP32 incoming
coefficients. Every later input is computed by the actual preceding block.
Empty-start four-position streams run with both trained text/image-span biases
and repeat twice. Failure/retry and snapshot cases use the same actual route
corpus. This is batch one, one query per step and world size one. It is not a
captured upstream backbone, image-encoder evaluation or generation benchmark.

The complete chain checks source candidates and index refresh in its four-token
context. Actual top-512 truncation, long ring wraps and 2,049-to-2,048 candidate
pruning retain their separate
[longer-context Attention evidence](long-context-validation.md). Those tests
are not represented as six-block long-context executions.

## Atomic publication and resources

All six windows, source compressed/key caches, candidates, selected IDs, sharing
metadata and position form one unpublished candidate. Publication follows all
six MoEs, the final residual download and the last reserve check. A failure
after layer 24 cannot commit just layers 20..23 or expose a partial chain output.
Expert residency may change; routing and committed model state cannot.

Ten injected failures cover early pressure, three completed source experts,
the first handoff, refreshed-index score download, three completed layer-24
experts, the final consumer's Attention output/experts/residual download,
combined primary cap refusal and late publication pressure. Completed-expert
counts identify the actual failure point in every layer. Retrying checks all
boundaries again. Snapshot restore covers all six states and index/candidate
owners; consumed, foreign and reset-expired handles are refused. Diagnostic
copies cannot mutate committed state.

Native host allowances total **9,728 MiB**, including the retained-fixture
2,048-MiB allowance, all six 320-MiB MoE caps and the 1,024-MiB diagnostic frame.
The optional original comparison adds **6,144 MiB**. Admission retains the
32-GiB available-memory floor. Combined call-time owned GPU caps are **2 GiB
primary / 384 MiB helper**, covering every HC, projection, both MoE/cache tiers
and nested compressor/index norm device. Initialization, CUDA contexts/modules
and optional reference/compiler allocations are outside that GPU ledger.
These are owned-buffer ledgers and sampled headroom checks, not RSS or OS limits.
The original comparison peaks at **1,660,849,784 primary / 226,488,960 helper
owned bytes**, and sampled available RAM stays above **135,238,295,552 bytes**
(125.95 GiB). Complete cleanup records are included in both raw reports.
The whole-runtime 96-GiB governor remains
[Issue #9](https://github.com/shell-term/vesper/issues/9).

## Original artifacts and reproduction

Existing source-20/consumer-21 artifacts and stores are reused, as are original
Attention fixtures for layers 24/25. Only missing per-layer tensors and actually
selected experts are imported. Plans for 22, then 23, then 24, then 25 execute
every preceding complete block through the current layer's actual gate before
that layer's experts are fetched. A prefix plan's `layers` records exactly which
blocks executed; its `origin` identifies the common target-chain corpus. No
consumer FFN output is invented to plan a later layer.

Each layer's import is capped at 48 observed experts; execution still routes
over all 384 original choices and refuses a missing expert before expert I/O.
It never prunes or forces routes. All original tensor bytes/scales and original
quantization/rounding rules are retained. No full shard/checkpoint, retraining
or additional lossy quantization is used. Every range is <=8 MiB.

The complete chain binds **1,132 original tensors / 3,988,557,840 bytes**,
including 157 selected experts. Preparing the extension fetches 426,734,432
nonexpert bytes and 2,011,668,480 expert bytes, reusing 610,122,928 verified
nonexpert bytes and the existing 940,032,000 expert bytes. Reproduction still
requires the same original fixtures; no original weight payload is published.

Expert preparation uses at most two workers under a 224-MiB host envelope:
two 64-MiB fetch/validation allowances, then a 96-MiB store builder. Partial
failure never builds a native store; verified immutable expert files can be
reused on retry. Inference and comparison fetch no weights.

First prepare the original 20/21 artifacts/stores and qualified Attention
bundle using the [preceding pair gate](ratio1-block-pair-validation.md).

```sh
python examples/prepare_block_chain.py fixtures --topology-fixtures artifacts/topology-fixture --block-root artifacts
```

For each target layer 22, 23, 24 and 25, in that order, run the following two
commands with the corresponding layer number. Later planning requires all
preceding selected-expert stores to be complete.

```sh
python examples/prepare_block_chain.py plan --topology-fixtures artifacts/topology-fixture --block-root artifacts --attention-bundle artifacts/native-attention-v1 --target-layer 22 --plan artifacts/chain-plans/layer22.json
python examples/prepare_block_chain.py experts --topology-fixtures artifacts/topology-fixture --block-root artifacts --target-layer 22 --plan artifacts/chain-plans/layer22.json
```

Then compare and independently replay every original golden:

```sh
python examples/compare_block_chain.py --reference-directory artifacts/reference --topology-fixtures artifacts/topology-fixture --block-root artifacts --attention-bundle artifacts/native-attention-v1 --plan-directory artifacts/chain-plans --output artifacts/refresh-chain.json
python examples/recheck_block_chain.py --reference-report results/2026-10-19-refresh-block-chain/chain.json --topology-fixtures artifacts/topology-fixture --block-root artifacts --attention-bundle artifacts/native-attention-v1 --output artifacts/refresh-chain-native-only.json
```

Native-only replay requires neither Torch nor TileLang. It checks exact original
fixture/store identities, source/compiler/binary seals, the complete corpus,
ordered routes, every actual handoff and every intermediate/output/state hash.
The qualified original-arithmetic PTX memory adapter remains explicit; the
untouched full-shape official sparse Attention launch still exceeds 4090 shared
memory. Full artifact/tokenizer/Engram import, whole-runtime memory governance,
all-forty-layer numerical execution, logits/sampling, model quality and end-to-end
speed remain separate M2/M3 requirements.
