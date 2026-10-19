# Complete compressed Transformer block

This gate connects **original layer 2** through HC, RMS, compressed Attention,
the original gate, offloaded FP4 experts, the resident FP8 shared expert, ordered
MoE accumulation and the final residual. It extends the
[layer-0 block](layer0-block-validation.md) and
[Attention sharing-role](attention-topology-validation.md) gates.
Progress belongs to [Issue #7](https://github.com/shell-term/vesper/issues/7).

## Scope and fidelity

The corpus has 32 successful single-token block calls: sixteen calls across two
repetitions of four-position text/image-span streams, twelve successful warm-up
and retry calls for six injected failures, and four snapshot/rollback calls.
Every stream starts with empty state. The first token retains a partial
ratio-2 group and skips the indexer; subsequent calls emit compressed KV and
index keys and use the original sparse selection. No fabricated layer-3 input
or consumer Attention is evaluated to run this source block.

Inputs are deterministic generated BF16 tensors with four independent HC lanes
and nontrivial incoming FP32 pre coefficients. They are **not captured layer-2
activations from a full-model execution**. Returned pre coefficients belong to
the next layer, not the next token in this corpus. The pinned configuration
places Engram in layers 1 and 14, so layer 2 has no Engram operation.

Layer 2 uses its own six FP32 HC tensors, two BF16 block norm vectors, thirteen
Attention tensors, three compressor tensors, five indexer tensors, gate weight,
trained text/image biases, FP8 shared expert and 25 selected FP4 experts.
Layer-0 weights are never substituted. HC/Attention/MoE layer identities are
checked before block admission. The current complete-block registration accepts
only layers 0 and 2; it does not qualify other blocks by shape alone.

All existing native CUDA binary identities and the six-kernel sparse Attention
bundle are retained. The official-arithmetic PTX memory adapter remains an
explicit qualified oracle. Untouched full-shape sparse Attention still exceeds
4090 shared memory; this gate does not change that launch limitation.

## Independent original controls

Each native boundary is compared with original operations, including every
selected expert's thirteen stages and the original ordered accumulation. A
separate unchanged `Attention.forward` controls twelve output/cache boundaries.
A separate unchanged `Block.forward`, with its own Attention/cache state and
actual original MoE callback, controls final BF16 output, next-layer FP32
coefficients, window KV, compressed KV, index keys, selected IDs and both
compressor partial arrays. These eight controls run for every successful call.
Whole committed state, including non-array sharing metadata, is also compared.

All **8,793 boundary comparisons / 70,536,278 storage elements match exactly**,
along with 256 complete-block controls, 384 complete-Attention controls and
32 whole-state comparisons. The original and framework-free processes agree
on every boundary, dtype/layout, ordered expert decision and committed state.

The reference uses pinned model/kernel sources, PyTorch source/build, TileLang,
cuBLAS, compiler and binary identities. HC Sinkhorn tracing adds stores only;
removing those additions recovers the original arithmetic AST. Incorrectly using
the current FFN's own pre coefficients, and omitting compressed entries from
block Attention, must change the checked values.

An independent process without Torch or TileLang installed/imported replays all
32 cases against the full original goldens, including intermediate storage
hashes, layouts, routing, output and complete state. Reports are not abbreviated
to final-output hashes. See the [result manifest](../results/2026-10-19-compressed-block/summary.json).

## Publication and resources

Window KV, compressor partials, compressed KV, index keys, selected IDs, sharing
metadata and position form one bounded host diagnostic candidate. It is
published only after the MoE, final residual download and final memory check
succeed. No output is returned on failure. Cache residency can change on a
failed attempt without changing model routing or committed model state.

Six failures cover early host pressure, index-key download, failure after three
completed experts, final residual download, combined GPU-cap refusal after MoE,
and late publication pressure. Every failure preserves all committed state and
position, releases temporary GPU allocations and retries exactly. Snapshot
rollback reproduces every block boundary and refuses consumed handles.

The combined native allowances are **3,584 MiB**, including a 768-MiB retained
fixture admission and the 1,024-MiB diagnostic frame allowance. The paired run
adds a 2,048-MiB reference diagnostic allowance. All admissions retain the
32-GiB system-available floor. Owned primary-GPU buffers during block calls
have a combined **512-MiB cap**, including the compressor/indexer's nested norm
devices. Component initialization, context/module memory, compiler/framework
memory and process RSS remain outside that GPU ledger. Allowances and sampled
available memory are not OS-level limits. Full-engine memory governance remains
[Issue #9](https://github.com/shell-term/vesper/issues/9).

Both processes peak at **315,674,308 owned primary-GPU bytes (301.05 MiB)**.
Sampled system-available RAM stays at least **101.08 GiB paired / 102.99 GiB
native-only**. All native owned buffers, frame reservations and context retains
are released. The [summary](../results/2026-10-19-compressed-block/summary.json)
records exact byte counts and complete cleanup evidence is in both reports.

Only 43,312,088 new control/gate/bias/shared bytes and 470,016,000 selected expert
bytes are fetched. The 142,936,832 Attention/compressor/indexer bytes are reused
from the existing layer-2 fixture. Every weight request is at most 8 MiB;
comparison and replay do no weight fetching. No full checkpoint/shard, extra
lossy quantization, retraining, expert pruning or cache-dependent routing bias
is used.

## Reproduction

First prepare the pinned source, exact reference runtime, registered
[Attention bundle](native-attention-validation.md) and
[topology fixtures](attention-topology-validation.md). Execution uses the
NumPy/native environment; original comparisons additionally need the exact
reference environment described in those reports.

```sh
python examples/prepare_compressed_block.py fixtures --fixtures artifacts/compressed-block-fixture
python examples/prepare_compressed_block.py plan --fixtures artifacts/compressed-block-fixture --topology-fixtures artifacts/topology-fixture --attention-bundle artifacts/native-attention-v1 --plan artifacts/compressed-block-plan.json
python examples/prepare_compressed_block.py experts --fixtures artifacts/compressed-block-fixture --plan artifacts/compressed-block-plan.json --expert-directory artifacts/block2-experts --store artifacts/block2-store
python examples/compare_compressed_block.py --reference-directory artifacts/reference --topology-fixtures artifacts/topology-fixture --fixtures artifacts/compressed-block-fixture --attention-bundle artifacts/native-attention-v1 --plan artifacts/compressed-block-plan.json --store artifacts/block2-store --expert-directory artifacts/block2-experts --output artifacts/compressed-block.json
python examples/recheck_compressed_block.py --reference-report results/2026-10-19-compressed-block/block.json --topology-fixtures artifacts/topology-fixture --fixtures artifacts/compressed-block-fixture --attention-bundle artifacts/native-attention-v1 --store artifacts/block2-store --output artifacts/compressed-block-recheck.json
```

Planning runs the real HC/Attention/RMS prefix and original gate before any
expert import. It records actual ordered IDs, weights and inputs for all eight
distinct conditions, refusing unions above 48 experts. Preparation fetches only
that union; execution refuses missing experts before expert I/O. Existing
fixtures are hash/layout/identity validated, and existing plans/stores are never
overwritten. Published sources and results contain no model weights or private
paths.

## Remaining integration

Source/consumer sharing across **complete consecutive blocks**, other compressed
profiles, all forty backbone weight bindings, original upstream hidden states,
Engram, logits/sampling, larger complete-block context, batching, model quality
and generation speed remain unfinished. This is a four-position source-block
gate, not a full-model or performance result. Issue #7 was still open at this component gate; see the later
[block-operation acceptance audit](block-ops-qualification.md). M2 remains open.
