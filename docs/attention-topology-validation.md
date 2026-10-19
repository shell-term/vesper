# Original KV and index sharing across Attention profiles

All four configured KV sources and all eight configured index sources now have
original-weight comparisons, together with eight selected consumers that reuse
an earlier layer's selected indices. Across **four independent sharing groups,
2,224 frames / 8,896 complete Attention-layer calls**, all **684,616 boundaries /
13,174,870,167 storage elements** match exactly. Every frame also passes
separately executed unchanged original full-forward controls, totaling **79,508
additional comparisons**. Native-only replay covers every boundary without
Torch or TileLang installed or imported.

This extends the [earlier connected profiles](compressed-attention-validation.md)
and [long-context probe](long-context-validation.md). It qualifies named Attention
profiles and sharing roles with generated independent layer inputs. The four
groups are executed separately, with one KV owner each; this is not a complete
backbone pass or hidden-state handoff between groups. Complete compressed
Transformer blocks, layer-14 Engram, full-model quality and generation remain
unfinished. Issue #7 was still open at this component gate; see the later
[block-operation acceptance audit](block-ops-qualification.md). M2 remains open.

## Pinned ownership and new execution path

The pinned configuration defines KV owners **2, 8, 14, 20**, and index sources
**2, 8, 14, 20, 24, 28, 32, 36**. Layers 2–19 use ratio 2; layers 20–39 use ratio
1. Layers 0/1 use the window-only profile. The topology helper records all forty
backbone ownership intervals and rejects non-backbone identities. The bounded
executor accepts only these four registered groups:

| Group | KV/key owner | Computes an index query | Reuses selected indices |
| --- | --- | --- | --- |
| 0: 2 → 3 | 2 | 2 | 3 reads 2 |
| 1: 8 → 9 | 8 | 8 | 9 reads 8 |
| 2: 14 → 15 | 14 | 14 | 15 reads 14 |
| 3: 20 → 21 → 24 → 25 → 28 → 29 → 32 → 33 → 36 → 37 | 20 throughout | 20, 24, 28, 32, 36 | 21 reads 20; 25 reads 24; 29 reads 28; 33 reads 32; 37 reads 36 |

An index source after layer 20 computes its own query and head weights while
reading **source 20's keys and candidate mask**. Ordinary consumers compute
their own Attention query, window KV and output, but reuse the most recent index
source's selected IDs. They do not have indexer parameters or execute an index
query. Each frame records the computed-query layer list, selected-index owners,
KV owners and bit identity of every reused selection. Wrong/missing owners and
operator-layer identities are rejected.

The empty first ratio-2 group retains the original full-forward guard: no
`Indexer.forward` call occurs until a compressed position exists. Subsequent
source/consumer state preserves original ordering, causal bounds and the window
offset. The candidate path remains source 20 followed by later index sources;
no cache-dependent router or score modification is introduced.

## Original bytes and independent controls

Source and configuration remain pinned to model revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`. Bounded per-layer fixtures contain
**251 tensors / 2,109,723,648 original weight bytes** across the sixteen named
profiles. This run reuses **412,949,504 bytes** and fetches **1,696,774,144 new
bytes**, in requests of at most 8 MiB. No full shard or checkpoint is downloaded.
Group loading is bounded and sequential; all sixteen profiles are not resident
together. Original weights and downloaded sources remain outside the repository.

Every projection retains the released FP8 bytes/scales. Each grouped `wo_a`
conversion is independently compared with unchanged original conversion code
and bound to its own hash. Ratio-2 compressor parameters are promoted from the
original BF16 bytes into FP32 as in the reference; ratio 1 retains BF16.
Original learned norms, quantization formats, intermediate rounding and sparse
Attention arithmetic are preserved. All ten earlier native CUDA binary
identities and the six-kernel Attention bundle remain unchanged.

The optional oracle verifies the exact configuration bytes and unchanged method
ASTs, then executes original `Attention.forward`, compressor and indexer bodies
with the original weights. Full-forward ownership flags are derived directly
from the pinned configuration. Controls check outputs, windows, actual queries,
combined KV/indices, Attention results, source caches, ratio-2 partial state and
the published selection after **every** layer. Each two-layer group has 20
controls per frame; the final group has 83.

There are also **35,712** original linear diagnostic controls, **4,442** index
query controls, **4,442** index sum controls, **5,873** index packed/in-place
quantization controls and **1,431** compressed-KV packed/in-place controls.
Failures raise immediately. The pinned BLAS/reduction/source and generated
quantizer/query CUDA identities are retained.

The official-arithmetic **Attention PTX memory adapter remains explicit** and
its full live qualification is included in every group report. Untouched
64-head/512-dimensional sparse Attention still exceeds 4090 shared memory.
This extension changes weight/profile binding and host orchestration; it does
not make the untouched full kernel executable or qualify another arithmetic
implementation.

## Corpus, sensitivity and failure recovery

Each group runs 556 frames:

- Six batch-1/2 streams with prefill lengths 1/2/3 and two subsequent decodes: 18 frames.
- Empty-start continuous 260-position streams at batch 1 and 2: 516 frames, crossing two window wraps each.
- Two explicit synthetic histories at position 1,024, each followed through position 1,027: 8 frames.
- Five failure-test warm frames and exact retries: 10 frames.
- Snapshot, advances and exact rollback/retry: 4 frames.

Across the groups this gives sixteen continuous-stream wrap events. Seeded calls
exercise actual top-512 truncation with both ratios. Synthetic history uses
banks from each owner's original compressor/key projection and each layer's
original window projection. Those banks and all state arrays are independently
rebuilt by the original component path and compared. They are **not** preceding
full-model executions or captured later-layer activations.

For each group's final ordinary consumer, omitting the reused compressed
indices, replacing shared compressed KV with that consumer's window, or omitting
the index offset all change BF16 Attention output. The reports retain all twelve
negative controls with exact mismatch counts. Candidate-block pruning at 2,049
blocks is covered by the earlier long-context 20→24 probe; this <=1,028-position
corpus does not claim new pruning sensitivity for layers 28/32/36.

Five failures per group cover early pressure, key readback, final-consumer
Attention, its output after native computation, and late publication pressure.
All twenty failed attempts preserve the complete committed state and position,
release scratch and retry exactly. Whole-state hashes include every array's
dtype, shape and bytes plus all sharing metadata. Four snapshot restorations,
twelve stale/duplicate-handle refusals and mutation of every returned diagnostic
also pass.

## Resources and complete records

State remains a bounded **host diagnostic frame** with a 1,024-MiB allowance,
prefill <=3, batch 1/2 and subsequent one-query decode. Capacity is fixed at
1,152 positions; the largest seeded corpus position count is 1,028. Native
components retain their separate GPU caps and 2-GiB allocation-time free-memory
floor. All host admissions retain the **32-GiB system-available floor**.

Including the retained 2,048-MiB fixture allowance, combined native host
allowances are **4,896 MiB** for a two-layer group and **8,992 MiB** for the
ten-layer group. With the reference, allowances are **5,984 / 10,080 MiB**.
Fixtures have a separate bounded loading admission. The summary gives the
minimum available RAM samples and individual owned-device peaks for every run.
All owned allocations/context retains and frame reservations are released.
The paired/native-only runs sampled at least **129.25 / 131.42 GiB** available
system RAM.

These allowances are not process RSS or OS limits. Serialized records,
caller-retained diagnostics and Python/framework/compiler/driver/module/context
overhead are outside the counters. Individual GPU peaks cannot be summed into
a measured simultaneous process peak. Full-engine memory governance remains #9;
other workloads, page files and GPU state are left under the user's control.

Paired reports retain every per-boundary golden using the same complete compact
schema as the long-context probe. Separate name registries identify native
boundaries and unchanged original controls. Every replay checks values, element
counts and dtype/shape roots, plus exact reference-file bytes. Pinned source,
fixture/conversion and coverage identities reject missing cases, altered
ownership, sizes, layouts, binaries or weights. Retrieval/reuse counters may
differ when another user recreates identical original fixtures.

All **194 CPU tests** pass, including ownership intervals, operator identity,
fixture headers before allocation, finite/future seed checks, atomic copy and
publication refusal, snapshots and tampered evidence. Current source modules
and all four license files are checked in the built package; weights and
generated binaries are excluded.
The previous native-only 103-frame and 1,441-frame Attention matrices, 49
compressor components/432 state calls and 84 HC cases/26 layer-0 block calls
still pass after the explicit layer-binding changes.

## Reproduce

Fetch once, optionally reusing the three earlier verified fixtures:

```sh
vesper fetch-topology-fixture --directory artifacts/topology-fixture \
  --reuse-attention artifacts/compressed-attention-fixture/native-compressed-attention.npz \
  --reuse-compression artifacts/compression-fixture/native-compression.npz \
  --reuse-indexer artifacts/indexer-fixture/native-indexer.npz
```

In the pinned optional reference environment, choose group 0/1/2/3:

```sh
vesper compare-attention-topology --group 0 \
  --reference-directory artifacts/reference --fixture-directory artifacts/topology-fixture \
  --attention-bundle artifacts/native-attention-v1 --device 0 \
  --output artifacts/group-2-3.json
```

In the ordinary NumPy-only environment with the registered CUDA Toolkit:

```sh
python examples/recheck_topology.py --reference-report artifacts/group-2-3.json \
  --fixture-directory artifacts/topology-fixture --attention-bundle artifacts/native-attention-v1 \
  --output artifacts/group-2-3-native-only.json
```

[Summary, all eight reports, exact hashes and resource observations](../results/2026-10-19-attention-topology/summary.json)

| Group | Complete paired goldens | Native-only replay |
| --- | --- | --- |
| 0 | [2 → 3](../results/2026-10-19-attention-topology/group-2-3.json) | [Replay](../results/2026-10-19-attention-topology/group-2-3-native-only.json) |
| 1 | [8 → 9](../results/2026-10-19-attention-topology/group-8-9.json) | [Replay](../results/2026-10-19-attention-topology/group-8-9-native-only.json) |
| 2 | [14 → 15](../results/2026-10-19-attention-topology/group-14-15.json) | [Replay](../results/2026-10-19-attention-topology/group-14-15-native-only.json) |
| 3 | [20 through 37, named layers](../results/2026-10-19-attention-topology/group-20-21-24-25-28-29-32-33-36-37.json) | [Replay](../results/2026-10-19-attention-topology/group-20-21-24-25-28-29-32-33-36-37-native-only.json) |

Next: compressed Attention inside complete Transformer blocks. These results do
not qualify all forty layer weight bindings, Engram, larger prefill, distributed
execution, full-model logits/sampling/quality or generation speed.
