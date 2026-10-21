# Original artifacts, tokenizer and Engram

> **Recorded PC evidence.** Results, requirements and next-step wording below
> describe this report's experiment/profile. They do not set the current
> development queue or require exact equality from every new candidate. See
> [current policy](quality-speed.md), [validation](validation.md) and
> [PC roadmap](roadmap.md). Raw evidence and original scope are unchanged.

[Issue #8](https://github.com/shell-term/vesper/issues/8) preserves the original
five acceptance conditions. Its complete evidence is in
[the manifest](../results/2026-10-19-engram/manifest.json) and
[the acceptance audit](../results/2026-10-19-engram/acceptance.json).

The original tokenizer, N-gram addresses, selected raw rows, Engram arithmetic
and token-history publication pass their bounded reference gates. **110 successful
session steps / 220 layer calls / 6,820 typed and shaped intermediate comparisons
match bit-for-bit.** An independent process without PyTorch, TileLang,
Transformers or SymPy installed reproduces all 110 steps. No retraining or
additional quantization is used.

These are independent layers 1 and 14 with deterministic generated HC hidden
inputs, not consecutive backbone execution. Batch sizes 1/2, query chunks 1/2/3,
decode boundaries, text/image masks, signed zero, zero/scaled inputs, repeated
calls and three-token carried history are covered; contexts are at most eight
positions. Whole-model logits, sampling, quality and generation speed remain
unverified. [Issue #9](https://github.com/shell-term/vesper/issues/9) owns global
allocation policy; [Issue #10](https://github.com/shell-term/vesper/issues/10)
integrates the complete model. M2 stays open.

## Released identities and addressing

All source requests use `deepseek-ai/DeepSeek-V4.1-Flash-1017` at revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`. The import validates the raw SHA-256
and byte counts of the root/inference configs, original tensor index, tokenizer,
tokenizer config, `inference/engram.py` and original MIT license. All 48 bounded
safetensors headers are cross-checked against 96,085 index entries, exact
dtype/shape/contiguous extents, shard assignments and the released LFS whole-file
byte counts. The qualified catalog digest is
`f2595189d8a80ca4e1245a3cbfbaa32920ee1a8daee71376f81c3ac8f4d7e4b9`.
The 48 released whole-shard SHA-256 identities are separately pinned. This
metadata check does not claim to hash 510 GB of local original shards.

The native tokenizer uses the original Rust tokenizer JSON and runtime
`tokenizers==0.23.3`, with no automatic BOS/EOS insertion or decode cleanup in
this profile. Its complete 129,280-entry compressed map matches the unchanged
original implementation, giving 99,092 entries and digest
`26b9be2936d236a124ba318a998c417bc7032e3e92a3107fe98deee49f1dc496`.
Six encode/decode controls include case/space normalization, accents, Japanese,
partial UTF-8 tokens in the exhaustive map, and special tokens. The original
CPU and CUDA `NgramHashState` both match all 42 addressing/history cases.

Three carried token IDs per batch row reproduce the original active cache tail.
Masked tokens block older lookbacks with the original pad behavior. Original
PCG64 multipliers, prime bucket sizes, XOR hashing, offsets and signed int64
operations are retained. Layer totals remain 384,006,168 and 384,016,682 rows.
Original row numbers are never compacted before hashing or reference lookup.
Snapshots are owned, consumed on rollback, expire on reset and have a 16-handle
limit. The returned diagnostic state is a copy.

## Original arithmetic and the full-shape oracle

The native path retains original FP8 row bytes and E8M0 scales, embedding FP32
products and BF16 conversion, Wkv per-32 activation quantization, original
FP8 MMA accumulation, BF16 projection, separate per-HC normalization, q/k weight
products, signed square-root/clamp/sigmoid gating, text masking and BF16 residual.
All **31 intermediates per layer** have a dtype, shape, element count and both
original/native SHA-256 digests in [original.json](../results/2026-10-19-engram/original.json).

The oracle executes unchanged original `ParallelEngramEmbedding.forward`,
`linear` and `Engram.forward` method bodies. Original `F.embedding` sees the
literal full table shapes, original strides and original row addresses. CUDA
virtual memory reserves 202,761,043,968 address bytes while backing only the
verified requested pages. On this Windows/RTX 4090 profile, 2 MiB page granularity
requires **3,888,119,808 physical bytes** for 1,854 pages, below a 4 GiB VMM cap.
The unrequested pages remain unmapped. This is an oracle storage arrangement,
not additional GPU capacity or a native speed claim. `F.embedding` and original
row addresses are not replaced/remapped. All requested raw row bytes are also
checked through unchanged full-shape `F.embedding` before the forward comparisons.

Each of 220 successful layer calls independently checks the unchanged embedding,
the FP32 projection rounded to unchanged `linear`, and the final unchanged
`Engram.forward`. Diagnostic formulas are not the only numerical reference.
The oracle uses the registered PyTorch 2.14.1+cu130 build/git revision, TileLang
0.1.12 and BF16 default dtype. Its existing TileLang specialization launch-handle
workaround is recorded; arithmetic is unchanged. The native binary uses the
registered NVRTC 12.4 / SM89 profile, with `fmad=false` and `ftz=false`.

Seven deliberately changed controls are all detected: normalization across HC
copies, removing signed square root, dropping row scales, dropping the zero
clamp, omitting the mask, ignoring masked history and clearing history between
chunks. Chunk shapes are compared separately with the corresponding original
shape; the report does not assume GPU floating-point reductions are invariant
under batching.

## Bounded row storage and failures

The small fixture contains 504 requested rows per layer, totaling only 266,112
original weight/scale bytes. A 512-byte record preserves each row's 256 FP8 bytes
and eight scale bytes with a manifest of original shard offsets and three
digests. Direct 4 KiB reads bypass file caching; there is no model-wide file map
and no buffered fallback. A 32-row LRU per layer is bounded to 128 rows at most;
the row component has an 8 MiB admission budget including metadata and scratch.
Missing rows are refused before I/O. It does not silently fetch an entire table.

For an already-existing full checkpoint, `register_existing` streams each of
the 48 originals through an aligned 64 KiB direct reader, verifies its released
whole-file SHA-256, and creates padded block-digest sidecars. It neither downloads
nor duplicates original weights. Registration requires explicit execution and a
new integrity directory. A failed registration has no final manifest; partial
new sidecars require inspection before a new registration attempt.

`OriginalShardRows` uses the same original addressing and bounded LRU for any
valid row, with direct verified reads from registered shard blocks. The integrity
index is streamed and hashed once, with bounded per-page digests; subsequent
index pages and requested data blocks are verified before row publication. It
keeps at most two 64 KiB data blocks, plus aligned scratch and bounded index-page
digests. Reads crossing block boundaries and exact, unaligned EOF tails are
supported without mapping or buffered fallback. This full-shard backend is
implemented with address/ownership unit checks; **full local shard registration
has not been executed**. The actual streaming primitive is checked on six
small original tensors / 471,040 bytes / 192 requested ranges in
[direct-io.json](../results/2026-10-19-engram/direct-io.json). CPU checks also
exercise corrupted data/index pages, ownership refusal, EOF tails and cleanup.

Ten injected failures cover host pressure, row I/O/hash corruption, failure after
layer 1, layer-14 cache pressure, projection/final download failures, missing
rows, combined GPU-cap refusal and final publication pressure. All refuse an
output, preserve history/position, release temporary GPU buffers and reproduce
the original boundaries on retry. Snapshot rollback reproduces every boundary.
Byte-cache warmth may change after a failed attempt; model history does not.

Native admissions total **2,096 MiB** across retained fixtures, two operators,
two row caches and the session. The combined native GPU cap is **384 MiB** with
a measured owned-buffer peak of **319,934,118 bytes**; each component also has
a 256 MiB cap. The parent-cap test requests 100 MiB that fits an individual
component but is refused by the combined ledger before backend allocation.
Native GPU/context and reservation counters return to zero. Original VMM pages
and virtual reservations are released. The oracle has a separate 2 GiB host
admission; its tensor-framework allocations are separate from native ledgers.
All probes keep a **32 GiB system-available floor**. These are component
admissions/counters, **not process RSS or OS limits**; CUDA runtime/module and
other application memory are not inferred from them. Global ownership remains #9.

## Full import capacity plan

[import-plan.json](../results/2026-10-19-engram/import-plan.json) records actual,
read-only disk-capacity samples without private paths or drive identities.
The original checkpoint has **510,286,023,000 tensor bytes** and
**510,296,708,312 shard-file bytes**, including both original Engram tables
(202,758,032,400 tensor bytes), all experts, other backbone tensors, MTP and
vision. Nothing is pruned or requantized. A fresh import needs one original copy,
the verified metadata and detached headers, **249,200,640 bytes** of block
integrity indices, and a **64 GiB disk reserve**: approximately **579.3 GB free**.

The workspace volume lacks that free capacity; the sampled existing alternative
volume has about 1.30 TB free and meets the capacity plan. Its media type and
read performance are unqualified. No new SSD is needed for current probes, and
capacity evidence alone does not justify buying one. Full checkpoint download
or registration has not started. Broader storage throughput and end-to-end
measurements belong to later integration/performance work.

## Reproduction

Install `.[artifacts]` for the Rust tokenizer; the native path needs NumPy,
the NVIDIA driver and the registered CUDA 12.4 compiler profile. The optional
reference additionally requires the exact documented PyTorch/CUDA build,
TileLang, Transformers, tokenizers and SymPy versions. Downloaded original
sources, weights and generated binaries remain ignored local artifacts.
The measured Windows oracle uses CUDA 13 headers and NVRTC 13.4, with
`TILELANG_EXECUTION_BACKEND=nvrtc` and matching `CUDA_HOME`/`CUDA_PATH`. Its
optional `--reference-nvrtc` argument preloads that installation's NVRTC DLL
when Windows library discovery needs it. Keep the native CUDA 12.4 installation
available as well; the two compiler profiles are independent. Installed
TileLang template/CUTLASS paths must match that package.

```powershell
python -m pip install -e ".[artifacts]"
python examples/prepare_engram.py --metadata artifacts/import-source --fixtures artifacts/engram-fixture
python examples/compare_engram.py --metadata artifacts/import-source --fixtures artifacts/engram-fixture --reference artifacts/reference --output artifacts/engram-original.json
python examples/recheck_engram.py --reference-report results/2026-10-19-engram/original.json --metadata artifacts/import-source --fixtures artifacts/engram-fixture --output artifacts/engram-native.json
python examples/check_artifact_io.py --fixtures artifacts/engram-fixture --scratch artifacts/io-scratch --output artifacts/direct-io.json
python examples/import_artifacts.py plan --metadata artifacts/import-source --target . --output artifacts/import-plan.json
python examples/qualify_engram.py --output artifacts/issue8-acceptance.json
```

Preparation requires a fresh selected-row destination; completed control imports
can be verified and reused. The bounded download comprises about 315 MB of
control tensors plus selected rows and metadata. A full original table is never
downloaded by this command. The optional full-checkpoint `register` command
requires `--originals`, `--integrity`, `--metadata`, `--output` and `--execute`;
all originals must already exist. It streams substantial data and is deliberately
separate from preparation/comparison probes.

The acceptance manifest pins all complete reports and the four attribution/
license files. License text uses LF for hashing so Git's CRLF/LF checkout
conversion is portable; every other byte remains checked. Released model,
source-artifact and numerical identities always hash their raw original bytes.
Its validator rejects missing/substituted cases, boundaries,
failure evidence, native compiler identity, framework independence and original
acceptance wording. CPU CI tests those refusals without model downloads or GPUs.
See [third-party notices](../THIRD_PARTY_NOTICES.md) for adapted original
DeepSeek/PyTorch formulas and the optional tokenizer dependency.
