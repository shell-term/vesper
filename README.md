# Vesper

**Big models. Original weights. Your hardware.**

Vesper is building a native-precision Mixture-of-Experts inference engine across
GPU memory, system RAM and SSD. The first target is
[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
on heterogeneous consumer NVIDIA GPUs.

[日本語](README.ja.md) · [Artifacts and Engram](docs/engram-artifacts-validation.md) · [Block-operation acceptance](docs/block-ops-qualification.md) · [Six consecutive blocks](docs/refreshed-block-chain-validation.md) · [Validation](docs/validation.md) · [Architecture](docs/architecture.md)

[Roadmap](docs/roadmap.md) · [Issues](https://github.com/shell-term/vesper/issues) · [Milestones](https://github.com/shell-term/vesper/milestones)

## Status

This is a **verification runtime**, not a working text-generation engine. Native
experts now run on the RTX 4090 through SSD, bounded RAM and two GPU cache tiers.
Issue #7's five original block-operation checks are qualified by seventeen
sealed original/native evidence groups. The latest gate executes complete
blocks **20 -> 21 -> 22 -> 23 -> 24 -> 25**, including every actual hidden/HC
coefficient handoff, source-20 KV/candidates, index refresh at 24 and reuse at 25.
Tokenizer and bounded Engram now pass 110 steps / 6,820 exact intermediate
comparisons, including original CPU/CUDA hashes, required-row direct reads,
ten failure/retries and independent replay without PyTorch. All 48 original
artifact identities and actual full-import capacity are planned; the full
checkpoint is not downloaded. See [Issue #8's evidence](docs/engram-artifacts-validation.md).
Full-model loading, memory governance, logits/sampling, model-wide
quality and generation performance remain M2/M3 work. See the
[acceptance evidence and exact scope](docs/block-ops-qualification.md).

Direct comparisons with the pinned official implementation give:

| Candidate | Scope | Result |
| --- | --- | --- |
| RTX 4090, native FP8 path | One complete expert, 128 conditions | 0 / 655,360 final-output mismatches; all checked intermediate bits match |
| SSD / RAM / 3080 Ti storage / 4090 compute | Three distinct experts, 232 successful calls | 0 / 1,187,840 final-output mismatches; all checked intermediate bits match |
| GPU SHA-256 upload audits | Same three experts, another 232 calls | All thirteen checked stages match; unchanged model kernel |
| RTX 3080 Ti, experimental BF16 path | 128-row W1 slice, 8 inputs | 14 / 1,024 output mismatches; unqualified for inference |

The expert checks include both input projections, asymmetric clamps, weighted
SwiGLU, the BF16 boundary, down-input quantization and the output projection.
Independent format checks also pass on both GPUs. Other Transformer-block
profiles, full-model quality and generation speed remain unverified. The 3080 Ti currently stores original bytes;
all model arithmetic in the cache probe runs on the 4090. See the
[cache report and complete results](docs/cache-validation.md), [resident expert report](docs/expert-validation.md)
and the [initial transfer/storage measurements](docs/first-probe.md).

Single-row routing postprocessing now matches all ordered output bits in 244
official gate conditions and 232 additional selection/normalization conditions,
including the previously unsupported ties. This is scoped to the pinned CUDA
PyTorch top-k path; other batch shapes remain pending.
See [issue #3](https://github.com/shell-term/vesper/issues/3)
and the [routing report](docs/router-validation.md).

The original BF16 gate matrix is now promoted exactly to FP32 and projected
through native cuBLAS without a PyTorch runtime dependency. Across 307 generated
inputs and both text/image masks, all 614 comparisons with the unchanged full
gate match logits, ordered expert IDs and weights bit-for-bit. See
[issue #4](https://github.com/shell-term/vesper/issues/4) and the
[gate report](docs/gate-validation.md). Full-model quality remains unverified;
the separate connected real-activation gate is described below.

Original shared-expert arithmetic and ordered accumulation now pass their scoped
reference gates: 54 FP8 shared-expert cases, 38 aggregation cases, and 16 supplied-
route conditions combining six original FP4 experts with the shared expert.
All checked intermediate bits match. A native-only process reproduces all 108
case goldens without PyTorch installed. See
[issue #5](https://github.com/shell-term/vesper/issues/5) and the
[shared/aggregation report](docs/moe-arithmetic-validation.md). Original gate and
bounded-offload integration is now implemented: 30 connected calls on generated
inputs match all 108 boundaries each, and four pressure/transfer/I/O failure
checks refuse partial outputs and recover exactly. The **real-activation gate
now also passes**: four captured original prefix inputs, both gate-bias settings
and another 30 calls match all 108 boundaries. A process without Torch reproduces
the full matrix. See [Issue #6](https://github.com/shell-term/vesper/issues/6), the
[real-input report](docs/real-moe-validation.md) and the earlier
[generated-input report](docs/connected-moe-validation.md). M1's scope is the MoE
sublayer; the later layer-0 block comparison is described below. Text generation
remains unfinished.

An explicit Attention storage candidate now executes the actual 64-head,
512-dimensional shape on the 4090 with 74 KiB of shared memory instead of
138 KiB. It now matches a separately qualified official-arithmetic reference
that retains the original PTX arithmetic/control instructions and adapts only
memory access. Untouched full-shape kernel launch still exceeds the device
limit. Qualified capture is explicit and includes its complete evidence/hash.
Head partitioning was rejected after intermediate differences. See
[Issue #7](https://github.com/shell-term/vesper/issues/7) and the
[Attention memory-adapter report](docs/attention-memory-validation.md).

The qualified sparse Attention candidate now also runs through Vesper's CUDA
Driver executor without Torch or TileLang installed. Across 36 conditions, all
six output/intermediate boundaries match the qualified oracle exactly, including
1,261,568 final BF16 elements. Both pressure/retry checks pass. Optional tools
build pinned local kernels once; execution checks their identities and needs
only NumPy and the driver. This is a bounded operation; the subsequent window
Attention-layer integration is described below. See the
[native Attention report](docs/native-attention-validation.md).

Native post-RoPE KV rounding and the 128-row sliding window now connect to that
sparse Attention operation: 398 KV updates and 128 selected Attention conditions
match all checked bits. Window wraps, exact snapshot restoration and failed
publication/retry also pass in a process without Torch/TileLang. This uses the
original per-32 rounding across all 512 values, including the RoPE tail. See
the [window KV report](docs/window-kv-validation.md) for that post-RoPE probe.

Native RMS normalization and rotary position math now pass 476 component cases,
using four original learned norm vectors and both original frequency profiles
at every position from 0 through 131,071. Another 56 conditions connect projected
inputs through KV normalization, RoPE and the window cache; 46 also cover query
RoPE, sparse Attention and inverse RoPE. All 3,420 checked boundaries match,
including FP32 intermediates. Independent replay passes without Torch/TileLang
installed. See the
[normalization and RoPE report](docs/normalization-rope-validation.md).

Original query/KV/output projections now connect the **complete layer-0 window
Attention path**. Four FP8 projections retain per-32 activation/weight scales;
the grouped output projection retains original BF16 conversion and input
rounding. Across 90 component cases and 309 successful layer calls (including
five failure retries), all 17,355 checked boundaries match, including four real
original-prefix inputs. Every layer call also matches unchanged
`Attention.forward` output/cache controls. A process without Torch/TileLang
reproduces the matrix, including two window wraps and failed-rollback recovery.
This probe covers the Attention sublayer; the complete-block follow-up is below.
See the [Attention projection/layer report](docs/attention-projection-validation.md).

Native Hyper-Connections and residuals now connect Attention and the tiered MoE
into a **complete original-weight layer-0 Transformer block**. Across 84 HC cases
and 26 successful single-token block calls, all 6,648 checked boundaries match
exactly. Every block call compares 225 boundaries and also matches unchanged
`Block.forward` hidden-state, next-layer coefficient and KV controls. Five
failure/retry checks and independent replay without Torch/TileLang pass. This
covers the uncompressed layer-0 profile, reusing 27 original experts. The
compressor/cache, indexer and bounded compressed-Attention follow-ups are below;
other complete-block profiles, Engram and full-model execution remain unfinished. See the
[complete-block report](docs/layer0-block-validation.md).

The original ratio-2 and ratio-1 compressors now also execute natively, through
learned projection/pooling, normalization, group-position RoPE and the model's
original FP4/E4M3 compressed-KV rounding. Across 49 component cases and 432
state calls, all 8,751 checked boundaries match. Five fault/retry conditions,
snapshot restoration and independent replay without Torch/TileLang pass. This
uses bounded host diagnostic cache copies and generated inputs with original
layer-2/layer-20 weights; the subsequent indexer and connected compressed-Attention
probes are described below. See the [compression report](docs/compression-validation.md).

Original index-key/query arithmetic, sparse selection and selected source-layer
handoffs now pass **261 component cases and 99 index frames**. All 11,189 checked
boundaries match, including FP4/E8M0 index rounding, BF16 score boundaries,
signed-zero top-k membership and latest-block candidate pinning. Layers 2/20/24
use original weights; the frame checks selected-index sharing 2-to-3 and 24-to-25,
plus key/candidate sharing 20-to-24. Five failure/retries, snapshot restoration
and independent replay without Torch/TileLang pass. The cache remains a bounded
host diagnostic copy and long histories are explicitly primed, generated data.
The subsequent complete compressed-Attention subset is below; other profiles
and full-model execution remain unfinished. See the [indexer report](docs/indexer-validation.md).

Original compressors/indexers now connect the **complete Attention profiles for
layers 2/20/24**, including compressed-KV/key/candidate sharing 20-to-24.
All 103 frames / 309 layer calls match 31,841 boundaries, with zero bit
differences. Every frame also matches separately executed unchanged full-forward
output/cache controls. Five failure/retries, snapshot restoration and independent
replay without Torch/TileLang pass. This is generated independent layer input,
one rank, batch 1/2, prefill at most three queries and context at most 64 tokens;
state remains bounded host diagnostic storage. The longer-context follow-up is
below; other profiles, Engram and full-model execution remain unfinished. See the
[connected compressed-Attention report](docs/compressed-attention-validation.md).

The same profiles now pass **1,441 longer-context frames / 4,323 layer calls**,
with **440,997 boundary comparisons and zero differences**. Empty-start generated
streams reach 1,152 tokens (batch 1) and 260 (batch 2), including ten window wraps
and real top-512 truncation. Explicit synthetic histories at 16,384 positions
exercise 2,049-to-2,048 candidate-block pruning through consumer-24 output;
they are not preceding full-model executions. Every frame matches unchanged
original controls and native-only replay. Failure/retry and snapshots pass.
All registered CUDA binaries remain unchanged. See the
[long-context report and complete evidence](docs/long-context-validation.md).

Original KV/index sharing now also covers **sixteen named Attention profiles**,
including KV sources 8/14, index sources 28/32/36 and consumers that reuse
selected IDs. Four independent sharing groups pass **2,224 frames / 8,896 layer
calls**, with **684,616 boundary comparisons and zero differences**. Each frame
also passes unchanged original full-forward controls; all boundaries reproduce
without Torch/TileLang. Twenty failure/retries and four snapshot restorations
pass. These are independent generated-input groups, with continuous 260-position
streams and explicit synthetic histories through 1,028 positions, rather than
complete compressed Transformer blocks. See the
[sharing topology report](docs/attention-topology-validation.md).

A **complete original layer-2 Transformer block** now connects compressed
Attention, HC and the tiered MoE. All 32 single-token calls match **8,793
boundaries / 70,536,278 storage elements**, with zero differences. Separate
unchanged `Block.forward` and `Attention.forward` controls match output,
next-layer coefficients, selected IDs and all caches/partial state. Six
failure/retries, snapshot restoration and independent replay without
Torch/TileLang pass. This uses layer 2's own weights, 25 gate-selected experts
and generated four-position HC streams. The consecutive-block follow-up is
below; other profiles, Engram and full-model quality/speed remain unfinished. See the
[complete compressed-block report](docs/compressed-block-validation.md).

**Complete consecutive blocks 2 and 3** now pass actual hidden-state/HC
coefficient handoff, source compressed-KV/index reuse and joint publication.
All 36 pair steps / 72 block calls match **18,447 boundaries / 133,921,032
storage elements**, with zero differences. Separate unchanged original blocks
provide 648 block and 720 Attention controls. Eight failures preserve both
states and retry exactly; snapshots and replay without Torch/TileLang pass.
Initial layer-2 inputs are generated four-position streams, with computed
layer-3 inputs and 52 selected original experts. Other complete roles, longer
chains, Engram and full-model quality/speed remain unfinished. See the
[consecutive-block report](docs/block-pair-validation.md).

**Complete ratio-1 blocks 20 and 21** also pass actual hidden/coefficient handoff
and joint publication, including source-20 candidate state from the first token.
All 36 pair steps / 72 block calls match **19,152 boundaries / 160,709,484
storage elements**, with zero differences, plus 648 unchanged block and 684
Attention controls. Eight failure/retries, snapshots, five changed-input/state
controls and replay without Torch/TileLang pass. The pair uses 26/24 actually
selected original experts and four-position generated entry streams. This is
separate from the 2/3 chain; the six-block index-refresh follow-up is described
in the [complete-chain report](docs/refreshed-block-chain-validation.md).
Longer complete contexts and the full model remain unfinished. See the
[ratio-1 complete-pair report](docs/ratio1-block-pair-validation.md).

Vesper uses the released weights; its design does not require retraining.
Model-wide output and quality verification remain required before that engine
is ready for use.

The goal is to preserve the released model's precision and semantics. Native FP4
expert weights are already part of the official checkpoint; using those bytes is
different from applying another lossy quantization. Vesper prohibits extra
cache-dependent router bias, expert pruning and lossy requantization.

## Start

Python 3.10+ is required. CPU checks need NumPy. GPU probes also need an NVIDIA
driver and a CUDA Toolkit containing NVRTC and its headers; a host C++ compiler
is not needed for Vesper's own probes. Initial GPU testing uses Windows and CUDA
12.4. Linux support is implemented but not yet tested on hardware.

```sh
python -m pip install -e .
python -m unittest discover -s tests -v
vesper inventory
vesper validate --output artifacts/components.json
```

Fetch a **348,160-byte original weight/scale slice**, with bounded HTTP Range
requests pinned to the source revision, then validate the scalar projection:

```sh
vesper fetch-fixture --directory artifacts/fixture
vesper validate --fixture artifacts/fixture/native-weight-slice.npz --output artifacts/native-slice.json
```

The fetcher also reads small metadata/header files. It refuses a server that
ignores the requested byte range. It never downloads a full checkpoint shard.
Downloaded model bytes and reference code stay under ignored `artifacts/`.

```sh
vesper bench-transfer --output artifacts/transfers.json
vesper bench-disk --output artifacts/disk.json
vesper bench-disk --file /path/to/an/existing/model.gguf --output artifacts/model-drive.json
```

`bench-disk` uses direct reads, with no buffered fallback. By default it creates
one unique 288 MiB scratch file, then removes only that file. `--file` opens an
existing model read-only and creates no scratch file. The test uses one aligned
expert-sized buffer and queue depth one.

## What is verified

- E2M1 FP4 codes, native E8M0 scales, signed zero and NaN classification.
- E4M3 conversion across all finite BF16 encodings and rounding midpoints.
- Per-32 activation quantization against the reference formula's specification.
- A scalar FP8 × FP4 projection against an independent high-precision block oracle.
- Host/GPU and explicit host-staged GPU/GPU transfers, checked for byte integrity.
- Bounded direct reads and original fixture hashes.

These checks do **not** prove full-model fidelity. An altered accumulation order,
activation rounding, KV cache or router can change model behavior despite exact
stored weights. See [the validation gates](docs/validation.md).

## Official reference comparison

Use a separate environment with a CUDA-enabled PyTorch >=2.10.0 and
`tilelang==0.1.12`, matching the pinned reference's requirements. Vesper does not
install a replacement PyTorch into another application's environment.

```sh
python -m pip install -e '.[reference]'
vesper compare-reference --fixture artifacts/fixture/native-weight-slice.npz --device 0 --output artifacts/official.json
vesper compare-reference --fixture artifacts/fixture/native-weight-slice.npz --device 1 --reference-device 0 --seed 42 --output artifacts/ampere-candidate.json
```

Check that the installed PyTorch build has CUDA enabled; selecting an appropriate
PyTorch CUDA build is platform-dependent. The command downloads the pinned
official kernel source with its license and compares exact output bits. Missing
dependencies, unsupported compiler/hardware paths, and mismatches must remain
visible in the report. Exit code 2 means the probe did not pass. The second command
uses the 4090 as the official oracle for the 3080 Ti candidate; the original FP8
reference is not assumed to run on Ampere. A specification pass cannot substitute
for this gate. Windows reference-compiler setup and its current limitations are
recorded in the [first measurement report](docs/first-probe.md#reproduction).

Fetch **one complete expert (18,800,640 original bytes)** and check its pipeline:

```sh
vesper fetch-expert-fixture --directory artifacts/expert-fixture
vesper compare-expert --fixture artifacts/expert-fixture/native-expert.npz --device 0 --output artifacts/expert.json
vesper trace-reduction --fixture artifacts/fixture/native-weight-slice.npz --device 1 --reference-device 0 --output artifacts/reduction.json
```

`compare-expert` reuses GPU-resident weights and work buffers. It is single-token
only. The complete 128-condition command is in [the report](docs/expert-validation.md#reproduction).
`trace-reduction` compares native FP8 and BF16 intermediate reductions; a mismatch
keeps exit code 2 even when the final BF16 value happens to agree. The Ampere path
remains unqualified. The optional TileLang 0.1.12 NVRTC oracle also needs per-kernel
launch-handle isolation, applied by Vesper without altering the official arithmetic
or source files and recorded in its reports.

## Bounded cache probe

Fetch three distinct experts, then construct a small aligned store. Each expert
contains 18,800,640 original bytes. The builder refuses existing destinations.

```sh
vesper fetch-expert-fixture --expert 1 --directory artifacts/expert-fixture-1
vesper fetch-expert-fixture --expert 2 --directory artifacts/expert-fixture-2
vesper build-store --fixtures artifacts/expert-fixture/native-expert.npz artifacts/expert-fixture-1/native-expert.npz artifacts/expert-fixture-2/native-expert.npz --directory artifacts/native-store
vesper validate-cache --store artifacts/native-store --fixtures artifacts/expert-fixture/native-expert.npz artifacts/expert-fixture-1/native-expert.npz artifacts/expert-fixture-2/native-expert.npz --device 0 --helper-device 1 --output artifacts/cache.json
```

The optional reference environment above is needed for `validate-cache`. It uses
a 96 MiB host-buffer cap, one expert per RAM/GPU tier and a 32 GiB system-available
floor. Transfers are synchronous and audited by readback. The test verifies
evictions, forced SSD reloads, pressure refusal and recovery; it does not run the
model router or estimate real cache-hit rates. Stores default to at most four
experts; connected probes require explicit limits up to 48. See
[the cache report](docs/cache-validation.md) for the original probe's limitations.

An optional audit path hashes consecutive 4 KiB chunks on each GPU and returns
147 KB of digests per expert instead of 18.8 MB of audit readback. Original
CPU-side file/tensor hashes remain the trust anchor. Host-copy checks are
deferred until GPU admission; every GPU upload must pass before publication or
model computation. The readback path remains the default comparison baseline.

```sh
vesper validate-integrity --devices 0 1 --output artifacts/integrity.json
vesper validate-cache --store artifacts/native-store --fixtures artifacts/expert-fixture/native-expert.npz artifacts/expert-fixture-1/native-expert.npz artifacts/expert-fixture-2/native-expert.npz --integrity gpu-sha256 --profile --output artifacts/cache-gpu-audit.json
vesper bench-cache --store artifacts/native-store --trials 6 --repeats 8 --output artifacts/cache-benchmark.json
```

Six alternating pairs of identical **warm synthetic traces** completed 5.02 times
faster with GPU audits (pair range 4.93–5.13). The comparison
includes all upload audits and diagnostics; it has no timed SSD misses and does
not estimate generation speed. Startup and first-source digest preparation are
separate costs. See [the profiling report and complete records](docs/cache-performance.md).

## Memory policy

The provisional full-engine design has a **96 GiB host-memory cap** and a **32 GiB
system-available floor**, including caches, pinned staging, metadata and scratch.
The full memory governor is not implemented yet. The cache runtime now enforces
its owned buffer cap, shrinks RAM under observed pressure and refuses requests if
the reserve remains unavailable. This ledger covers explicit buffers and a
metadata allowance; Python, CUDA contexts, compilation, returned diagnostics and
the optional official oracle remain outside it. It is not an OS memory limit.

The initial transfer probes cap explicit allocations at 256 MiB per GPU and check a 1 GiB
free-memory reserve. GPU contexts and modules consume additional memory. Transfer
and disk probes use an 18.8 MB data buffer. They do not reset GPUs or stop other
workloads. Measurements can therefore include contention.

The single-expert runtime has a stricter 64 MiB explicit GPU allocation cap; the
128-condition run peaked at 18,903,528 bytes. This excludes CUDA/PyTorch contexts
and reference compiler allocations. Its weights remain in ignored artifacts.
The cache probe peaked at 76,251,136 host-buffer bytes (72.7 MiB), 18,903,528
explicit bytes on the 4090 and 18,800,640 on the storage-only 3080 Ti.
The optional GPU audit mode adds original digest tables, one pinned digest
readback and comparison scratch: its peak is 76,985,536 host-buffer bytes
(73.4 MiB), 19,050,408 explicit bytes on the 4090 and 18,947,520 on the 3080 Ti.

The complete layer-0 block probe has separate combined native host allowances of
1,136 MiB and a 512-MiB cap on owned primary-GPU buffers during block calls.
Its measured peak is 272,182,716 bytes. Contexts/modules, optional oracle memory,
process RSS and component initialization are outside that combined GPU figure;
see the [block resource scope](docs/layer0-block-validation.md#state-publication-and-memory).

The complete compressed layer-2 block has separate native allowances of
3,584 MiB, including retained fixtures, and the same combined 512-MiB owned
primary-GPU cap during block calls. The paired original comparison adds a
2,048-MiB reference allowance. These are diagnostic admissions, not RSS or OS
limits; see the [compressed-block resource scope](docs/compressed-block-validation.md#publication-and-resources).

Each complete pair admits 4,832 MiB of native host allowances including
retained fixtures, plus 3,072 MiB for the optional reference comparison. Combined
owned buffers during pair calls are capped at 768 MiB primary / 128 MiB helper.
The 2/3 run peaks at 586,739,584 / 75,496,320 bytes and the 20/21 run at
571,013,000 / 75,496,320 bytes. Sampled available RAM stays above 129 GiB
in both. See the [pair resource scope](docs/block-pair-validation.md#atomic-state-and-bounded-resources)
and [ratio-1 measurements](docs/ratio1-block-pair-validation.md#publication-and-resources)
for exclusions and the distinction between allowances and process RSS.

The complete 20..25 chain admits **9,728 MiB** of native host allowances,
including retained fixtures; the optional original comparison adds **6,144 MiB**.
Combined owned buffers during chain calls are capped at **2 GiB primary /
384 MiB helper** and peak at 1,660,849,784 / 226,488,960 bytes. The original
comparison's sampled available RAM remains above 125.95 GiB. The 32-GiB floor
still applies; contexts/modules and optional compiler allocations are outside
the GPU ledger. See the [complete-chain resource scope](docs/refreshed-block-chain-validation.md#atomic-publication-and-resources).

## Development

The current implementation uses Python for orchestration and NumPy oracles, with
CUDA kernels compiled directly through NVRTC. Scalar specification kernels and
small Tensor Core comparison kernels establish baselines; they are not intended
as the final high-throughput implementation.

The first design hypotheses are to execute an expert on the GPU holding its
weights, prefetch deterministic Engram rows, and choose speculative verification
length by its actual expert-I/O cost. They are hypotheses, not promised speedups.

See [CONTRIBUTING.md](CONTRIBUTING.md). Code is MIT licensed; source model weights,
CUDA libraries and adapted reference code retain their own notices. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
