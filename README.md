# Vesper

**Big models. Original weights. Your hardware.**

Vesper is building a native-precision Mixture-of-Experts inference engine across
GPU memory, system RAM and SSD. The first target is
[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
on heterogeneous consumer NVIDIA GPUs.

[日本語](README.ja.md) · [Real-input MoE](docs/real-moe-validation.md) · [Cache performance](docs/cache-performance.md) · [Validation](docs/validation.md) · [Architecture](docs/architecture.md)

[Roadmap](docs/roadmap.md) · [Issues](https://github.com/shell-term/vesper/issues) · [Milestones](https://github.com/shell-term/vesper/milestones)

## Status

This is a **verification runtime**, not a working text-generation engine. Native
experts now run on the RTX 4090 through SSD, bounded RAM and two GPU cache tiers.
Direct comparisons with the pinned official implementation give:

| Candidate | Scope | Result |
| --- | --- | --- |
| RTX 4090, native FP8 path | One complete expert, 128 conditions | 0 / 655,360 final-output mismatches; all checked intermediate bits match |
| SSD / RAM / 3080 Ti storage / 4090 compute | Three distinct experts, 232 successful calls | 0 / 1,187,840 final-output mismatches; all checked intermediate bits match |
| GPU SHA-256 upload audits | Same three experts, another 232 calls | All thirteen checked stages match; unchanged model kernel |
| RTX 3080 Ti, experimental BF16 path | 128-row W1 slice, 8 inputs | 14 / 1,024 output mismatches; unqualified for inference |

The expert checks include both input projections, asymmetric clamps, weighted
SwiGLU, the BF16 boundary, down-input quantization and the output projection.
Independent format checks also pass on both GPUs. Layers, full-model quality and
generation speed remain unverified. The 3080 Ti currently stores original bytes;
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
sublayer; full Transformer blocks and text generation remain unfinished.

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
only NumPy and the driver. This is a bounded operation, with native projections
and full-layer integration still pending. See the
[native Attention report](docs/native-attention-validation.md).

Native post-RoPE KV rounding and the 128-row sliding window now connect to that
sparse Attention operation: 398 KV updates and 128 selected Attention conditions
match all checked bits. Window wraps, exact snapshot restoration and failed
publication/retry also pass in a process without Torch/TileLang. This uses the
original per-32 rounding across all 512 values, including the RoPE tail. Native
normalization, RoPE, projections and compressed/indexed KV remain pending. See
the [window KV report](docs/window-kv-validation.md).

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

Current GPU probes cap explicit allocations at 256 MiB per GPU and check a 1 GiB
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
