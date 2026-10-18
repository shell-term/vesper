# Vesper

**Big models. Original weights. Your hardware.**

Vesper is building a native-precision Mixture-of-Experts inference engine across
GPU memory, system RAM and SSD. The first target is
[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
on heterogeneous consumer NVIDIA GPUs.

[日本語](README.ja.md) · [Expert validation](docs/expert-validation.md) · [Validation](docs/validation.md) · [Architecture](docs/architecture.md)

## Status

This is a **verification runtime**, not a working text-generation engine. One
complete native expert now runs on the RTX 4090 with resident original weights.
Direct comparisons with the pinned official implementation give:

| Candidate | Scope | Result |
| --- | --- | --- |
| RTX 4090, native FP8 path | One complete expert, 128 conditions | 0 / 655,360 final-output mismatches; all checked intermediate bits match |
| RTX 3080 Ti, experimental BF16 path | 128-row W1 slice, 8 inputs | 14 / 1,024 output mismatches; unqualified for inference |

The expert checks include both input projections, asymmetric clamps, weighted
SwiGLU, the BF16 boundary, down-input quantization and the output projection.
Independent format checks also pass on both GPUs. Layers, full-model quality and
generation speed remain unverified. See the [new report and complete results](docs/expert-validation.md)
and the [initial transfer/storage measurements](docs/first-probe.md).

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

## Memory policy

The provisional full-engine design has a **96 GiB host-memory cap** and a **32 GiB
system-available floor**, including caches, pinned staging, metadata and scratch.
The full memory governor is not implemented yet. The current reservation ledger
only covers allocations explicitly owned by a probe; it is not an OS memory limit.

Current GPU probes cap explicit allocations at 256 MiB per GPU and check a 1 GiB
free-memory reserve. GPU contexts and modules consume additional memory. Transfer
and disk probes use an 18.8 MB data buffer. They do not reset GPUs or stop other
workloads. Measurements can therefore include contention.

The single-expert runtime has a stricter 64 MiB explicit GPU allocation cap; the
128-condition run peaked at 18,903,528 bytes. This excludes CUDA/PyTorch contexts
and reference compiler allocations. Its weights remain in ignored artifacts.

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
