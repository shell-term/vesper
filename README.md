# Vesper

**Big models. Original weights. Your hardware.**

Vesper is building a native-precision Mixture-of-Experts inference engine across
GPU memory, system RAM and SSD. The first target is
[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
on heterogeneous consumer NVIDIA GPUs.

[日本語](README.ja.md) · [First measurements](docs/first-probe.md) · [Validation](docs/validation.md) · [Architecture](docs/architecture.md)

## Status

This is the first **verification scaffold**, not a working text-generation engine.
CUDA format and projection probes run on RTX 4090 and RTX 3080 Ti.
Independent specification checks pass on both GPUs, including an original weight
slice. Direct comparison with the pinned official kernel gives:

| Candidate | W1 output mismatches across 8 inputs | State |
| --- | --- | --- |
| RTX 4090, FP8 Tensor Core path | 0 / 1,024 | Pass for this slice and these inputs |
| RTX 3080 Ti, experimental BF16 Tensor Core path | 14 / 1,024 | Failed; unqualified for inference |

This is one 128-row W1 slice, not a complete expert or model. Full-model quality
and generation speed are unverified. See the [report and complete result files](docs/first-probe.md).

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

## Memory policy

The provisional full-engine design has a **96 GiB host-memory cap** and a **32 GiB
system-available floor**, including caches, pinned staging, metadata and scratch.
The full memory governor is not implemented yet. The current reservation ledger
only covers allocations explicitly owned by a probe; it is not an OS memory limit.

Current GPU probes cap explicit allocations at 256 MiB per GPU and check a 1 GiB
free-memory reserve. GPU contexts and modules consume additional memory. Transfer
and disk probes use an 18.8 MB data buffer. They do not reset GPUs or stop other
workloads. Measurements can therefore include contention.

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
