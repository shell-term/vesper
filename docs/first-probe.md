# First measurements — 2026-10-18

Vesper has a runnable verification scaffold, not a text-generation engine. The
first native FP8 Tensor Core projection on the RTX 4090 matched the official
kernel for all eight tested inputs. The RTX 3080 Ti BF16 candidate failed and is
not qualified for inference. No tolerance was introduced to hide the difference.

## Exact scope

The source is [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170),
revision `1bae0e1088993b9b82de08f59de5be5a34e42170`. Only the first 128 rows of
`layers.0.ffn.experts.0.w1` were fetched: 327,680 packed FP4 bytes and 20,480 native
scale bytes. Small metadata and safetensors headers were read separately. The
full checkpoint was not downloaded.

Each input is one 5,120-element vector sampled with NumPy's default RNG and
rounded to BF16. Seeds: `20261018, 0, 1, 42, 127, 512, 4096, 65535`. The original
activation quantizer and `fp4_gemm` were run from the pinned, hash-checked
`inference/kernel.py`, without editing that source. The official oracle ran on
the RTX 4090 for both candidates. Comparisons use exact stored bits.

| Candidate | Activation bytes / scales | Projection output bits | Outcome |
| --- | --- | --- | --- |
| RTX 4090, `gemv_fp4_mma` | All 8 inputs match | 0 / 1,024 mismatches | Pass within this scope |
| RTX 3080 Ti, `gemv_fp4_bf16_mma` | All 8 inputs match | 14 / 1,024 mismatches | Fail; 1 of 8 inputs passes |

This does not cover a complete expert, other projections, router, attention,
Engram, KV cache, batched inputs, model logits or evaluation workloads. A pass
here cannot establish unchanged model quality.

The independent specification checks passed on both GPUs: six checks per GPU,
including finite BF16-to-E4M3 conversion space, rounding midpoints, signed zero,
scale encodings and the original W1 slice. Those checks exercise the scalar
specification kernel, not the Tensor Core candidates above.

An earlier scalar candidate passed its independent oracle but differed from the
official kernel in 2 of 128 outputs for the first input. Changing the Ada path to
native FP8 MMA produced the eight-input match. Exactly widening FP8 and FP4
operands to BF16 did not make the Ampere candidate match. The observations
justify retaining the official arithmetic gate; they do not yet isolate the
cause of every discrepancy or prove universal equivalence of the Ada path.

## Transfers and storage

Hardware: RTX 4090 (SM 8.9), RTX 3080 Ti (SM 8.6), Intel Core i7-12700KF, about
159.8 GiB physical RAM, Windows. CUDA Driver API version: 13.4. Vesper's probe
kernels used NVRTC from CUDA 12.4 with contraction and flush-to-zero disabled.
The optional oracle used PyTorch `2.14.1+cu130`, TileLang `0.1.12` and CUDA 13.4
NVRTC. There is no direct peer access between these two GPUs in this machine.

| Measurement | Median result | Scope |
| --- | --- | --- |
| Pinned host → GPU | 12.34–12.36 GB/s | 18,800,640 bytes, 12 repeats per GPU |
| GPU → pinned host → other GPU | 0.0268–0.02735 ms | 20,480 bytes, both directions |
| GPU → pinned host → other GPU | 2.981–3.043 ms | 18,800,640 bytes, both directions |
| Model volume, sequential direct read | 3.87 GB/s | Expert-sized request, queue depth 1 |
| Model volume, random direct read | 4.41 GB/s | Expert-sized request, queue depth 1 |
| Model volume, random 4 KiB direct read | 0.14385 ms | 64 reads, queue depth 1 |

GB/s is decimal. Transfer contents were checked byte-for-byte. Direct reads
checked the first 4 KiB at every tested offset outside the timed region. The
model-volume test opened an existing, unrelated model file read-only. The
workspace-volume test used one unique 288 MiB scratch file and removed it.
Neither test reads DeepSeek routing traces or measures Engram lookup end to end.
Drive firmware caches, ordinary system activity and host-call overhead can
affect these small synchronous measurements. No other jobs were stopped.

One complete native expert occupies 18,800,640 bytes; a BF16 input and output
together occupy 20,480 bytes. Moving activations to resident weights is worth
testing, but these transfer measurements alone do not establish a speedup. SSD
traffic depends on actual expert reuse. There is no measured token rate yet.

## Memory

The proposed full-engine host cap is 96 GiB, with at least 32 GiB of system
available memory. This is a design constraint, not an implemented full-engine
governor. Current probes use small explicit reservations; GPU contexts, modules
and optional PyTorch/TileLang allocations are additional.

The component probe's peak explicit device buffers were 353,696 bytes per GPU.
Transfer and storage probes reserve an 18,800,640-byte host buffer. No full-model
RAM cache has been allocated. The source fixture is just 348,160 bytes of model
data plus its local container/metadata.

The isolated project environment also passed 16 CPU specification/boundary tests
on Python 3.13 with NumPy 2.5.4. A wheel was built and checked for the CUDA source,
Vesper's MIT license and the DeepSeek license/attribution files. The configured
Windows/Linux CI matrix has not yet run on GitHub.

## Reproduction

Follow the install and fixture commands in [README](../README.md), then run
`validate`, `bench-transfer`, `bench-disk`, and the optional official comparison.
The published [result directory](../results/2026-10-18/README.md) contains the
complete comparison records, fixture provenance and probe outputs without model
bytes or machine-specific paths.

To run the same eight-input comparison matrix in a configured reference
environment:

```sh
python scripts/compare_reference_matrix.py --fixture artifacts/fixture/native-weight-slice.npz --devices 0 1 --reference-device 0 --output artifacts/official-matrix.json
```

On the first Windows experiment, TileLang's default host-compiler path failed.
The successful oracle used `TILELANG_EXECUTION_BACKEND=nvrtc` with CUDA 13.4
runtime, NVRTC, CCCL and `cuda-python` installed in an isolated reference area.
TileLang template/CUTLASS/CUDA include paths were ASCII relative paths, and its
temporary/cache directory was an ASCII absolute path. The initial Unicode-path
build failed; this setup is not yet an automated, portable installer. A normal
CUDA-enabled PyTorch installation is also required. Vesper's own Driver/NVRTC
probes do not need PyTorch, TileLang or a host C++ compiler.

## Next implementation gates

1. Locate the first divergent block in the Ampere candidate and compare its
   reduction and scaling operations against the unchanged official oracle.
   Keep that path experimental unless the gate passes.
2. Extend the 4090 comparison to all three projections and a complete expert,
   including activation/clamp/weighted SwiGLU behavior and accumulation order.
3. Build bounded resident caches and measure real routing misses and bytes read.
   Use those traces to choose transfer placement and prefetch policy.
4. Add layers and model execution, then compare logits and workloads before
   reporting any generation throughput or model-quality claim.
