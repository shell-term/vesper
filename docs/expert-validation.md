# One complete expert — 2026-10-18

The RTX 4090 native pipeline matched the pinned official implementation in all
128 tested conditions, including checked FP32 intermediates and all 655,360
final BF16 output values. The experimental RTX 3080 Ti BF16 replacement remains
unqualified. No comparison tolerance was introduced.

This is one original expert with one token per call. It does not establish
layer/model fidelity or generation throughput.

## Complete-expert scope

Source: `deepseek-ai/DeepSeek-V4.1-Flash-1017`, revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`. Exactly layer 0, routed expert 0 was
fetched: three native FP4 matrices plus their E8M0 scales, totaling 18,800,640
bytes. Each HTTP Range read is bounded below 8 MiB. No full checkpoint shard,
tokenizer, Engram table or other expert was downloaded for this test.

| Projection | Output × input elements | Native format |
| --- | --- | --- |
| W1 (gate) | 2,304 × 5,120 | Packed E2M1, E8M0 scales per 32 |
| W3 (up) | 2,304 × 5,120 | Packed E2M1, E8M0 scales per 32 |
| W2 (down) | 5,120 × 2,304 | Packed E2M1, E8M0 scales per 32 |

Inputs use seeds `20261018, 0, 1, 42, 127, 512, 4096, 65535`, each scaled by
`0.0625, 1, 16, 128` and rounded to BF16. Each of these 32 inputs is run without
a routing weight and with FP32 routing weights `0.25, 1.5, 0`. Thus the matrix
covers 128 calls, including repeated execution and weight-buffer reuse.

The runtime uploads the original packed expert once and reuses GPU work buffers.
It performs native FP8 input quantization, W1/W3, the trained asymmetric clamps
(gate maximum 10; up limited to [-10, 10]), FP32 SiLU/product/routing weight,
the BF16 boundary, down-input quantization, and W2.

The oracle executes the original `linear` and `Expert.forward` function bodies
selected from hash-checked `inference/model.py`. The selection removes unrelated
module imports/classes; it does not rewrite those bodies. Projection wrappers
record their arguments/results. The GEMMs and activation quantizer come from
the unchanged pinned `inference/kernel.py`.

All comparisons passed:

- Input E4M3 bytes and E8M0 scales; both full input projections.
- Clamped FP32 gate/up, FP32 SiLU, product and weighted product.
- The BF16 down input, its E4M3 bytes/scales, and the final BF16 projection.

The FP32 diagnostic trace is recomputed with the same eager PyTorch operations.
Its BF16 boundary is also checked against the actual down-input tensor captured
from the original `Expert.forward` execution. The [complete records](../results/2026-10-18-expert/README.md)
retain all per-case check counts, source hashes and fixture provenance.

## Ampere discrepancy localized

The W1 slice regression still gives 0/1,024 official-output differences on the
4090 and 14/1,024 on the 3080 Ti candidate. New trace kernels record the FP32
block dot, scaled block, and running sum for each 32-element scaling block.

Across eight inputs and 163,840 block dots:

- Native FP8 versus widened-BF16 dots differ in 19,019 positions.
- Widened-BF16 dots match the float64 independent dot oracle in this sample.
- The entire BF16 trace matches between the 4090 and 3080 Ti.
- Trace final outputs match the corresponding original Vesper projection probes.

The first difference is already present in the block dot, before scale
application and the final BF16 conversion. Flushing the rare FP8 subnormal input
codes did not remove the discrepancy. Simple shared-exponent/truncation models
also failed to explain every native dot; there is no qualified software
emulation yet.

This evidence rules out treating wider BF16 arithmetic as an equivalent
replacement in this sample. It does not identify all native hardware rounding
steps. NVIDIA's [PTX MMA specification](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma)
leaves accumulation order and rounding unspecified for these operations.
Preserving source bytes alone cannot certify interchangeable arithmetic.

## Optional oracle runtime correction

Repeated complete-expert execution exposed a separate host-runtime problem in
the installed TileLang 0.1.12 NVRTC adapter. Its `kernels` handle table is a
shared class dictionary. W1/W3 and W2 use the same generated function name with
different N/K dimensions, so registering W2 replaced an earlier launch handle.
The first expert call passed, and later calls used the wrong specialization.
The packed weights and scales themselves retained their hashes.

Vesper now snapshots the handle table onto each compiled NVRTC adapter as soon
as its factory returns. This preserves the CUDA kernel, model source, arguments
and arithmetic. The workaround is explicit in the comparison report. A CPU
regression test reproduces the handle collision without launching a wrong-shape
GPU kernel. Both repeated full-expert checks and the earlier single-slice gate
were rerun after this correction.

The correction is limited to the optional reference integration. It changes no
installed application environment or downloaded source file. A dedicated
reference environment remains advisable; see the [Windows setup notes](first-probe.md#reproduction).

## Resource scope

The native expert's peak explicit device allocation was **18,903,528 bytes**
under a 64 MiB cap, with the existing 1 GiB VRAM reserve check. CUDA contexts,
modules, optional PyTorch allocations and compilation are additional. The source
fixture is 18,800,640 model bytes plus its local container/manifest. Archive
member sizes and NPY headers are checked before NumPy can allocate tensors.

The isolated reference used PyTorch `2.14.1+cu130`, TileLang `0.1.12`, NumPy
`2.5.2` and CUDA 13.4 NVRTC; Vesper's own kernels used CUDA 12.4 NVRTC with
contraction and flush-to-zero disabled. The standalone diagnostics/CPU tests use
the project's NumPy `2.5.4` environment. Source/compiled hashes are in the records.

This is not the proposed 96 GiB full-engine RAM governor. There is no full-model
cache, asynchronous offload scheduler or token-rate measurement. Other jobs were
not stopped, and no GPU reset was performed.

## Reproduction

Install Vesper and configure the optional CUDA PyTorch/TileLang environment as in
[README](../README.md) and the [initial setup report](first-probe.md#reproduction).

```sh
vesper fetch-expert-fixture --directory artifacts/expert-fixture
vesper compare-expert --fixture artifacts/expert-fixture/native-expert.npz --device 0 --reference-device 0 --seeds 20261018 0 1 42 127 512 4096 65535 --input-scales 0.0625 1 16 128 --routing-weights 0.25 1.5 0 --output artifacts/expert-matrix.json
vesper trace-reduction --fixture artifacts/fixture/native-weight-slice.npz --device 1 --reference-device 0 --seed 20261018 --output artifacts/reduction.json
```

The expert command also includes the unweighted case. The reduction diagnostic
requires the small W1 fixture from `fetch-fixture`. It returns exit code 2 when
intermediate reductions differ, including a case whose final BF16 happens to
match. A final-output coincidence does not clear the intermediate gate.

Next, bounded cache residency/eviction and transfer scheduling can be developed
around this resident expert. The Ampere replacement must pass its own gate
before being dispatched for native expert computation. Router, aggregation,
shared expert, attention, Engram and end-to-end quality remain separate work.
