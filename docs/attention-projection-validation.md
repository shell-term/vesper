# Original-weight native window Attention validation

Issue [#7](https://github.com/shell-term/vesper/issues/7), milestone M2.
Original Query, KV and output projections now connect the complete **layer-0,
uncompressed window Attention** path. This extends the
[normalization/RoPE](normalization-rope-validation.md),
[window KV](window-kv-validation.md) and
[sparse Attention](native-attention-validation.md) component evidence.
Hyper-Connections, residuals, compressed/indexed/shared KV and full Transformer
blocks remain unfinished. Issue #7 and M2 stay open; text generation is unfinished.

## Results

| Check on RTX 4090 | Result |
| --- | --- |
| Four original FP8 projections and grouped BF16 projection | 90 component cases; all checked bits match |
| Successful complete window Attention calls | 309: 304 corpus calls plus five failure retries |
| Boundaries per layer call | 55, including FP8/scales, FP32 diagnostics, BF16 boundaries and KV state/indices |
| Combined component/layer comparisons | 17,355 boundaries / 194,331,461 storage elements; zero differences |
| Diagnostic versus unchanged original FP8 `linear` BF16 output | 1,308 identity controls pass |
| Diagnostic versus unchanged `Attention.forward` output and cache | Both controls pass in all 309 calls |
| Original-prefix Attention inputs | Four inputs across two two-token sequences; all layer boundaries match |
| Five failed layer calls | No partial output; committed KV restored; exact retries pass |
| Failed rollback publication | Further calls blocked until explicit successful recovery |
| Independent process with neither Torch nor TileLang installed/imported | Same component, layer and failure/retry matrix passes |

Storage-element counts include intermediate arrays and repeated diagnostic
boundaries, not final model outputs or tokens. All comparisons use exact bits,
including signed zero; tolerances are not used.

The component corpus covers 1, 2, 3, 4, 6 and 8 input rows plus zero, signed-zero,
mixed-scale, near-epsilon, subnormal-square and large-value inputs. The layer
corpus covers batches one/two, prefill lengths one/two/three followed by decode,
four stress modes, and a continuous 260-token sequence crossing two full ring
wraps. Larger prefills, distributed execution and other layer/compression
profiles are not qualified by this probe.

## Original arithmetic and oracle

The source is DeepSeek-V4.1-Flash-1017 revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`. Original fixture hashes, extracted
method AST hashes, generated-input hashes, compiler options and binary identities
are retained in the complete records. No new weight bytes are downloaded; the
existing bounded layer-0 prefix fixture is reused.

| Projection | Original logical weight shape | Retained arithmetic |
| --- | --- | --- |
| `wq_a` | 1,280 × 5,120 | E4M3 weights, per-32 × 32 E8M0 weight scales; per-32 E4M3 activation quantization |
| `wq_b` | 32,768 × 1,280 | Same original FP8 path after learned query RMSNorm |
| `wkv` | 512 × 5,120 | Same original FP8 path before learned KV RMSNorm |
| `wo_a` | 8,192 × 4,096 | Original conversion to BF16; eight independent groups of 1,024 × 4,096 |
| `wo_b` | 5,120 × 8,192 | Original FP8 path after grouped BF16 output |

Each FP8 projection retains the original K=32 Tensor Core result, activation
scale multiply, weight scale multiply, ordered FP32 add and BF16 output. Its
optional diagnostic uses the original kernel's supported FP32-output profile,
then checks that BF16 conversion matches unchanged original `linear` output.
The original BF16 output remains the boundary passed to the next operation.

The `wo_a` weight is not used directly as FP8. Native conversion preserves the
original FP32 E4M3 × E8M0 multiply followed by BF16 storage. All converted weight
bytes match the unchanged conversion statements, with BF16 tensor SHA-256
`6969c8d5adbf4ed357443f417f6adfb75ee0ebe559d3776bd7d44762b452c8f3`.
After inverse RoPE, all output activations pass through the original per-32 FP8
quantize/dequantize-to-BF16 boundary before grouped einsum. A native cuBLAS
strided-batch call preserves the pinned ATen layout and BF16 output, including
the token/group strides.

These boundaries follow pinned DeepSeek
[`model.py`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/inference/model.py),
[`kernel.py`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/inference/kernel.py)
and
[`convert.py`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/inference/convert.py).
The cuBLAS binding follows PyTorch revision
`5c4886908584029761b579af026dcfb627c84070`:
[`Blas.cpp`](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/cuda/Blas.cpp)
and
[`CUDABlas.cpp`](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/cuda/CUDABlas.cpp).
Third-party licenses are retained in distributions.

The diagnostic chain is checked against unchanged `Attention.forward`, using
actual original projections and weights throughout. Every successful call
checks both its final BF16 output and exact window-cache bytes. Sparse Attention
uses the independently qualified official-arithmetic PTX memory adapter and
reruns its full qualification. The untouched full-shape kernel still exceeds
4090 shared memory; this does not claim that it executed.

Four real input vectors are captured from unchanged original `Block.forward`
at the Attention input boundary, using original embeddings, Hyper-Connections
and block normalization. Native Attention then reproduces the complete layer
on those exact inputs. Hyper-Connections in this capture are **oracle code**;
this does not qualify native Hyper-Connections, the residual or full block.
The derived input vectors are included for independent replay; original weights
and downloaded source stay in ignored artifacts.

| Deliberately rejected replacement | Final grouped BF16 differences |
| --- | --- |
| Omit output-activation quantize/dequantize | 15,318 / 16,384 |
| Replace grouped BF16 math with FP32 math | 3 / 16,384 |

Higher precision alone does not preserve the original model's output bits.

## Layer publication and recovery

The layer owns one bounded native window cache and reuses opened projection,
normalization and sparse Attention executors. Query/KV projections and
normalization run before window publication. A saved original window snapshot
then protects the remaining Attention/output-projection work. The layer advances
its committed position only after the complete result succeeds.

Host pressure and a third-allocation failure refuse early. Failures in grouped
compute and post-KV output download restore exact pre-call cache bytes and
position. If rollback's own download fails, the saved snapshot remains live,
the layer returns no result and blocks further calls. Explicit `recover()` must
restore the committed state before retry; explicit reset may instead start a
new sequence. This is a bounded window-layer transaction, not whole-model or
sampler rollback.

## Resources and qualified execution

| Resource | Bound or observation |
| --- | --- |
| Projection host allowance / owned GPU cap | 512 MiB / 256 MiB |
| Projection peak owned GPU buffers | 202,367,744 bytes, including temporary weight conversion |
| Normalization / sparse Attention peaks in this corpus | 886,028 / 2,570,056 owned GPU bytes |
| Batch-one / batch-two KV peaks | 400,944 / 801,888 owned GPU bytes |
| Combined native host allowances | 656 MiB |
| Combined native plus optional reference allowances | 1,232 MiB |
| System-available floor / native free-VRAM floor | 32 GiB / 2 GiB |
| Paired run minimum sampled available RAM | 124,261,744,640 bytes, about 115.73 GiB |
| Independent replay minimum sampled available RAM | About 118.45 GiB; exact value in the raw record |
| Owned buffers/reservations after close | Zero; owned CUDA context retains released |

Individual device peaks are not a simultaneous-process peak. Allowances track
owned buffers and scratch estimates, not OS process RSS or total driver VRAM.
Python/caller-owned diagnostic arrays, CUDA contexts/modules/compilers and
optional framework allocations are outside these counters. The future
full-engine 96-GiB cap remains Issue #9. No actual memory exhaustion, workload
stopping, page-file change or GPU reset is used.

Native projection/conversion compilation uses NVRTC **12.4**, SM89 and a pinned
CUBIN. Grouped BF16 uses cuBLAS version **120402**, default math mode and explicit
workspace, with unregistered profiles refused. The optional reference uses the
pinned PyTorch **2.14.1+cu130** build; the separately qualified Attention builder
uses NVRTC **13.4**. Execution of that local bundle needs no Torch/TileLang.
Hardware evidence is Windows/4090; Linux CI checks CPU contracts. The diagnostic
host-array pipeline is not optimized for throughput. No end-to-end token-rate
or model-quality measurement is claimed.

## Reproduction and complete records

Prepare the existing bounded original prefix fixture and qualified Attention
bundle as described in the [real-input MoE](real-moe-validation.md) and
[native Attention](native-attention-validation.md) reports. In the configured
optional reference environment:

```sh
vesper compare-projections --reference-directory artifacts/reference \
  --prefix-fixture artifacts/prefix-fixture/native-prefix.npz \
  --attention-bundle artifacts/native-attention-v1 --device 0 \
  --output artifacts/projections.json
```

Then in the ordinary NumPy-only environment with CUDA Toolkit 12.4/cuBLAS:

```sh
python examples/recheck_projections.py \
  --reference-report artifacts/projections.json \
  --prefix-fixture artifacts/prefix-fixture/native-prefix.npz \
  --attention-bundle artifacts/native-attention-v1 --device 0 \
  --output artifacts/projections-native-only.json
```

Independent replay binds exact report bytes, fixture provenance, derived real
inputs, coverage, compiler/binary/cuBLAS profiles and each boundary hash. CPU
tests cover layout/range refusal, incomplete weights, interrupted layer
publication, failed rollback blocking/recovery, source identity and incomplete
or tampered evidence.

- [Paired comparisons, complete goldens and unchanged-forward controls](../results/2026-10-18-attention-projections/projections.json)
- [Independent native-only replay](../results/2026-10-18-attention-projections/native-only-recheck.json)
- [Summary, raw-file hashes and resource observations](../results/2026-10-18-attention-projections/summary.json)

Next: native Hyper-Connections and residual boundaries, followed by complete
layer-0 Transformer-block comparison with the already qualified MoE sublayer.
Compressed KV, sparse index selection/source-layer sharing, Engram, full-model
logits/sampling and workload performance remain later gates.
