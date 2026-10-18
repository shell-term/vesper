# Native RMS normalization and RoPE validation

Issue [#7](https://github.com/shell-term/vesper/issues/7), milestone M2.
Native RMS normalization and rotary position math now connect projected inputs
to the previously qualified [window KV](window-kv-validation.md) and
[sparse Attention](native-attention-validation.md) operations. Original
query/KV/output projections are still missing. This is bounded component and
chain evidence; Issue #7 and M2 remain open, and text generation is unfinished.

## Results

| Comparison on RTX 4090 | Result |
| --- | --- |
| RMS normalization with four original learned weight vectors | 136 cases; all seven boundaries match |
| Original window and compressed/YaRN frequency profiles | 64 chunks; every position 0 through 131,071 in each profile matches |
| Forward and inverse rotary operation | 276 cases; frequency, FP32 rotation and BF16 output bits match |
| Projected-input normalization/RoPE/window KV chain | 56 conditions, batch one and two |
| Chain conditions also including query RoPE, sparse Attention and inverse RoPE | 46 of those 56; all checked boundaries match |
| Combined normal comparisons | 3,420 boundaries / 151,253,882 storage elements; zero bit differences |
| Instrumented versus unchanged original RMS / rotary methods | 195 / 425 identity controls pass |
| Deliberately changed arithmetic | All four negative controls detected |
| Pressure, allocation and download failure | Three refused calls; no partial result or owned-buffer leak; exact retries |
| Independent process without Torch or TileLang installed | Same corpus, chain and failure/retry checks pass |

The comparison count includes FP32 diagnostic arrays and BF16 storage elements;
it is not a count of final model outputs. The three successful failure retries
add 21 boundary comparisons outside the normal total. Comparisons use storage
bits, including signed zero, rather than a numerical tolerance.

## Original normalization and position behavior

The oracle uses DeepSeek revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`. It extracts unchanged
`RMSNorm.forward`, `precompute_freqs_cis` and `apply_rotary_emb` bodies from
[`inference/model.py`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/inference/model.py).
Only the following four BF16 learned norm vectors are read from the existing
prefix fixture, totaling **24,064 original bytes**:

- `layers.0.attn.q_norm.weight`, width 1,280.
- `layers.0.attn.kv_norm.weight`, width 512.
- `layers.0.attn_norm.weight`, width 5,120.
- `layers.0.ffn_norm.weight`, width 5,120.

The loader validates the bounded ZIP member, NPY header, manifest and tensor
hash before accepting each vector. It does not load the other prefix matrices
or fetch new weight bytes. Inputs are generated BF16 values, not captured
full-layer activations.

Normalization converts BF16 input to FP32, squares it, computes the original
FP32 mean and `rsqrt(mean + 1e-20)`, applies the learned weight in the original
order, then rounds to BF16. All seven boundaries are compared. The native mean
uses the qualified PyTorch build's vector-of-four accumulators, block geometry
and reduction order, including its dependence on row count. It does not replace
the mean with a float64 sum. The source is pinned to PyTorch revision
`5c4886908584029761b579af026dcfb627c84070`; see
[`Reduce.cuh`](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/cuda/Reduce.cuh).

The norm corpus covers 18 row counts from 1 through 514, rank-two through
rank-four layouts, zeros/signed zeros, mixed scales, values around epsilon,
subnormal inputs/squares, large values and all finite BF16 encodings. The
executor accepts only the bounded, contiguous shapes and finite inputs covered
by its public validation contract.

RoPE uses 64 rotary dimensions, adjacent complex pairs and factor 16. The
window profile has theta 10,000 and `original_seq_len=0`; the original
uncompressed Attention path disables YaRN. The compressed/YaRN frequency
profile has theta 160,000, `original_seq_len=65,536`, beta-fast 32 and beta-slow
1. The latter's corrected dimensions are 15 and 25. Both profiles compare
inverse frequencies, position products and sine/cosine pairs at all 131,072
positions. This tests compressed-profile **position math**, not compressed KV
construction or retrieval.

The pinned ATen tensor/scalar division multiplies by a rounded reciprocal in
the YaRN ramp. A direct division changes one inverse frequency. Complex
rotation also preserves the qualified CUDA build's specific fused multiply-add
order before BF16 conversion; using unfused arithmetic changes intermediate
bits. The native implementation follows those observed operations explicitly.
See pinned
[`BinaryDivTrueKernel.cu`](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/cuda/BinaryDivTrueKernel.cu)
and
[`complex.h`](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/headeronly/util/complex.h).
It generates only the requested frequencies on demand; it retains no full
131,072-position table in the native runtime.

Rotary cases cover forward/inverse direction, rank-three/four shapes, one/two
batches, 1/16/32/64 heads, width 64/512, long prefill and positions around window,
4,096, original-context and 131,072 boundaries. Nonrotary prefixes are copied
bit-for-bit. Qualification is tied to the named compiler, hardware and bounded
corpus, not a claim of universal equivalence on other backends or shapes.

| Rejected replacement | Detected storage differences |
| --- | --- |
| Higher-precision mean reduction | 4 / 17 FP32 values |
| Default epsilon `1e-6` | 1,024 / 1,024 BF16 values |
| Unfused complex rotation | 4,040 / 16,384 FP32 values |
| Direct division in the YaRN ramp | 1 / 32 FP32 inverse frequencies |

## Connected boundary and remaining gates

Each batch size tests prefill lengths 1, 2, 127, 128, 129, 255 and 257, followed
by three single-token decode steps. Inputs begin **after projection**. Original
learned KV normalization and original rotation feed the unchanged reference
`Attention._window_kv` method, with only the KV projection explicitly replaced
by an identity at this boundary. Native normalization, RoPE, per-32 rounding,
causal indices and exact window bytes are checked against it.

For sequences of at most three queries, generated already-projected queries
also pass through original/native RoPE, sparse Attention and inverse RoPE.
Attention compares its five FP32 diagnostics and BF16 output with the
independently qualified official-arithmetic PTX memory adapter, whose complete
qualification is rerun. The untouched full-shape Attention kernel still exceeds
the 4090 shared-memory limit; this does not claim that it executed. Longer
prefills check normalization/RoPE/KV boundaries but omit the connected Attention
step in this bounded probe.

Original query/KV/output projections and their FP8/BF16 boundaries must be
connected next, then compared at original-weight Attention-layer and prefix
outputs. Compressed KV, sparse indexing/source-layer sharing, Hyper-Connections,
residuals, full blocks, logits and sampling remain separate requirements.
Original weights and these component passes alone do not establish full-model
quality or end-to-end speed.

## Resources and execution profile

| Resource | Bound or observed value |
| --- | --- |
| Norm/RoPE host allowance / system-available floor | 96 MiB / 32 GiB |
| Norm/RoPE owned GPU cap / free-VRAM floor | 64 MiB / 2 GiB |
| Norm/RoPE peak explicitly owned GPU buffers | 42,123,288 bytes, including diagnostic arrays |
| Separate Attention / batch-two KV owned peaks | 1,713,440 / 2,110,496 bytes |
| Combined native host allowances | 144 MiB |
| Combined optional reference and native allowances | 336 MiB |
| Minimum sampled available RAM, paired run | 125,184,446,464 bytes, about 116.59 GiB |
| Minimum sampled available RAM, independent replay | 126,316,961,792 bytes |
| Owned GPU buffers / host reservations after close | Zero; owned CUDA context retains released |

Individual device peaks are not summed into a simultaneous-process peak. Host
allowances and cleanup counters are not process RSS or total driver/framework
limits. CUDA contexts/modules, compilation, optional framework allocations and
caller-owned returned arrays are excluded. The full-engine 96-GiB cap remains
future work in Issue #9. Failure injection does not exhaust actual memory,
stop other workloads or reset GPUs.

The native runtime needs NumPy, the CUDA Driver and NVRTC **12.4**, targeting
SM89. It refuses unregistered compiler/CUBIN profiles. The optional oracle uses
PyTorch **2.14.1+cu130**, CUDA **13.0**, at the pinned source revision. Its
separate Attention compiler uses NVRTC **13.4** and an existing qualified local
bundle. Source/compiler/binary hashes are recorded in the complete results.
Hardware evidence is Windows/RTX 4090; Linux CI covers CPU contracts only.
The five earlier expert/router/shared/aggregation/KV CUBINs were recompiled and
retain their previously qualified hashes.
The diagnostic host-array interface is a validation boundary, not an optimized
resident pipeline, and no timing/token-rate measurement is claimed.

## Reproduction and complete records

Prepare the bounded original prefix fixture described in the
[real-input MoE report](real-moe-validation.md) and the qualified local Attention
bundle from the [native Attention report](native-attention-validation.md).
Use the configured optional reference environment:

```sh
vesper compare-normalization --reference-directory artifacts/reference \
  --prefix-fixture artifacts/prefix-fixture/native-prefix.npz \
  --attention-bundle artifacts/native-attention-v1 --device 0 \
  --output artifacts/normalization.json
```

Then use the ordinary NumPy-only environment with the NVRTC 12.4 toolkit:

```sh
python examples/recheck_normalization.py \
  --reference-report artifacts/normalization.json \
  --prefix-fixture artifacts/prefix-fixture/native-prefix.npz \
  --attention-bundle artifacts/native-attention-v1 --device 0 \
  --output artifacts/normalization-native-only.json
```

Independent replay binds the exact reference-report bytes, selected original
weight provenance, compiler/CUBIN/source identities, Attention bundle, full
coverage, generated inputs and each intermediate hash. CPU checks cover
layout/range/refusal contracts, bounded norm-only fixture loading, source
tampering, complete evidence and cleanup behavior.

- [Paired comparisons, complete goldens and reference qualification](../results/2026-10-18-normalization-rope/normalization.json)
- [Independent native-only replay](../results/2026-10-18-normalization-rope/native-only-recheck.json)
- [Summary, raw-file hashes and resource records](../results/2026-10-18-normalization-rope/summary.json)
