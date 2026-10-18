# Original compressor and compressed-KV validation

Issue [#7](https://github.com/shell-term/vesper/issues/7), milestone M2.
The original **layer-2 ratio-2** and **layer-20 ratio-1** compressors now execute
without Torch/TileLang at runtime. Their pre-RoPE latents, native rotary math,
original FP4 rounding and bounded compressed-KV publication pass exact reference
comparisons. This follows the [uncompressed layer-0 block](layer0-block-validation.md).
It does not qualify a complete compressed Attention layer: sparse index
selection and source-layer sharing remain unfinished. Issue #7 and M2 stay open.

## Results

| Check on RTX 4090 | Result |
| --- | --- |
| FP4 compressed-KV rounding | 25 cases; original packed codes, E4M3 scale bytes and in-place BF16 output match |
| FP32 pair softmax/pooling | 24 cases; all seven diagnostic boundaries match |
| Successful state updates | 432: 414 core calls, five fault-test prefills, five retries and eight snapshot-stream calls |
| Combined component/state comparisons | 8,751 boundaries / 38,584,338 storage elements; zero bit differences |
| Diagnostic versus unchanged original compressor/cache methods | 434 control groups pass, including two reference-only prefix replays after rollback |
| Original packed/scaled versus original in-place FP4 | 364 reconstruction controls pass |
| Five whole-call failures | Partial-group state, cache and position remain unchanged; all retries match |
| Snapshots | Both ratios restore exact partial-group/cache bits; six duplicate/expired-handle refusals pass |
| Changed-arithmetic controls | All five detected |
| Independent process without Torch/TileLang installed or imported | Entire component, state, fault and snapshot matrix passes |

Counts include diagnostic arrays and repeated cache contents, not distinct model
outputs or tokens. Comparisons use exact storage bits, including signed zero
and the incomplete group's negative-infinity score state. No tolerances are used.

The quantization corpus covers all **35,410 finite BF16 encodings within the
admitted magnitude range of 2688**, signed zero, subnormals, the nonzero scale
floor, FP4 midpoint ties and E4M3 scale transitions. The bounded quantizer refuses
larger magnitudes and nonfinite inputs; it makes no claim about overflow-scale
behavior outside that domain. Pooling covers batches one/two, one/three/eight
groups, near-tied or saturated scores, cancellation and signed-zero products.

The 46 state streams cover both original ratios, batches one/two, initial prefill
lengths 1/2/3/7/8/15/16, single-token decode, zero/signed-zero/near-epsilon/mixed-
scale inputs and continuous 128-token sequences in both profiles. Odd prefills
retain their trailing projection/score pair until the next token completes it.
These are generated component inputs, not captured layer-2/layer-20 activations
from a complete model. Larger contexts, larger prefills and distributed execution
remain outside this probe.

## Source arithmetic and oracle

The source is DeepSeek revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`. The fixture contains five original
BF16 tensors: two layer-2 projection matrices, its learned norm, one layer-20
projection matrix and its learned norm. Only **15,730,688 original bytes** were
fetched in bounded HTTP ranges; the complete checkpoint was not downloaded.
Weights and downloaded sources remain ignored local artifacts.

The two paths intentionally use different math:

- Ratio 2 promotes the original BF16 projection weights and inputs exactly to
  FP32, projects both KV and gate scores, then softmaxes each pair in FP32.
  Products and their sum retain the original FP32 rounding; only the pooled
  vector is converted to BF16 before the learned RMS normalization.
- Ratio 1 uses a direct BF16 projection followed by the learned norm. It has no
  pooling gate or partial-group state. Promoting every path to FP32 is not
  assumed equivalent to the original implementation.
- The pre-RoPE latent is preserved for a future index-key consumer. Group `j`
  rotates at its **first** token position, `j × ratio`, with the original
  compressed/YaRN frequency profile. Native rotary execution now accepts these
  explicit, nonconsecutive group positions; the default consecutive-position
  path retains its previous behavior and binary.
- Compressed KV uses the checkpoint's original **16-value FP4 groups and E4M3
  scales**, with a `6 × 2^-9` maximum floor. Scale rounding, FP4 ties, saturation
  and final BF16 dequantization follow the original kernel. This is the model's
  trained quantization, not an additional lossy format chosen by Vesper.

The optional oracle executes unchanged `linear`, `Compressor.forward` and
`Attention._compress_kv` bodies. Its diagnostic chain is checked against the
unchanged methods' latent, partial states and actual quantized cache writes.
**Index selection is explicitly replaced with an empty-index callback** that
records the latent before in-place rotation/quantization. This isolates cache
construction and verifies that boundary; it does not qualify `Indexer.forward`,
index-key computation, candidate/top-k selection or complete Attention output.

Original packed and in-place FP4 kernels execute separately, and their output
bits also reconstruct identically. Original method AST and generated CUDA hashes
are pinned; no store-instrumented quantization oracle is substituted. The pooling
diagnostics separately expose max, exponential, denominator, weights, products
and both FP32/BF16 pooled outputs, while unchanged `Compressor.forward` controls
the resulting latent and state transitions.

The native profile is SM89, NVRTC 12.4 and cuBLAS 120402/default math and workspace.
The optional oracle uses PyTorch revision
`5c4886908584029761b579af026dcfb627c84070` (2.14.1+cu130), TileLang 0.1.12 and
NVRTC 13.4. The pinned ATen spatial-softmax and reduction ordering is retained.
Original/native compiler, source, binary, fixture and boundary hashes bind
independent replay. Unregistered profiles are refused.

| Deliberate change | Detected differences |
| --- | --- |
| Replace ratio-2 FP32 projection with BF16 | 1,536 / 1,536 FP32 projection values |
| Round pool products to BF16 early | 495 / 1,536 BF16 pooled values |
| Replace compressed-KV groups of 16 with 32 | 538 / 1,536 BF16 KV values |
| Replace compressed-KV E4M3 scales with E8M0 | 1,310 / 1,536 BF16 KV values |
| Rotate at the group's last token | 94 / 1,536 BF16 values |

## State publication and resources

Model arithmetic executes on CUDA. The small cache and partial-group state are
**bounded host diagnostic copies**; this is not a GPU-resident, asynchronous
production cache. Updates stage all state and diagnostic outputs before one
publication step. Failures during projection allocation, rotary execution,
rounded-value download, or observed memory pressure at entry/publication leave
the old cache and position intact. The five injected failures do not exhaust
real machine memory. Snapshots retain one bounded copy; reset invalidates handles.
Returned diagnostics own their arrays and cannot mutate committed state. A reset
allocation failure also preserves the previous state and snapshot.

Combined native host allowances are **240 MiB**: 128 MiB for compression,
96 MiB for normalization/rotation and 16 MiB for one opened state. Including the
optional oracle, they total **496 MiB**. The independent 32-GiB system-available
floor is retained. Minimum sampled available RAM was 119.64 GiB in the paired
run and 121.19 GiB in native-only replay; exact byte observations are in the raw
records. These allowances are not RSS limits or the full-engine 96-GiB governor.

Compression and normalization each cap owned GPU buffers at 64 MiB and retain a
2-GiB free-VRAM reserve. Their separate component peaks are 35,848,192 and
263,552 bytes. Adding component peaks is not a measured simultaneous peak. CUDA
contexts/modules, compiler/framework allocations and returned diagnostics are
outside those ledgers. Owned GPU allocations, context retains and host
reservations are released after the probe. No other workload was stopped, GPU
reset or page-file change was needed. Full-engine memory governance remains
[#9](https://github.com/shell-term/vesper/issues/9).

## Reproduction and complete evidence

Fetch only the five bounded tensors:

```sh
vesper fetch-compression-fixture --directory artifacts/compression-fixture
```

The fetcher refuses an existing destination or a server ignoring byte ranges.
In the separate optional reference environment:

```sh
vesper compare-compression --reference-directory artifacts/reference \
  --fixture artifacts/compression-fixture/native-compression.npz --device 0 \
  --output artifacts/compression.json
```

Then in the ordinary NumPy-only environment with the registered CUDA Toolkit:

```sh
python examples/recheck_compression.py \
  --reference-report artifacts/compression.json \
  --fixture artifacts/compression-fixture/native-compression.npz \
  --output artifacts/compression-native-only.json
```

CPU checks cover layout/position/range refusal, malformed fixture headers before
array allocation, headroom refusal, snapshot identity, diagnostic ownership,
atomic reset failure and incomplete/tampered source, coverage or golden records.

- [Paired comparisons, complete goldens and unchanged-method controls](../results/2026-10-18-compression/compression.json)
- [Independent native-only replay](../results/2026-10-18-compression/native-only-recheck.json)
- [Summary, raw-file hashes and resource observations](../results/2026-10-18-compression/summary.json)

Next: index-key/query computation, sparse/candidate selection, source-layer
sharing, then connected compressed Attention. Engram, other block profiles,
full-model logits/sampling, quality and generation speed remain unfinished.
