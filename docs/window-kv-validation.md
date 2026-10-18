# Native sliding-window KV validation

Issue [#7](https://github.com/shell-term/vesper/issues/7), milestone M2.
Native rounding and 128-row window state now connect to the previously qualified
[sparse Attention operation](native-attention-validation.md). Inputs start at an
explicit **post-projection, post-normalization, post-RoPE BF16 boundary**. The
later [normalization and RoPE report](normalization-rope-validation.md) qualifies
native learned normalization and rotation from projected inputs. The
[projection/layer follow-up](attention-projection-validation.md) connects the
original layer-0 projections. The [complete layer-0 block follow-up](layer0-block-validation.md)
adds native Hyper-Connections, residuals and MoE. Compressed KV, sparse
indexing/sharing and other block profiles remain separate gates.
Issue #7 and M2 remain open; text generation is unfinished.

## Results

| Comparison on RTX 4090 | Result |
| --- | --- |
| Original/native KV transitions, batch one and two | 398 updates pass |
| Six KV boundaries and selected connected Attention boundaries | 3,156 comparisons / 93,644,910 compared storage elements; zero bit differences |
| Rounded KV BF16 values within those comparisons | 2,676,736 values; zero differences |
| Connected sparse Attention conditions | 128 conditions; all six boundaries match |
| Final Attention BF16 values within those comparisons | 6,193,152 values; zero differences |
| Failed appends / failed rollback publication | Six / two injections preserve the active state and recover exactly |
| Snapshot restoration and invalidation checks | Eight state events pass |
| Independent process without Torch or TileLang installed | Same updates, selected Attention and recovery matrix pass |

The corpus covers prefill lengths 1, 2, 31, 32, 33, 127, 128, 129, 255, 256 and
257, each followed by four decode steps in both batch sizes. A continuous
one-token prefill plus 259 decode steps crosses two complete window wraps.
Additional inputs cover mixed per-group scales, zeros, signed zeros and every
one of the 65,280 finite BF16 encodings. The latter is a rounding stress test;
original overflow results are retained, and it is not fed into the finite-input
Attention executor. Exact cache bytes, padding and slot order are compared.

## Original semantics retained

The original `_window_kv` rounds **all 512 post-RoPE elements**, including the
64-element RoPE tail, in **32-element groups**. It does not use one scale for
the whole vector. Each group uses the original `1e-4` amax floor, power-of-two
E8M0 scale, E4M3 conversion and BF16 quantize/dequantize boundary. The cache's
physical payload is BF16 containing those rounded/dequantized values; storing
the unrounded input would change model arithmetic.

The optional oracle executes the unchanged pinned `act_quant` and
`Attention._window_kv` / `get_window_topk_idxs` method bodies. Only projection,
normalization and rotation have explicit identity stand-ins to expose the
post-RoPE input boundary. This does not qualify those missing native operations.
FP8 codes/scales are checked with the unchanged out-of-place quantizer; rounded
BF16 cache values use the original fused in-place form. Source and extracted
method AST hashes are recorded. No model weights are fetched by this probe.

During prefill, Attention sees the whole rounded chunk, with one causal index
row per query. The last 128 rows seed their original modulo-128 slots. Decode
writes one row at `position % 128` and retains the original oldest-first slot
order, including leading `-1` padding while filling. The executor does not
reorder slots for cache convenience or assume permutations preserve FP32 bits.
Invalid positions, skipped tokens and multi-token chunks at nonzero positions
are refused before publication. A new sequence uses an explicit reset.

Three deliberately wrong quantizers are detected in a 1,536-value control:

| Rejected change | BF16 differences |
| --- | --- |
| Omit rounding | 1,436 |
| Use one scale per 512 elements | 810 |
| Skip the RoPE tail | 176 |

Connected Attention compares BF16 output and five FP32 diagnostics against the
independently qualified official-arithmetic PTX memory adapter. That adapter
reruns its complete qualification in this probe. The untouched full-shape
official kernel still exceeds 4090 shared memory; it is not claimed to execute.
Full-layer/full-model quality remains untested.

## Publication, rollback and resources

`NativeWindowKV` owns two resident GPU window buffers and one snapshot buffer.
It quantizes into temporary buffers and writes an inactive window. Kernels,
downloads and temporary-buffer cleanup must all succeed before the active
buffer and logical position change. This preserves the active state if pressure,
allocation or publication download fails. Retry starts from the same position.

There is one bounded snapshot slot. A snapshot saves exact cache bits and
position. Rollback copies into the inactive buffer and publishes atomically;
failed publication leaves the active state and live snapshot intact. Successful
rollback consumes the handle. Foreign/copied/consumed handles are rejected;
discard and reset invalidate them. A replacement branch is then compared with
the restored original reference state. This is a KV runtime extension using
saved original state; the upstream model does not provide a rollback API.
Sampler, routing, compressed KV and whole-model speculative rollback are not
covered.

| Resource | Bound or observed value |
| --- | --- |
| Resident windows plus snapshot, batch two | 786,432 bytes |
| KV peak explicitly owned device buffers, batch two | 2,110,496 bytes |
| Separate Attention peak owned buffers in this probe | 1,713,440 bytes |
| KV device buffer cap / free-VRAM floor | 8 MiB / 2 GiB |
| KV host allowance / system-available floor | 16 MiB / 32 GiB |
| Combined native KV + Attention host allowances | 48 MiB |
| Combined optional oracle and native host allowances | 144 MiB |
| Minimum sampled available RAM during paired run | 124,640,350,208 bytes, about 116.08 GiB |
| Owned buffers/host reservations after close | Zero; owned context retains released |

These are owned-buffer allowances and sampled free memory, not process RSS or
total driver VRAM ceilings. CUDA context/module/compiler/framework allocations
are excluded. Returned arrays belong to the caller. The full-engine 96-GiB
policy remains future work in Issue #9; this probe does not load the model.
Failure injection does not exhaust real RAM/VRAM or reset GPUs.

The KV path uses Vesper's own CUDA source, NVRTC **12.4**, SM89, explicit rounding
and a pinned CUBIN hash. The source is compiled once per opened runtime, with no
Torch/TileLang dependency. The separate Attention build/oracle uses NVRTC 13.4;
its existing local bundle is reused. Unknown KV compiler binaries are refused.
The four earlier expert/router/shared/aggregation CUBINs were recompiled and
retain their previous hashes. Hardware evidence is Windows/4090 only; Linux CI
covers CPU contracts. Timing and end-to-end token rates are not measured.

## Reproduction and complete records

Prepare a qualified local Attention bundle as in the
[native Attention report](native-attention-validation.md), and use the configured
optional reference environment:

```sh
vesper compare-window-kv --reference-directory artifacts/reference \
  --attention-bundle artifacts/native-attention-v1 --device 0 \
  --output artifacts/window-kv.json
```

Then use the ordinary NumPy-only environment with the NVRTC 12.4 toolkit:

```sh
python examples/recheck_window_kv.py --reference-report artifacts/window-kv.json \
  --attention-bundle artifacts/native-attention-v1 --device 0 \
  --output artifacts/window-kv-native-only.json
```

The independent process binds exact report bytes, source/compiler/binary
profiles, Attention bundle, positions and generated-input/boundary hashes. CPU
checks cover causal/ring indices, layout/range refusals, all-finite encoding
coverage, snapshot identity/publication failure, source hashes, complete
evidence and cleanup. The small hardware corpus is not universal equivalence.

- [Paired comparisons, complete goldens and reference qualification](../results/2026-10-18-window-kv/window-kv.json)
- [Independent native-only replay](../results/2026-10-18-window-kv/native-only-recheck.json)
- [Summary, raw-file hashes and resource records](../results/2026-10-18-window-kv/summary.json)

The [normalization and RoPE follow-up](normalization-rope-validation.md) now
connects those operations from projected inputs. The subsequent
[projection/layer probe](attention-projection-validation.md) adds original
query/KV/output projections and complete window Attention comparisons.
The [complete layer-0 block probe](layer0-block-validation.md) adds native
Hyper-Connections, residuals and MoE. Compression/index selection/sharing and
other block profiles remain required; full-model integration is unfinished.
