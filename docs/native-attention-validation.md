# Native sparse Attention validation

Issue: [#7](https://github.com/shell-term/vesper/issues/7), milestone M2.
This completes a bounded native sparse Attention operation. The subsequent
[window KV report](window-kv-validation.md) qualifies bounded native post-RoPE
rounding, ring state and rollback. The later [normalization and RoPE
report](normalization-rope-validation.md) adds original learned normalization and
rotation from projected inputs; the [projection/layer
follow-up](attention-projection-validation.md) connects the original layer-0
projections. The [complete layer-0 block follow-up](layer0-block-validation.md)
adds native Hyper-Connections, residuals and MoE. The [compression probe](compression-validation.md)
qualifies original compressor/cache construction. Indexing/sharing and connected
compressed Attention still need integration. Issue #7 and M2 remain open;
generation is unfinished.

## Result

On the RTX 4090, the CUDA Driver executor matches a
[qualified official-arithmetic PTX reference](attention-memory-validation.md):

| Check | Result |
| --- | --- |
| Three head/dimension profiles, twelve conditions each | 36 conditions pass |
| BF16 output plus five FP32 boundaries | 216 comparisons / 5,888,960 elements, zero bit differences |
| Final BF16 outputs within those comparisons | 1,261,568 elements, zero bit differences |
| Instrumented versus plain native output | Another 36 comparisons; zero differences |
| Combined normal comparisons | 252 comparisons / 7,150,528 elements; zero differences |
| Pressure before allocation and failure on fourth allocation | Both refuse output, free prior allocations and retry exactly |
| Independent process without Torch or TileLang installed | Same 36 conditions and both failure/retry checks pass |

Each process makes 74 successful native calls and two expected failed calls.
The five FP32 boundaries are QK scores, softmax probabilities, running softmax
state, value accumulation and output before BF16 conversion. Checks compare
storage bits, including signed zero and masked-score infinities; tolerances are
not used. Inputs are finite BF16/FP32. They cover masking, all-empty selections,
duplicate indices, ties, positive/negative sinks, signed zero and scale changes.

The oracle first reruns its complete qualification, including literal original
PTX arithmetic/control retention and hardware matrix-load controls. It reproduces
all six original cached binaries. Small shapes also execute the untouched
official kernel. The untouched 64-head, 512-dimensional kernel still needs
138 KiB of shared memory and cannot launch on the 4090. The full-shape oracle
uses its explicitly qualified integer-only memory adapter; this is not a claim
that the untouched full-shape kernel executed.

## Execution and provenance

`NativeAttention` uses NumPy, Python's standard library and the CUDA Driver.
It accepts contiguous uint16 arrays containing original BF16 bits, FP32 sinks
and int32 indices. It preserves the 64-wide KV tile, head count, reductions,
casts and formulas. Query storage moves to registers in the explicitly tested
candidate, reducing full-shape shared memory to 75,776 bytes. Heads are not split.

The optional build/oracle step uses TileLang 0.1.12 and NVRTC **13.4**, with the
pinned DeepSeek source revision. It recompiles each candidate and requires byte
identity with the already-qualified loaded binary. Six source and binary hashes,
the exact ordered launch ABI, SM89 target and shared-memory sizes are pinned in
the executor. Runtime opening verifies every binary, the complete qualification
receipt, model/source identities and retained DeepSeek license before loading
any CUDA module. An unknown compiler binary requires a new qualification profile.
Local hashes identify artifacts; they are not publisher signatures.

Generated CUDA/PTX/CUBINs are local build artifacts and are not committed or
downloaded at runtime. The exported bundle contains no model weights. Building
requires the qualified optional tools; using an existing bundle needs neither
Torch, TileLang nor NVRTC. This is a framework-free execution path with an
optional framework-based build and oracle, not a new independent compiler.

Hardware qualification is currently Windows, SM89, driver API 13040. Torch's
oracle build reports CUDA 13.0; that is separate from the measured NVRTC 13.4
compiler. Linux CI checks CPU contracts only. Other GPUs/compiler profiles have
not been qualified.

## Resource and failure scope

Calls are synchronous and serialized. All per-call GPU buffers have scoped
cleanup, and no KV state is published. Supported bounds are batch 1..2, sequence
1..3, KV rows 1..257, top-k 1..193, and profiles 64x256, 16x512 or 64x512. These
are probe bounds, not advertised long-context or prefill support.

| Native resource | Measured or enforced |
| --- | --- |
| Owned GPU buffer peak | 2,897,168 bytes |
| Device buffer cap / free-VRAM floor | 16 MiB / 2 GiB |
| Native host allowance / system-available floor | 32 MiB / 32 GiB |
| Combined optional oracle and native host allowances | 96 MiB |
| Minimum sampled available RAM during paired comparison | 125,264,261,120 bytes, about 116.66 GiB |
| Owned host/device allocations after close | Zero; native-only context/module handles released |

The host allowance accounts for bounded owned buffers and artifact loading. It
is not a process RSS ceiling. CUDA context/module/compiler/framework internals
are excluded from owned-buffer counters. The full-engine memory policy remains
Issue #9. Failure checks inject pressure; they do not exhaust real RAM or VRAM.

No throughput optimization is claimed. The current host-array API transfers
inputs/results per call. Resident device buffers and native KV integration are
subsequent work; end-to-end prefill, decode and speed are unmeasured.

## Reproduction and evidence

Configure the qualified optional reference environment as described in the
[memory-adapter report](attention-memory-validation.md). Choose a **new** bundle
directory; export refuses to overwrite an existing one:

```sh
vesper compare-native-attention --reference-directory artifacts/reference \
  --bundle artifacts/native-attention-new --device 0 \
  --output artifacts/native-attention.json
```

Then use the ordinary NumPy-only environment and the same bundle:

```sh
python examples/recheck_attention.py \
  --reference-report artifacts/native-attention.json \
  --bundle artifacts/native-attention-new --device 0 \
  --output artifacts/native-attention-only.json
```

The recheck binds the exact reference-report bytes, kernel manifest, input hashes
and all six oracle boundary hashes. CPU checks cover old/new input identity,
bundle corruption and provenance rejection, launch ABI, finite/layout/index
contracts, incomplete goldens, bit comparisons and failed-open cleanup.

Complete immutable evidence:

- [Paired comparison and oracle goldens](../results/2026-10-18-native-attention/attention.json)
- [Framework-free replay](../results/2026-10-18-native-attention/native-only-recheck.json)
- [Rerun oracle qualification and binary bindings](../results/2026-10-18-native-attention/qualification.json)
- [Kernel manifest](../results/2026-10-18-native-attention/kernel-manifest.json)
- [Summary and raw-file SHA-256 values](../results/2026-10-18-native-attention/summary.json)

These finite generated-input component checks do not prove universal numerical
identity or full-model quality. Original model activations, projections and
complete layer-0 blocks have the follow-up evidence above. Compressed/indexed
Attention and other block profiles remain separate gates.
