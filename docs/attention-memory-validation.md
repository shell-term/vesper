# Official Attention arithmetic with a qualified memory adapter

> **Recorded PC evidence.** Results, requirements and next-step wording below
> describe this report's experiment/profile. They do not set the current
> development queue or require exact equality from every new candidate. See
> [current policy](quality-speed.md), [validation](validation.md) and
> [PC roadmap](roadmap.md). Raw evidence and original scope are unchanged.

The **64-head × 512-dimensional** register-query candidate now matches a
separately constructed reference that retains the official generated **PTX
floating-point instructions, their registers, loops, branches and barriers**.
Only memory access is adapted. All checked intermediate/output bits match.

This qualifies a storage-adapted official-arithmetic reference on the measured
SM 8.9 / driver / compiler profile. Untouched full-kernel launch remains blocked
by the same **138 KiB request / 99 KiB device limit**. These are separate gates;
the earlier [partial experiment](attention-storage-validation.md) is retained.

## Why this comparison is stronger

Changing the Query allocation in TileLang can change code generation. A second
CUDA-level memory adapter matched every tested value, but inspection found that
the compiler unrolled the score loop differently (48 versus 40 static MMA
instructions in the complete kernel). That result alone was not accepted as
proof that the generated arithmetic program was retained.

The final adapter starts **after** CUDA compilation. It pins the original CUDA
source hashes and reproduces all **six original cached CUBIN binaries byte for
byte**. It then adapts the original PTX, keeping every floating-point arithmetic
and control instruction literally unchanged. The full uninstrumented profile
retains **1,887 floating-point arithmetic instructions**, including the original
MMA instructions and loop expansion. The patch is reversible to every original
PTX byte; extra instructions use a separate integer/predicate register namespace.

The Query slab moves from on-chip shared memory to a private **64 KiB GPU-global
workspace per batch/sequence CTA**. Other shared accesses move down by 64 KiB,
preserving their relative layout and aliases. This brings dynamic shared memory
to **74 KiB**; it does not reduce model-weight storage. Added operations transport
raw integer bits and do not perform floating-point arithmetic.

The global matrix-load mapping follows the [NVIDIA PTX ISA's `ldmatrix` row/lane
layout](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-ldmatrix).
The exact production PTX load emitter is also compiled into an independent
hardware-load comparison probe. Four patterns cover all BF16 encodings, bit
bases, permutations and random data across all K steps, lanes and six CTAs:

- **786,432 packed 32-bit load values:** zero differences against hardware `ldmatrix`.
- **786,432 BF16 storage values:** zero differences against the original Query-copy layout.
- An intentionally wrong column mapping produces **196,608 differences**.

## Numeric evidence and scope

Three profiles use twelve generated conditions each: 64×256, 16×512 and the
actual 64×512 shape. Conditions cover masks, empty rows, duplicate indices,
partial blocks, ties, signed zeros, small/large inputs and sinks, with bounded
batch/sequence shapes. The smaller two profiles also execute the untouched
original kernels to qualify the adapter and the driver's PTX execution path.

Across **36 conditions / 456 comparisons / 11,583,616 checked elements**, all
bits match. Stores expose QK scores, probabilities, five softmax state vectors,
per-block value accumulators and FP32 values before BF16 rounding. Instrumented
and uninstrumented outputs also match. Counts include different comparison
types; this is not a claim that the untouched 64×512 kernel executed.

The measured environment is TileLang 0.1.12, **NVRTC 13.4**, driver API 13040,
PyTorch 2.14.1+cu130 and NumPy 2.5.2. Torch's CUDA build version and the actual
NVRTC compiler version are recorded separately. Unknown CUDA/PTX profiles,
opcodes, parameter layouts and modified qualification evidence are rejected.

The host reservation is **64 MiB**, with a 32 GiB available-memory floor. Torch
peak allocations are **9,130,496 bytes**, excluding driver/compiler memory.
Reservations are not RSS/OS limits. This experiment downloads no model weights,
does not qualify native Attention, and measures no generation speed or quality.

## Reproduction and capture

With the qualified optional reference environment:

```sh
vesper compare-attention-memory --device 0 \
  --reference-directory artifacts/reference --output artifacts/attention-memory.json
vesper capture-prefix --fixture artifacts/prefix-fixture/native-prefix.npz \
  --attention-reference ptx-memory --output artifacts/prefix-inputs.json
```

Capture explicitly selects the adapter and reruns its complete qualification.
It executes unchanged layer-0 prefix methods with original weights and embedding
rows, includes the full qualification record/hash and stops before MoE.
Four inputs from two token-ID sequences repeat bit-for-bit. Token IDs are
selected directly; this does not qualify tokenization, images or full-model
generation. The default capture path still performs the untouched-kernel guard.

[Raw adapter evidence](../results/2026-10-18-attention-memory/attention-memory.json)
and [summary / SHA-256](../results/2026-10-18-attention-memory/summary.json) are
complete, sanitized results. [Issue #7](https://github.com/shell-term/vesper/issues/7)
still includes native Attention, indexing/RoPE, Hyper-Connections and KV state.
