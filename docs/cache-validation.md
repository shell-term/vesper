# Tiered expert cache validation — 2026-10-18

> **Recorded PC evidence.** Results, requirements and next-step wording below
> describe this report's experiment/profile. They do not set the current
> development queue or require exact equality from every new candidate. See
> [current policy](quality-speed.md), [validation](validation.md) and
> [PC roadmap](roadmap.md). Raw evidence and original scope are unchanged.

Three different original experts now execute on the RTX 4090 after movement
between SSD, RAM, RTX 3080 Ti storage and RTX 4090 compute. All **232 successful
calls** match the unchanged official expert in all thirteen checked stages,
including **1,187,840 final BF16 output values with zero mismatches**.

This is a single-token cache prototype. It does not implement a model layer,
router, attention, Engram or text generation. The controlled request pattern is
synthetic and provides no estimate of real router locality or generation speed.

## Original bytes and computation

The model is `deepseek-ai/DeepSeek-V4.1-Flash-1017`, revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`. The fixtures identify layer 0 experts
0, 1 and 2. Each contains six original tensors, including packed FP4 bytes and
E8M0 scales, totaling 18,800,640 bytes. Together they form a 56,401,920-byte aligned
store. No additional quantization, pruning or cache-dependent routing is used.

The store validates expert IDs, tensor source names, layouts, full-block hashes
and all six original tensor hashes. Reads bypass the OS buffered file-cache
path; there is no buffered fallback. The drive/controller's internal cache is
not controlled, so these are direct-file read counts, not measured NAND traffic.

The 4090 executes the previously checked native FP8 arithmetic. The 3080 Ti
stores original bytes only: its BF16 arithmetic candidate remains unqualified.
The cards do not expose direct peer access on this machine. Transfers therefore
use portable pinned host buffers and are synchronous. Every upload is read back
and hashed before its cache entry becomes visible.

The official comparison uses the original `linear()` and `Expert.forward()` AST
bodies and pinned kernel source. The previously documented TileLang 0.1.12 NVRTC
launch-handle isolation remains active; neither arithmetic nor source is changed.
The compiled Vesper kernel hash matches the previous resident-expert experiment.

## Controlled cases

The repeating expert order is `0, 1, 0, 2, 2, 1, 0`. Each call compares input FP8
values/scales, W1 and W3 BF16 outputs, five FP32 SwiGLU intermediates, the BF16
down-input boundary, its FP8 values/scales, and the final W2 BF16 output.

| Phase | Successful calls | Direct-store reads | Result |
| --- | ---: | ---: | --- |
| One slot in each GPU and one RAM expert | 224 | 3 | All checked bits match; all three identities retained after initial reads |
| Injected pressure and recovery | 1 | 1 additional | RAM cleared; one request refused before I/O; restored observation permits exact reload |
| Zero RAM, one slot in each GPU | 7 | 5 | Forced evictions/reloads preserve all checked bits |

The 224-case phase crosses seeds `20261018, 0, 1, 42`, input scales `1, 16`, route
weights `None, 0.25, 1.5, 0`, and the seven-call pattern. The recovery and zero-RAM
phases reuse the last input/route, including the zero-weight condition. Successful
calls came from SSD 9 times, RAM 95 times, helper GPU 64 times and compute GPU
64 times. There is also one intentionally refused request, excluded from the
232 successful output comparisons.

Pressure injection changes only the available-memory observation to one byte
below the configured floor. The cache releases its RAM entry and refuses work
when the observation remains low. Restoring the actual provider triggers an
original-byte reload. No real memory exhaustion, page-file change, GPU reset or
interruption of other workloads is used.

## Resource limits

| Explicit owned state | Peak bytes |
| --- | ---: |
| Host buffers plus metadata allowance | 76,251,136 (72.7 MiB) |
| RTX 4090 weights and expert workspace | 18,903,528 |
| RTX 3080 Ti weight storage | 18,800,640 |

The probe's host-buffer cap is 96 MiB, separate from the provisional **96 GiB**
full-engine design. Host state comprises two 18,800,640-byte pinned buffers, one
equally sized direct-I/O scratch buffer, one RAM expert and a 1 MiB metadata
allowance. With zero RAM the reservation is 57,450,496 bytes. RAM allocation is
lazy and admission checks both the cap and a 32 GiB available-memory floor.

This ledger excludes Python objects, CUDA contexts/modules, compiler work,
returned diagnostic arrays and the three resident official comparison experts.
It is not a whole-process or OS memory limit. Actual available host memory was
sampled separately and stayed at least 99.09 GiB during the successful checks.
Both runs finish with zero tracked host reservations and zero live explicit
device allocations. Driver/PyTorch caches and contexts are outside that claim.

The library exposes provisional 96 GiB / 80 GiB / 32 GiB host cap / RAM capacity /
available-floor defaults, but this probe stores only three experts and allocates
at most one RAM expert. It does not validate operation at full-engine scale.
Fixed GPU slots currently refuse work under low GPU headroom; they do not shrink.

## Timing limits and next work

Observed median call times were 0.99 ms for a compute-GPU hit, 44.33 ms from the
helper GPU, 69.19 ms from RAM and 77.60 ms from the direct store. These calls
include complete-block hashes, upload audit readbacks, evictions and diagnostic
output copies. Source groups have different eviction work and few store samples;
the figures are not an isolated bandwidth comparison or a cache-speedup claim.

Synchronous staging and integrity audits dominate the transfer paths. Next work
should separate transfer, hashing and computation timing, then test overlap and
prefetch under the same exact-output checks. Useful full-model scheduling also
requires the unchanged router and aggregation order; a synthetic trace cannot
choose a predictor or establish production cache capacity.

## Reproduction and records

Follow the [README setup](../README.md#official-reference-comparison) and the
[Windows reference-runtime notes](expert-validation.md#reproduction). Fetch
experts 0, 1 and 2 with `fetch-expert-fixture --expert N`, build a new store using
`build-store --fixtures ... --directory ...`, then run `validate-cache` as shown
in the [cache commands](../README.md#bounded-cache-probe). The validator fixes its
small caps and test pattern; it requires three distinct, matching fixtures.

The small-store builder currently accepts one to four complete expert fixtures
and refuses an existing output directory. A full-checkpoint importer, asynchronous
storage, helper-GPU computation and full inference are not implemented.

[Complete JSON records](../results/2026-10-18-cache/README.md) preserve every case,
all checks, per-tier traffic counters, fixture provenance and cleanup results.
The 128-case resident-expert matrix was rerun after the executor refactor and
again passed. CPU boundary tests cover failed/corrupt transfers, allocation and
pressure failures, exclusive identities, no-helper operation, nested request
refusal and slot ownership across threads.
