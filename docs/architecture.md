# Architecture under investigation

This page describes the PC design and hypotheses under the
[current quality/speed policy](quality-speed.md). It does not change another
platform's qualification gates. Existing exact prototype behavior below is a
comparator implementation, not a ban on declared approximate alternatives.

## Requirements

1. Preserve as much useful output quality as possible while improving generation speed. The PC operator permits measured quality changes for speed/memory gains; retain the released-precision path as a comparator and declare approximate numerical profiles separately.
2. Leave host memory available for other work.
3. Maximize throughput within those constraints.

The name Vesper and the open-source direction are selected. The mechanisms below
remain hypotheses until measurements justify them.

## Memory tiers

Keep original packed weights and scales immutable on disk; any transformed
candidate assets have a separate profile and provenance. Keep selected expert blocks in
a bounded RAM cache and GPU caches. Engram is a separate, huge table whose rows
are addressed by token IDs. Avoid a model-wide mmap cache silently consuming the
RAM intended for other applications.

The current expert source is already packed FP4 with E8M0 scales. FP8 execution
does not mean these experts are stored as uncompressed eight-bit weights.
Further precision reduction needs actual quality and latency measurements;
smaller files alone do not establish a faster or acceptable model. See the
[quality/speed workflow](quality-speed.md) for candidate and validation rules.

The provisional full-engine host cap is 96 GiB, initially up to 80 GiB for expert
cache and 16 GiB for other owned state. A 32 GiB available-memory floor applies
independently. Runtime pressure must shrink caches or reject work. Fixed
allocations, pinned staging, metadata, scratch and Engram row cache all count.

The [whole-runtime governor](memory-policy-validation.md) now enforces a shared
admission policy across components/threads. The 80/16 GiB split remains a sizing
hypothesis, not a tested full-model allocation or required fixed partition.
Typed resources and conservative existing component envelopes share the 96 GiB
ceiling; process residency/private commit and driver/compiler/reference effects
are observed separately. Retained KV/history are never pressure-evicted.
Representative resource geometry and scoped forty-layer numerical runs are
recorded. Broader task-quality evaluation remains #10; a charge counter does
not establish the actual memory cost of retaining every context.

## Implemented cache prototype

The [three-expert probe](cache-validation.md) uses exclusive tiers: the compute
GPU, a storage-only helper GPU and RAM. SSD remains the immutable backing store.
Each request promotes the exact requested expert to the compute GPU. A displaced
compute entry moves to the helper, whose displaced entry may enter RAM. LRU
replacement never changes the expert ID, route weight or arithmetic.

Two portable pinned buffers keep promotion bytes separate from demotion bytes.
One fixed aligned direct-I/O buffer avoids mapping a model-wide file cache. GPU
slot metadata is published only after a completed, audited upload. A serialized
request holds its slot until computation finishes; nested requests are refused.
Failed transfers can discard a cache copy, but the original SSD copy remains.

RAM arrays are allocated lazily. Admission checks the buffer cap and available
memory; request boundaries release RAM entries under pressure and refuse work if
the available-memory floor cannot be restored. Fixed buffers are included in the
ledger. Python/driver/compiler objects and diagnostic outputs are outside it;
there is no whole-process OS memory guarantee. GPU pressure currently refuses
work rather than resizing fixed slots. This prototype has no asynchronous
prefetch, routing predictor, layer aggregator or Engram cache.

The [profiled GPU-audit option](cache-performance.md) retains the same cache
capacities and movement policy. Whole-file-block/tensor SHA-256 checks first
authenticate source bytes on the CPU. The store then derives immutable 4 KiB
SHA-256 tables under the buffer budget. Every subsequent GPU upload hashes all
chunks on that GPU and compares the returned digests before publication. RAM and
other intermediate host copies defer checks to this admission boundary. They
cannot enter model computation without a matching GPU audit. This is a change
in verification placement, not in model arithmetic or routing.

## Move work to resident weights

A native expert includes 18,800,640 bytes of FP4 data and E8M0 scales. One
5120-element BF16 input and output total 20,480 bytes. This motivates sending a
token's activation to the GPU holding the expert instead of moving its weights.

This is expert parallelism, not a claim of a newly invented algorithm. Its value
on an asymmetric consumer pair depends on cache hits, the helper GPU's compute,
host staging and latency. The initial 4090/3080 Ti pair reports no direct peer
access, so an explicit portable pinned-host path is measured.

For the released-precision comparator, keep the required accumulation order;
GPU-local partial sums can change rounding. Exact Ampere compute needs a verified
path for the reference's FP8 operations. A deliberately approximate compute or
aggregation profile instead needs its own quality/speed evaluation. The recorded
Ampere candidate failed its exact gate and remains unqualified; the implemented
helper stores/audits bytes and performs no model arithmetic. See deferred #12.

The [session-lifetime experiment](session-lifetime.md) retains forty bounded MoE
owners rather than recreating them every call. Its short comparison is promising
but not a production speed result. Measured integrity/chunk preparation and
helper transfers motivate testing a shared serial staging pool, integrity work
placement and helper use. Extra compression, prefetch and overlap are candidates
to select from evidence, rather than mandatory steps in that order.

## Schedule by useful work per byte

Speculative verification loads the union of experts selected by all its tokens.
A larger draft can therefore increase disk traffic without enough accepted
tokens to pay for it. Measure span lengths and disable speculation where it loses.
For exact speculation, preserve the configured target profile's distribution
and rollback semantics. Intentionally changing that target is a separate
approximate model experiment, not exact acceleration of the released comparator.

Previously collected reading leads include [SpecFit](https://arxiv.org/abs/2610.15417)
and [JigSawPT's report](https://github.com/JigSawPT/deepseek-v41-flash-on-5090).
Their existence is not qualification of any Vesper profile; measure the selected
mechanism on the actual workload/hardware before adopting it.

## Ideas currently deprioritized

In an initial 3 MiB checkpoint-prefix sample, zlib level 1 saved about 2% of
native FP4 weight bytes. That is a small, nonrepresentative sample, but it offers
no evidence for fitting the original expert pool into RAM with generic lossless
compression. Revisit only with stronger measurements.
