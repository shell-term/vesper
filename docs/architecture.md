# Architecture under investigation

## Requirements

1. Preserve the released model's precision and semantics.
2. Leave host memory available for other work.
3. Maximize throughput within those constraints.

The name Vesper and the open-source direction are selected. The mechanisms below
remain hypotheses until measurements justify them.

## Memory tiers

Store original packed weights and scales on disk. Keep selected expert blocks in
a bounded RAM cache and GPU caches. Engram is a separate, huge table whose rows
are addressed by token IDs. Avoid a model-wide mmap cache silently consuming the
RAM intended for other applications.

The provisional full-engine host cap is 96 GiB, initially up to 80 GiB for expert
cache and 16 GiB for other owned state. A 32 GiB available-memory floor applies
independently. Runtime pressure must shrink caches or reject work. Fixed
allocations, pinned staging, metadata, scratch and Engram row cache all count.

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

## Move work to resident weights

A native expert includes 18,800,640 bytes of FP4 data and E8M0 scales. One
5120-element BF16 input and output total 20,480 bytes. This motivates sending a
token's activation to the GPU holding the expert instead of moving its weights.

This is expert parallelism, not a claim of a newly invented algorithm. Its value
on an asymmetric consumer pair depends on cache hits, the helper GPU's compute,
host staging and latency. The initial 4090/3080 Ti pair reports no direct peer
access, so an explicit portable pinned-host path is measured.

Keep expert contributions separate and aggregate in the required order. Arbitrary
GPU-local partial sums could change rounding and violate the reference check.
Ampere also needs a verified arithmetic path for the reference's FP8 operations.
The current Ampere candidate failed that check, so the implemented helper moves
original bytes through pinned host memory and performs no model arithmetic.

## Schedule by useful work per byte

Speculative verification loads the union of experts selected by all its tokens.
A larger draft can therefore increase disk traffic without enough accepted
tokens to pay for it. Measure span lengths and disable speculation where it loses.
Change scheduling and draft extent, never the target's router or distribution.

[SpecFit](https://arxiv.org/abs/2610.15417) is a relevant reference, measured on a
different model with host-memory offload. Transfer to this SSD-bound model is
unproven. Existing DeepSeek experiments also report neutral or negative gains
from generic DSpark use; [JigSawPT's report](https://github.com/JigSawPT/deepseek-v41-flash-on-5090)
is useful for its negative measurements, not an automatic fidelity certificate.

## Ideas currently deprioritized

In an initial 3 MiB checkpoint-prefix sample, zlib level 1 saved about 2% of
native FP4 weight bytes. That is a small, nonrepresentative sample, but it offers
no evidence for fitting the original expert pool into RAM with generic lossless
compression. Revisit only with stronger measurements.
