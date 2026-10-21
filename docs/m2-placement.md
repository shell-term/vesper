# M2 bounded helper-placement cycle

The PC operator targets 20 accepted decode tokens/s for one conversation,
while limiting development/measurement time and measuring output quality.
That remains a stretch target, not a measured result or this cycle's completion
condition. [M2 / #11](https://github.com/shell-term/vesper/issues/11) ends with
one controlled comparison and an adoption/rejection decision. M3's complete
reasoning/tool cases remain pending in #10.

## Hypothesis and selected candidate

The [earlier retained-session run](session-lifetime.md) records 17.72 seconds
of helper H2D/D2H sections across five whole-model calls, versus 1.87 seconds
in recorded expert-kernel sections. These are scoped host wall-time sections,
not a complete GPU profile. Its cache has only 17 helper hits among 1,200 expert
selections. Test whether storing those helper slots in RAM saves transfer/audit
time without increasing selected-payload store reads enough to offset the gain.

`host-retained` retains all forty MoE contexts like the `retained` comparator.
Per layer, both keep two primary-GPU slots and ten resident expert payloads
in total. The comparator has two helper-GPU slots plus six RAM slots; the
candidate has zero helper slots plus eight RAM slots. Original SSD bytes,
model arithmetic, trained routing, selected expert count, activation/KV
representation, accumulation and sampling are unchanged (`released-fp4`).
All original source checks and primary-GPU upload audits remain enabled.
The candidate compiles the same thirteen qualified Ada images; the comparator
also needs the qualified Ampere SHA-256 storage-audit image. No helper model
compute is introduced. This is an opt-in benchmark storage profile.

Both configurations keep the 96 GiB engine host admission cap, 32 GiB
available-memory floor, 2 GiB pinned cap and the same component reservation
ceilings. Actual residency/private commit and both GPUs are measured; moving
two slots per layer to RAM intentionally changes the allocation by tier.
Admission counters are not OS quotas.

## Fixed measurement plan

- Reuse the already qualified raw-text five-call request and cached original
  comparator; no new original execution or full intermediate trace.
- Four fresh native processes in order `retained`, `host-retained`,
  `host-retained`, `retained`, with two independently prepared index directories.
  Preparation copies/fetches zero expert payload bytes.
- Each timed process receives a 300-second cooperative budget. Stop at four
  attempts; no automatic extra retries, sweep or long quality capture. The
  planned 20-minute model-run budget excludes implementation/preparation/CPU
  checks, and setup/cleanup can overrun a checkpoint.
- Check all 646,400 logit elements, 200 ordered top-six routes/weight bits,
  selected-ID feedback/RNG, completion and final complete model state against
  the existing short reference. Verify the expected compiled-image set.
- Record setup, request/engine TTFT, prefill and subsequent decode separately,
  plus actual process/GPU memory, store reads, tier transfers and cleanup.
  OS/driver caches are not reset; alternating order reduces but does not remove
  their influence. Diagnostic tensor materialization remains timed.

The request has only two post-prefill decode calls per process and no task
score. Repeated timing applies to this input only. It cannot establish general
or production throughput, broad quality, 20 tps, or complete #10. Required
comparisons must complete; failures/timeouts remain incomplete. Adopt the
storage profile only within the measured scope if its benefit is consistent
and its exact-output/resource checks pass. A negative completed result can
reject it and still finish the finite investigation.
