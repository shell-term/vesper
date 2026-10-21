# PC quality and speed workflow

The PC objective is useful generation speed with as much output quality as
possible, within the 96 GiB engine host cap and 32 GiB available-memory floor.
Quality variation is permitted when the measured speed/memory tradeoff is
worthwhile. Exact intermediate equality is a reference tool rather than the
acceptance rule for every candidate. This direction does not change Spark's
qualification gates or retrospectively broaden existing results.

## Comparator and candidate identity

Keep original tensor bytes immutable and retain a released-precision comparator.
Existing exact expert, block, short-generation and coding results qualify only
their recorded implementations and inputs.

Each experiment records a numerical profile independently of the workspace
execution profile `native`. Declare changed weight/scale precision, activation
or KV representation, expert count, routing and sampling. An approximate
candidate cannot reuse a released-precision qualification merely because its
decoded answer happens to match. No lossy candidate is implemented or promoted
by this workflow document.

The expert tensors are already packed **FP4 with E8M0 scales**. The current FP8
compute path unpacks those original tensors for arithmetic. An additional
compression experiment therefore starts from an already compressed source.
Measure decompression, staging and device execution as well as stored bytes.
Lossless compression may cost more CPU time than it saves in I/O; lossy
compression may affect outputs. Both require measurements.

## Development checks

| Situation | Check |
| --- | --- |
| Local edit | Relevant CPU/resource/lifecycle tests |
| Exact execution optimization | A short qualified reference with final logits, selected IDs, routing/RNG and final state; investigate mismatches |
| Approximate numerical candidate | Named task outputs/scores against the comparator, numerical differences and failures; exact equality is not required |
| Timing run | Bounded real generation, explicit startup/prefill/TTFT/decode timing, memory and actual I/O; omit routine full intermediate fingerprinting |
| Unexplained mismatch or major release gate | Opt-in detailed tracing of the affected operators/positions; broader checks when the result justifies them |

Preserve the full prompt and generation budget of a named quality task. A
shortened smoke test is a different scope and cannot complete an earlier
reasoning/coding/tool-use evaluation. Do not rescore interrupted runs as passes.
The earlier full coding comparison remains a comparator. The interrupted
67-call reasoning reference is partial, unscored and has no completed reference
execution receipt; its preparation and private records are retained.

## Measurement and selection

First measure the execution path rather than infer token speed from cache-hit
ratios or synthetic transfers. The current diagnostic path recreates each
layer's MoE execution/cache context on every call. Retaining bounded contexts
across calls is the first precision-preserving candidate; it must be measured,
not assumed faster. Weight integrity, cleanup and the host/device reserve
checks remain active.

Report setup separately from request TTFT and decode. State whether Python
startup, reference checks, diagnostic tensor materialization and output
fingerprinting are included. Use identical requests, sampling, budgets and
hardware for comparisons. Report cache state and order; one short A/B pair is
exploratory rather than a repeated performance conclusion.

Compare output quality, TTFT, decode throughput, memory and storage/transfer
traffic together. Retain the useful alternatives rather than selecting solely
on byte reduction. Introduce lossy transformations only as named experiments
with their actual changes and measured effects. Small task scores do not
establish broad reasoning/coding/tool quality.

Track the comparator's open evaluations in [#10](https://github.com/shell-term/vesper/issues/10)
and measured candidates in [#11](https://github.com/shell-term/vesper/issues/11).
Scoped performance work can use already qualified short inputs while broader
#10 evaluations remain open; this does not close #10 or M2.

## Initial session-lifetime experiment

`python -m vesper.session_benchmark prepare` creates only source indexes for
the already qualified five-call raw-text input. It reuses the legacy input
explicitly and copies/fetches zero expert payload bytes. New output is confined
to the bound PC workspace's `artifacts/pc-heterogeneous/performance/` tree.

Run `python -m vesper.session_benchmark baseline` and then
`python -m vesper.session_benchmark retained` in separate native processes.
Both use the same request, six-expert RAM allowance per layer, two primary and
two helper slots, GPU upload audits and the 96 GiB engine admission cap / 32 GiB floor.
The benchmark explicitly admits up to 2 GiB of pinned staging memory: forty
retained layer owners currently have two buffers each (about 1.40 GiB total).
These buffers count toward the host cap. Retaining all forty owners also keeps
their conservative legacy component allowances live; actual RSS and private
commit are measured separately. A future serial staging pool could
share them; the default runtime's 1 GiB pinned ceiling remains unchanged.
Only context lifetime changes. The experimental retained path holds those
bounded contexts across tokens; it is not the default trajectory provider.

Each process has a default **300-second cooperative budget**, checked before
each MoE layer and model call. Setup/cleanup operations can overrun a checkpoint;
this is not an OS execution deadline. An exhausted budget is reported as such,
with no complete-reference match or completed quality claim.
Resource admission refusals are likewise incomplete results with cleanup and
memory observations, rather than completed output comparisons.

The quick released-precision check compares every full-vocabulary logit hash,
ordered routes/weights, selected-ID feedback and RNG offset, plus final complete
model state. It omits full intermediate fingerprinting. Diagnostic operators
still materialize intermediate tensors, and that cost is included; these
numbers are measurements of this prototype, not a finished production engine.
The raw continuation has no reasoning/coding/tool task score. No new lossy
numerical profile is implemented by this context-lifetime experiment.
