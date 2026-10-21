# Longer whole-model trajectory validation

> **Recorded PC evidence.** Results, requirements and next-step wording below
> describe this report's experiment/profile. They do not set the current
> development queue or require exact equality from every new candidate. See
> [current policy](quality-speed.md), [validation](validation.md) and
> [PC roadmap](roadmap.md). Raw evidence and original scope are unchanged.

The complete, previously declared Python-expression task passes independent
original/native execution and actual completion scoring. Its unchanged
chat-format prompt has **30 tokens** and requests at most 32 new tokens.
Both executions produce **`xs[-1]` and EOS in 34 successful model calls**, with all
**178,423 typed boundaries / 962,450,386 storage elements** matching exactly.
This qualifies one small expression task; broader coding, reasoning, tool-use,
chunk prefill and end-to-end performance remain open.

This is the **pc-heterogeneous** track. Select the
[PC workspace](../platforms/pc-heterogeneous/README.md) and pass
`python -m vesper.workspace check` first. New private runs and public results
use `artifacts/pc-heterogeneous/` and `results/pc-heterogeneous/`. Execution
receipts bind the raw report hash to the actual platform/model, source commit,
dirty digest and run ID. Shared code does not transfer qualification to Spark.

The preceding [longer-input/storage gate](sequence-storage-validation.md)
qualifies indexed short generation and a separate synthetic-hidden Engram
comparison. Those results alone do not qualify this longer whole-model task.

## Complete coding-task evidence

The [file-bound summary](../results/pc-heterogeneous/2026-10-20-coding-trajectory/summary.json)
binds the [original capture](../results/pc-heterogeneous/2026-10-20-coding-trajectory/original.json),
[native capture](../results/pc-heterogeneous/2026-10-20-coding-trajectory/native.json),
all per-call files, forty index receipts, input plan and preparation records.
The summary's context records evidence publication. The three separate
execution receipts retain the actual preparation/reference/native source
contexts; publication does not retrospectively change an execution identity.

| Observed check | Result |
| --- | --- |
| Complete original chat prompt | 30 original token IDs; unchanged declared request |
| Successful single-query calls | 34 through all forty layers and both Engrams |
| Exact typed boundaries / storage elements | 178,423 / 962,450,386 |
| Complete FP32 logits / ordered top-six routes | 4,395,520 / 1,360 |
| Distinct selected original experts | 3,067 |
| Selected IDs including EOS | `[38753, 26913, 19, 63, 1]` |
| Actual decoded completion | `xs[-1]` |
| Declared expression cases | `[2, 5, 9] -> 9`, `[7] -> 7`, `[-3, 0] -> 0`: all pass |
| Failed whole-token attempt | One after final logits/sampling at position 29; complete state/RNG restored; healthy retry matches |
| Actual native expert reads | 8,400 / 157,925,376,000 bytes, including the failed attempt |
| Native compiled-image cache | 9,876 hits / 14 misses; all fourteen images match the preceding qualified set |
| Reference/native HTTP calls and fetched weight payloads | Zero in both independent executions |

Each original/native call independently records every typed math boundary,
ordered route, complete model-state digest and sampling RNG offset. Native
execution supplies its own encoded prompt, route choices, sampled IDs and
feedback. The same answer alone is not the comparison criterion.

The reference retains the previously qualified integer-only Query memory
adapter described in [the official-arithmetic memory gate](attention-memory-validation.md).
Its floating-point/control instructions are retained. This is not an
untouched-full-kernel claim. Original-address embedding paging retains
`F.embedding`, original row offsets and BF16/FP8 views.

The first preparation attempt, started before platform isolation, failed on a
checkpoint connection reset before producing a complete plan or qualified
capture. Completed immutable expert objects and 1,243,541 bytes of unfinished
input caches were reused explicitly; copied rows are revalidated by original
address and SHA-256. The new PC preparation imported 433 additional experts
(8,140,677,120 payload bytes), with zero expert payload copies between banks.
The reader now retries transient transport failures at most four times, with
1/2/4-second backoff. Certificate, range and payload-validation failures still
refuse immediately. New imports leave no persistent NPZ copies.

The [published-evidence tests](../tests/test_coding_trajectory_published.py)
recheck all file bindings and both complete captures. They also refuse a
rehashed altered original completion and a corrupted failed-token recovery
record even when successful math is unchanged.

## Bounded preparation and offline execution

[Preparation](../src/vesper/trajectory_prepare.py) stages actual original
embeddings, both Engrams, all forty blocks and the final original head/sampler.
It resolves the actual unmodified top-six route between stages. This phase may
fetch missing original payloads; it is not independent reference evidence or
an offline model-quality result.

Complete task prompts, their token-ID lists and scoring cases are pinned to the
preceding independent original CPU [message-format gate](message-format-validation.md).
Both whole-model executions must retain those complete inputs; a shorter prompt,
lower reasoning effort or altered tool schema cannot silently qualify.

The [expert bank](../src/vesper/expert_bank.py) reuses existing immutable packed
sources. Each new expert is checked by the existing six-tensor importer,
packed once and published by a directory rename. A temporary NPZ belongs to
one TemporaryDirectory and is removed after packing; no growing union of
expert payloads is recopied. Prepared indexes contain only metadata.

New packed payloads have a separate **64 GiB storage cap**, with a **64 GiB
disk-free reserve** and bounded transient imports. This storage cap is not a RAM
allocation. At most four expert imports share eight payload-transfer workers.
Failed imports retain no published token; completed immutable objects can be
reused when preparing a fresh run. Raw model bytes remain ignored artifacts.

The subsequent [original execution](../src/vesper/trajectory_reference.py)
uses unchanged upstream `generate`, `Transformer.forward`, `Block.forward`,
gate, expert, Engram and sampling arithmetic. An offline loader supplies the
same original tensor bytes from the bank, checking payload and per-tensor
SHA-256 plus provenance. The original gate chooses its own experts. No native
planned route or sampled ID supplies an original execution decision.

The original prompt chunk is serialized by the already documented adapter;
this qualifies the single-query path rather than chunk prefill. Intermediate
prompt calls do not consume sampling RNG. NgramHash admits 512 input positions;
the upstream generation adapter separately provides 513 token-array slots,
including the output beyond the final admitted input position.

[Native replay](../src/vesper/trajectory_native.py) independently encodes the
request, computes every route and feeds its own sampled IDs back. It has no
reference frameworks installed/imported and no HTTP fallback. Original
provenance is an integrity guard, never a substitute for routing or math.

## Original-address row transport

Both Engrams retain the [paged original-address transport](sequence-storage-validation.md).
The full input-embedding view now uses the same bounded mechanism, including
rows that straddle two physical pages. Original virtual shapes, BF16/FP8 views,
row offsets and `F.embedding` remain unchanged. At most 24 logical rows per
table are active. All five paged tables share the **512 MiB physical backing
cap** and the device-free reserve; virtual reservation is reported separately.

The historical reference profile and its original eager-table default are
unchanged. Missing logical rows are rejected even when a neighboring row's
physical page is already mapped. No zero row or lossy row remapping exists.

## Scoped compilation reuse

The optional [compilation cache](../src/vesper/compile_cache.py) holds immutable
NVRTC cubin bytes in one process. It caches no CUDA module, context, pointer,
model weight or mutable launch handle. The default compiler remains uncached.
Architecture, compiler arguments and the sealed compiler/source inputs bind
each entry. Every hit checks the immutable binary's digest and source/binary
stamps. Misses and final close check every sealed input by content, including
all toolkit headers; changing a compiler input rejects the scope. A completed
capture is written only after that final check succeeds.

There are at most 32 entries / 32 MiB of cubin payload, covered by a 96 MiB
conservative host admission. Binary checksums and the actual compiler metadata
are reportable; local toolkit paths and cubins are not committed. Cache hits
alone do not establish exact model outputs or any measured token rate.

The [complete cached short replay](../results/2026-10-20-compilation-cache/native.json)
passes all 26,250 original typed boundaries / 646,400 FP32 logits from the
existing five-call corpus, with 1,476 cache hits and fourteen actual compiled
variants. The [faster-checking compiler-only follow-up](../results/2026-10-20-compilation-cache/compiler.json)
reproduces every one of those fourteen cubins byte-for-byte. Its CPU compilation
check does not run the GPU model. The
[file-bound summary](../results/2026-10-20-compilation-cache/summary.json)
also records a rejected initial probe-list mistake: the Ampere entry requested
the expert variant instead of the actual integrity variant. No qualified
capture came from that mistaken compile-only inventory. The helper GPU still
stores/audits bytes; no Ampere model arithmetic is qualified here.

The first cache checker walked the toolkit inventory on every hit; its full
replay includes that overhead. The follow-up validates all headers on misses
and final close, while checking only the immutable binary and source/binary
stamps on hits. These executions are correctness evidence, not a matched
comparison of TTFT, decode throughput or end-to-end cache performance.

That short cache gate completed before workspace isolation was introduced;
its historical result path and original unbound captures are retained. They
are not retrospectively stamped as newly bound GPU executions. A PC task can
explicitly import a completed legacy input snapshot with `migrate-input`:
only embeddings, selected rows and index metadata are copied, with zero expert
payload copies and an unchanged logical plan identity. New original/native
execution then uses the PC namespace and fresh execution-context receipts.

## Streaming evidence and state recovery

[Evidence files](../src/vesper/trajectory_evidence.py) retain all typed/shaped
math hashes and whole-token state for every original/native call. Each call is
a separate file, bounded to 4 MiB and bound by its byte count and SHA-256. The
manifest binds the ordered complete file list, original source/method identities,
weights/index provenance, decoded sampled IDs and deterministic task score.
This avoids keeping the growing corpus's intermediate fingerprints in RAM.

Discovery does not self-qualify by recalculating checksums. The independent
original identity and complete goldens must be pinned after inspection.
Validation rejects missing/changed calls, early termination, substituted
feedback IDs, changed RNG progression and omitted typed boundaries.

Native replay also injects an actual failure after the last prompt call's final
logits and sampling. The token transaction must restore the complete model,
hash history and RNG; the healthy retry must match the original call. A failed
call produces no continuation ID. This is additional work, distinct from
successful call counts and from CPU capacity tests.

The conservative native/preparation admission cap is **64 GiB**, the original
reference cap **32 GiB**, and the system-available floor remains **32 GiB**.
Report actual sampled RSS/private commit, available RAM, GPU ownership and
cleanup separately from admission counters. The default full-engine cap remains
96 GiB. No OS memory quota or full-checkpoint download is implied.

Actual observations from this task are separate from those admission caps:

| Sampled observation | Original reference | Independent native |
| --- | --- | --- |
| Resident process peak | 9.37 GiB | 7.98 GiB |
| Private process commit peak | 23.34 GiB | 19.95 GiB |
| Minimum system-available RAM | 93.68 GiB | 87.46 GiB |
| Host admission cap / system-available floor | 32 / 32 GiB | 64 / 32 GiB |
| Charged host bytes after close | 0 | 0 |

The original's five paged tables peak at **194 MiB of physical backing** under
their shared 512 MiB cap. All 6,141 mapped pages are evicted; physical and
virtual reservations are zero after close. Framework tensor allocation returns
to its zero-byte baseline. Both policies close with no leases, reclaimers,
in-flight operations or monitor errors. These are sampled observations and
owned-allocation checks, not an OS quota or a measured generation rate.

## Reproduction

Existing pinned controls, tokenizer and earlier immutable stores are required.
Use a fresh run name; complete plans, call records and captures are not replaced.
The optional original reference environment is separate from native execution.

```powershell
python -m vesper.workspace check
python -m vesper.trajectory_probe prepare --run coding-new --case coding-python-expression
# Qualified original environment; configure its original NVRTC runtime.
python -m vesper.trajectory_probe reference --run coding-new
# Inspect and pin a new original capture before independent native recheck.
python -m vesper.trajectory_probe native --run coding-new
python -m vesper.trajectory_probe check --run coding-new
python -m unittest discover -s tests -p test_coding_trajectory_published.py -v
python -m unittest discover -s tests -v
```

One expression task cannot establish general coding, reasoning or tool-use
quality. The two other declared tasks, broader inputs and end-to-end performance
remain separate work. Issue #10 and M2 remain open.
