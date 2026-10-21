# Longer input storage and indexed expert replay

> **Recorded PC evidence.** Results, requirements and next-step wording below
> describe this report's experiment/profile. They do not set the current
> development queue or require exact equality from every new candidate. See
> [current policy](quality-speed.md), [validation](validation.md) and
> [PC roadmap](roadmap.md). Raw evidence and original scope are unchanged.

Two independent gates pass. The indexed expert source reproduces the existing
**five-call, forty-layer text continuation**, with all **26,250 typed boundaries**
unchanged. A separate **63-token input/Engram comparison** matches **3,906 typed
boundaries / 30,247,560 storage elements** against the original implementation.
The latter uses synthetic hidden inputs; it does **not** execute a 63-call
Transformer trajectory, generate a completion or score model quality.

Issue #10 and M2 remain open. These results remove storage/admission obstacles
to longer evaluation without claiming that the three quality tasks have run.
Original weights, scales, activation/KV rounding, trained routing and sampling
are retained. There is no retraining, expert pruning or added quantization.

## Indexed original expert files

The [indexed store](../src/vesper/indexed_store.py) references immutable packed
files through a bounded, layer-local metadata manifest. It retains one aligned
direct-I/O buffer and one payload file handle, switching the handle when an
actual selected expert resides in another file. Replacement opens before the
old handle closes; a failed open retains the old source and leaves the target
unchanged. There is no buffered-I/O or network fallback.

The index permits at most the layer's **384 original expert identities**. It
does not alter the router's choices. Source paths are relative to an explicit
root, escaping paths are refused, source manifests/extent/provenance are bound,
and duplicate expert identities must agree on original payload hashes and
provenance. Every read retains the existing full payload/tensor SHA checks.
Index construction reads metadata and copies **zero expert payload bytes**.
The legacy 48-entry store and its default runtime configuration are unchanged.

The actual [native replay](../results/2026-10-20-indexed-generation/native.json)
uses **786 distinct original experts**, representing **14,777,303,040 logical
bytes**, from 80 existing packed-file receipts. Twenty-nine layers actually
read experts from two different original files. Across all forty layers, it
performs **1,200 full expert reads / 321 payload opens or switches**, with no
new expert downloads and no copied expert payloads.

Against the [previous sealed original execution](../results/2026-10-19-generation/original.json),
all **138,434,360 storage elements, 646,400 complete FP32 logits and 200 ordered
routes** match. The independent native driver again generates IDs
`[1014, 260, 223]`, decoded as `" time a "` including its trailing space.
This is the existing `Once upon a` corpus, rather than a new longer trajectory.
The [file-bound summary](../results/2026-10-20-indexed-generation/summary.json)
binds the raw replay, construction record and all forty executed index files.

## Explicit longer input profile

The opt-in [serial profile](../src/vesper/sequence.py) admits at most **512 model
calls**, 512 distinct input embeddings, 12,288 Engram rows per layer and 384
original experts per layer. The historical eight-call GenerationRequest profile
and all of its sealed evidence remain unchanged. A larger admission number is
not evidence of a prepared or numerically qualified whole-model run.

The three [declared quality tasks](../src/vesper/message_cases.py) now fit this
explicit profile's request admission. Their full prompts remain 63/30/293
tokens and request at most 190/61/388 serial model calls. The reasoning request
keeps the upstream default effort **75**. No prompt, tool schema or requested
generation limit was shortened to fit a fixture.

Actual preparation for the 63-token reasoning input imports **49 original
embedding rows** and **1,496 original Engram rows per layer 1/14**. It fetches
1,281,408 payload bytes in bounded ranges, with **no new expert payloads**.
Selected Engram metadata has an explicit 12 MiB parse limit and a conservative
66 MiB reader admission; the resident row cache remains **32 rows per layer**.
This preparation does not supply the routes or generated-token rows required
for a future whole-model reasoning trajectory.

## Original-address paging: measured failure and correction

The first reference attempt eagerly backed all selected original-address rows.
It was refused by the existing **4 GiB physical VMM cap**, before producing a
qualified capture. A direct driver query established that the actual minimum
allocation granularity here is **2 MiB**. Assuming that one tiny logical row
costs only its data bytes would therefore substantially understate residency.
The failed process has no captured before-exit cleanup snapshot; the published
summary records that limitation instead of asserting measured zero cleanup.

The optional [paged reference transport](../src/vesper/paged_tables.py) keeps
the original full tensor shapes, virtual row addresses and unchanged
`F.embedding` lookup. It makes only the next requested rows resident, with at
most **24 active logical rows per table**, a **512 MiB physical cap** and a
2 GiB device-free reserve. An unavailable logical row is rejected even when
another row happens to share its mapped physical page.

The successful capture reserves **204,086,444,032 virtual bytes** across the
full-shaped views, while actual physical backing peaks at **260,046,848 bytes
(248 MiB)**. These are different quantities. There are 504 activations and
10,820 page maps; all 10,820 are evicted by close. Four per-table close callbacks
own current mappings without accumulating one callback per historical page.
Physical/virtual ownership, active rows and owned context all return to zero.
The separate PyTorch device allocation peak is 316,781,568 bytes, with its
allocation returning from a zero baseline to zero after cleanup.

## Independent longer component comparison

The [original capture](../results/2026-10-20-extended-input/original.json) uses
the original AutoTokenizer, unchanged CPU NgramHash, original
`ParallelEmbedding.forward`, `ParallelEngramEmbedding.forward`, `Engram.forward`
and linear arithmetic. All prepared source rows are first checked through the
original full-address embedding lookup in bounded chunks. The complete
compressed vocabulary, 63 input IDs and prefix hash histories match; the actual
input-embedding lookup matches **322,560 BF16 elements**.

Both actual Engram layers then process all 63 prefix positions, for **126
original Engram executions**. Their hidden inputs are deterministic, finite
synthetic BF16 values including signed zeros, **not hidden states from the
forty-layer model**. All 62 typed/shaped intermediate boundaries per position
match the [independent native execution](../results/2026-10-20-extended-input/native.json).

At position 32, the native test corrupts bytes returned by the actual layer-14
direct reader after layer 1 has executed. The original-row SHA check refuses
them, no hash history is published, and a healthy retry matches every original
boundary and history. Both processes disable Python HTTP and fetch no weights
during comparison. Torch, TileLang, Transformers and SymPy are absent from the
native environment.

The [component summary](../results/2026-10-20-extended-input/summary.json) binds
the complete original/native/preparation files. Qualification pins the complete
original identity `64a1c54c777885661cb174f08b4257342c9e5c3c392929b4cd1ac4bc1e43e976`
and math goldens `e5a193ae67e448e146bc9219ab8afdf2b1e7fea70defb90e09cc073aa4e3c7e6`.
Rehashing an altered capture cannot qualify it. Regression tests reject missing
math/history, changed index bytes, skipped reads, false whole-model claims,
reference-framework/HTTP use and retained allocations.

## Measured host resources

| Process | Admission cap | Peak charged admission | Sampled resident peak | Sampled private-commit peak | Minimum sampled available RAM |
| --- | --- | --- | --- | --- | --- |
| Original 63-token component | 8 GiB | 2.56 GiB | 1.34 GiB | 3.36 GiB | 135.73 GiB |
| Native 63-token component | 8 GiB | 3.22 GiB | 0.53 GiB | 1.72 GiB | 136.03 GiB |
| Native indexed five-call whole model | 64 GiB | 46.42 GiB | 8.00 GiB | 19.94 GiB | 128.51 GiB |

Admissions are conservative accounting, **not RSS or operating-system quotas**.
All policy charges, pinned/device ownership, leases, reclaimers and in-flight
operations are zero after close; monitors stop. The full-engine default remains
**96 GiB host / 32 GiB system-available floor**. These bounded preparations fit
existing storage without an additional SSD; capacity for a larger route import
must be checked separately. The helper GPU still stores/audits original bytes;
this does not qualify Ampere model arithmetic. No token-rate claim is made.

## Reproduction and remaining work

The existing pinned metadata, controls and packed stores are prerequisites.
Use fresh run names: complete plans/captures are never replaced. Original and
native execution use separate environments as in the preceding generation gate.

```powershell
# Native environment: prepare original input rows in bounded HTTP ranges.
python -m vesper.sequence_probe prepare-input --run reasoning-new
# Optional original reference environment; configure the qualified NVRTC runtime.
python -m vesper.sequence_probe reference --run reasoning-new
# Independent native environment without the reference frameworks.
python -m vesper.sequence_probe native --run reasoning-new
python -m vesper.sequence_probe check --run reasoning-new
# Reuse the earlier all-layer original capture with indexed existing stores.
python -m vesper.sequence_probe indexed-prepare --run indexed-new
python -m vesper.sequence_probe indexed-native --run indexed-new --reference results/2026-10-19-generation/original.json
python -m unittest discover -s tests -v
```

CPU tests additionally exercise 49 synthetic full-sized expert objects across
the historical 48-entry boundary, a 1,025-row fixture across the historical
1,024-row bound, 1,000 paged activations and a 512-call transaction driver using
a fake engine. Those are capacity/failure tests, not model-quality evidence.

Next, integrate bounded original-route preparation with the larger profile,
capture actual whole-model hidden/logit/state trajectories for the complete
quality requests, and score their real completions. Generated-token embeddings,
Engram rows and expert routes must be admitted without remote calls in native
execution. Reference token-output capacity and model-call capacity must be
budgeted separately at the profile boundary. Compilation reuse and measured
TTFT/decode performance remain Issue #11 work after the relevant exact-output
gates. Neither new gate closes #10 or M2.
