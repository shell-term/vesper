# Bounded natural-language text continuation

> **Recorded PC evidence.** Results, requirements and next-step wording below
> describe this report's experiment/profile. They do not set the current
> development queue or require exact equality from every new candidate. See
> [current policy](quality-speed.md), [validation](validation.md) and
> [PC roadmap](roadmap.md). Raw evidence and original scope are unchanged.

This gate advances the forty-layer backbone from fixed BOS/PAD inputs to a
short raw-text completion. The request is **`Once upon a`**, encoded by the
original tokenizer as **`[16600, 4465, 260]`**, with three new tokens,
temperature **0.7** and seed **2026**. Each side chooses its own continuation
and feeds the sampled ID into its next complete model call. No reference
trajectory is supplied to the native driver.

Evidence preparation and independent execution are separate operations.

## Verified original and native result

The original loop produces these three tokens; the last token is a space:

```json
{"prompt":"Once upon a","completion":" time a ","selected_ids":[1014,260,223],"stop_reason":"max_new_tokens"}
```

The input IDs in the last two calls come from the earlier actual selections:

| Position | Input ID | Sampling requested | Selected ID | RNG offset after | Typed boundaries |
| --- | --- | --- | --- | --- | --- |
| 0 | 16600 | No | None | 0 | 5,247 |
| 1 | 4465 | No | None | 0 | 5,247 |
| 2 | 260 | Yes | 1014 | 4 | 5,252 |
| 3 | 1014 | Yes | 260 | 8 | 5,252 |
| 4 | 260 | Yes | 223 | 12 | 5,252 |

Independent native replay matches **all 26,250 typed/shaped boundaries /
138,434,360 storage elements**, **646,400 complete FP32 logits**, **200 original
Block calls / ordered router decisions**, four original input embedding rows,
120 Engram rows per layer and **786 original experts**. The original source,
method identities, grouped weight conversions, output fixture and Engram
controls are unchanged from the earlier all-forty-layer qualification.

The [sealed original capture](../results/2026-10-19-generation/original.json),
[complete native replay](../results/2026-10-19-generation/native.json) and
[file-bound summary](../results/2026-10-19-generation/summary.json) include all
intermediate hashes, the actual decoded result and resource measurements.
All probabilities, exponential noise, scores, selected IDs and RNG offsets
match exactly. Both independent executions disable Python HTTP; all four
reference frameworks are absent from the native environment. The native
transaction commits exactly five calls with no pending recovery or cleanup.

## Measured resources

| Process | Host admission cap | Peak charged admission | Sampled resident peak | Sampled private-commit peak | Minimum sampled available RAM |
| --- | --- | --- | --- | --- | --- |
| Original reference | 32 GiB | 21.13 GiB | 9.41 GiB | 23.90 GiB | 120.05 GiB |
| Native replay | 64 GiB | 46.39 GiB | 8.00 GiB | 19.99 GiB | 122.35 GiB |

These conservative admissions are **not RSS or operating-system quotas**.
The default full-engine policy remains **96 GiB host / 32 GiB available floor**;
this opt-in probe retains that floor and uses its smaller explicit cap. All
policy charges, pinned allocations, owned device allocations, leases,
reclaimers and in-flight operations are zero after close, and the monitor stops.
Native explicit device payload peaks are **10,442,542,756 bytes** on the 4090
and **37,748,160 bytes** on the helper. The helper stores and audits original
bytes; this does not qualify Ampere model arithmetic. The original framework's
device peak is **11.85 GiB**, with allocation returning from baseline **0 bytes**
to **0 bytes** after its own tensor/BLAS-workspace cleanup. VMM and driver/code
storage are distinct from framework allocation counters.

Preparation uses 786 experts, four input embeddings and 240 Engram rows, and
does not download the full checkpoint. Its minimum sampled available RAM is
**120.20 GiB**; free disk remains approximately **165.72 GiB** afterward. Current
bounded probes fit the existing workspace drive. The previously documented
full-import capacity requirement remains a separate decision.

## Exact scope

The runtime executes all original blocks 0..39, both Engrams at 1/14, every
original ordered top-six route, the six selected experts and shared expert,
actual HC hidden/coefficient handoffs, final norm, complete 129,280-entry FP32
logits and the original stochastic sampler. Original weights, activation/KV
rounding and RNG progression are retained without retraining.

The native driver encodes raw text without inserting BOS, chat, role or thinking
markers. This is a raw text-continuation test, not the upstream chat encoder.
Input tokens before the final prompt token update the full model state without
consuming sampling RNG. Subsequent inputs come from actual generated IDs. EOS
stops the driver and is omitted from decoded completion, as upstream. The final
selected ID is returned without an unnecessary extra model call.

The fixture scope is at most **eight complete single-query calls**, with at
most 48 distinct original experts per layer, eight input embeddings and 1,024
Engram rows per table. These are explicit bounded snapshot limits, not claims
about the model's supported context. An unprepared row/expert is rejected; no
zero substitute, approximate route or network fallback exists during replay.

## Independent original execution

The reference executes the **unchanged pinned upstream `generate` function**,
including its prompt mask, token feedback, EOS handling and completion slicing.
It uses the original Transformers tokenizer independently of the native Rust
wrapper, and checks prompt encoding before running. It does not read planned
sampled IDs or the native driver's completion.

An explicit memory adapter serializes the original loop's first prompt chunk.
Each query calls the unchanged `Transformer.forward`, `Block.forward` and
qualified original arithmetic methods. Intermediate prompt calls set original
temperature zero, discard their greedy selection, and must leave RNG unchanged.
The last prompt query and generated-token queries use temperature 0.7. This is
a comparison of **serial prompt ingestion**, not numerical qualification of
the upstream chunk-prefill path or its different matrix shapes.

The existing original sparse-Attention memory adapter is still required for
the 4090. This gate does not assert that the oversized unmodified full kernel
launches. Original embedding and Engram shapes and addresses are preserved by
CUDA virtual memory, with only the requested original pages backed. Original
auxiliary target-layer output is recorded but not compared as native MTP.

For every query, the original full Transformer logits are also checked against
an independently executed original output tail. Replaying the original sampler
from its saved RNG state must reproduce its selection and RNG advancement.
Complete typed/shaped intermediate hashes, ordered routes and committed KV,
compression/index, Engram history and RNG state are compared on the native side.

## Bounded artifact preparation

The separate planning operation may fetch original byte ranges between model
stages. It is not independent model evidence. It imports only actual required
embedding rows, Engram rows and original selected experts; all metadata/source
extents and payload hashes are retained. Existing qualified BOS/PAD fixtures
and sealed reports are preserved.

One temporary six-expert store is used per planning call. Final immutable unions
are built once after tracing, avoiding repeated copies of growing unions.
Checked expert files may be hard-linked from previous preparations, with a copy
fallback. Four admitted expert importers share at most eight transfer workers;
the six exact original tensor ranges are overlapped within their existing host
allowances, and the unchanged importer checks headers/layout/hashes afterward.
Preparation preserves at least **64 GiB free disk space**, checks a
bounded fixture limit before growth, and refuses to replace completed stores.

Both independent executions disable Python HTTP requests. The native side
requires PyTorch, TileLang, Transformers and SymPy to be absent. It uses the
existing twelve qualified native binaries; no CUDA arithmetic changes belong
to this gate. The reference clears its own framework tensors and private BLAS
workspaces and verifies return to its original GPU allocation baseline.

## Reproduction

Prepare the already documented original control, output and metadata fixtures
first. Install `.[artifacts]` for native tokenization. In the native environment:

```powershell
python -m vesper.generation_probe prepare --artifacts artifacts --run story-v1
```

In the separately qualified reference environment, run:

```powershell
python -m vesper.generation_probe reference --artifacts artifacts --run story-v1
```

The Windows CUDA-13 reference profile can supply `--nvrtc-bin` pointing to its
NVRTC directory. The original report must pass inspection and match the sealed
identity/golden constants before native replay is accepted. This is an explicit
qualification step; capturing arbitrary new data does not bless it automatically.

Return to the native environment and run:

```powershell
python -m vesper.generation_probe native --artifacts artifacts --run story-v1
python -m vesper.generation_probe check --artifacts artifacts --run story-v1
```

The CLI refuses to replace completed reports. Evidence metadata reads are
bounded to 64 MiB, and embedding/Engram metadata extents are checked before
reading. Preparation can require several
GiB of original downloads and is separate from offline replay. On restricted
Windows shells, select a writable `TEMP`/`TMP` directory when running CPU tests.

## Remaining work

This small completion does not establish broad model quality, reasoning/coding/
tool-use performance, arbitrary chat prompts, long histories, chunk-prefill,
vision or DSpark/MTP parity. No TTFT or tokens-per-second claim follows from
this diagnostic path. MoE instances still reopen per layer and state inspection
copies diagnostic tensors; persistent scheduling and throughput are later work.
Issue #10 and M2 retain their outstanding original acceptance requirements.
