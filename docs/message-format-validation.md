# Original V4.1 text-message format and evaluation admission

> **Recorded PC evidence.** Results, requirements and next-step wording below
> describe this report's experiment/profile. They do not set the current
> development queue or require exact equality from every new candidate. See
> [current policy](quality-speed.md), [validation](validation.md) and
> [PC roadmap](roadmap.md). Raw evidence and original scope are unchanged.

The independent CPU comparison passes **30 message encodings, 11 response
parses and 13 malformed-input rejections**. Original AutoTokenizer and native
PinnedTokenizer agree on **2,336 input token IDs**, including three complete
quality-task prompts. This qualifies the recorded text-format corpus. **No
model inference or model-quality evaluation ran in this gate.**

The preceding [raw-text generation gate](text-generation-validation.md) remains
the actual five-call, three-token all-forty-layer numerical/generation evidence.
Neither that result nor this format comparison qualifies arbitrary chat,
reasoning, coding or tool-use quality. Issue #10 and M2 remain open.

## Independent original and native execution

The CPU oracle imports the entire unchanged upstream
[`encoding/encoding.py`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/encoding/encoding.py)
at the same pinned checkpoint revision as the inference sources. Its raw file
has **37,962 bytes**, SHA-256
`13231c592e958fb39e2af6001dae69d90e7a36b265702b9b390236fb387599e2`.
The accompanying upstream encoding README is also size/hash checked. These two
source files total 50,126 bytes and remain ignored artifacts. The pinned MIT
license is checked before the oracle is imported; its notice is preserved in
[the message-adapter addendum](../third_party/DeepSeek-MESSAGES-NOTICE.md).

The original process uses **Transformers 5.19.0 / tokenizers 0.23.3** with
`AutoTokenizer.from_pretrained(local_files_only=True, trust_remote_code=False)`.
CUDA is uninitialized before and after capture. The separate native process
uses the Rust tokenizer directly; Torch, TileLang, Transformers and SymPy are
absent from its environment. Both executions disable Python HTTP throughout
comparison, make zero HTTP calls and fetch no model weight payloads.

The native [text adapter](../src/vesper/messages.py) reproduces the selected
format without loading the optional upstream encoder. No guessed generic
chat template is substituted. The sealed evidence includes complete prompt
strings, token ID arrays, parsed response objects and rejection outcomes:

- [Original CPU capture](../results/2026-10-20-message-format/original.json)
- [Independent native comparison](../results/2026-10-20-message-format/native.json)
- [File sizes, hashes and preflight summary](../results/2026-10-20-message-format/summary.json)

The complete original report has canonical SHA-256
`177d00d1b6ede9e415c5034c1018eaf60f8cba053a9cf37ae643196d0ade4bc2`.
Discovery and qualification are separate: rehashing a changed capture cannot
make it qualified. Native validation binds to the original raw file, its
qualified identity, complete results and pinned tokenizer files.

## Recorded format coverage and limits

The corpus checks chat and thinking modes, original BOS, numeric reasoning
effort 1/50/100 and low/high/max aliases, default effort, Unicode and whitespace,
adjacent user text, response-format schema rendering, mid-conversation system
messages, reminders and an unfinished assistant prefix. History checks include
dropped/preserved earlier reasoning and the automatic preservation used when
tools are defined.

Tool cases include the V4.1 leading spaces in DSML ` calls`, ` invoke` and
` parameter` tags, typed JSON/string arguments, namespaces and descriptions,
empty argument objects, double-encoded argument JSON, text tool results and
reordering multiple results to their original call order. Parser cases retain
the original ordered argument JSON and cover both EOS-terminated text and the
upstream accepted closed-tool-block form without EOS. The 13 rejection cases
cover invalid reasoning budgets, namespace conflicts, image placeholders,
missing terminators, duplicate parameters, trailing content and legacy DSML.
This is a declared corpus; it does not establish equivalence for every malformed
input the upstream encoder might tolerate.

The public adapter accepts a bounded text-only subset: at most 32 messages,
64 KiB of serialized input metadata, 16 tools/calls per message, and 4,096
characters in the rendered prompt. Response parsing is limited to 16,384
characters. It rejects unsupported fields, images, classification tasks and
existing-cache context. It does not execute tools, add a default system
instruction, change reasoning effort, repair malformed responses or start
generation. Formatted prompts still need admission by an execution profile.

## Actual quality-task preflight

The [fixed corpus](../src/vesper/message_cases.py) declares a small arithmetic
reasoning task, a Python last-element expression task and an `add` tool task.
Their original and native formatted inputs match exactly. They keep complete
instructions and the original tool schema/template; none was shortened to
fit the existing fixture. The reasoning task retains the original default
effort **75 (high)**; the runner does not lower it to reduce the required work.

| Category | Original prompt tokens | Requested new-token limit | Maximum model calls requested | Existing fixture limit | Model evaluation |
| --- | ---: | ---: | ---: | ---: | --- |
| Reasoning | 63 | 128 | 190 | 8 | Not run: outside execution scope |
| Coding | 30 | 32 | 61 | 8 | Not run: outside execution scope |
| Tool use | 293 | 96 | 388 | 8 | Not run: outside execution scope |

For single-query serial ingestion the requested maximum is
`prompt_tokens + max_new_tokens - 1`; EOS could end a future run earlier. Even
these prompts alone exceed the current eight-call execution fixture. Native
preflight also exercises the actual GenerationRequest admission and confirms
rejection. Each evidence record has `model_inference_run: false`, no completion
and no score. No reasoning/coding/tool-use success rate is reported.

The [small deterministic scorers](../src/vesper/evaluation.py) are ready to
evaluate captured outputs: exact final integer, three list-expression cases,
and one exact tool name/typed-argument object. Scorer unit examples are authored
fixtures, never model answers. An unfinished generation receives `incomplete`;
missing/malformed EOS receives no passing score. The Python scorer interprets
only a small AST subset for list indexing, integer index arithmetic and `len`;
other expressions are `unscorable`, and arbitrary generated code is never
executed. Tools are parsed and checked without execution.

## Execution work still required

The following describes the historical eight-call fixture at this gate. The
subsequent [longer-input/storage work](sequence-storage-validation.md) adds an
explicit larger admission profile, indexed experts and a qualified 63-token
Engram component comparison. Actual whole-model quality trajectories remain
unprepared and unscored.

A larger model-call integer alone is insufficient. The current immutable
selected-expert store allows 48 experts per layer, the prepared embedding set
allows eight IDs, and selected Engram fixtures allow 1,024 rows per layer. The
reference hash buffer is also an eight-call profile. A faithful larger run must
prepare the actual original routes/rows, stream or partition their immutable
stores within the host budget, and extend both sides' explicit profiles without
changing original arithmetic. It must compare actual logits, sampling and state
over the longer trajectory and preserve cancellation/rollback behavior.

The provisional full-engine memory policy remains **96 GiB managed host /
32 GiB system-available floor**. This CPU format test neither loads the model
nor measures that full runtime; it consumes no additional checkpoint storage.
The next execution design must budget disk as well as resident/pinned scratch
and avoid duplicating growing expert-store unions. Storage capacity will be
checked before any larger payload import. No full checkpoint was downloaded.
Compilation reuse and steady-state throughput remain separate measured work
under Issue #11; this gate makes no generation-speed claim.

## Reproduction

Use a new run name to retain completed reports. The optional original
environment and native environment are separate processes with the same pinned
metadata under `artifacts/import-source`:

```powershell
# Original environment: Torch, Transformers 5.19.0 and tokenizers 0.23.3.
python -m vesper.message_probe reference --artifacts artifacts --run message-v1
# Independently qualify that capture against the pinned complete identity.
# Native environment: tokenizers 0.23.3, no reference frameworks installed.
python -m vesper.message_probe native --artifacts artifacts --run message-v1
python -m vesper.message_probe check --artifacts artifacts --run message-v1
python -m unittest discover -s tests -p test_messages.py -v
```

The reference command may fetch only the two small pinned format source files;
its execution after preparation is offline. Reports are bounded to 2 MiB before
reading, and completed captures are never replaced. CPU regression tests also
reject altered/rehashed goldens, incomplete native evidence, wrong reference
binding, framework/HTTP use and false claims of model evaluation.
