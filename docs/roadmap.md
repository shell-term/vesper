# Roadmap and progress tracking

This M0..M3 roadmap is the **PC track**. Spark uses separate S0..S3 milestones
and its [own entry](../platforms/dgx-spark/README.md). Select the target through
[workspace rules](../platforms/README.md) before reading/updating progress.
Shared code does not transfer platform qualification.

[Issues](https://github.com/shell-term/vesper/issues) are the source of task state.
[Milestones](https://github.com/shell-term/vesper/milestones) group acceptance
gates. This document explains their scope and workflow; it does not duplicate a
mutable task-status list or estimate whole-model completion from issue counts.

| Milestone | Required outcome | Work |
| --- | --- | --- |
| [M0: Verified component baseline](https://github.com/shell-term/vesper/milestone/1) | Scoped native 4090 expert parity, bounded tiered cache and integrity evidence | [Expert](https://github.com/shell-term/vesper/issues/1), [cache/audits](https://github.com/shell-term/vesper/issues/2) |
| [M1: Exact routed MoE sublayer](https://github.com/shell-term/vesper/milestone/2) | Gate scores/projection, selected IDs/weights, routed accumulation, shared expert and bounded integration | [Routing](https://github.com/shell-term/vesper/issues/3), [projection](https://github.com/shell-term/vesper/issues/4), [aggregation](https://github.com/shell-term/vesper/issues/5), [connected MoE](https://github.com/shell-term/vesper/issues/6) |
| [M2: Single-token model and bounded memory](https://github.com/shell-term/vesper/milestone/3) | Attention/indexing/KV, Hyper-Connections, original artifacts/Engram, runtime memory policy and reference logits/sampling | [Block operations](https://github.com/shell-term/vesper/issues/7), [artifacts](https://github.com/shell-term/vesper/issues/8), [memory](https://github.com/shell-term/vesper/issues/9), [decode](https://github.com/shell-term/vesper/issues/10) |
| [M3: Measured end-to-end performance](https://github.com/shell-term/vesper/milestone/4) | Actual routing/I/O, prefill, TTFT, decode, memory and qualified optimizations | [Real-workload performance](https://github.com/shell-term/vesper/issues/11) |

M1 concerns the feed-forward MoE sublayer, not a complete Transformer block.
M0 closes scoped component checks only. A failed optional candidate does not
become qualified because another path passes. The
[Ampere model-compute investigation](https://github.com/shell-term/vesper/issues/12)
remains deferred; the helper GPU can store/hash bytes without running experts.

Issue #7's original five checks qualify block operations within their explicit
component and connected-chain scopes; see the
[sealed acceptance audit](block-ops-qualification.md). They do not require or
prove a complete forty-layer model. Issue #8 owns tokenizer/original-artifact
and Engram import, #9 owns the whole-runtime memory policy, and #10 integrates
those dependencies with all-layer numerical/logit/sampling comparisons.
Completing #7 leaves M2 open. Longer complete-chain histories and broader
execution shapes remain integration requirements, even when their component
Attention operations already pass.

Issue #8's five original checks are audited separately in
[the artifacts/Engram report](engram-artifacts-validation.md): complete original
tokenizer mapping, 42 original CPU/CUDA addressing cases, 110 Engram session
steps, framework-free replay, bounded direct reads and actual full-import
capacity. Full-shard registration is implemented but not executed on the entire
checkpoint; small probes remain separate. Its component caps and combined Engram
GPU ledger are inputs to #9, not the whole-runtime governor. Completing #8 also
leaves M2 open for #9 and #10.

Issue #9's six original checks are audited in the
[memory-policy report](memory-policy-validation.md). The shared 96 GiB / 32 GiB
policy covers component admissions, typed resources and state publication.
All 15,360 original expert allocation identities are traversed with touched
synthetic bytes under an 8 GiB probe cap; original Engram and complete 20..25
numerical replays also pass under the governor. Separate process/reference
measurements and complete cleanup are recorded. This is a resource-scale gate;
all-forty-layer numerical/logit/sampling integration remains Issue #10 and M2
stays open until that required work passes.

Issue #10 now has qualified output/sampling components and a concrete local
streamed-reference design; see the [complete report](output-sampling-validation.md).
The full original FP32 head matches 4,654,080 logits; stochastic sampling matches
all probability/noise/score/token/RNG boundaries in 90 conditions, with independent
native replay and failure/state tests. This satisfies the reference-design
requirement, while all-layer integration, whole-model comparisons, decoding and
scoped evaluations remain open. Component passes do not close #10 or M2.
The [connected original 0..3 prefix](token-prefix-validation.md) now includes
embedding/Engram placement, actual hidden/HC handoffs, both window rings and
source-2 KV/index reuse. Complete original/native boundaries and nine failed-token
rollbacks pass under the governor, including diagnostic full-head/RNG state.
The attached prefix head follows only four blocks. Its historical cancellation
label was corrected with a fresh complete replay requiring actual runtime
cancellation. The subsequent [all-forty-layer gate](backbone-validation.md)
connects both Engrams and the real head/sampler: two fixed BOS/PAD calls match
10,504 typed boundaries, all 80 router decisions and 258,560 logits. Four late
failures restore whole-model state, including actual cancellation, and healthy
retry reproduces every second-token boundary. These are scoped numerical/state
results. The subsequent [bounded raw-text continuation](text-generation-validation.md)
executes the original upstream generation loop and a separate native driver:
five serial calls / three generated tokens match all 26,250 typed boundaries,
138,434,360 elements, 646,400 logits and 200 router decisions, including full
sampling/RNG and model state. Both replays are offline. Native sampled resident
RAM peaks at 8.00 GiB with 122.35 GiB minimum sampled available RAM. This is one
raw-text corpus with a serialized prompt adapter. Broader histories, chat/chunk
prefill and reasoning/coding/tool-use evaluations remain under #10. Its original
checks and M2 stay open until the remaining acceptance requirements pass.

The [V4.1 message format gate](message-format-validation.md) now qualifies
30 encodings, 11 parses, 13 rejection cases and 2,336 token IDs against the
unchanged original CPU encoder/tokenizer. Three complete quality tasks are
prepared with deterministic scorers. Their prompts contain 63/30/293 tokens;
their maximum requested single-query trajectories are 190/61/388 calls, beyond
the historical eight-call fixture. These evaluations did not run. Longer bounded
expert/Engram/embedding stores and reference/native trajectories remain #10
work; formatting/scorer tests do not close its model-quality requirement.

The [longer-input/storage follow-up](sequence-storage-validation.md) now admits
512 calls explicitly and indexes existing expert payloads without copying them.
Indexed all-forty-layer replay preserves all 26,250 short-generation boundaries.
A separate 63-position original/native Engram comparison passes 3,906 boundaries
on synthetic hidden inputs, with 248 MiB peak original-address physical row
backing. Larger whole-model route preparation, actual generated trajectories
and reasoning/coding/tool scores remain open; #10 and M2 remain open.

The [trajectory runner](trajectory-validation.md) implements bounded immutable
expert-object imports, original-address input-embedding paging and per-call
evidence files. Compilation reuse passes the existing complete short generation
and a separate fourteen-image exact-binary recheck; this is no throughput claim.
The first actual coding-task preparation is in progress. Its independent
original/native whole-model comparison and score remain pending; the other two
declared tasks remain not run. Neither #10 nor M2 is complete.

## Working on an issue

1. State the concrete outcome, acceptance checks, resource limits and dependencies.
2. Mark active work `status:in-progress`. Reference its issue in commits.
3. Use `status:needs-validation` when code exists but required evidence is missing
   or a relevant exact-output gate still fails. A partial pass does not close it.
4. Link complete sanitized results, source revision, hardware/toolchain, scope,
   exclusions and known failures. Keep original model bytes and private paths out.
5. Add `verified` and close only when every acceptance condition for that issue's
   stated scope passes. A milestone closes only after its required issues do.

`priority:P0` covers fidelity/correctness/resource requirements. `priority:P1`
covers performance after the relevant correctness gate. `priority:P2` is optional
investigation. `status:deferred` preserves noncritical work without implying it
blocks the qualified path. Area labels aid filtering; they are not milestones.

Do not invent deadlines, whole-model completion percentages or throughput from
synthetic cache tests. Separate implementation, specification, official-reference
parity, model quality and end-to-end speed, as in [the validation gates](validation.md).
If new evidence changes scope, update the issue and its acceptance criteria with
the reason before claiming completion. Keep current task state on GitHub.
