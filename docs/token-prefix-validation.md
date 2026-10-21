# Connected original token prefix and whole-token recovery

> **Recorded PC evidence.** Results, requirements and next-step wording below
> describe this report's experiment/profile. They do not set the current
> development queue or require exact equality from every new candidate. See
> [current policy](quality-speed.md), [validation](validation.md) and
> [PC roadmap](roadmap.md). Raw evidence and original scope are unchanged.

Vesper now connects original embedding rows through complete blocks **0 -> 1 ->
2 -> 3**, applying original Engram **before block 1** and carrying every actual
hidden tensor and HC pre coefficient into the next block. Two serial text-token
calls match independent original methods at **2,037 typed/shaped boundaries /
14,697,514 storage elements**. Native replay also restores the entire committed
token state across nine injected failures and reproduces every second-token
boundary. This advances [Issue #10](https://github.com/shell-term/vesper/issues/10);
the remaining 36 blocks, full-model generation and quality evaluation are open.

The complete sanitized [original goldens](../results/2026-10-19-token-prefix/original.json),
[native replay](../results/2026-10-19-token-prefix/native.json) and
[file-bound summary](../results/2026-10-19-token-prefix/summary.json) contain the
full evidence. They contain tensor metadata and hashes, not checkpoint weights.

**Cancellation correction:** the historical `cancel_after_sampling` record
caught a `TypeError` from an extra argument to `RuntimeMemory.cancel`. It proves
generic late-failure rollback, not runtime cancellation. The call and its CPU
test are corrected to require `RuntimeCancelled` and verify that cancellation
remains active until explicit resume. The historical JSON files are preserved;
their cancellation label must not be treated as cancellation qualification.
The [corrected complete replay](../results/2026-10-19-token-prefix-cancellation/native.json)
now catches **`RuntimeCancelled`**, verifies cancellation before resume, restores
the committed state and matches all **1,046** second-token boundaries. All nine
failure conditions and the real ring-checkpoint probe were rerun; the
[correction receipt](../results/2026-10-19-token-prefix-cancellation/summary.json)
binds the complete new report to the unchanged original goldens.

## Exact scope

- Released model/revision: `deepseek-ai/DeepSeek-V4.1-Flash-1017`,
  `1bae0e1088993b9b82de08f59de5be5a34e42170`.
- Rank one, batch one, one query per call, text only; the measured token stream
  is `[0, 2]`. These two special-token inputs test state and arithmetic rather
  than reasoning, coding or tool-use quality. This is not batched prompt prefill.
- Blocks 0/1 have independent original window Attention weights/rings. Blocks
  2/3 share the original source-2 compressed KV and selected indices; the second
  token crosses the ratio-two compression boundary. All four original gates,
  correction biases, six selected experts, shared experts and reduction orders
  are retained. The prepared expert unions contain 11, 11, 11 and 12 experts.
- Initial HC mixing is the unchanged **one-hot stream-zero** mix. Block 1 reads
  the actual Engram result, not an independent synthetic hidden input. Following
  blocks read the actual preceding FFN pre coefficients.
- To exercise the complete output/RNG transaction, the previously qualified final
  norm, full FP32 head and temperature-0.7 sampler are attached **after block 3**
  as a diagnostic. These logits and sampled IDs are **not** the forty-layer
  model's outputs. No text generation endpoint or accuracy claim is introduced.
- `token_prefix` is an explicit layer-1 binding profile. Default component
  allowlists and the seventeen sealed Issue #7 evidence groups are preserved;
  no arbitrary unqualified backbone layers are admitted.

## Independent original comparison

The original run finishes and records complete goldens before native replay.
It uses the pinned tokenizer/hash implementation, original full-address selected
Engram rows, original `Engram.forward`, `Block.forward`, `Attention.forward`, HC,
MoE, final norm/head and sampling methods. Each instrumented block is separately
checked against the unchanged original method, including its committed cache.
The four layers receive the reference run's own preceding outputs and routes.
Prepared native expert IDs do not override or restrict the reference router;
an unprepared reference selection fails instead of being redirected.

The reference Attention path uses the separately qualified arithmetic-preserving
PTX storage adapter described in [the Attention gate](attention-memory-validation.md).
The unmodified large-resource kernel and model-wide quality remain distinct gates.
Original arithmetic sources and their existing DeepSeek/PyTorch notices remain
pinned. The cuBLAS/NVRTC/CUDA kernels and original grouped-output conversions are
checked; no compute format or learned weight is changed.

Three changed controls are rejected: omitting Engram before block 1, using layer
0's projection weights for window 1, and using stale HC pre coefficients at block
2. The new reference binding also fixes a helper-name collision: integer layer
metadata had hidden the old window-step method. The method is now `window_step`;
the pinned original arithmetic and existing recorded goldens are unchanged.

Original identity: `9c982d7b7f04ab3023bfaefb8ddca56b6300a852256ed7cf7590cf465faa5317`.
Complete golden identity: `78dc3767fc0438bb8565a811e5921402624dab0cacbbed1772abfecac3b897e5`.
Native replay requires both identities and checks every boundary's name, shape,
dtype, element count and raw storage hash. Torch, TileLang, Transformers and SymPy
are absent from the independent native environment.

## Whole-token publication

`TokenTransaction` snapshots both window rings, the compressed/index frame,
Engram token history and sampler seed/offset before computation. History and
model position are published only after the final diagnostic result exists.
Failures restore all logical state; cache temperature may change without changing
the learned routing. Failed recovery retains its handles and blocks new calls
until explicit recovery succeeds. Cancellation can be restored under the same
memory limits; resuming execution still requires explicit policy resume.

Window Attention already uses one GPU snapshot slot for its own step. A separate
opaque outer checkpoint now holds the exact rounded ring on the host, at most
128/256 KiB for batch one/two, inside the existing 16-MiB component allowance.
Restore writes an inactive ring and verifies its bits before publishing it. It
does not re-quantize, repeat RoPE, or consume the inner Attention snapshot slot.
Foreign/expired handles are refused. The real-device probe crosses positions
128 -> 130 -> 128, injects readback failure, and replays the next append exactly
for both batches, including finite-BF16 encoding coverage in the initial input.

The nine complete-prefix conditions are: failure after window 0, Engram 1,
window 1 or block 3; head readback failure; failure or cancellation after
sampling; window-1 restore transfer failure; and an injected component observation
below the RAM reserve at publication. All restore the first-token state and then
pass all **1,046** second-token boundaries: **9,414** retry comparisons. Restore
transfer failure retains only window 1's handle; the reserve condition retains
all five state-owner handles. No partial token is returned. Separate protocol
tests cover failed snapshot acquisition, history commit, deferred cleanup and
invalid/stale handles. Failure to discard an unneeded snapshot after publication
does not falsely report an already committed token as failed.

## Resources and reproduction

The live original/native probes each use a **24-GiB conservative host admission
cap**, a **32-GiB observed available-RAM floor**, and the shared device governor.
The full-runtime defaults remain 96/32 GiB. The reports record actual process
RSS/private commit, explicit GPU payload peaks and admissions separately. These
are not an OS quota or a claim of physically allocating 24/96 GiB. The low-RAM
condition is injected; physical system exhaustion is not induced. All component
charges, device payloads, live operations/reclaimers and the monitor are cleaned
up. Current bounded preparation fits the existing working drive; no additional
SSD or full-checkpoint import is needed for this gate.

Reuse the existing prefix, topology 2/3, complete block/MoE, Engram, tokenizer,
output and native-Attention fixtures from the linked earlier gates. Prepare the
additional layer-1 controls and measured routes outside forward execution:

```powershell
python examples/prepare_token_prefix.py --artifacts artifacts --output artifacts/token-prefix-plan.json
# Optional pinned original-reference environment, as in the earlier gates:
python examples/capture_token_prefix.py --artifacts artifacts --output artifacts/token-prefix-original.json
# Native environment, without the four reference frameworks:
python examples/recheck_token_prefix.py --artifacts artifacts --reference-result results/2026-10-19-token-prefix/original.json --output artifacts/token-prefix-native.json
```

Preparation fetches at most two bounded original ranges concurrently, reuses
checked existing experts and refuses unregistered store identities. Missing
experts or Engram rows fail inside computation; the forward path performs no
checkpoint download. The original run stays local and starts no paid remote
compute. Reference dependency/compiler setup follows the earlier pinned gates.
Neither input stream, the partial diagnostic head nor this state coordinator
fulfills Issue #10's remaining all-forty-layer, deterministic generation and
reasoning/coding/tool-use evaluation acceptance checks.
