# Routing postprocessing progress — 2026-10-18

Tracked in [issue #3](https://github.com/shell-term/vesper/issues/3), under
[M1: exact routed MoE](https://github.com/shell-term/vesper/milestone/2).
This is an **unqualified diagnostic candidate**. The issue remains open: native
routing refuses selected top-k ties, while the original gate handles them.

## Isolated scope

The input is one supplied FP32 projection-logit vector with 384 elements.
The probe covers sqrt-softplus, trained text/image selection bias, top-six
selection, unbiased weight gathering, normalization with `1e-20` and route
scaling by 1.5. The pinned inference configuration omits `gate_temp` and
`norm_topk_prob`; the original `ModelArgs` defaults are 1.0 and true.

The reference extracts the unchanged `Gate.forward` body from the hash-checked
official source. Its `linear()` call is explicitly replaced by an identity
function, so supplied FP32 logits reach the original postprocessing. A separate
diagnostic trace repeats those operations and checks that its result matches
the unchanged function. **Gate projection is not tested**; it is
[issue #4](https://github.com/shell-term/vesper/issues/4). Neither a connected
MoE sublayer nor a Transformer block/model is established by this check.

The source model is `deepseek-ai/DeepSeek-V4.1-Flash-1017`, revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`. The fixture contains the original
layer-0 `bias` and `bias_vl`, 1,536 bytes each: **3,072 model bytes** total.
Checkpoint dtype, shape, tensor identity, ranges and hashes are validated. The
released `inference/config.json` hash is
`2e84f45cf1dac8c7fcbb200e96667d4b913275690668ed496f24c7747207a809`.
No model weights are committed. Native arithmetic uses NVRTC with `fmad=false`
and `ftz=false` on the RTX 4090; the oracle uses PyTorch `2.14.1+cu130`.

## Results

| Check | Observed result |
| --- | --- |
| All 384 unbiased FP32 scores | 0 / 44,544 bit differences |
| Bias-corrected selection scores | 0 / 44,544 bit differences |
| FP32 normalization denominator | 0 / 116 bit differences |
| Selected expert sets | Identical in all 116 conditions |
| FP32 weight associated with each selected expert ID | Identical in all 116 conditions |
| Complete ordered output for non-tied conditions | 87 / 87 pass |
| Selected-tie conditions | 29 unsupported; original-bias decision calls refuse them |

Eight seeds crossed with four logit scales and both text/image masks give 64
random conditions. Additional conditions cover zero/extreme/tiny logits,
softplus's threshold, a ramp, synthetic zero-bias ties/near-ties and logits that
approximately cancel each real trained bias at nine margins. The latter use
unequal unbiased scores to test the distinction between ranking and weighting.
There are 116 reference forwards and 228 completed native inspections, including
repeated calls through the decision-admission guard. Four synthetic zero-bias
conditions are diagnostic-only; the decision API always uses the original bias.

The ordered diagnostic tuples differ in 128 / 696 index positions and 108 / 696
weight positions, all in tied cases. Those positional differences disappear
when associating each weight with its expert ID. These are permutations in this
test set, not observed changes to the selected expert sets or weight maps.
Every non-tied case matches all checked ordered output bits.

## Why the tie gate remains open

The original biases are not unique: the 384 text entries have 63 distinct values,
and the image-span entries have 138. Ties appear with real trained biases, not
only with the synthetic all-zero bias. The candidate's ascending-ID choice is
an inspection convention; it is not presented as the official tie algorithm.
The [PyTorch top-k API](https://docs.pytorch.org/docs/2.14/generated/torch.topk.html)
does not promise stable indices for tied values across invocations.

`MoE.forward` accumulates routed contributions in ascending expert-ID order,
so a top-k tuple permutation by itself does not prove a changed MoE output.
However, normalization reduces the gathered raw scores in floating point, and
ties at the selection boundary can affect the expert set. The gate to qualify
is the selected ID/weight map and every ordering dependency actually consumed by
the unchanged model, not an arbitrary cosmetic order. The present test set does
not establish that general contract or a working decision path for tied inputs.

`RouterProbe.inspect()` exposes clearly marked diagnostic outputs. The separate
experimental `decision()` method raises `UnsupportedRoutingTie` whenever the
selected ranking is tied. It never returns the diagnostic choice for a tied
original-bias input. This probe is not connected to inference or the byte cache;
no existing qualified expert/cache path is changed. The next work is to qualify
a tie-capable decision path and confirm its weight mapping/normalization with
the official MoE aggregation, preserving the reference reduction behavior.

The command reports `status: partial`, `native_candidate_qualified: false` and
**exit code 2**. CPU/CI success is separate from this open numerical/behavioral
gate. A source-only implementation does not complete issue #3 or M1.

## Resources and reproduction

The native probe peaks at 7,760 explicit device bytes under a 1 MiB allocation
cap, retains a 1 GiB free-device reserve and finishes with zero live explicit
allocations. CUDA contexts/modules, compilation, the PyTorch oracle and returned
diagnostic arrays are outside this buffer count. Fixture fetching reserves a
small 64 MiB host allowance and keeps the 32 GiB available-memory floor; it uses
bounded HTTP ranges and never fetches a full shard or layer.

```sh
vesper fetch-router-fixture --directory artifacts/router-fixture
vesper compare-router --fixture artifacts/router-fixture/native-router.npz --device 0 --output artifacts/router.json
```

The comparison requires CUDA PyTorch, but does not use TileLang or load the
entire model module. The complete
[result and fixture provenance](../results/2026-10-18-routing/README.md) retain
all conditions, exact comparisons, mismatch details and admission outcomes.
CPU tests cover selection-bias separation, selected/boundary ties, invalid
inputs, fixture tampering/configuration, archive bounds and tie refusal.
