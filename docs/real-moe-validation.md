# Connected MoE with real prefix activations

> **Recorded PC evidence.** Results, requirements and next-step wording below
> describe this report's experiment/profile. They do not set the current
> development queue or require exact equality from every new candidate. See
> [current policy](quality-speed.md), [validation](validation.md) and
> [PC roadmap](roadmap.md). Raw evidence and original scope are unchanged.

The remaining **Issue #6 real-activation gate passes**. Four actual layer-0
prefix activations and both text/image gate-bias settings give eight conditions.
**30 successful calls match all 108 boundaries each: 3,240 comparisons and
153,600 final BF16 elements, with zero mismatches.** Four injected failures
return no partial output; complete retries reproduce the original bits.

A separate process with **PyTorch neither installed nor imported** reproduces
all 30 successes and four expected failures against the complete stage hashes.
The native model-kernel CUBIN hashes are unchanged from the earlier generated-
input validation. This is the scoped routed-MoE sublayer gate for M1, not a full
Transformer block, text generation, model-quality or speed result.

## Where the real inputs come from

The unchanged original layer-0 methods execute with original embedding rows,
normalizations, Hyper-Connection weights, projections, RoPE and window state.
The four token IDs form two sequences: `(1, 42)` and `(1000, 4096)`. Each sequence
uses single-token prefill at position 0 and decode at position 1; KV state resets
between sequences. Capture stops at the original MoE input boundary. All four
vectors repeat bit-for-bit across a second complete capture.

Attention uses the explicitly qualified [PTX memory adapter](attention-memory-validation.md).
It retains the original generated floating-point/control instructions and
validates every changed bit-transport operation; all six original binary builds
are reproduced byte for byte. The complete adapter qualification is embedded in
the capture and bound by a canonical SHA-256. Untouched 64×512 Attention launch
still exceeds the 4090's shared-memory limit; it is not claimed to have run.

Token IDs are selected directly. Image-bias choices are routing controls applied
to the captured vectors; no image encoding or tokenizer qualification is claimed.
Layer 0 has no Engram/compressed-KV path. Other layers, larger batches, full
prefill, indexing/compression/sharing, rollback and native block operations remain
under [Issue #7](https://github.com/shell-term/vesper/issues/7) and M2.

The plan binds the exact capture-file hash and qualified reference profile.
Missing, incomplete, failed or modified qualification evidence is refused before
using the vectors. A supplied capture cannot silently replace the import plan.

## Original experts and cache/failure coverage

The actual gate selects **27 distinct experts: 507,617,280 original bytes**.
Four existing fixtures are verified/reused; **23 experts / 432,414,720 bytes** are
fetched with bounded HTTP ranges. The explicit store limit remains 48 experts.
No full checkpoint shard is downloaded and no route is changed for cache hits.

The same matrix as the [generated-input experiment](connected-moe-validation.md)
covers cold reads, changing real selections, repeated calls, primary/helper/RAM
sources, eviction and zero-RAM forced reloads. Failures cover pressure before the
gate, pressure after three experts, a corrupted fourth upload and failure before
the fourth physical store read. Every retry recomputes the complete sublayer.

The unchanged original `linear`, `Gate.forward`, `Expert.forward` and `MoE.forward`
methods execute for the MoE oracle. The actual original gate runs inside that
MoE method. Selected IDs, weights, every expert's stages, ordered sums, shared
output and final BF16 bits are checked. Original trained routing bias is retained.

## Resources

Both the paired and native-only runs complete **105 direct store reads /
1,974,067,200 bytes**. These are controlled validation traces, not production
cache locality or token speed.

Native host allowance peaks at **260 MiB under a 320 MiB cap**, with a **32 GiB**
available-memory floor. The 4090's summed component peaks are **89,832,676 bytes**;
the storage-only 3080 Ti peaks at **37,748,160 bytes**. All owned cleanup counters
return to zero. Prefix plus Attention-reference allowances total **576 MiB**;
the separate MoE oracle allowance is **512 MiB**. These ledgers exclude framework,
compiler and driver overhead and are not RSS/OS limits or the full-engine governor.
The minimum sampled system-available memory across these runs is in the summary
(over 116 GiB). Other applications are not stopped.

## Reproduce

Prepare the original gate/router/shared fixtures as in the component reports.
The frozen public capture includes the qualified provenance needed by native-only
validation; using it does not require installing Torch.

```sh
vesper prepare-connected-store \
  --plan results/2026-10-18-real-moe/plan.json \
  --expert-directory artifacts/connected-experts --store artifacts/real-connected-store
python examples/recheck_connected.py \
  --reference-report results/2026-10-18-real-moe/connected-moe.json \
  --plan results/2026-10-18-real-moe/plan.json \
  --captured-inputs results/2026-10-18-real-moe/prefix-inputs.json \
  --store artifacts/real-connected-store \
  --gate-fixture artifacts/gate-fixture/native-gate.npz \
  --router-fixture artifacts/router-fixture/native-router.npz \
  --shared-fixture artifacts/shared-fixture/native-shared.npz \
  --output artifacts/real-native-only.json
```

To rerun the paired oracle, use `vesper compare-connected` with the same plan,
capture, store/fixture arguments and the qualified optional reference environment.
New capture uses the explicit `capture-prefix --attention-reference ptx-memory`
path and reruns complete Attention qualification. CPU CI does not run GPU parity.

Complete [results](../results/2026-10-18-real-moe), including the original prefix
vectors, full per-boundary checks, import accounting and independent recheck,
are linked from [summary / SHA-256](../results/2026-10-18-real-moe/summary.json).
