# Single-row routing validation — 2026-10-18

> **Recorded PC evidence.** Results, requirements and next-step wording below
> describe this report's experiment/profile. They do not set the current
> development queue or require exact equality from every new candidate. See
> [current policy](quality-speed.md), [validation](validation.md) and
> [PC roadmap](roadmap.md). Raw evidence and original scope are unchanged.

[Issue #3](https://github.com/shell-term/vesper/issues/3) now has passing evidence
for its isolated postprocessing scope: **244 official gate conditions and 232
additional synthetic selection conditions match every checked output bit**.
The previously unsupported 29 selected-tie conditions now pass, including their
ordered expert IDs and FP32 weights. Gate projection, connected MoE execution
and model-level quality remain untested.

## Scope and reference profile

Input is one supplied finite FP32 projection-logit vector of 384 elements. The
probe covers temperature 1.0, sqrt-softplus, trained text/image selection bias,
top-six selection, unbiased weight gathering, the six-element normalization
reduction with `1e-20`, and route scaling by 1.5. The pinned inference config
omits temperature/normalization flags; original `ModelArgs` supplies the checked
defaults. The native runtime needs NumPy and CUDA, not PyTorch.

The oracle extracts the unchanged, hash-checked `Gate.forward` method. Its
`linear()` call is explicitly replaced by an identity to accept supplied logits.
The fixture contains the original layer-0 `bias` and `bias_vl`, 1,536 bytes each:
**3,072 model bytes**, with checkpoint identities, ranges and hashes checked.
No additional model tensors were needed for this tie-handling work.

- Model: `deepseek-ai/DeepSeek-V4.1-Flash-1017`
- Revision: `1bae0e1088993b9b82de08f59de5be5a34e42170`
- Inference config SHA-256: `2e84f45cf1dac8c7fcbb200e96667d4b913275690668ed496f24c7747207a809`
- Native device/compiler: RTX 4090, NVRTC 12.4, `fmad=false`, `ftz=false`
- Oracle: PyTorch `2.14.1+cu130`, CUDA 13.0, NumPy 2.5.2
- PyTorch revision: `5c4886908584029761b579af026dcfb627c84070`

The reference command refuses another PyTorch revision instead of certifying an
unexamined tie policy. This is a single-row, finite FP32, NVIDIA CUDA profile;
other batch shapes, backends, builds and nonfinite inputs are outside the gate.
The [top-k API](https://docs.pytorch.org/docs/2.14/generated/torch.topk.html)
does not promise stable tied indices across invocations or builds.

## Reproducing the actual tie path

For this shape, the pinned NVIDIA CUDA implementation uses `sbtopk`, not the
ROCm atomic-compaction or warp-merge paths. A 384-thread block has one gather
iteration. Finite FP32 values are converted to ordered integer keys, preserving
the distinction between signed zeros for threshold selection. It gathers keys
strictly above the kth key in ascending ID order, then fills the remaining slots
from equal keys in ascending ID order.

The six gathered values are then sorted by ATen's **32-cell bitonic network**,
including invalid padding and its unstable equal-key swaps. Sorting the six
values with a stable rule gives a different tuple. Vesper reproduces that
network and the observed six-element FP32 reduction tree. Native code currently
executes the small selection/network serially; this is a correctness baseline,
not a measured routing-speed optimization.

Seven hash-checked PyTorch source/license files pin dispatch, key conversion,
gather and sorting behavior. Fetching is bounded to 256 KiB per file and sources
are cached in ignored artifacts. Adapted CPU/CUDA code retains the complete
[PyTorch license](../third_party/PyTorch-LICENSE), also included in wheels.

## Measured results

| Check | Observed result |
| --- | --- |
| Original `Gate.forward` conditions | 244 / 244 pass |
| Unbiased and bias-corrected scores | 0 / 93,696 bit differences each |
| Gate ordered expert IDs | 0 / 1,464 differences |
| Gate FP32 route weights | 0 / 1,464 bit differences |
| Gate normalization denominators | 0 / 244 bit differences |
| Gate selected-tie conditions | 29 / 29 pass |
| Original-bias decision API calls | 240 admitted, including 27 tied calls |
| Additional synthetic selection conditions | 232 / 232 pass, including 222 tied cases |
| Synthetic selection ordered IDs and weights | 0 / 1,392 differences each |
| CPU top-six specification against CUDA PyTorch | 232 / 232 pass |

The 244 gate conditions retain all 116 historical inputs and add 128 conditions
from 16 more seeds at four scales. They cover text/image masks, extreme/zero/
tiny logits, the softplus threshold, near-ties, synthetic zero-bias ties and
logits that approximately cancel each real trained bias at nine margins. Four
zero-bias conditions remain diagnostic-only; the decision API always uses the
original trained bias.

The separate 232-condition selection probe supplies synthetic unbiased and
selection scores directly to unchanged PyTorch top-k/gather/reduction operators.
It stresses discrete ties, 0–5 strictly higher experts with 2–384 boundary peers,
signed zero, subnormals, finite extremes, repeated identical calls and raw
weights spanning many powers of two. It does not certify the score transform or
claim that those artificial selection scores came from the trained bias.

There are 244 reference gate forwards, 232 reference selection calls and 716
native inspections. Every primary and secondary comparison passes, including
ordered IDs, raw gathered scores, denominators and route-weight bits. The
comparison reports `pass` and exit code 0 without relaxing a tolerance.

## Ordering and remaining work

The original MoE accumulates routed experts by ascending expert ID. The earlier
partial probe had matching ID-weight maps but different tuples in tied cases;
those permutations alone were not evidence of model-quality loss. The new path
matches the original ordered tuples as well, preserving the normalization
order instead of relying on a permutation-equivalence argument. The historical
[partial results](../results/2026-10-18-routing/README.md) remain unchanged.

This establishes supplied-logit postprocessing only. Original gate projection
is [issue #4](https://github.com/shell-term/vesper/issues/4); actual expert/shared-
expert arithmetic and accumulation are [issue #5](https://github.com/shell-term/vesper/issues/5).
Connected MoE, Transformer blocks and whole-model quality remain separate gates.
No cache-dependent bias, pruning, expert replacement or lossy requantization is
introduced. The router is not yet connected to the expert byte cache.

## Resources and reproduction

Native explicit device allocations peak at 7,760 bytes under a 1 MiB cap with a
1 GiB free-device reserve; zero explicit allocations remain after cleanup. The
32-cell sorting scratch uses compiler-managed kernel storage, outside that
allocation ledger. CUDA contexts/modules, compiler, PyTorch oracle and returned
diagnostics are also excluded. This is not an OS/process memory-cap claim.

```sh
vesper fetch-router-fixture --directory artifacts/router-fixture
vesper compare-router --fixture artifacts/router-fixture/native-router.npz --device 0 --output artifacts/router.json
```

The comparison requires the pinned CUDA PyTorch build but does not use TileLang
or load the full model module. Complete
[results, provenance and hashes](../results/2026-10-18-routing-ties/README.md)
retain all 476 conditions. CPU tests cover source/fixture tampering, resource
bounds, invalid inputs, selection-only bias use and recorded tie-order goldens.
CPU CI is separate from these manual GPU/reference checks.
