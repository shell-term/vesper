# Original output and sampling, and the streamed model-reference design

The output/sampling component gate for
[Issue #10](https://github.com/shell-term/vesper/issues/10) passes. This includes
the **complete original output matrix**, all 129,280 logits, stochastic sampling,
RNG-state recovery and independent replay without reference frameworks.
**Issue #10 remains open:** forty-layer execution, model-level comparisons,
working generation and reasoning/coding/tool-use evaluations are unfinished.
The [complete evidence summary](../results/2026-10-19-output-sampling/summary.json)
preserves this distinction and the original remaining acceptance requirements.

## Original output tail

The released head is stored as BF16 but `ParallelHead` keeps its weight in
**FP32** and computes `F.linear(x.float(), weight)` into FP32 logits. The native
candidate promotes the original bits exactly and uses FP32 cuBLAS, with TF32
disabled. It computes every vocabulary row; it does not truncate the vocabulary.
Bounded host conversion chunks avoid allocating another complete FP32 host copy.
The full resident GPU head weighs 2,647,654,400 bytes.

The connected tail covers unchanged `Block.hc_pre`, the final learned
`RMSNorm.forward`, `ParallelHead.forward` and `sample`. It uses the existing
qualified HC collapse and normalization kernels. The collapse is weight
independent; the probe reuses its layer-0 operator, not layer-0 HC mix weights
as a replacement for the final layer's coefficients.

Twenty-four conditions cover eight original embedding rows at batch one/two,
generated HC streams at three magnitude scales, and all-zero streams. These are
output-component inputs, not the hidden states of a preceding forty-layer run.
All **312 typed/shaped boundaries** and **4,654,080 FP32 logits** match original
methods bit for bit. Another **48 connected sampling calls / 144 boundaries**
cover greedy and temperature 0.7 selection from those actual head logits.

A deliberately substituted BF16 GEMM differs in **129,272 of 129,280 logits**
in its control condition and is rejected. Passing the greedy token alone would
not qualify that replacement.

The fixture fetches 1,323,919,360 original bytes: the head, final norm and eight
embedding rows. Source revision, tensor addresses, all raw SHA-256 values and
the complete fixture identity are pinned. Each network read is at most 8 MiB;
verified partial chunks can resume, existing foreign partials are preserved,
and concurrent importers are refused by an OS file lock. Preparation admits
640 MiB for catalog/transfer overhead and retains the 64 GiB free-disk reserve.
This does not download the complete 510 GB checkpoint.

## Original stochastic selection

The pinned original `sample` uses greedy argmax at zero temperature. Otherwise
it divides by `max(temperature, 1e-5)`, computes full-vocabulary FP32 softmax,
draws exponential noise, divides probabilities by that noise, and selects the
maximum. The native implementation follows those actual operation boundaries,
the original CUDA reduction order and its Philox execution policy.

Ninety conditions cover seeds 0, 2026 and 2^64-1; batch one/two; flat, banded
and large logits; and temperatures 0, 1, 0.7, 1e-6 and 2. All **378 checked
boundaries** match, including every FP32 probability, exponential draw and
selection score, each selected ID, and the generator offset before/after every
call. Greedy calls consume no random values. Native RNG state is explicit and
isolated from other processes' generators.

PyTorch's pinned CUDA `at::log(float)` uses `__logf`. Substituting ordinary
`logf` changes **109,666 of 129,280 exponential values** in the negative control.
It is rejected even when the selected token happens to agree. The accepted
implementation uses the original operation, including the near-one clamp in
the exponential transform. All six relevant PyTorch source files are fetched
with bounds and checked against their pinned raw hashes.

Four real native failure/retry modes cover host admission pressure, the third
GPU allocation, probability readback and cancellation after score computation.
Failed work returns no result, frees temporary payload buffers and leaves the
committed seed/offset unchanged. Retrying reproduces all original boundaries.
A bounded identity-checked snapshot restores the second original draw exactly;
foreign, copied and consumed handles are refused. Cancellation recovery retains
the shared memory policy, and ordinary work still requires explicit resume.

## Resource and runtime scope

The native-only runs have none of Torch, TileLang, Transformers or SymPy
installed or imported. Original comparisons use a separate pinned reference
environment, Torch 2.14.1+cu130 at commit
`5c4886908584029761b579af026dcfb627c84070`. Native compilation uses NVRTC 12.4,
SM89, `--fmad=false`, `--ftz=false`; FP32 output uses cuBLAS 12.4.2, math mode 0.
Sampling currently requires the qualified 4090 profile: 128 SMs, 1,536 maximum
threads per SM. CUDA/cuRAND headers remain in the user's Toolkit installation.

| Recorded native run | Output plus sampling | Sampling alone |
| --- | --- | --- |
| Explicit probe host ceiling | 16 GiB | 2 GiB |
| Peak global host charge | 9,026,142,208 bytes | 201,326,592 bytes |
| Primary GPU payload peak | 2,674,872,536 bytes | 5,171,200 bytes |
| System available-memory floor | 32 GiB | 32 GiB |
| Final host/GPU charges | Zero | Zero |

The full-runtime default remains 96 GiB / 32 GiB. These are smaller comparison
ceilings. Component admissions, actual resident/private-commit observations,
compiler/driver overhead and optional-reference overhead remain separate;
see the [memory policy](memory-policy-validation.md). The report contains full
process observations rather than presenting a reservation as an OS memory cap.
No other workload was stopped, no GPU reset was needed and page-file settings
were unchanged. These component runs establish neither model quality nor
generation throughput.

## Streamed full-model comparison design

The original acceptance requirement to design a comparison without assuming
the whole checkpoint fits in RAM is addressed by the
[metadata-derived reference plan](../results/2026-10-19-output-sampling/reference-plan.json).
It binds all 96,085 original tensor descriptors and all forty layer roles.
The measured tail gates above establish its complete-output/sampling endpoint;
the rest of the reference/model executor still needs implementation and tests.

1. **Freeze a local, explicit workload.** Use the pinned original tokenizer and
   prompt encoder, rank one, batch one, one query per original `forward` call,
   text inputs and an initial context bound of 1,152. Preserve the exact token
   IDs, call shapes, seed and temperature. Serial prompt processing is compared
   to serial original calls; it does not prove equality with batched prefill.
2. **Prepare artifacts separately from forward execution.** Stream original
   packed bytes/scales from authenticated backing artifacts. All forty layers'
   non-Engram-table/non-routed-expert controls total **7,199,083,968 bytes**;
   these controls plus the complete head and final norm total
   **8,522,921,408 bytes** on disk. A layer's six selected original experts
   occupy **112,803,840 packed bytes**. Embeddings and Engram retain original row
   addresses and read required rows. These are sizing facts, not execution proof
   or a promise that only this much data suffices for every workload.
3. **Run the original oracle in its own local process.** Execute unchanged
   original methods, binding one layer/current expert set to resident data as
   needed. Use the explicitly qualified original-arithmetic Attention memory
   adapter where the untouched launch exceeds 4090 shared memory. Never replace
   the original router with a recorded native decision: each side independently
   computes IDs/weights. Missing artifacts stop preparation/verification; they
   never change the route. No remote inference endpoint or paid compute is used.
4. **Seal full original boundaries before native replay.** Record inputs, all
   per-layer HC/Attention/MoE boundaries, KV/index/candidate ownership and state,
   router sets/weights, complete logits and sampling state. Hold only bounded
   current tensors/diagnostics on the host/GPU and write complete typed/shaped
   goldens. Native execution must reproduce all forty stages and the final
   output using its own arithmetic and state, without the reference frameworks.
5. **Publish a whole-token transaction.** Window layers 0/1, compressed KV/index
   sources, Engram history, sampler seed/offset and model position must commit
   together. Preserve bounded recovery handles/copies across layer failures.
   Test failures after early and late blocks, replay/rollback and continuation
   across prompt/decode boundaries. Do not treat already tested sublayer
   transactions as proof of this full transaction.
6. **Run actual generation/evaluation gates.** Test deterministic repeated
   decoding, source-equivalent stochastic continuation and scoped reasoning,
   coding and tool-format workloads. Publish actual prompts, tokens, logits,
   limitations and failures before claiming model-level fidelity or generation.

The plan requires neither all 510 GB in RAM nor all 288.8 GB of routed expert
bytes on the GPU. Actual full-model host/process/driver peaks must still be
measured under the common 96 GiB ceiling and 32 GiB available reserve. The
allocation census does not pre-qualify that future executor. Full artifact
registration/capacity remains a separate import decision, as in the
[artifact report](engram-artifacts-validation.md).

## Reproduce and attribution

After preparing the original metadata and layer-0 prefix fixture from the
[artifact](engram-artifacts-validation.md) and
[layer-0 reports](layer0-block-validation.md), use:

```console
python examples/prepare_output.py --metadata artifacts/import-source --output artifacts/output-fixture
python examples/plan_decode_reference.py --metadata artifacts/import-source --output artifacts/reference-plan.json
```

In the pinned optional reference environment:

```console
python examples/compare_output.py --metadata artifacts/import-source --fixtures artifacts/output-fixture --prefix artifacts/prefix-fixture/native-prefix.npz --reference artifacts/reference --output artifacts/output-original.json
python examples/compare_sampling.py --reference artifacts/reference --output artifacts/sampling-original.json
```

In the independent native environment:

```console
python examples/recheck_output.py --fixtures artifacts/output-fixture --prefix artifacts/prefix-fixture/native-prefix.npz --reference-report results/2026-10-19-output-sampling/output-original.json --output artifacts/output-native.json
python examples/recheck_sampling.py --reference-report results/2026-10-19-output-sampling/sampling-original.json --output artifacts/sampling-native.json
python -m unittest discover -s tests -v
```

Complete published evidence: [original output](../results/2026-10-19-output-sampling/output-original.json),
[native output](../results/2026-10-19-output-sampling/output-native.json),
[original sampling](../results/2026-10-19-output-sampling/sampling-original.json),
[native sampling](../results/2026-10-19-output-sampling/sampling-native.json),
[reference plan](../results/2026-10-19-output-sampling/reference-plan.json) and
[summary/file hashes](../results/2026-10-19-output-sampling/summary.json).

Original model methods retain the [DeepSeek license](../third_party/DeepSeek-LICENSE).
Sampling reduction/distribution specialization follows the pinned PyTorch
[SoftMax.cu](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/cuda/SoftMax.cu),
[distribution policy](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/cuda/DistributionTemplates.h),
[exponential transform](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/core/TransformationHelper.h),
[numeric operations](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/NumericUtils.h) and
[block reduction](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/cuda/block_reduce.cuh).
The complete [PyTorch license](../third_party/PyTorch-LICENSE) remains in source
and wheel distributions. No model weights or generated CUDA binaries are
included in the repository or wheel.
