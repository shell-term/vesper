# Connected MoE validation — 2026-10-18

> **Recorded PC evidence.** Results, requirements and next-step wording below
> describe this report's experiment/profile. They do not set the current
> development queue or require exact equality from every new candidate. See
> [current policy](quality-speed.md), [validation](validation.md) and
> [PC roadmap](roadmap.md). Raw evidence and original scope are unchanged.

**Historical generated-input experiment.** Its real-activation gate has since
passed; see the [real-input follow-up](real-moe-validation.md). At this initial
publication, Issue #6 remained open. The
original gate, audited byte cache, selected routed experts, shared expert and
ordered accumulation are connected. On generated BF16 inputs, **30 successful
calls match all 108 checked boundaries each**, including 153,600 final BF16
elements. Four injected failures return no output, and retries match the same
original reference. A separate process without PyTorch reproduces the matrix.

The unchanged official layer-0 Attention specialization cannot run on the tested
RTX 4090: it requires **141,312 bytes (138 KiB)** of dynamic shared memory per
block, while CUDA reports an opt-in maximum of **101,376 bytes (99 KiB)**. Real
Direct MoE input capture was therefore blocked. Generated inputs did not satisfy
that acceptance check, and M1 stayed open. See [issue #6](https://github.com/shell-term/vesper/issues/6)
and the related [Attention work](https://github.com/shell-term/vesper/issues/7).

The subsequent [Query storage experiment](attention-storage-validation.md)
executes a full-shape candidate with 74 KiB of shared memory and passes smaller-
profile intermediate comparisons. Its full-original gate is still unavailable;
the candidate is not substituted into this report's real-activation gate.

## Connected path and original artifacts

Each call converts its FP32 input to the same finite BF16 row used by both the
gate and experts, or accepts original BF16 storage bits directly. The native
gate uses the original BF16 matrix, exact FP32 promotion and the qualified
cuBLAS profile. The router uses the trained text/image bias and returns the
original ordered six IDs and FP32 weights. Cache residency never changes them.

All selected IDs must exist in the pinned store before any expert I/O starts.
An absent expert refuses the call; `forward()` never downloads weights or
substitutes another expert. Selected experts execute in ascending ID order with
routing weights inside SwiGLU before down-input rounding/quantization. Their
BF16 outputs accumulate in FP32; the original FP8 shared expert runs last before
the final BF16 conversion. A retry zeroes the accumulator and recomputes all six
experts. Partial sums from a refused attempt never become a model output.

The source remains `deepseek-ai/DeepSeek-V4.1-Flash-1017`, revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`. Four generated vectors from seeds
20261018, 42, 0 and 65535, at scales 1, 1/16, 16 and 1, are tested with both
text/image bias choices. These eight conditions select **30 distinct original
layer-0 experts**, totaling **564,019,200 bytes**. The selected union is computed
by actual original-weight gate projection and routing, without forced IDs.

Import is a separate explicit step. It fetches only those 30 experts through
bounded, revision-pinned ranges; an existing verified expert is reused. A store
still defaults to four experts. Larger imports/readers require an explicit limit
of at most **48 experts / 902,430,720 weight bytes**. The builder streams one
verified fixture at a time and refuses an existing destination. Every native
store read checks original block/tensor hashes. Every GPU admission checks
original SHA-256 chunk digests before publication or compute.

The original gate/bias and FP8 shared fixtures are reused. No further weight
quantization, pruning or cache bias is introduced. The RTX 3080 Ti stores and
hashes original bytes; model arithmetic runs on the RTX 4090.

## Unchanged reference and checked boundaries

The oracle executes unchanged `linear`, `Gate.forward`, `Expert.forward` and
`MoE.forward` bodies extracted from hash-checked source. The real original gate
is called inside `MoE.forward`. It is not replaced with supplied IDs. Separate
diagnostics check gate projections/scores, every expert stage and every indexed
FP32 addition against the original method's actual output.

Each successful call checks 108 boundaries: eight gate/input arrays, thirteen
stages for each of six routed experts, thirteen shared-expert stages, and nine
accumulation boundaries. All 3,240 boundary checks pass with exact bits. Thirty
calls exercise repeated and changed cache states; they are not thirty unique
inputs. Whole-model activations and output quality remain untested.

The oracle is PyTorch `2.14.1+cu130`, revision
`5c4886908584029761b579af026dcfb627c84070`, CUDA 13.0, TileLang 0.1.12 and NumPy
2.5.2. FP32 precision is `highest`, backend is cuBLAS and tuning/TF32 overrides
are disabled. The named reference profile is enforced. Original TileLang launch
handles retain the existing per-specialization isolation; arithmetic is unchanged.
The oracle retains at most six routed experts at once. The existing routed,
router, shared and accumulation CUDA kernels are unchanged from their qualified
component gates. Native-only rechecks use NumPy 2.5.4 and CUDA without installed
or imported PyTorch.

## Cache states, refusal and retry

| State / fault | Observed result |
| --- | --- |
| Cold original store | Gate-selected experts load through direct reads and pass upload audits |
| Two primary + two helper + two RAM slots | Controlled reverse-order cache preparation exposes primary/helper/RAM hits in one ascending-ID MoE call, with no additional store reads |
| Changing gate decisions and repeated calls | All gate, expert, shared and accumulation bits match |
| One primary + one helper slot, zero RAM | Forced store reloads preserve the same output |
| Pressure before gate | RAM cache is released; no expert/store I/O or output occurs |
| Pressure after three experts | The fourth expert is refused; no partial output; full retry matches |
| Corrupted fourth primary upload | GPU SHA-256 rejects it before publication/compute; full retry matches |
| Failure before fourth OS read | Three reads complete, the next is refused; full retry matches |

Pressure changes only the available-memory observation, without allocating
machine memory. Transfer corruption and read failure are deliberate harness
injections. Gate ID/weight bits are checked on interrupted attempts where the
gate has run. The independent native-only process repeats all 30 successes and
four failures, checking the original fixture provenance and all stage hashes.
All owned cleanup counters return to zero in both processes.

The two native phases complete **108 direct store reads / 2,030,469,120 bytes**.
These counters include failed attempts that finished a verified read before an
upload was rejected. A pre-read failure contributes no completed read. Reads
bypass the OS buffered path; device firmware caching is outside these counters.
Cache preparation and generated inputs are controlled experiments. These data
do not measure production locality or generation throughput.

## Resource scope

The connected probe has a **320 MiB parent host cap** and reserves a conservative
**260 MiB allowance** for fixtures/per-call buffers, the gate, accumulator and
the cache's independent 128 MiB cap. The cache's measured ledger peaks at
99,751,936 bytes in the six-slot phase. Parent allowances cover child caps;
adding parent and child ledger values would double-count reservations.

The system-available floor is **32 GiB**. The minimum sampled availability during
the reference/native checks is **130,656,772,096 bytes**. Explicit native buffers
peak at 89,832,676 bytes on the 4090 and 37,748,160 bytes on the storage-only
3080 Ti. The connected primary cap is 128 MiB; allocations and calls check a
1 GiB GPU reserve. Zero-RAM and six-slot phase counters are published separately.

These are explicit buffers and conservative reservation allowances. They exclude
Python/NumPy runtime and metadata, caller-retained diagnostics, cuBLAS internal
state, CUDA contexts/modules/compiler and the optional official oracle. The
oracle has a separate 512 MiB host allowance. These ledgers do not enforce RSS
or implement the future full-engine 96 GiB memory governor.

## Remaining real-input gate

The optional prefix experiment selects the original layer-0 Attention,
Hyper-Connections and normalization weights plus four original embedding rows,
totaling 130,736,856 bytes. Layer 0 has no Engram or compressed-attention source.
The capture path stops the unchanged `Block.forward` at its MoE input. It also
preserves the official `convert.py` dequantization of `wo_a` to BF16, rather than
treating its released FP8 bytes as an already converted BF16 matrix.

The initial attempt reached the unchanged sparse-attention launch and failed its
dynamic shared-memory attribute request. A no-launch preflight now inspects the
generated original host wrapper and queries CUDA device attribute 97. It records
the 141,312-byte requirement and 101,376-byte limit without altering kernel code,
head count, block shape or arithmetic. Preflight runs before capture loads its
weight fixture. **Zero real MoE inputs have been captured.** Successful prefix
capture on a compatible profile remains unvalidated by this experiment.

The complete blocked result and pinned hashes for `model.py`, `kernel.py`,
`convert.py`, `generate.py` and configuration are published. A faithful Attention
path or separately qualified capture route was required to resolve this
constraint. The [subsequent PTX memory adapter](attention-memory-validation.md)
now supplies that qualified route; this original generated-input report remains
unchanged evidence of the earlier scope.

## Reproduction and complete evidence

Use the [optional reference environment](first-probe.md#reproduction) for the
original comparisons. The existing original gate/router/shared fixtures are
required. Plan/import and the native-only check do not need PyTorch.

```sh
vesper plan-connected --gate-fixture artifacts/gate-fixture/native-gate.npz --router-fixture artifacts/router-fixture/native-router.npz --max-experts 48 --output artifacts/connected-plan.json
vesper prepare-connected-store --plan artifacts/connected-plan.json --expert-directory artifacts/connected-experts --store artifacts/connected-store --output artifacts/connected-import.json
vesper compare-connected --plan artifacts/connected-plan.json --store artifacts/connected-store --gate-fixture artifacts/gate-fixture/native-gate.npz --router-fixture artifacts/router-fixture/native-router.npz --shared-fixture artifacts/shared-fixture/native-shared.npz --expert-directory artifacts/connected-experts --output artifacts/connected-moe.json
python examples/recheck_connected.py --reference-report artifacts/connected-moe.json --plan artifacts/connected-plan.json --store artifacts/connected-store --gate-fixture artifacts/gate-fixture/native-gate.npz --router-fixture artifacts/router-fixture/native-router.npz --shared-fixture artifacts/shared-fixture/native-shared.npz --output artifacts/native-connected.json
vesper prefix-preflight --output artifacts/prefix-preflight.json
```

For published goldens, use the committed `connected-moe.json` and `plan.json`
from [the complete results](../results/2026-10-18-connected-moe/) in the recheck.
Do not overwrite an existing store. A blocked preflight returns exit code 2;
the generated-input component comparison alone did not clear #6's real-input gate.
CPU tests cover input rounding, import limits, provenance, missing-expert refusal,
failure cleanup and complete retry. CPU CI does not rerun the CUDA matrix.
