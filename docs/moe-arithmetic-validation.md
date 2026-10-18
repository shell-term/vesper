# Shared expert and MoE accumulation validation — 2026-10-18

[Issue #5](https://github.com/shell-term/vesper/issues/5) now has passing
exact-bit evidence for the original FP8 shared expert, ordered accumulation,
and six original FP4 experts combined with that shared expert. The input routes
are supplied by the test. Connecting the original gate and bounded offload is
[issue #6](https://github.com/shell-term/vesper/issues/6); model generation,
quality and speed remain untested.

| Check | Result on RTX 4090 |
| --- | --- |
| Original shared expert | 54 generated BF16 inputs; all 13 boundaries match; 0 / 276,480 final-element differences |
| Aggregation with supplied BF16 contributions | 38 cases, 351 token rows; every indexed FP32 write, shared addition and final BF16 conversion matches; 0 / 1,797,120 final-element differences |
| Six original routed experts + original shared expert | 16 single-token conditions; all expert stages and aggregation boundaries match; 0 / 81,920 final-element differences |
| Separate native process without PyTorch installed | All 108 case goldens match, including intermediate hashes |

The six-expert checks include 96 weighted routed-expert calls, with 491,520
final elements, and 16 shared-expert calls. Their contributions are then added
and compared with actual `MoE.forward`. These are finite generated inputs and
deliberately supplied routes, not activation/routing traces from a full model.

## Original bytes and arithmetic

The source is `deepseek-ai/DeepSeek-V4.1-Flash-1017`, revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`. Only selected tensors are downloaded
with exact HTTP ranges of at most 8 MiB. A server that ignores ranges is refused.

The layer-0 shared expert is **FP8 E4M3**, unlike the routed FP4 experts. Each of
its three weight matrices contains 11,796,480 original bytes. Its E8M0 scales
are per **32 × 32 weight block**, with 11,520 bytes per projection. The six
shared tensors total **35,424,000 bytes (33.78 MiB)**. Full tensor identities,
shapes, offsets and SHA-256 digests are in the published report. The fixture
loader verifies these layouts and hashes before admission. No shared weight
is converted to FP4 or otherwise requantized.

The new FP8 projection uses `mma.sync.m16n8k32` E4M3 arithmetic and the original
K=32 grouping, scale multiplication and FP32 addition order. It reuses the
qualified per-32 activation quantizer and SwiGLU implementation. The native
shared expert has no routing weight. Its gate/up projections return BF16;
clamping and SwiGLU run in FP32 before the same BF16 down-input boundary.

Routed expert weights multiply the FP32 SwiGLU product **before** conversion to
the down-projection input dtype and its activation quantization. Multiplying an
unweighted final expert output instead is a failed negative control: it differs
in **80 / 96 calls and 351,213 final BF16 elements** in the original-weight test.

## Accumulation order and participation

The single-rank reference iterates selected experts in **ascending expert ID**,
regardless of their top-k rank. Each returned BF16 contribution is widened and
added to its token's FP32 accumulator. The shared BF16 result is added last.
Only then is the result converted to BF16, with the original input shape restored.

`MoEAccumulator` enforces that order, rejects missing contributions and prevents
an early shared addition. It accepts completed expert outputs; it never applies
a post-projection route weight. Current calls synchronize each kernel and expose
diagnostic snapshots. This is a verification path, not a throughput optimization.

The aggregation matrix covers 0, 1, 2, 8 and 32 tokens, sparse expert IDs through
383, experts with no participating tokens, a token participating in six experts,
multiple tokens sharing an expert, signed zeros and BF16 subnormal contributions.
Eight-token cases also exercise the reference's restored `[2, 4, 5120]` shape.
The empty batch uses supplied empty expert outputs; it does not qualify an empty
FP8 GEMM. Batched expert/GEMM parity is outside this result.

Three permutations of one cancellation-sensitive input produce the same correct
output. Deliberately adding in route-rank order, rounding to BF16 after each
expert, or adding the shared expert first each produces detectable differences.
These controls ensure that a final-output pass is sensitive to the required
order and rounding boundaries.

An expert ID repeated **within one token's top-k** is rejected. Original top-k
selection returns unique IDs. Repeating an ID would cause overlapping CUDA
advanced-index writes in `y[idx] += ...`; this probe does not invent a scatter-sum
interpretation for that invalid routing input.

## Unchanged reference bodies and the adapter correction

The oracle extracts unchanged `linear`, `Expert.forward` and `MoE.forward` AST
bodies from the hash-checked original files. The synthetic aggregation check
supplies gate results and BF16 expert callbacks to isolate indexed accumulation.
The original-weight check additionally executes all seven actual original
`Expert.forward` bodies inside `MoE.forward`, comparing their outputs with the
separate stage diagnostics. CUDA dispatch tracing records each actual indexed
write and shared addition; it does not rewrite the selected method body.

The first shared-expert attempt failed after changing matrix shapes. A separate
adapter inspection confirmed that TileLang 0.1.12's NVRTC adapter stores launch
handles in a shared class table: creating `[5120, 2304]` replaces the earlier
`[2304, 5120]` launch handle. The existing isolation for activation/FP4 kernels
now also covers `fp8_gemm_kernel`. The two inspections compile specializations
and compare handle tables, without launching a mismatched kernel. Instance
tables preserve the earlier handle. All 54 shared cases then pass with the same
native FP8 cubin; no arithmetic, source hash or comparison tolerance changed.

The complete failed attempt is retained as **historical, invalid-oracle evidence**,
along with both handle inspections. It is not presented as a mathematical parity
failure or a passing reference run. A CPU regression test covers all three
factories and prevents launch-table sharing from returning.

The measured oracle is PyTorch `2.14.1+cu130`, revision
`5c4886908584029761b579af026dcfb627c84070`, CUDA 13.0, TileLang 0.1.12 and NumPy
2.5.2. The native-only recheck uses NumPy 2.5.4, CUDA Driver/NVRTC and no installed
or imported PyTorch. Native kernels compile with `fmad=false`, `ftz=false`.
Other hardware/compiler/reference profiles require their own parity gate.

| Native cubin | SHA-256 |
| --- | --- |
| Existing routed expert, unchanged | `e4d5330f1f0f9893463370ba2361f4002b06f7fc496b94490fdbc8ef5ce1ccbf` |
| Shared FP8 projection | `ffa252ef6bf601ef5f290382cad9cd023d6208f5eeb3a940df33c34b4ac390bf` |
| Ordered aggregation | `1548b2f434ca560b26b7d42b35cd82d54d6c264c995e75a04f840711ab895771` |

## Resource scope

The shared fetch reserves 128 MiB of host space. Reference comparisons reserve
192 MiB for the shared probe, 196 MiB including the accumulator for synthetic
aggregation, and 516 MiB including the accumulator for the six-expert probe.
These are conservative reservation allowances, not measured process RSS or
OS-enforced limits. A 32 GiB system-available floor is checked before reservations
and before each reference case. The minimum sampled availability was
132,550,332,416 bytes (123.45 GiB); other workloads were not stopped.

Explicit native GPU allocations peak at 35,526,888 bytes for the shared expert,
1,310,848 bytes for the 32-token accumulator, and **148,989,020 bytes (142.09 MiB)**
across six routed experts, the shared expert and a one-token accumulator in the
original-weight check. Expert executors each enforce 64 MiB; the accumulator
enforces 4 MiB. Each device allocation checks a 1 GiB free-memory reserve.
The independent native-only process retains a 32-token workspace and peaks at
150,258,904 explicit bytes across its components.

Explicit device allocation counters and the accumulator's host reservation return
to zero after the checks. Python/NumPy runtime and metadata, CUDA modules/contexts
and compilation, caller-retained diagnostics, and the optional PyTorch/TileLang
oracle are outside those native buffer ledgers. The full-engine 96 GiB governor,
concurrent workloads and pressure/offload integration remain separate gates.

## Reproduction and complete evidence

Use the optional reference environment described in
[the initial report](first-probe.md#reproduction). Reference dependencies are not
needed for native-only rechecks. Original weights stay in ignored artifacts.
The three earlier expert fixtures are reused; fetch experts 3–5 to make six.

```sh
vesper fetch-shared-fixture --directory artifacts/shared-fixture
vesper fetch-expert-fixture --expert 3 --directory artifacts/expert-fixture-3
vesper fetch-expert-fixture --expert 4 --directory artifacts/expert-fixture-4
vesper fetch-expert-fixture --expert 5 --directory artifacts/expert-fixture-5
vesper compare-shared --fixture artifacts/shared-fixture/native-shared.npz --output artifacts/shared-expert.json
vesper compare-aggregation --output artifacts/moe-aggregation.json
vesper compare-original-accumulation --fixtures artifacts/expert-fixture/native-expert.npz artifacts/expert-fixture-1/native-expert.npz artifacts/expert-fixture-2/native-expert.npz artifacts/expert-fixture-3/native-expert.npz artifacts/expert-fixture-4/native-expert.npz artifacts/expert-fixture-5/native-expert.npz --shared-fixture artifacts/shared-fixture/native-shared.npz --output artifacts/original-expert-accumulation.json
python examples/recheck_moe_arithmetic.py --shared-fixture artifacts/shared-fixture/native-shared.npz --fixtures artifacts/expert-fixture/native-expert.npz artifacts/expert-fixture-1/native-expert.npz artifacts/expert-fixture-2/native-expert.npz artifacts/expert-fixture-3/native-expert.npz artifacts/expert-fixture-4/native-expert.npz artifacts/expert-fixture-5/native-expert.npz --reports-directory results/2026-10-18-moe-arithmetic --output artifacts/native-only-recheck.json
```

The comparison commands return exit code 2 for blocked dependencies or failures.
Run `examples/inspect_fp8_handles.py --output artifacts/handles-class.json` and
then, in a fresh process, the same command with `--isolate` and a separate output
file to reproduce the adapter inspection. It compiles both shapes but launches
neither kernel; the optional reference environment is required.

Every checked boundary and expected hash is published in
[the complete results](../results/2026-10-18-moe-arithmetic/), including the
historical failed run. The native-only result records hashes of the exact public
reference files. The public repository includes no model tensors or downloaded
reference source. CPU CI checks guards and fixtures; it does not rerun CUDA parity.
