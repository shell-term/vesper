# Native Hyper-Connections and complete layer-0 block validation

Issue [#7](https://github.com/shell-term/vesper/issues/7), milestone M2.
Native Hyper-Connections (HC) and residuals now connect the qualified
[window Attention layer](attention-projection-validation.md) and
[tiered MoE](real-moe-validation.md) into a complete original-weight layer-0
Transformer block. The qualified block profile is **one rank, batch one, one
token per call, uncompressed window KV**. The subsequent
[compression probe](compression-validation.md) qualifies original compressor/cache
construction. Other layers, indexed/shared KV and Engram remain separate gates. Issue #7 and M2 stay open; text generation
and full-model quality are unfinished.

## Results

| Check on RTX 4090 | Result |
| --- | --- |
| HC components | 84 cases: 42 mixes, 21 pre-collapse and 21 residual expansion |
| Complete successful block calls | 26: 16 core calls, five fault-test prefills and five retries |
| Boundaries per block call | 225, including ordered routing, six experts, shared expert, residuals and KV |
| Combined component/block comparisons | 6,648 boundaries / 56,874,975 storage elements; zero bit differences |
| Diagnostic versus unchanged original `Block.forward` | All 26 calls match final BF16 hidden state, next-layer FP32 pre-mix and KV state |
| Diagnostic versus unchanged original `Attention.forward` | Both output/cache controls pass in all 26 calls |
| HC diagnostics versus unchanged original methods | 243 control groups / 433 comparisons pass |
| Deliberately changed arithmetic | All four negative controls detected |
| Whole-block failure/retry | Five refused calls preserve committed KV or require explicit recovery; all five retries match |
| Independent process without Torch/TileLang installed or imported | Same 84 HC cases, 26 block calls and failure matrix pass |

Counts include intermediate arrays, repeated diagnostic boundaries and the final
output aliases. They are not token counts or independent final-output samples.
Comparisons use exact storage bits, including signed zero, without tolerances.
This finite corpus does not establish equivalence for every possible input.

HC tests cover one, two, three, four, six and eight rows; zero, signed-zero,
mixed-scale, near-epsilon, cancellation and ordinary generated inputs. The block
matrix uses original embedding rows for sequences `(1, 42)` and `(1000, 4096)`,
with two repeats and both text/image gate-mask settings. Layer-0 input is the
embedding repeated across four HC copies, with initial pre-mix `(1, 0, 0, 0)`.
That initial pre-mix is supplied for each token; the returned pre-mix belongs to
the next layer, not the next token. The image-mask checks use these embedding
inputs and do not establish image-understanding quality.

Across those calls the unchanged router selects 27 distinct original experts.
The native path executes its original ordered IDs and weights through the
audited SSD/RAM/3080 Ti storage/4090 compute cache. The 3080 Ti stores and hashes
original bytes; it does not perform model arithmetic in this probe. Existing
fixtures and the 507,617,280-byte aligned expert store are reused; no new model
weight bytes or full checkpoint shards were downloaded for this comparison.

## Original arithmetic and oracle

The source is DeepSeek revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`. The optional oracle extracts unchanged
`Block.hc_mixes`, `hc_pre`, `hc_post` and `Block.forward`, and executes the actual
original Attention/MoE methods with original weights. An independently evaluated
unchanged `Block.forward` controls the diagnostic chain's final hidden state,
next-layer coefficients and cache. The gate projection is not replaced with an
identity or supplied routes.

For the full-shape Attention operation, the oracle remains the separately
[qualified PTX memory adapter](attention-memory-validation.md), which retains
literal official arithmetic/control instructions. The untouched full-shape
Attention kernel still requires more shared memory than the 4090 can launch.
Passing the complete block does not change that limitation.

The new native HC path preserves these details:

- FP32 promotion and square/mean over all `4 × 5120 = 20480` values, followed by
  the original `1e-20` epsilon, reciprocal square root and FP32 linear projection.
- The pinned ATen two-dimensional reduction tree: vector-of-four accumulation,
  block-x reduction, then block-y reduction. A flat 512-thread tree differs for
  this width and some row counts, even though a previous normalization shape
  uses a different tree.
- Explicit fused multiply-add in the HC affine transform, sigmoid/softmax
  behavior, `1e-6` HC epsilon and all 20 Sinkhorn iterations. The probe observes
  all 40 row/column normalization stages and their reduction values.
- FP32 pre-collapse products and the original four-term sum; residual
  `comb[source, target]` orientation, ordered FP32 products/additions and only the
  final BF16 conversion. Earlier product rounding changes output bits.
- The original coefficient handoff: incoming pre-mix feeds this block's
  Attention, Attention's newly computed pre-mix feeds this block's FFN, and
  FFN's newly computed pre-mix is returned for the next block.

Sinkhorn instrumentation adds stores and output arguments only. Removing those
additions restores the original arithmetic AST exactly. Both original and
instrumented factory AST/generated-CUDA hashes are pinned, and every diagnostic
call also checks the unchanged original outputs. This is an explicit diagnostic
variant, not a silently rewritten reference.

The registered native profile is SM89, NVRTC 12.4 and cuBLAS 120402 with default
math/workspace policy. The optional oracle uses the pinned PyTorch revision
`5c4886908584029761b579af026dcfb627c84070` (2.14.1+cu130), TileLang 0.1.12 and
NVRTC 13.4. Compiler, binary, source, original-fixture and routing identities are
recorded and checked before independent replay. Unregistered profiles are
refused rather than assumed equivalent.

| Deliberate change | Detected differences |
| --- | --- |
| Flatten the HC mean reduction tree | 2 / 8 FP32 values |
| Replace 20 Sinkhorn iterations with one | 128 / 128 FP32 combination values |
| Transpose residual source/target orientation | 163,550 / 163,840 BF16 values |
| Round residual products to BF16 early | 62,533 / 163,840 BF16 values |

## State publication and memory

The Attention executor now accepts a finalizer that runs before its bounded KV
snapshot is discarded. The complete block finishes MoE and the final residual
inside that transaction. A failure after Attention cannot commit a partial
block. Cache evictions may remain after a failed call because they change
performance state, not committed model state or routing.

The five injected faults cover observed host pressure, MoE failure after
Attention, final-residual download failure, combined owned-GPU-cap refusal and
rollback's own download failure. The last case blocks another call until
explicit recovery succeeds. All retries reproduce the original output and KV
bits. Pressure is injected into the observation function, and the oversized
allocation is refused before CUDA allocation; real machine exhaustion is not
attempted.

Combined native host allowances are **1,136 MiB**. With the optional oracle they
total **1,968 MiB**. All retain the 32-GiB system-available floor. Minimum sampled
available RAM was 126,929,694,720 bytes (118.21 GiB) in the paired comparison and
129,180,069,888 bytes (120.31 GiB) in independent replay. These are observations
and explicit allowances, not process RSS limits or the future full-engine
96-GiB policy; [#9](https://github.com/shell-term/vesper/issues/9) remains required.

During block calls, a combined primary-GPU guard checks every owned allocation
against **512 MiB** before allocating. Peak combined owned primary buffers were
272,182,716 bytes (259.57 MiB). This excludes CUDA context/module/framework
allocations and component initialization before the block opens. Each component
also retains its own limits and free-VRAM reserve. Owned allocations, host
reservations and context retains are released after the probe; zero cleanup
counters are not proof of zero process RSS or total VRAM use.

The path downloads diagnostic arrays at component boundaries. It is a fidelity
and recovery probe, not a resident high-throughput executor. End-to-end speed,
token rates and full-model quality were not measured.

## Reproduction and evidence

Use the existing bounded prefix, gate/router/shared and 27-expert fixtures from
the [real-input MoE report](real-moe-validation.md), and its qualified Attention
bundle. The bundle builder uses the optional reference environment once; native
execution needs only NumPy, the CUDA Driver and the registered toolkit/cuBLAS.

In the optional reference environment:

```sh
vesper compare-block --reference-directory artifacts/reference \
  --prefix-fixture artifacts/prefix-fixture/native-prefix.npz \
  --attention-bundle artifacts/native-attention-v1 \
  --store artifacts/real-connected-store \
  --gate-fixture artifacts/gate-fixture/native-gate.npz \
  --router-fixture artifacts/router-fixture/native-router.npz \
  --shared-fixture artifacts/shared-fixture/native-shared.npz \
  --expert-directory artifacts/connected-experts --device 0 \
  --output artifacts/block.json
```

Then in the ordinary NumPy-only environment:

```sh
python examples/recheck_block.py --reference-report artifacts/block.json \
  --prefix-fixture artifacts/prefix-fixture/native-prefix.npz \
  --attention-bundle artifacts/native-attention-v1 \
  --store artifacts/real-connected-store \
  --gate-fixture artifacts/gate-fixture/native-gate.npz \
  --router-fixture artifacts/router-fixture/native-router.npz \
  --shared-fixture artifacts/shared-fixture/native-shared.npz \
  --output artifacts/block-native-only.json
```

Independent replay binds exact report bytes, complete coverage, ordered expert
IDs/weight bits, original inputs/fixtures and every boundary hash. CPU tests
cover refusal before CUDA access, coefficient/layout contracts, combined cap
admission and guard cleanup, finalizer rollback and tampered/incomplete evidence.

- [Paired comparisons, complete goldens and unchanged-method controls](../results/2026-10-18-layer0-block/block.json)
- [Independent native-only replay](../results/2026-10-18-layer0-block/native-only-recheck.json)
- [Summary, raw-file hashes and resource observations](../results/2026-10-18-layer0-block/summary.json)

The [compression follow-up](compression-validation.md) now qualifies original
compressed-KV construction/rounding with bounded diagnostic state. The
[indexer follow-up](indexer-validation.md) separately qualifies bounded index
selection and named key/candidate/index handoffs. The later
[connected compressed-Attention probe](compressed-attention-validation.md)
qualifies profiles 2/20/24 and compressed-KV sharing 20-to-24 at context <=64;
larger contexts and their full-block integration remain next.
Other block profiles, larger prefill/batching, Engram, full-model
logits/sampling, full-engine memory governance and workload performance remain
open gates.
