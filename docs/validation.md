# Validation gates

These gates describe evidence, not a list of interchangeable accuracy scores.

| Gate | Evidence required | Current evidence |
| --- | --- | --- |
| Released artifacts | Pinned checkpoint, exact weight/scale bytes, matching tokenizer and Engram | Original component fixtures, all 48 headers / 96,085 tensor descriptors and released whole-file identities; complete tokenizer map and bounded Engram pass; full local checkpoint registration not run |
| Format specification | All FP4 codes, scale encodings, FP8/BF16 rounding boundaries, signed zero | Pass on Ada and Ampere |
| Projection specification | Native weight slice against an independent block arithmetic oracle | Pass on Ada and Ampere |
| Official kernels | Per-operation exact comparisons with the pinned TileLang code | W1 slice: 4090 passes 8 inputs; 3080 Ti candidate fails, 14/1,024 output mismatches |
| Full expert | Activation quantization, clamps, weighted SwiGLU and three projections | One native expert on 4090 passes 128 single-token conditions, including intermediate bits |
| Tiered expert storage | Identity, original bytes, eviction/reload and unchanged expert computation | Three native experts pass 232 calls across SSD/RAM/two GPUs, including intermediate bits |
| Device integrity audits | Original-byte digest anchors, padding/chunk boundaries, corruption refusal and unchanged computation | Both GPUs pass 96 payload cases and 10 fault injections; optional path passes another 232 official comparisons |
| Routing postprocessing | Scores, trained bias, ordered expert IDs/weights and tie handling from supplied FP32 logits | 244 official gate conditions and 232 selection conditions pass all checked bits; pinned single-row CUDA top-k profile only |
| Original gate projection and routing | Original BF16 weights promoted to FP32, actual linear projection and connected expert selection | 307 generated inputs across both masks give 614 exact-bit passes against unchanged linear/Gate.forward; one row, named cuBLAS/PyTorch builds |
| Shared expert and ordered accumulation | Original shared dtype, routing-weight placement, FP32 additions in expert-ID order and final BF16 rounding | 54 FP8 shared cases, 38 aggregation cases and 16 original-six-expert supplied-route conditions match all checked bits |
| Connected MoE sublayer | Gate projection, routing, accumulation order, shared expert and bounded offload | Generated and real-prefix inputs each pass 30 calls / 108 boundaries and four refusal/retry checks; native-only rechecks match |
| Attention Query storage candidate | Exact intermediates with executable original profiles, then full-shape original parity | 24 smaller-profile cases match all checked bits; full 64 × 512 candidate executes at 74 KiB shared memory; full original gate blocked at 138 KiB |
| Storage-adapted official Attention arithmetic | Original PTX arithmetic/control retention, exact bit transport and intermediate comparisons | All six original binaries reproduce; 36 conditions / 456 comparisons match; explicit capture supplies four real layer-0 vectors; untouched full-kernel launch remains blocked |
| Native sparse Attention operation | Framework-free dispatch, exact boundaries and bounded failure recovery | 36 conditions, 216 oracle-boundary comparisons, zero differences; independent NumPy-only replay passes; subsequent layer-0 integration below |
| Native post-RoPE window KV | Original rounding, ring order, publication failure, saved-state rollback and selected Attention | 398 updates / 128 connected conditions, 3,156 comparisons, zero differences; native-only replay passes; compressed KV pending |
| Native RMS normalization and RoPE | Original learned norm weights, reduction/rounding order, frequency profiles, rotation and projected-input KV/Attention chain | 476 component cases and 56 chain conditions, 3,420 boundary comparisons, zero differences; both frequency profiles cover positions 0 through 131,071; native-only replay passes |
| Original-weight native window Attention layer | Original query/KV/output projections, grouped BF16 conversion, complete forward output/cache, real inputs and atomic failure recovery | 90 component cases and 309 successful layer calls (304 corpus calls + five retries), 17,355 comparisons, zero differences; four real prefix inputs and two window wraps; independent native-only replay passes; layer-0, one rank, batch 1/2, prefill at most three queries |
| Native Hyper-Connections and residuals | Original FP32 projection/reduction, Sinkhorn iterations, coefficient handoff and BF16 boundaries | 84 component cases pass; all four changed-arithmetic controls detected; independent native-only replay passes |
| Full Transformer block | Connected Attention, MoE, Hyper-Connections, residuals and layer-specific Engram/compressed profiles | Uncompressed layer 0 passes 26 calls; compressed layer 2 passes 32 calls; consecutive 2/3 passes 36 pair steps / 18,447 boundaries and ratio-1 20/21 passes 36 pair steps / 19,152 boundaries, plus unchanged Block.forward controls; other profiles and Engram pending |
| Original compressor and compressed KV | Original learned FP32/BF16 projections, softmax pooling, partial groups, group-position RoPE, FP4/E4M3 rounding and state publication | 49 component cases and 432 state calls, 8,751 comparisons, zero differences; five refusal/retries and both snapshot profiles pass; independent native-only replay passes; generated inputs and host diagnostic cache only; subsequent indexer probe below |
| Native indexers and named index sharing | Original key/query/weight projections, FP4/E8M0 rounding, BF16 score boundaries, ordered top-512, candidate blocks, source/consumer ownership and atomic state | Layers 2/20/24 pass 261 component cases and 99 frames, 11,189 comparisons, zero differences; unchanged forward/sharing controls, five refusal/retries, snapshot restoration and native-only replay pass; bounded generated-input/primed-history diagnostic scope; connected compressed-Attention follow-up below |
| Connected compressed Attention | Complete original layer projections, compressor latents, indexers, concatenated window/compressed KV and output, source sharing and atomic state | Profiles 2/20/24 pass 103 frames / 309 layer calls, 31,841 comparisons, zero differences; unchanged full-forward/cache controls, five refusal/retries, snapshot restoration and native-only replay pass; generated independent layer inputs, context <=64, other profiles/larger contexts/complete blocks pending |
| Longer-context connected compressed Attention | Window wraps, actual top-512 truncation, reachable candidate pruning, original full-forward controls and state recovery | Profiles 2/20/24 pass 1,441 frames / 440,997 boundaries with zero differences; continuous empty-start generated streams reach 1,152/260 tokens, explicit synthetic histories reach 16,392 positions; native-only replay, five failure/retries and snapshots pass; full-model histories/other profiles/complete blocks pending |
| Named Attention KV/index ownership roles | All configured KV/index sources, ordinary selected-index consumers, original full-forward controls and transactional state | Sixteen profiles in four independent groups pass 2,224 frames / 684,616 boundaries with zero differences; continuous 260-position streams, explicit seeded histories through 1,028 positions, twenty failure/retries, snapshots and native-only replay; complete compressed blocks/full backbone pending |
| Complete compressed source block | Layer-specific HC/Attention/MoE, exact actual gate, complete output and transactional source state | Original layer 2 passes 32 calls / 8,793 boundaries / 70,536,278 storage elements with zero differences; 640 unchanged Block/Attention controls, six failure/retries, snapshots and native-only replay pass; four-position generated HC streams, consecutive-block follow-up below, full model pending |
| Complete consecutive source/consumer blocks | Actual hidden/pre-coefficient handoff, shared compressed KV/selected IDs, both complete MoEs and atomic pair state | Original blocks 2/3 pass 36 pair steps / 72 block calls / 18,447 boundaries with zero differences; 648 Block controls, 720 Attention controls, eight failures, snapshots and native-only replay pass; four-position initial generated HC streams, other complete roles/longer chains/full model pending |
| Complete ratio-1 source/consumer blocks | Actual hidden/pre-coefficient handoff, source-20 candidate state, original KV/index reuse and atomic pair state | Original blocks 20/21 pass 36 pair steps / 72 block calls / 19,152 boundaries with zero differences; 648 Block controls, 684 Attention controls, eight failures, snapshots, five changed-state controls and native-only replay pass; generated entry, four-position context; independent from 2/3, refreshed-index blocks/full model pending |
| Complete source-to-refreshed-index chain | Every consecutive hidden/pre handoff, source KV/candidates, layer-24 selection and layer-25 reuse, atomic six-block state | Original blocks 20..25 pass 40 chain steps / 240 block calls / 59,600 boundaries / 415,577,840 storage elements with zero differences; 2,160 unchanged Block and 2,040 Attention controls, ten failure/retries, snapshots, seven changed-state controls and native-only replay; generated entry, four-position context, batch one |
| Original Issue #7 acceptance | All five original checks bound to complete original goldens and fresh framework-free native replay | Seventeen sealed evidence groups pass the [acceptance audit](block-ops-qualification.md); configured role coverage is distinct from all-forty-layer numerical execution, full-model quality and generation |
| Original tokenizer, artifact import and Engram | Original compressed map/hashes/masks/row addressing, bounded reads, actual capacity plan and unchanged original output/state | All 129,280 map entries, 42 CPU/CUDA hash cases, 110 Engram steps / 6,820 boundaries pass; seven changed controls detected; ten failure/retries and snapshots pass; independent native replay without reference frameworks, six small original direct-I/O tensors / 192 ranges and five original [Issue #8 criteria](engram-artifacts-validation.md); full original-shard registration/download and backbone integration not run |
| Full model | Router sets/weights, attention/indexing/KV, logits and sampling | Not implemented |
| Output tail and original sampling | Exact HC collapse/final norm, complete FP32 head/logits, stochastic probabilities/noise/scores/IDs and RNG state | [Output/sampling gate](output-sampling-validation.md): 24 head conditions / 4,654,080 logits, 48 connected sampling calls and 90 independent sampler conditions pass; four failure/retries, RNG snapshot, two rejected arithmetic controls and independent native replay; no preceding forty-layer execution or model-quality/generation claim |
| Workloads | Reproducible reasoning, coding and tool-use evaluations | Not run |
| Resource policy | All runtime-owned admissions plus pressure behavior and concurrency with other work | All six original [Issue #9 checks](memory-policy-validation.md) pass: shared 96 GiB / 32 GiB policy, touched synthetic resource traversal of all 15,360 original expert identities and full-backbone retained-state shapes, native numerical regressions, separate process/reference observations and zero-charge cleanup; live probes use 8/16 GiB caps, not a 96 GiB allocation or all-layer numerical inference; no OS quota |
| End-to-end speed | Prefill, TTFT, decode, actual storage traffic and memory on named hardware | Not run |
| Warm cache timing | Equal capacities/traces, alternating measurement order, complete audits | Six pairs: median call-time ratio 5.02; no model-token-rate claim |

## Reference

The initial source is `deepseek-ai/DeepSeek-V4.1-Flash-1017`, revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`. The official `inference/kernel.py`
quantizes BF16 activations to E4M3 with per-32 E8M0 scales, multiplies per-32
FP8/FP4 blocks, applies scales, accumulates in FP32 and writes BF16.

Vesper's scalar projection uses a deliberate, explicit FP32 accumulation order.
Its NumPy oracle computes each inner block dot in float64 before FP32 rounding.
Agreement is useful evidence, but the official Tensor Core reduction order may
differ. Do not loosen a tolerance or call a high correlation "exact" to clear a
failed gate. Record the discrepancy and its cause.

The initial scalar candidate passed its specification oracle but differed from
the official kernel. The Ada FP8 MMA candidate then matched all eight tested
inputs. The experimental Ampere BF16 MMA candidate still differed. This supports
retaining reference arithmetic as a gate; it does not establish universal Ada
parity or identify every cause of the Ampere differences. See the
[first measurements](first-probe.md) for scope, negative results and raw data.

The [complete-expert follow-up](expert-validation.md) passes the 4090 pipeline
within its stated scope. Its repeated-shape checks also exposed shared launch
handles in the optional TileLang 0.1.12 NVRTC adapter. The harness isolates those
handles while preserving the official model and CUDA kernel source/arithmetic.
The workaround is explicit in the result, not a tolerance or an oracle rewrite.

Preserving bytes is necessary but insufficient. In particular, do not replace
the original FP8 activation behavior with a supposedly more precise path and
assume model behavior is unchanged. Near-tied router scores can select different
experts after a small numerical change.

## Measurement rules

- Report the source revision, hardware, compile options and raw mismatch count.
- Compare GPU paths with the same inputs; keep NaN classification separate from finite bit identity.
- Report cold/first-content and resident/cache-reuse performance separately.
- Mark derived bandwidth ceilings as assumptions, never as measured token rates.
- Measure actual cache misses and physical I/O before building a predictor.
- Validate speculative sampling against the unchanged target. Batch shape and rollback are part of the check.
- Keep weights, local paths and identifiers out of committed results.

## Initial resource scope

Unit tests cover memory reservation refusal, checkpoint range refusal and fixture
tamper detection. These are boundary checks, not proof of the future full-engine
memory governor. All probes must retain their explicit, small allocation limits.

The [cache follow-up](cache-validation.md) adds exclusive-tier identity checks,
corrupted/failed transfer injection, cap refusal, RAM release and retry. Its
real-weight run compares all thirteen expert stages with the unchanged official
expert while switching source tiers. Available-memory pressure is injected into
the observation function; no real machine memory exhaustion is attempted. The
actual observed minimum available memory is reported separately. Buffer cleanup
counters reach zero; that is not proof that CUDA contexts or Python RSS vanish.

The [GPU-audit follow-up](cache-performance.md) measures CPU SHA-256 as the main
baseline cost and moves repeated verification to GPU admission. Its digests are
derived only after the original full-block and six tensor hashes pass. All
uploaded chunks must match before a cache slot becomes visible. Intermediate
RAM copies can be unverified until this point. Unit tests explicitly corrupt
RAM and transfers and require refusal before compute-slot publication. Device
tests compare standard SHA-256 digests and inject single-bit faults on both GPUs.
These tests are scoped engineering checks, not a cryptographic certification.

The [router probe](router-validation.md) isolates postprocessing by replacing
only the reference projection with supplied FP32 logits. It now reproduces the
pinned CUDA top-k gather order and 32-cell bitonic network, including boundary
ties. Ordered tuples and normalization match exactly, so no permutation-based
equivalence is assumed. The original partial report is retained as historical
evidence. Qualification covers one row, 384 finite FP32 logits and top-six
selection against the named PyTorch build; other shapes/builds and connected
MoE/model execution remain separate gates.

The [gate probe](gate-validation.md) now uses the original gate matrix and
unchanged `linear` plus `Gate.forward` bodies without an identity substitution.
Native cuBLAS with explicit FP32/default-math settings reproduces the observed
reference projection and routing bits. Inputs are generated boundary vectors,
not recorded full-model activations. That gate alone does not establish expert/
shared-expert accumulation, connected MoE execution, whole-model quality or speed.

The [shared/aggregation probe](moe-arithmetic-validation.md) adds exact FP8 shared-
expert arithmetic and the single-rank original expert-ID accumulation order. Its
six-expert test supplies gate outputs while executing unchanged original
`MoE.forward` and all seven original `Expert.forward` bodies. Cancellation controls
detect reordered additions, early BF16 rounding and post-projection route weights.
Zero participation and repeated participation across distinct experts are covered;
duplicate expert IDs within a token are rejected. Original gate/offload integration,
batched GEMM parity, model quality and generation speed are not established.

The [connected probe](connected-moe-validation.md) executes actual original gate
decisions through the audited byte cache and all expert/accumulation stages.
Generated inputs pass, including partial-call refusal and complete retry, with
an independent native-only recheck. At initial publication the unchanged prefix
could not supply real MoE inputs on the 4090: its 138 KiB dynamic shared-memory
request exceeds the 99 KiB opt-in limit. Issue #6 and M1 therefore stayed open. A generated-input
pass was not substituted for that missing real-activation acceptance condition.

The [Attention storage experiment](attention-storage-validation.md) changes only
Query allocation scope in the arithmetic AST. Intermediate/output comparisons
pass for two smaller original profiles; the actual 64-head, 512-dimensional
candidate executes within the 4090 limit. Repeat and QK component checks do not
replace its unavailable full-original comparison. The candidate remains explicit
and is not used by prefix capture. Head partitioning is rejected because it
changes observed FP32 reduction results. That initial experiment did not clear
the real-input gate.

The subsequent [qualified PTX memory adapter](attention-memory-validation.md)
retains literal official arithmetic/control instructions and proves the changed
bit transport before capture. Its four actual prefix vectors now pass the
[connected real-input matrix](real-moe-validation.md), including all failure and
native-only checks. This satisfies the scoped M1 MoE gate; native Attention,
complete-block operations and full-model quality remain separate M2 work.

The [normalization and RoPE follow-up](normalization-rope-validation.md) uses four
original learned norm vectors, the pinned ATen mean reduction order and the
original complex rotation rounding. Both window and compressed/YaRN frequency
profiles match at every tested position from 0 through 131,071. The connected
chain starts with generated projected KV/query inputs, so it qualifies native
normalization, rotation, window KV and selected Attention boundaries without
claiming original query/KV/output projections or compressed KV execution.
Four deliberately changed arithmetic controls are detected. Full-layer/prefix
outputs and Hyper-Connections were separate requirements, addressed by the
subsequent scoped probes below. Full-model quality remains unverified.

The [Attention projection/layer probe](attention-projection-validation.md) now
connects all original layer-0 Attention weights and normalizations through both
output projections. Every successful call compares 55 boundaries and checks
diagnostics against unchanged `Attention.forward` output/cache. This includes
four inputs captured from the unchanged original block prefix; Hyper-Connections
in that capture remain oracle operations. A BF16-to-FP32 grouped-math replacement
changes three final values in the negative control. Layer failure tests preserve
committed KV or block further calls until explicit recovery. The subsequent
complete-block evidence is below; compressed/indexed/shared KV and full-model
quality remain pending.

The [complete layer-0 block probe](layer0-block-validation.md) connects native
HC mixes, all 20 Sinkhorn iterations, pre-collapse and residual expansion to
the original-weight Attention and tiered MoE. All 84 HC cases and 26 complete
single-token calls match: 6,648 boundary comparisons / 56,874,975 storage elements,
zero differences. A separately executed unchanged `Block.forward` controls
final hidden state, next-layer pre-mix and KV state. Store-only Sinkhorn
instrumentation restores the original arithmetic AST exactly, and four altered
arithmetic controls are detected. Five whole-call faults refuse partial
publication and recover exactly; independent native-only replay passes. This
qualifies the uncompressed layer-0 single-token profile, not every block,
compressed/indexed/shared KV, Engram, full-model quality or generation speed.

The [compression probe](compression-validation.md) now qualifies original
layer-2 ratio-2 and layer-20 ratio-1 compressor/cache construction independently.
It preserves original FP32 versus BF16 projection, pair softmax/pooling, learned
normalization, group-first-token rotary positions and 16-value FP4/E4M3 rounding.
All 49 component cases and 432 state calls match, including incomplete groups,
failure/retry and snapshots. Unchanged original compressor/cache bodies control
latents and state writes, with an explicit empty-index callback. This does not
qualify index-key/query arithmetic, candidate/top-k selection, source sharing or
complete compressed Attention. Those indexer operations are a separate probe
below. The cache is a bounded host diagnostic copy;
original-weight tests use generated inputs, not captured later-layer activations.

The [indexer probe](indexer-validation.md) separately qualifies layers 2/20/24
through original key/query projections, FP4/E8M0 rounding, BF16 score stages,
candidate-block and ordered top-512 selection. Named handoffs cover indices
2-to-3 and 24-to-25, plus keys/candidates 20-to-24. Every successful frame also
matches separately executed unchanged indexer/wrapper controls. Six arithmetic
or membership changes are detected; whole-frame failures and snapshot rollback
preserve both key owners and sharing state. The 261 components and 99 frames
match every checked bit in paired and native-only processes. This is generated
input with explicit diagnostic long-history priming, at most 16,392 index
positions and the pinned single-block CUDA top-k profile. It does not yet
connect compressors through complete compressed Attention or compressed-KV
sharing; that later subset is described below. Every layer/context/distributed
profile and full-model quality/speed remain separate requirements.

The [connected compressed-Attention probe](compressed-attention-validation.md)
joins original profiles 2/20/24 through final output, including source-20 KV,
index-key and candidate handoffs to 24. Each frame also matches 28 controls from
separately executed unchanged full Attention methods and their real sharing
globals. All 103 frames and 31,841 native/oracle boundaries match, including
failure/retry, whole-frame snapshot restoration and native-only replay. The
64-token bound means larger-context window wraps and actual top-512/candidate
pruning still require connection tests. Other profiles, complete Transformer
blocks and full-model quality remain unfinished. State is bounded host
diagnostic storage, with no generation-throughput claim.

The [complete compressed-block probe](compressed-block-validation.md) connects
the original layer-2 HC, Attention, gate and offloaded MoE through the final
residual. Its 32 calls match every checked boundary and all committed state;
separate unchanged Block/Attention methods provide eight/twelve controls per
call. Six failures refuse publication and retry exactly; snapshot replay and a
process without Torch/TileLang reproduce all goldens. The gate uses four-position
generated HC streams, 25 actually selected original experts, layer-specific
weights and bounded host diagnostic state. Consecutive-block sharing is covered
by the follow-ups below; other profiles, original upstream activations, Engram, model output/quality and
generation speed remain unfinished. Issue #7 was still open at this component gate; see the later
[block-operation acceptance audit](block-ops-qualification.md). M2 remains open.

The [consecutive-block probe](block-pair-validation.md) now connects complete
blocks 2/3 through actual returned hidden states and FP32 pre coefficients,
source KV/index reuse, both original MoEs and joint state publication. All
36 pair steps match every checked intermediate/output/state boundary and
separately executed unchanged original block/Attention controls. Eight failures,
including consumer failures after the source completed, preserve both states
and retry exactly. Snapshot restoration and native-only replay pass. The initial
layer-2 HC input remains generated and context is four positions; later blocks
receive computed inputs. Other complete compressed roles, upstream/full-model
history, larger context/batching, Engram and model output/quality/speed remain
separate requirements. Issue #7 was still open at this component gate; see the later
[block-operation acceptance audit](block-ops-qualification.md). M2 remains open.

The [ratio-1 complete-pair follow-up](ratio1-block-pair-validation.md) covers
blocks 20/21 with their own original weights and actual inter-block hidden/pre
handoff. All 36 pair steps / 19,152 boundaries / 160,709,484 storage elements
match, with 648 unchanged Block controls, 684 Attention controls and 36
whole-state comparisons. The source candidate mask is controlled from the first
token. Eight failures, snapshot restore, five altered handoff/KV/index/candidate
controls and native-only replay pass. Initial input is generated and context is
four positions. The 2/3 and 20/21 pairs remain independent chains; refreshed-index
blocks with actual intervening-layer execution, larger complete contexts,
Engram and full model/output/quality/speed remain unfinished.

The [complete 20..25 follow-up](refreshed-block-chain-validation.md) executes all
intervening complete blocks, refreshes the index at 24 and reuses it at 25.
It matches all 59,600 boundaries and unchanged original methods, with atomic
six-block failure recovery. The later
[Issue #7 acceptance audit](block-ops-qualification.md) retains all five original
checks and validates all seventeen original/native evidence groups. This closes
the scoped block-operation gate; original full-model artifacts/Engram, global
memory governance, all-layer output/logit/sampling comparisons and generation
work remain under Issues #8/#9/#10 and M2/M3.
