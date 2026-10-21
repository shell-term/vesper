# Spark repeated preparation and nonfinal-input output work

日本語案内: 補助演算器の再利用と、入力途中だけ不要な出力計算を省く経路を実装しました。統合CPU213件・実機CPU101件と、6回のモデル計算による数値一致・解放を確認。対象部分は短縮しましたが、全体速度は別の準備処理の変動が大きく、安定した改善は未確認です。S2・長い品質確認・画像対応は未完了です。

This iteration follows the [previous measured short loop](spark-short-loop-progress.md)
under #20/#21/#24. Accuracy is first. It does not adopt approximate weights,
pruning, routing changes or a new quality tolerance. Existing originals,
services, historical receipts and the 32 GiB physical-available floor remain
protected. Long quality checks and final all-question trials remain deferred.

## Implemented changes

### Reuse two small native auxiliaries in S2

`--reuse-profile s2` now retains at most one router and one single-token MoE
accumulator per runtime. Every layer still authenticates its fixtures and
uploads **both complete trained biases**, including the protected image-routing
bias. Existing native `begin()` zeros accumulation before contributions.
Gate/shared/expert weights and their original arithmetic remain unchanged.
The PC/default `auxiliary_pool=None` path still creates fresh owners.

The two native device layouts total **48,724 bytes**, independently charged
once. Host admissions retain one metadata MiB plus the native accumulator's
four MiB; these are not measurements of total context/module/process overhead.
Everything shares the same physical pool and existing cap. Idle entries may be
reclaimed; active borrowers cannot be evicted. Borrower cleanup and runtime-pool
retention are reported separately. Healthy cleanup drops a failed pool before
a fresh retry; failed cleanup retains its owner and refuses reuse.

Source/workspace, compiler/library/binary, device and workspace identities are
rechecked before reuse. No expert payload cache or cache-dependent routing
bias is introduced. This targets the previous roughly 3.217 s router-plus-
accumulator opening region across 40 layers; that is a hypothesis about an
opportunity, not a measured current-model speedup.

### Explicitly omit the stateless output tail on nonfinal input

`--prefill-tail skip_output_tail` omits the final collapse, norm and vocabulary
projection only at actual nonfinal input positions. It executes all 40 blocks,
both Engrams and transactional KV/index/history work, without advancing RNG.
The last input and every generated-token feedback call retain the complete
head and sampler. `full` is the default; ordinary `step(sample=False)` and PC
behavior are unchanged.

Reports mark the omitted work explicitly and reject a missing or malformed
head without that admitted marker. This is a separate diagnostic-work profile,
not a matched full-head speed comparison or a quality qualification. It does
not add batched prefill, extend the existing context limits or enable images.

## Affected CPU checks

213 integrated CPU/fixture tests pass with no failures, errors or skips across
the auxiliaries, ConnectedMoE, Spark composition/backends/prefill/probe/request,
shared generation/transaction/router/aggregation/message/sequence, memory policy
and execution-target contracts. Faults cover partial initialization, bias upload,
pressure, changed identities, late observation, cleanup failure and healthy
retry. These are not model-quality or GPU numerical results.

The private finite-worker revision also passes 13 CPU/mock checks. Its earlier
zero-model preflight and worker revision are retained. The current worker keeps
remaining child PID/ownership after a failed cleanup instead of falsely clearing
it; this is execution-control evidence, not model qualification.

## Finite hardware comparison

The fixed plan permits exactly these short text observations, each in a fresh
process, temperature 0 and seed 2026, using the unchanged 18-ID chat request:

| Source and mode | Maximum model calls | Main comparison |
| --- | ---: | --- |
| Preserved `226b6106` S2, full tail | 2 | Old full at positions 0 and 1 |
| New committed S2, full tail | 2 | Same-position, same-diagnostic preparation comparison |
| Same new S2, skipped nonfinal tail | 2 | Separate work-reduction comparison; retained backbone only |

Each source first runs fresh Attention reference/native and head native/reference/
comparison gates. Maximum text work is six model calls total, with no selected
output, EOS, quality/GPQA run, download, service change or automatic retry. Each
source job has a 900 s overall ceiling, each child at most 300 s, and one 40 s
cleanup deadline within the job deadline. The existing GPU lock is inherited
by children and source/binding/executable/worker/plan identities are checked.

Old/new full records must agree at both positions across routes/weights, all
40 block output boundaries, both Engrams, final hidden output and full logits.
New full/skip records compare the retained backbone boundaries separately;
the absent head is explicitly outside that comparison. Current ordinary EOS,
later positions, thinking quality, images and sustained continuation tps remain
unqualified. Fresh source-bound partial evidence does not restamp older passes.

One observation per configuration and fixed source/mode order leave OS cache,
storage and process/driver state uncontrolled. Report initialization, each
model-call position, total wall and cleanup separately. The same-source
baseline option is not an old-source control. Do not infer a sustained tps
score, broad quality or a reliable GPQA completion estimate from two positions.

## Hardware results

The completed comparison used old source
`226b6106ff372fbddb1699d82084b4a75776388a` and deployed candidate source
`0b4f64d8366b8c467478920db8c61b1ccadf2731`. The candidate's native Spark
environment passed **101 CPU tests**, an overlapping subset of the local 213,
not another 101 distinct contracts. Both sources passed the five fresh gates.
Exactly six text model calls consumed only the first two BOS/User input
positions; they selected no output. The full-head comparison matches at both
positions, and skipped-tail execution matches the retained boundaries exactly.
The preserved previous `226b6106` one-call arrays also match position zero.

| Observation | Total including initialization/cleanup | Call at position 0 | Call at position 1 |
| --- | ---: | ---: | ---: |
| Old S2, full tail | 92.342 s | 41.029 s | 40.858 s |
| Candidate S2, full tail | 104.667 s | 46.245 s | 47.854 s |
| Candidate S2, skipped nonfinal tail | 88.195 s | 39.810 s | 38.095 s |

The targeted router-plus-accumulator opening region decreased, over two calls,
from **6.773 s** to **2.607 s** with full tail and **2.739 s** with skipped tail.
Each candidate process opened only two auxiliary owners, recorded 158 reuse
hits/160 borrows and 80 complete trained-bias uploads. Its ledger peak increased
by one metadata MiB, not an additional whole-model weight cache. Native sum
zeroing and original routed outputs remain exact within this two-position scope.

However, **same-work whole-model improvement is not established**. Total
candidate full-tail time increased by 13.35% in this observation. Projection
opening time over two calls was **16.404 / 33.228 / 16.510 s** for old full,
candidate full and candidate skip, respectively. MoE opening decreased from
26.064 s to 21.513/21.816 s, while MoE forward/close was 29.793/29.540/29.887 s.
The full-to-skip wall reduction of 16.472 s coincides with 16.718 s less
projection-opening time; do not attribute it to head omission alone. Candidate
skip's 4.49% shorter total than old full is a fixed-order observation with
different work, not an isolated, stable or sustained speedup result.

A separate post-run 12-pair alternating CPU guard check measured median full
source-context lookup **12.292 ms** (ten Git children) versus binding/module
batch validation **5.474 ms** (five). Multiplying their median difference by
80 borrows gives only about 0.545 s; this component check does not explain the
observed whole-run variation or measure the same cost inside inference. No new
guard-policy change is adopted from that comparison. The next priority is to
split projection opening into authenticated reads, validation, conversion,
allocation/upload and synchronization, with matched order/cache conditions.

All owned workers exited and the GPU lock was released after each job. Every runtime ended
with zero owned charges/leases/reclaimers/device allocations and stopped
monitors. Minimum sampled physical availability was **41,044,447,232 bytes**,
above the unchanged 32 GiB floor. Existing services and checkpoint files were
unchanged. Deployment's strict guard initially stopped on an intended prior
archive-index expansion; the original index was retained, all 100 previously
tracked evidence files remained byte exact, and 113 published historical Git blobs
plus the seven new old-source records were verified. Legacy Windows working
tree line endings were not misrepresented as cross-platform raw byte identity.

The [exact archive manifest](../results/dgx-spark/2026-10-21-prefill-reuse/archive-manifest.json),
[cross-source audit](../results/dgx-spark/2026-10-21-prefill-reuse/cross-source-derived-audit.json)
and [compact measurements](../results/dgx-spark/2026-10-21-prefill-reuse/measurement-summary.json)
retain the original sources, all 15 model/gate records, the exact native CPU
record, frozen worker and separate derived evidence. Publication-only commits
do not update the deployed runtime or restamp historical results.

No current-source ordinary EOS, thinking/extended-input behavior, practical
quality, image input or accepted continuation tps is qualified by this loop.
The 20 tps aspiration remains unmeasured. A token-admissible request is not proof
that the proposed 600 s/question GPQA budget is viable. Further improvement and
small measured checks precede quality inference; any final all-question trial
still requires a measured total-duration estimate and the user's decision.
