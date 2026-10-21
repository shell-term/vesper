# Spark S2 measured preparation and candidate progress

日本語案内: S2の準備改善・圧縮候補の実測です。全40層の1回計算は一致しましたが、S2完了・通常回答の完了・20 tps達成を意味しません。画像入力は必須のS3 #25として保持しています。

Status: **S2 remains open**. This report records bounded evidence, not profile adoption, practical-quality qualification or a 20 tokens/s claim.

Initial inventory/component/calibration records use runtime source `c0439df8cf6961461f052a21be2d3e715cadb16f`, clean-source digest `9d908ecfb6b256def8b49a7c504e6c889c4b0e41fe6ce3e01863dd7b61a20aa0`. The final section records fresh checks of source `5f35873c8a4c3a8754eb3915e22e5382dbbc4c46` separately. Model: `deepseek-ai/DeepSeek-V4.1-Flash-1017`, original revision `1bae0e1088993b9b82de08f59de5be5a34e42170`, Spark/native. Original native precision and historical S0/S1 evidence remain unchanged.

The [archive manifest](../results/dgx-spark/2026-10-21-s2-preparation/archive-manifest.json) lists twelve exact source-bound records, including the separate derived recovery audit and external audit experiment, plus the frozen audit worker. Original raw UTF-8 bytes, hashes, contexts and originating failure remain intact; archive publication is not new execution. Earlier S0/S1 evidence and the [a84 native component report](../results/dgx-spark/2026-10-21-s2-native-components/components-r01.json) remain separate.

Required image input is tracked by [#25](https://github.com/shell-term/vesper/issues/25). Current execution remains text-only. Vision/aligner/marker/routing weights are retained, and text-only numerical evidence cannot qualify image behavior.

## Authenticated metadata inventory

The metadata-only inventory reports **96,085 tensors / 510,286,023,000 tensor bytes**. Original shard containers total **510,296,708,312 bytes**; the **10,685,312-byte** difference is header/container overhead, not an established removable saving. No GPU or tensor-payload read was requested by this census.

| Structural group | Tensors | Tensor bytes |
| --- | ---: | ---: |
| Routed experts | 92,160 | 288,777,830,400 |
| Engram tables | 4 | 202,758,032,400 |
| Other Engram tensors | 8 | 315,043,840 |
| Shared experts | 240 | 1,416,960,000 |
| Core layer tensors | 1,003 | 5,467,080,128 |
| Embedding, head and final norm | 3 | 2,647,664,640 |
| MTP | 2,401 | 7,932,874,632 |
| Vision | 259 | 823,685,120 |
| Aligner | 4 | 146,821,120 |
| Learned image markers | 3 | 30,720 |

Routed-expert storage comprises 271,790,899,200 bytes of original packed FP4 codes (catalog dtype `I8`) and 16,986,931,200 bytes of E8M0 scales. Structural classification **does not prove execution dependency, absence of use or safe removal**. Image input is required under #25: vision, aligner, learned markers and trained image-routing behavior must be preserved. MTP's runtime dependency is not established by this census either.

## Bounded lossless codec samples

The CPU probe sampled 12 original experts: layers 0/20/39, experts 0/127/255/383. Each level reconstructed all **225,607,680 original sample bytes exactly**. Independent frames are at most 64 KiB; modeled storage includes 4 KiB alignment, index records and container headers.

| zlib level | Modeled storage / original | Modeled saving | Aggregate decode time | Decoded original bytes/s |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.957516 | 4.248% | 0.739868 s | 0.305 GB/s |
| 6 | 0.953704 | 4.630% | 0.728996 s | 0.309 GB/s |

These are fixed-sample CPU results, not whole-checkpoint ratios or model speed measurements. Codec execution took 11.171 seconds in total. No compressed model was written or adopted. The modest modeled saving and measured CPU decode cost do not by themselves justify adding this codec to active decode; real storage integration, transfer overlap, locality and end-to-end cost remain unmeasured. This result does not reject all lossless formats or other candidate classes.

## Source-bound native calibration

Fresh source-bound Attention reference/native checks passed 36 cases and two native fault/healthy-retry cases. Head comparisons passed 24 exact checks across eight inputs in separate native/reference processes. Attention qualifies the original-arithmetic memory adapter; the untouched full kernel remains blocked by GB10 shared-memory capacity.

| One-call process | Total including initialization/cleanup | Model-call time | Result |
| --- | ---: | ---: | --- |
| Baseline preparation | 75.863 s | 64.683 s | Partial: one model call |
| S2 preparation reuse | 57.114 s | 45.214 s | Partial: one model call |
| S2 with cProfile | 66.139 s | 52.675 s | Partial: one model call |

The unprofiled pair differs by **18.749 seconds / 24.714%** in total wall time. OS/driver/cache order was uncontrolled; fresh processes are not proof of matched cold caches. This is an observed pair, not an isolated or general speedup claim. The profiled run includes profiler overhead.

The audit confirms exact semantic arrays across all three calls: all 40 layers, both Engrams and the full vocabulary head. **Each run stops at the one-call budget before sampling: selected IDs are empty, no generated-token feedback or EOS is exercised, and functional EOS acceptance remains false.** Full-model practical quality and sustained accepted-token throughput remain unqualified.

MoE opening was 27.012 versus 15.321 seconds in the unprofiled pair; projection opening was 15.266 versus 9.477 seconds. Those are inclusive opening observations, not catalog-only attribution. The profiled run still shows substantial hashing, file/proc reads and source-check overhead; inclusive profiler entries must not be added as independent costs.

All three calls show final owned host charge and live leases zero, with the memory monitor stopped. S2 admission peak was 8,953,973,484 bytes; minimum sampled available memory across the three calls was 41,128,902,656 bytes, above the 32 GiB floor. CPU inventory/codec also report owned cleanup. Cooperative admissions and sampled availability are not OS hard caps or continuous physical-memory guarantees.

## Worker exit recovery and remaining gates

The originating worker remains **failed**: its profiled phase expected process exit 2, while the installed cProfile CLI consumed the module's `SystemExit(2)` and returned 0. A separate exit-behavior probe reproduced that behavior. The derived audit validates the already completed raw evidence; it made **zero new model calls**, did not restamp the originals and did not alter the original worker state. Audit SHA-256: `b805dfc4458e6c8393caf01e42876a368575493be591ce3ec40bb3d25465d473`.

S2 still needs a predeclared candidate/quality decision under #23/#24, measured selection and a viable agreed evaluation budget, plus the selected bundle's complete ordinary EOS/feedback/cleanup regression under #20. A native selection or evidence-based candidate rejection is valid; this report adopts neither an approximate profile nor a numeric quality tolerance. Required image execution and text/image quality remain #25/#22 gates; text-only evidence cannot qualify altered shared experts/backbone for images. The 20 tokens/s text-continuation aspiration is neither demonstrated nor a mandatory S2 threshold.

## Actual readback versus GPU-SHA experiment

A separately frozen [external worker](../results/dgx-spark/2026-10-21-s2-preparation/audit-experiment-worker.py) ran the two actual integrity modes with identical S2 preparation, fresh bounded assets/catalog/backends and a fresh compiler cache per mode. It made no complete model calls. The [raw result](../results/dgx-spark/2026-10-21-s2-preparation/s2/audit/s2-audit-r01.json) records all 324 exact diagnostic arrays and layer-0 revisit equality, measured cache counters, source/plan identity and clean terminal resource checks. The parent enforced one 300-second child budget with one 40-second cleanup allowance; the child inherited the held GPU lock until exit. CPU cancellation/observer fault checks are distinct from this successful GPU execution.

| Actual audit mode | Full audit readback | Digest readback | Three MoE visit times, summed | Whole component mode |
| --- | ---: | ---: | ---: | ---: |
| Readback | 338,411,520 B | 0 B | 2.350 s | 6.218 s |
| GPU-SHA256 | 0 B | 2,643,840 B | 2.902 s | 5.281 s |

GPU-SHA hashed the same 338,411,520 payload bytes on device and reduced audit transfer volume, but the targeted MoE visits were slower in this pair. Overall component time is confounded by fixed readback-first order and projection/driver warmup; it does not establish a GPU-SHA speedup. Both modes retain source authentication and diagnostics. **The experiment adopts no audit-mode change.** It does not assess full-model generation speed, image behavior or practical quality. All owned resources closed; each mode maintained the 32 GiB sampled physical-available floor and existing services were unchanged.

## Next native preparation work

The working implementation now batches artifact/module validation within one synchronous scope operation, with a fresh full binding/checkout/registry check per batch. Public source-context/result/progress checks remain unchanged. Linux physical-memory observations still reopen the kernel file every time; parsing stops after the two required fields and malformed/missing values fail closed. CPU mutation, path/module refusal, cancellation/retry and memory-floor regressions pass. They introduce no long-lived workspace authorization cache, cached physical-available value, removed source authentication or relaxed memory floor. Their new-source Spark checks are recorded below. Candidate/readiness decisions remain [#23](https://github.com/shell-term/vesper/issues/23), [#24](https://github.com/shell-term/vesper/issues/24) and [#20](https://github.com/shell-term/vesper/issues/20).

## Fresh checks after validation batching

Runtime source `5f35873c8a4c3a8754eb3915e22e5382dbbc4c46` passed fresh Attention adapter reference/native checks and the full head's 24 comparisons. The finite r04 worker then ran exactly two one-call partials, S2 followed by baseline, with the original 18-token chat request, temperature 0 and seed 2026. All 40 layers, both Engrams and full-head semantic arrays matched between modes and against the earlier c043 S2 partial. See the [exact nine-record archive](../results/dgx-spark/2026-10-21-s2-validation-r04/archive-manifest.json), [finite worker summary](../results/dgx-spark/2026-10-21-s2-validation-r04/s2/calibration/s2-native-calibration-r04.json) and separate [local cross-source array audit](../results/dgx-spark/2026-10-21-s2-validation-r04/cross-source-array-audit.json). The local audit performs no new Spark execution and does not restamp old results.

| One-call preparation | Total including initialization/cleanup | Model-call time |
| --- | ---: | ---: |
| Baseline | 69.682 s | 59.295 s |
| S2 reuse | 56.356 s | 45.907 s |

This pair differs by 13.326 seconds in total. Fresh process/assets/compiler lifetimes were declared, but the S2-first order and OS/driver/storage caches were not randomized/reset. It is an observed bounded pair, not a general speedup or sustained-token-rate claim. Both partials stop before any selected output; no EOS or generated-token feedback was exercised.

Inclusive opening observations for baseline versus S2 were HC 2.548/0.976 seconds, MoE 25.396/12.806 seconds and projections 12.042/13.026 seconds. Projection opening increased in this pair, so reuse does not reduce every measured phase. MoE compute including its native diagnostics remained 14.990/14.930 seconds; the opening measurements must not be called catalog-only savings.

| Reused owner | Identity, bounded lifetime and pressure behavior | Actual r04 observations |
| --- | --- | --- |
| Authenticated model catalog | One runtime; pinned model/revision, checked workspace/source scope, registration and all 57 metadata/header/manifest content fingerprints. Admit at most 512 MiB retained metadata plus the separately bounded 512 MiB initial loader; return at most 4,096 detached tensor records per selection. Rehash content on every selection, invalidate/drop on changed/corrupt/short inputs, retain first failure and require a fresh owner after invalidation. No persistent cache or tensor payload retention. | One catalog load and registration load, 42 selections/hits; 44 fingerprint passes. Estimated retained Python metadata 59,658,836 B under its conservative admission. |
| HC/projection backends | At most two entries in one runtime; original kernel/source, compiler library/options, device, model and cuBLAS/math identity. Host allowances total 640 MiB; per-owner device caps are 64/256 MiB within the shared unified-memory policy. Retain handles/modules/workspaces; layer weights remain with the borrowing operator. Evict idle entries under pressure; active/failed/poisoned owners refuse reuse until safe cleanup. | Two opens, 80 borrows and 78 hits. The two retained cuBLAS workspaces total at most 17,039,360 B; live borrowed layer allocations are separate. |
| Compiled kernel cache | Existing source/compiler/flags/device identity and a 64 MiB runtime-owned bound; a hit conveys no numerical qualification. | 12 entries and 200 hits before close. |

The reproducible runtime switch is `--reuse-profile baseline|s2` on `vesper.spark_text_probe generate`, with the same local authenticated assets and fresh source-bound Attention receipts. Default baseline and PC paths remain unchanged. See [catalog/read ownership](../src/vesper/spark_artifacts.py), [backend ownership](../src/vesper/spark_operator_backends.py), [workspace batch checks](../src/vesper/workspace.py) and their [artifact](../tests/test_spark_artifacts.py), [backend](../tests/test_spark_operator_backends.py), [workspace](../tests/test_workspace.py) and [memory-observation](../tests/test_resources_observation.py) tests. CPU/mock cases cover changed/corrupt/short metadata and source bytes, same-size/restored-mtime changes, wrong scope/module/path, first-failure retention, fresh retry, pressure eviction/refusal, partial initialization, cancellation, failed cleanup poisoning and healthy close retry. Actual Spark evidence covers successful component/all-layer arithmetic and cleanup, plus the separately identified Attention faults; no whole-model hardware corruption injection is claimed. The assembled ordinary EOS/feedback check remains under #20.

The earlier c043 S2 total was 57.114 seconds and its model-call time was 45.214 seconds. The latest minor changes therefore **do not demonstrate a material further gain in the S2 path**: total wall fell slightly while the model-call observation rose slightly. CPU tests demonstrate fewer nested Git operations (five per fresh checked batch rather than 45 for the three-path scope), but larger original-weight reads/authentication and operator preparation remain. Faster host checks alone do not establish useful interactive generation speed or the 20 tokens/s aspiration.

The S2 partial's owned admission peak was 8,953,973,484 bytes and sampled available memory stayed at or above 41,200,312,320 bytes. Both processes report zero final owned host/pinned/device charges, leases, reclaimers and in-flight operations, healthy stopped monitors and all worker children exited. Existing services and model files were unchanged. CPU checks covered 86 workspace/artifact/tensor cases (two platform/privilege skips) and 40 observation/memory cases; independent reviews found no material blocker. The [source5f CI workflow](https://github.com/shell-term/vesper/actions/runs/39544814444) passed. Fault scopes remain distinct: those CPU/mock checks are not whole-model hardware fault injection.

The preserved r02 worker failed before spawning a child because an evidence-path guard rejected the intentional venv interpreter symlink. Its original failed summary is archived unchanged; its `running` phase row and false `owned_children_exited` field arose before the launch failure handler and do not describe a launched child. A separate read-only cleanup audit verified no phase log/child PID and a released GPU lock. The r03 identity-only preflight then identified the standard root-owned `/usr/bin/python3` alias, with no GPU/model call. Final r04 validates that bounded executable chain, protected parents, read-only binary hash/inode/version, venv config and post-hash link identity. Artifact/evidence path rules remain unchanged. Failed attempts were preserved, and no automatic retry occurred within a declared worker plan.

The user chose further short checks and acceleration and deferred long quality checks. The proposed 1,200-second ordinary EOS budget, four-case 192-call / three-hour screen and its rubrics were **not adopted**. S2 remains open: final quality/candidate selection and ordinary EOS/readiness checks are still required later. Image input remains required under #25; these text-only checks do not implement or qualify it.
