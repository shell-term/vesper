# Spark S2 measured preparation and candidate progress

日本語案内: S2の準備改善・圧縮候補の実測です。全40層の1回計算は一致しましたが、S2完了・通常回答の完了・20 tps達成を意味しません。画像入力は必須のS3 #25として保持しています。

Status: **S2 remains open**. This report records bounded evidence, not profile adoption, practical-quality qualification or a 20 tokens/s claim.

Runtime source for the records below: `c0439df8cf6961461f052a21be2d3e715cadb16f`, clean-source digest `9d908ecfb6b256def8b49a7c504e6c889c4b0e41fe6ce3e01863dd7b61a20aa0`. Model: `deepseek-ai/DeepSeek-V4.1-Flash-1017`, original revision `1bae0e1088993b9b82de08f59de5be5a34e42170`, Spark/native. Original native precision and historical S0/S1 evidence remain unchanged.

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

The working implementation now batches artifact/module validation within one synchronous scope operation, with a fresh full binding/checkout/registry check per batch. Public source-context/result/progress checks remain unchanged. Linux physical-memory observations still reopen the kernel file every time; parsing stops after the two required fields and malformed/missing values fail closed. CPU mutation, path/module refusal, cancellation/retry and memory-floor regressions pass. These separate changes require new source-bound Spark checks before runtime qualification. They introduce no long-lived workspace authorization cache, cached physical-available value, removed source authentication or relaxed memory floor. Candidate/readiness decisions remain [#23](https://github.com/shell-term/vesper/issues/23), [#24](https://github.com/shell-term/vesper/issues/24) and [#20](https://github.com/shell-term/vesper/issues/20).
