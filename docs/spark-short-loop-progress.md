# Spark short-loop implementation progress

日本語案内: 少量評価の準備、思考モード・長さの明示指定、認証付き読み出しの余分なコピー削減を実装しました。以下のCPU検査・入力適合確認と、Spark実機の数値・速度・品質の合格は区別します。S2は未完了です。

## Implemented scope

- Bulk authenticated reads borrow the existing aligned scratch while verifying
  every data-block and index-page SHA, then copy verified bytes to the caller.
  The previous extra payload-sized temporary allocation/copy is removed. The
  legacy owned-bytes read API remains, and no persistent payload cache,
  changed model arithmetic or reduced authentication is introduced.
- The Spark text entry accepts chat/thinking and original reasoning effort
  1–100, separate input/output/model-call limits and a finite duration budget.
  Defaults remain chat and the historical 512-call scope. An explicit candidate
  context can reach **1,152 positions**, the current shared-frame capacity;
  it requires a duration budget and does not qualify later positions on hardware.
  Shared/PC request limits are unchanged.
- Duration, call and completion-character stops retain partial results and
  cleanup evidence. One in-flight call can overrun the cooperative duration
  budget. Original serialization/tokenization still bounds user text to 4,000
  characters, serialized input to 4,096 and completion parsing to 16,384.
- The offline [small evaluation protocol](spark-small-evaluation.md) prepares
  frozen GPQA subsets, grades final-channel answers, compares matched native
  runs and estimates full-suite duration from measured small runs. It cannot
  launch a model, external judge or all-question trial. Unfinished/unsupported
  cases are ungraded; their status and coverage remain visible.

## Bounded checks before Spark deployment

158 focused CPU tests pass across authenticated storage, Spark assets, cache,
storage, text requests/probes, shared message/generation/sequence regressions
and the new quality protocol. These are CPU/fixture checks, not GPU or practical
quality scores. An independent review found and corrected preparation/cleanup
duration overruns being reported as successful preparation.

An alternating 12-pair Windows CPU experiment used synthetic 64 MiB data and
8 MiB verified spans. Baseline/new median read-verification-copy time was
**8.286/6.479 ms**; both hashed exactly 128 data blocks plus one index page and
returned identical bytes. Python allocation peaks were 8,395,394/7,981 bytes;
the reservation-ledger peak decreased by exactly 8 MiB. This observed local
component improvement is not a Spark or model tokens/s result. Storage caches,
the host and the synthetic payload limit generalization.
The [exact CPU record](../results/dgx-spark/2026-10-21-short-loop/windows-reader-cpu.json)
and [archive manifest](../results/dgx-spark/2026-10-21-short-loop/archive-manifest.json)
retain the original source-byte hashes, order, I/O/hash counts and cleanup.

The private author GPQA Diamond source is pinned to commit
`56686c06f5e19865c153de0fdb11be3890014df7`, CSV SHA256
`41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305`.
The proposal fixes 12 development and six disjoint held-out questions without
filtering by observed correctness or runtime fit. The existing pinned Spark
tokenizer processed all 18 under thinking/effort 100, with **zero model/GPU
calls**: inputs occupy 162–475 tokens. Seventeen fit the proposed 1,152-position
context with a 768-token continuation; one needs 1,242 model calls and remains
unsupported. It is retained, not replaced or given a smaller hidden output cap.
No answers, question text or gold key are published with this report.

## Open qualification

Fresh Spark component/all-layer partial revalidation of the new runtime is
pending. The deployed preceding source was `5f35873`; its historical records
remain bound to that source. Thinking-mode ordinary EOS, extended-context
positions, sustained continuation speed, practical quality and image input are
not qualified by the CPU checks or tokenization. Current execution is text-only.

The concrete subset counts and time/token budgets are a reviewable development
proposal, not an adopted approximate-quality tolerance. Longer quality runs
remain deferred. Before a final all-question trial, present a total-duration
estimate from actual small-run measurements and obtain the user's decision.
Final findings return to improvement and revalidation. Existing originals,
services, historical results and the 32 GiB physical-available floor remain intact.
