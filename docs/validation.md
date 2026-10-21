# PC validation workflow

Choose evidence for the claim being made. The current [PC policy](quality-speed.md)
allows measured quality variation for useful speed/memory gains. Full bitwise
equality protects the released-precision comparator; it is not the acceptance
rule for every approximate candidate. Other platforms keep their own gates.

## Checks by purpose

| Work or claim | Required evidence |
| --- | --- |
| Documentation or progress cleanup | Links, platform/issue scope, consistent acceptance, preservation of historical results; no model replay solely for prose |
| Runtime/resource change | Relevant lifecycle, corruption/refusal, pressure, rollback/retry and cleanup checks; actual memory observations when affected |
| Released-precision execution optimization | Short qualified comparator: full logits, routes/weights, feedback/RNG and final state; investigate mismatches |
| Approximate numerical candidate | Declared profile/changes, actual named task outputs/scores and failures against a valid comparator, numerical differences, speed and memory; equality is not required |
| Timing claim | Explicit startup/prefill/TTFT/decode scope, actual I/O and memory, requests/settings/cache state, resource policies and order; controlled repeated trials for general conclusions |
| Unexplained discrepancy or deliberate qualification | Detailed tracing of the affected operators/positions, expanded only when a concrete uncertainty warrants it |

Local edits use relevant CPU checks. CI retains full CPU regression. Reuse valid
qualified references; do not repeat an independent full-model reference capture
for every timing experiment. Full per-token/per-layer fingerprints are opt-in
diagnostic evidence. Model execution still needs real source/hardware context;
a format/unit test is not GPU or quality evidence.

## Quality and candidate selection

Keep the original assets immutable. A numerical profile is separate from the
workspace execution profile `native`. Record changed weight/scale precision,
activation/KV representation, expert count, routing, accumulation and sampling.
Compare actual output quality, speed and memory together. A failed exact profile
remains failed; deliberately designed approximate candidates do not inherit its
qualification or turn it into a pass.

Complete the prompt and generation budget of each named quality task. A new
short case has its own name/scope and cannot replace an unfinished longer case.
Interrupted/time-limited attempts remain incomplete and unscored. Cached
comparator outputs can be reused when their provenance, settings and actual
coverage support the comparison. State whether a new original execution ran.

The [raw-text continuation](text-generation-validation.md) and
[complete Python-expression task](trajectory-validation.md) are recorded
comparators. One expression task is not broad coding/reasoning/tool quality.
The 67-call reasoning capture is interrupted and unscored; its completed input
preparation is not an independent original/native quality pass. Reasoning/tool
evaluation remains explicit work in [#10](https://github.com/shell-term/vesper/issues/10),
without requiring exhaustive intermediate equality as the default quality method.

The [session-lifetime measurement](session-lifetime.md) records an actual
five-call timing comparison and matching short outputs. Its single trial per
variant, different admission ceilings and diagnostic overhead preclude a general
or production speed claim. It does not complete M2 quality coverage or M3.

## Reporting rules

- Identify source revision, platform/hardware/toolchain, numerical profile,
  inputs/settings and original/cached comparator provenance.
- Keep exact mismatch counts, task scores and performance as separate measures.
  Never call a tolerance/correlation result exact or infer quality from byte identity.
- State whether setup/imports, output checks, diagnostic materialization and
  cleanup are included. Separate cold process from OS/driver cache state.
- Compare controlled requests and budgets where possible. If memory policy,
  representation or sampling changes, report the difference and its tradeoff.
- Distinguish selected-payload direct reads, total process I/O and physical SSD
  traffic. Derived bandwidth ceilings are assumptions, not measured token rates.
- Measure actual process memory separately from cooperative reservations;
  preserve the 96 GiB host cap and 32 GiB available-memory floor.
- If speculation is selected, validate rollback and the configured target
  profile's distribution. A changed target is a separate approximate experiment.
- Keep weights, private paths, credentials and machine identities out of results.
  Preserve failed/incomplete records and the original execution contexts.

The [released-precision evidence index](reference-evidence.md) catalogs earlier
component and integration results. Its historical coverage notes do not define
today's queue. Use the [roadmap](roadmap.md) and guarded PC Issues for task state.
