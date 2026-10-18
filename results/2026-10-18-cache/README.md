# Cache evidence — 2026-10-18

The [measurement report](../../docs/cache-validation.md) explains the controlled
trace, exact comparison, memory scope, timing limits and reproduction commands.
These complete JSON records contain no model bytes or private paths. Earlier
[expert](../2026-10-18-expert/README.md) and [initial probes](../2026-10-18/README.md)
remain separate historical records.

| Record | Contents |
| --- | --- |
| [summary.json](summary.json) | Counts, cleanup outcomes and hashes of the complete records |
| [tiered-expert.json](tiered-expert.json) | 232 successful calls, all thirteen checks per call, per-tier sources/timings/traffic, pressure refusal and recovery, explicit allocation peaks and cleanup |
| [native-store-provenance.json](native-store-provenance.json) | Three distinct original expert identities, six tensor hashes/ranges each, aligned store offsets and aggregate hashes; no weights |
| [resident-expert-regression.json](resident-expert-regression.json) | Rerun of all 128 prior resident-expert conditions after separating arithmetic from residency; all checked bits match |

The helper GPU stores original bytes without doing model arithmetic. The request
order and memory-pressure observation are synthetic. The one intentionally
refused pressure request is counted separately from 232 successful comparisons.
No real router locality, full-layer/model fidelity or generation speed is claimed.
