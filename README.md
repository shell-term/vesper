# Vesper

**Big models. Your hardware.**

Vesper is building a Mixture-of-Experts inference engine across GPU memory,
system RAM and SSD. The first PC target is
[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
on RTX 4090 + RTX 3080 Ti. It uses released weights and does not require
retraining to use the engine.

[日本語](README.ja.md) · [PC workspace](platforms/pc-heterogeneous/README.md) ·
[DGX Spark workspace](platforms/dgx-spark/README.md)

Choose a [platform workspace](platforms/README.md) before reading or updating
progress. The PC evidence below does not qualify Spark.

## PC direction

Preserve as much useful output quality as possible while improving generation
speed and leaving memory for other work. Quality variation is permitted when
the measured speed/memory tradeoff is useful.

- Keep a released-precision comparator and immutable original assets.
- Evaluate additional compression, changed precision or reduced computation as
  separately named candidates. Compare actual quality, speed and memory;
  bit equality is not every candidate's acceptance rule.
- Use short representative checks routinely. Detailed full tracing is diagnostic
  or deliberate qualification work, rather than a prerequisite for every change.
- Keep the provisional 96 GiB engine host admission cap and 32 GiB available-RAM
  floor. Measure actual process memory separately from reservation counters.

Original expert tensors are already packed FP4 with E8M0 scales; further
compression needs measurements. The canonical [quality/speed policy](docs/quality-speed.md)
defines candidate identity and selection. Platform-specific gates remain separate.

## Recorded PC evidence

This is a research prototype. Broad task quality, arbitrary workloads and
practical production throughput are still under development.

| Scope | Recorded result |
| --- | --- |
| Released-precision components and integration | Exact scoped expert, MoE, Attention/KV/HC, Engram and memory checks; [evidence index](docs/reference-evidence.md) |
| Actual generation and recovery | Five-call raw-text continuation and a complete 34-call Python-expression task; [generation](docs/text-generation-validation.md), [task/state evidence](docs/trajectory-validation.md) |
| First session-lifetime measurement | 0.0472 to 0.0762 decode tokens/s; output/final state match on the short input. One trial each, different admission ceilings, more resident memory; [full measurement and limitations](docs/session-lifetime.md) |

The session experiment observes 15.80 GiB peak resident RAM and 117.74 GiB
minimum available RAM on the retained path. It is not practical production speed
or broad quality qualification. The 3080 Ti currently stores/audits bytes;
its model-compute candidate remains unqualified. The interrupted reasoning
capture is unscored and does not resume automatically.

## Current work and documentation

[PC Issues](https://github.com/shell-term/vesper/issues?q=is%3Aissue+label%3Aplatform%3Apc-heterogeneous)
own task state; the [roadmap](docs/roadmap.md) explains milestone scope.

- [M2 / #11: one bounded quality/speed optimization cycle](https://github.com/shell-term/vesper/issues/11)
- [M3 / #10: remaining representative task-quality evaluation](https://github.com/shell-term/vesper/issues/10)
- [#12: optional 3080 Ti compute investigation](https://github.com/shell-term/vesper/issues/12)

Run the short optimization cycle before the remaining long reasoning/tool cases.
The [bounded plan](docs/roadmap.md#next-execution-order-and-time-limits) specifies
one candidate, repeated short comparisons and a stopping budget. Completing M2
means a measured adoption/rejection decision, not finished production throughput.

Use [validation](docs/validation.md) to choose checks and
[architecture](docs/architecture.md) for implementation hypotheses. Individual
measurement reports describe their recorded scopes; their historical next steps
are not the current development queue.

## Start

Python 3.10+ and NumPy are required for CPU checks:

```sh
python -m pip install -e .
python -m unittest discover -s tests -p test_moe_session.py -v
```

GPU probes additionally need an NVIDIA driver and a CUDA Toolkit with NVRTC and
headers. See [component commands and reference reproduction](docs/probes.md)
for explicit fixture preparation and toolchain requirements. Those commands
reproduce individual probes; they are not a turnkey full-model chat interface.
Full CPU regression runs in CI; local development uses checks relevant to the
change. Documentation-only cleanup does not require model execution.

The engine code is MIT licensed. Model assets and adapted reference code retain
their own notices in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Model
weights, downloaded reference sources, private paths and credentials are excluded
from the repository.
