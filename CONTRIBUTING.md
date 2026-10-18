# Contributing

Vesper is at the verification stage. Small, independently measurable changes are
most useful. Start with a reproducer and the evidence needed to evaluate it.

Run `python -m unittest discover -s tests -v` before proposing changes. CUDA
changes also need relevant real-GPU reports with hardware, compile options and
exact mismatch counts. GPU probes are manual; ordinary CI runs CPU checks only.

Do not trade model fidelity for a higher speed number. Preserve original
weights, quantization, routing and sampling. Explain any change to arithmetic or
accumulation order and keep failed reference gates visible.

Performance reports should include prefill/decode distinction, cache state,
actual I/O, memory and workload contention. Keep assumptions separate from
measurements. Do not commit checkpoints, user prompts, credentials, machine
identifiers or absolute local paths. Use original source licenses when adapting
third-party code.
