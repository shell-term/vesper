# Third-party notices

## DeepSeek reference formulas and model format

The per-32 activation scale calculation and operation ordering in the probes
follow the MIT-licensed DeepSeek reference implementation. The format and
arithmetic oracles are independently implemented; the CUDA scale formula is
adapted from `fast_log2_ceil` / `fast_round_scale` in:

- Model: `deepseek-ai/DeepSeek-V4.1-Flash-1017`
- Revision: `1bae0e1088993b9b82de08f59de5be5a34e42170`
- Sources: `inference/kernel.py`, `inference/model.py`
- Copyright (c) 2023 DeepSeek
- License: [third_party/DeepSeek-LICENSE](third_party/DeepSeek-LICENSE)

The optional reference check downloads the original kernel with its license into
ignored artifacts. The small fixture downloader reads original model bytes;
those bytes retain the source model's license and are not included in this repo.
The complete-expert probe follows the clamp and weighted SwiGLU order in
`model.py`. Its optional oracle extracts and executes unchanged `linear` and
`Expert.forward` function bodies from the hash-checked original source.
The routing postprocessing probe follows `Gate.forward`'s score/bias/gather and
normalization formulas. Its optional oracle executes the unchanged method body
with an explicit identity projection to isolate supplied-logit postprocessing.

## Runtime dependencies

CUDA libraries and headers are supplied by the user's CUDA installation and are
not redistributed here. NumPy, PyTorch and TileLang retain their upstream licenses.
Vesper does not contain Strata, Maya, Xolotl or llama.cpp source code. They are
research references, not dependencies or code authors of this implementation.
