# Third-party notices

## DeepSeek reference formulas and model format

The per-32 activation scale calculation and operation ordering in the probes
follow the MIT-licensed DeepSeek reference implementation. The format and
arithmetic oracles are independently implemented; the CUDA scale formula is
adapted from `fast_log2_ceil` / `fast_round_scale` in:

- Model: `deepseek-ai/DeepSeek-V4.1-Flash-1017`
- Revision: `1bae0e1088993b9b82de08f59de5be5a34e42170`
- Sources: `inference/kernel.py`, `inference/model.py`, `inference/convert.py`, `inference/generate.py`
- Copyright (c) 2023 DeepSeek
- License: [third_party/DeepSeek-LICENSE](third_party/DeepSeek-LICENSE)

The optional reference check downloads the original kernel with its license into
ignored artifacts. The small fixture downloader reads original model bytes;
those bytes retain the source model's license and are not included in this repo.
The complete-expert probe follows the clamp and weighted SwiGLU order in
`model.py`. Its optional oracle extracts and executes unchanged `linear` and
`Expert.forward` function bodies from the hash-checked original source.
The shared-expert probe follows the original FP8 block-32 scaling and reuses
those same unchanged bodies with original FP8 weights. The aggregation probe
follows `MoE.forward`'s expert-ID iteration and FP32/BF16 boundaries; its optional
oracle executes the unchanged method with supplied gate outputs.
The connected probe additionally executes the actual original gate inside the
unchanged MoE method. The experimental prefix capture extracts unchanged
Attention/Block/normalization methods and the `wo_a` conversion statements;
the original generation/configuration files anchor its runtime defaults. Its
resource preflight is not a native Attention qualification.
The optional Attention storage experiment generates a derivative of the pinned
`sparse_attn_kernel`: the Query allocation changes from shared storage to a
fragment, with an inverse-AST check; diagnostic variants also add intermediate
stores. It retains the MIT notice in generated code and does not replace the
original prefix reference or qualify full-model behavior.
The subsequent optional memory adapter pins the generated official CUDA and
changes only shared/global memory handling in its compiled PTX. Floating-point
arithmetic and control instructions are retained byte for byte. Its unsigned
matrix-load mapping is independently implemented from NVIDIA's public PTX ISA
specification; NVIDIA compiler/runtime code is not redistributed. Qualified
prefix capture explicitly records this storage-adapted reference profile.
The routing postprocessing probe follows `Gate.forward`'s score/bias/gather and
normalization formulas. Its optional oracle executes the unchanged method body
with an explicit identity projection to isolate supplied-logit postprocessing.

## PyTorch CUDA top-k ordering

The single-row top-six CPU specification and native routing kernel adapt the
finite FP32 key conversion, gather order and small bitonic sorting network in
PyTorch ATen CUDA. Source revision:
`5c4886908584029761b579af026dcfb627c84070` (the qualified reference build).
Sources are `TensorTopK.cpp`, `TensorTopK.cu`, `Sort.cu`, `SortUtils.cuh`,
`SortingCommon.cuh` and `SortingRadixSelect.cuh` under
[`aten/src/ATen/native/cuda`](https://github.com/pytorch/pytorch/tree/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/cuda).
The complete upstream copyright, redistribution conditions and disclaimer are
retained in [third_party/PyTorch-LICENSE](third_party/PyTorch-LICENSE), including
binary distributions. This scoped adaptation does not claim stable top-k ties
for other PyTorch versions, backends or shapes.

The gate projection binding follows the same revision's FP32 `cublasSgemm`
calling convention and default workspace budget in `aten/src/ATen/cuda/CUDABlas.cpp`,
`aten/src/ATen/native/cuda/Blas.cpp` and `aten/src/ATen/cuda/CublasHandlePool.cpp`.
Its oracle executes the unchanged DeepSeek `linear` and `Gate.forward` bodies
with real weights and projection, without a projection substitution.

## Runtime dependencies

The optional native Attention bundle builder compiles the explicitly adapted
DeepSeek Query-storage candidate described above through TileLang. Generated
source and binaries remain local artifacts; the exporter retains the pinned
DeepSeek license alongside them. Vesper's committed CUDA Driver executor and
bundle/input validation code do not vendor TileLang or CUDA runtime source.

CUDA libraries (including cuBLAS) and headers are supplied by the user's CUDA installation and are
not redistributed here. NumPy, PyTorch and TileLang retain their upstream licenses.
Vesper does not contain Strata, Maya, Xolotl or llama.cpp source code. They are
research references, not dependencies or code authors of this implementation.
