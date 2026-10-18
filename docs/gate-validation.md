# Original gate projection validation — 2026-10-18

[Issue #4](https://github.com/shell-term/vesper/issues/4) has passing evidence for
the original layer-0 gate matrix and connected single-row routing. **307 input
vectors across both text/image masks give 614 exact-bit passes** against the
unchanged official `linear` and `Gate.forward`. This reference uses the real
projection, with no identity substitution. A second process without PyTorch
installed reproduces all 614 published projection/routing goldens.

Inputs are generated boundary vectors, not activations captured from a running
full model. Connected expert/shared-expert execution and model quality remain
untested. The native projection uses cuBLAS; this is not a new custom GEMM.

## Original artifacts and exact promotion

The pinned model is `deepseek-ai/DeepSeek-V4.1-Flash-1017`, revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`. The gate tensor is
`layers.0.ffn.gate.weight`, original dtype BF16, shape `[384, 5120]` and
**3,932,160 bytes (3.75 MiB)**. Its SHA-256 is
`f9c67489d1b40f9921b6d040ddd67858a9e68ab63c9116049b7d5c265d8f8c44`.
Only this tensor is read through a bounded HTTP range; no full shard/model is
downloaded. The existing original text/image biases add 3,072 fixture bytes.

Original BF16 storage bits are retained. The reference explicitly calls
`x.float()` and `weight.float()` before the non-quantized linear branch. Vesper
promotes the matrix to FP32 exactly; all 1,966,080 promoted values match the
oracle's bits. This promotion introduces no extra lossy quantization. For BF16
input cases, the oracle receives an actual BF16 tensor and its FP32 conversion
is compared with the decoded input supplied to the native projection.

## Native call and reference profile

The native runtime loads the installed CUDA Toolkit cuBLAS directly; it needs
NumPy and CUDA, not PyTorch. The call follows the pinned ATen FP32 path:

| Setting | Native value |
| --- | --- |
| API / data | `cublasSgemm_v2`, FP32 inputs and output |
| Matrix operation | `m=384, n=1, k=5120`, `transa=T`, `transb=N` |
| Leading dimensions | `lda=5120, ldb=5120, ldc=384` |
| Scalars | Host pointers, `alpha=1`, `beta=0` |
| Stream / math mode | Default stream, `CUBLAS_DEFAULT_MATH` (0) |
| Explicit workspace | 8,519,680 bytes, matching the pinned Ampere/Ada default |
| Native library | `cublas64_12.dll`, cuBLAS 12.4.2 (`120402`) |
| Reference library | `cublas64_13.dll`, cuBLAS 13.0.0 (`130000`) |

Both libraries' file sizes/hashes are recorded. They are provided by the local
CUDA/reference installations, not bundled with Vesper. Numerical agreement is
measured for this hardware/profile; it is not assumed across cuBLAS versions.
The native router cubin is unchanged from the previously qualified tie path.

The oracle uses RTX 4090, PyTorch `2.14.1+cu130`, revision
`5c4886908584029761b579af026dcfb627c84070`, CUDA 13.0 and NumPy 2.5.2.
Its precision is `highest`, backend is cuBLAS, tuning is off and workspace/TF32
override variables are unset. The comparison blocks other reference profiles
instead of silently certifying them. Native forced-TF32 configuration is also
refused. The three relevant ATen source files and seven top-k source/license
files are hash-checked with bounded reads.

The unchanged source methods execute with real BF16 weights and trained FP32
biases. A separate diagnostic repeats projection, score transform, gather and
normalization and checks its weights against the actual `Gate.forward` result.
No TileLang dependency or full model-module import is needed for this oracle.

## Measured results

| Check | Observed result |
| --- | --- |
| Original weight BF16-to-FP32 promotion | 0 / 1,966,080 bit differences |
| Activation promotion and linear logits | All 614 comparison conditions pass |
| FP32 logit comparisons | 0 / 235,776 bit differences |
| Unbiased and bias-corrected scores | 0 / 235,776 bit differences each |
| Ordered expert IDs | 0 / 3,684 differences |
| FP32 route weights | 0 / 3,684 bit differences |
| Normalization denominators | 0 / 614 bit differences |
| Decisions returned through the native API | All IDs/weight bits match |
| Conditions with selected routing ties | 18 / 18 pass |
| Separate native process, PyTorch absent | 614 / 614 goldens pass |

The 307 input vectors comprise 32 random seeds at four scales in both BF16 and
FP32 forms (256 vectors), 18 edge vectors, 12 one-hot columns, 12 vectors aligned
with original weight rows, eight row-difference vectors, and a final zero-input
reuse check. Edge cases include positive/negative constants, a ramp, alternating
signs, signed zero, tiny/subnormal values and mixed magnitudes. Both text and
image masks are checked for every vector: 332 BF16-input and 282 FP32-input
comparison conditions. These counts include repeated projection comparisons
for the two masks; they are not 614 independent projection inputs.

The native projection runs 307 times and its router runs 1,228 inspections,
including repeated decision calls. There are 614 original gate forwards and
1,228 reference linear calls including diagnostics. Every checked stage matches
without a tolerance; the command returns `pass` and exit code 0.

## Independent native runtime check

The separate NumPy 2.5.4 environment has no PyTorch installed or imported. It
regenerates every input, verifies its exact input hash and compares the native
logit hash, ordered IDs and weight bits with the reference records. All 614
conditions pass, with the same cuBLAS 12.4.2 library and zero owned buffers after
cleanup. The example uses no reference computation or network requests.

```sh
vesper fetch-gate-fixture --directory artifacts/gate-fixture
vesper fetch-router-fixture --directory artifacts/router-fixture
vesper compare-gate --fixture artifacts/gate-fixture/native-gate.npz --router-fixture artifacts/router-fixture/native-router.npz --device 0 --output artifacts/gate.json
```

With Vesper installed, the native-only recheck can use the published goldens:

```sh
python examples/recheck_gate.py --fixture artifacts/gate-fixture/native-gate.npz --router-fixture artifacts/router-fixture/native-router.npz --reference-report results/2026-10-18-gate/gate-projection.json --output artifacts/gate-native-only.json
```

## Resource scope and remaining gates

The projection's explicit device allocations peak at **16,406,016 bytes**,
including the promoted matrix, input/output and cuBLAS workspace. The router
adds **7,760 bytes**. Their allocation caps are 32 MiB and 1 MiB, each retaining a
1 GiB free-device reserve. The reference preflight requires 1.5 GiB free.

Native host reservations peak at 64 MiB under a 64 MiB cap, retaining the 32 GiB
system-available floor. Fixture fetching also has a 64 MiB reservation. All owned
device buffers and host reservations finish at zero. These are explicit buffer/
reservation ledgers: cuBLAS internal state, CUDA contexts/modules/compiler,
PyTorch oracle, Python/NumPy metadata and accumulated diagnostic records are
excluded. They do not establish a process/OS cap or full-engine memory policy.

CPU tests cover original-byte integrity, archive bounds/header bombs, invalid
weights/inputs, host-cap refusal, driver/workspace failure cleanup, forced TF32
and incompatible oracle profiles. The wheel includes the new modules and all
licenses, with no model arrays or NVIDIA binaries.

This prototype transfers logits through the host between projection and routing;
it is a correctness baseline. GPU-buffer integration, actual routed/shared-
expert arithmetic and accumulation are [issue #5](https://github.com/shell-term/vesper/issues/5)
and [issue #6](https://github.com/shell-term/vesper/issues/6). Whole-model quality,
attention/KV behavior, other batch shapes/backends and generation speed remain
unverified. Complete [records and provenance](../results/2026-10-18-gate/README.md)
retain every comparison, library/source profile, input/logit hashes and failures.
