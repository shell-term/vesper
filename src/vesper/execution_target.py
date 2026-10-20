"""Explicit execution profiles; a fresh Spark build is never a qualification.

The default retains each operator's published SM89 compiler/binary gate.  The
opt-in GB10 profile admits a freshly compiled candidate only after checking its
actual device, compiler, source and options.  Reference qualification is a
separate step: this module cannot manufacture or promote a qualification receipt.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
from pathlib import Path
import platform
import re


PC_TARGET = "pc-sm89"
SPARK_TARGET = "spark-sm121-candidate"
_FLAGS = ("reduction_trace", "expert", "integrity", "routing", "shared", "aggregation",
          "kv", "normalization", "projections", "hc", "compression", "indexer", "engram", "sampling")
_NORMALIZED_FLAGS = ("kv", "normalization", "projections", "hc", "compression", "indexer", "engram", "sampling")


def apply_execution_source_profile(source, architecture, *, has_normalization):
    """Match the observed SM121 original complex-multiply rounding order.

    PC source bytes stay identical. This changes only the imaginary component's
    choice of fused multiply in Spark RoPE, established against same-device
    original PyTorch; it does not introduce a lower-precision approximation.
    """
    if not isinstance(source,bytes) or type(has_normalization) is not bool:
        raise ValueError('Expected native source bytes and an explicit normalization flag')
    if architecture=='121' and has_normalization:
        original=b'float im = fmaf(real, d, __fmul_rn(imag, c));'
        replacement=b'float im = fmaf(imag, c, __fmul_rn(real, d));'
        if source.count(original)!=1:
            raise ValueError('Pinned Spark RoPE imaginary multiply boundary changed')
        return source.replace(original,replacement)
    return source


def kernel_source_sha256(*, execution_target=None, **flags):
    """Reproduce compile_probe's input identity, without compiling or loading CUDA."""
    if any(key not in _FLAGS or type(value) is not bool for key, value in flags.items()):
        raise ValueError("Expected explicit supported native-kernel flags")
    names = ["format_probe.cu"]
    for flag in ("reduction_trace", "expert", "integrity", "routing", "shared", "aggregation"):
        if flags.get(flag):
            names.append(("reduction" if flag == "reduction_trace" else flag) + "_probe.cu")
    if flags.get("kv") or flags.get("projections") or flags.get("engram"):
        names.append("kv_probe.cu")
    if flags.get("normalization") or flags.get("hc") or flags.get("engram"):
        names.append("normalization_probe.cu")
    if flags.get("projections") or flags.get("indexer") or flags.get("engram"):
        names.append("projection_probe.cu")
    for flag in ("hc", "compression", "indexer", "engram", "sampling"):
        if flags.get(flag):
            names.append(flag + "_probe.cu")
    directory = Path(__file__).parent / "kernels"
    source = b"\n".join((directory / name).read_bytes() for name in names)
    if any(flags.get(flag) for flag in _NORMALIZED_FLAGS):
        source = source.replace(b"\r\n", b"\n")
    source=apply_execution_source_profile(source,resolve_execution_target(execution_target).architecture,
        has_normalization=any(flags.get(key,False) for key in ('normalization','hc','engram')))
    return hashlib.sha256(source).hexdigest()


@dataclass(frozen=True)
class ExecutionTarget:
    name: str = PC_TARGET

    def __post_init__(self):
        if self.name not in (PC_TARGET, SPARK_TARGET):
            raise ValueError("Unknown native execution target")

    @property
    def is_candidate(self):
        return self.name == SPARK_TARGET

    @property
    def architecture(self):
        return "121" if self.is_candidate else "89"

    def describe(self):
        return {"execution_target": self.name,
                "qualification": "unqualified_candidate" if self.is_candidate else "existing_pinned_component_profile",
                "spark_reference_qualification": "not_run" if self.is_candidate else "not_applicable",
                "qualification_receipt_issued": False,
                "target_selection_changes_model_arithmetic": False}

    def validate_device(self, info, inventory=None):
        required = "12.1" if self.is_candidate else "8.9"
        if not isinstance(info, dict) or info.get("compute_capability") != required:
            raise ValueError("Native execution device differs from the selected target")
        if self.is_candidate:
            if platform.system() != "Linux" or platform.machine().lower() not in ("aarch64", "arm64"):
                raise ValueError("The Spark candidate requires Linux AArch64")
            if (not isinstance(inventory, dict) or len(inventory.get("devices", [])) != 1
                    or inventory["devices"][0] != info or "GB10" not in str(info.get("name", ""))
                    or type(inventory.get("driver_api_version")) is not int or inventory["driver_api_version"] < 13000):
                raise ValueError("The Spark candidate requires a single GB10 and a CUDA 13-capable driver")
        observed = {key: deepcopy(info[key]) for key in ("ordinal", "name", "compute_capability", "total_bytes") if key in info}
        if isinstance(inventory, dict) and "driver_api_version" in inventory:
            observed["driver_api_version"] = inventory["driver_api_version"]
        return observed

    def validate_blas(self, version, math_mode):
        if type(version) is not int or type(math_mode) is not int or math_mode != 0:
            raise ValueError("Native execution requires the original default FP32 cuBLAS math mode")
        if (not self.is_candidate and version != 120402) or (self.is_candidate and not 130000 <= version < 130100):
            raise ValueError("Native execution cuBLAS version differs from the selected target")
        return {"version": version, "math_mode": math_mode, "TF32_enabled": False,
                "qualification": "unqualified_candidate" if self.is_candidate else "existing_pinned_component_profile"}

    def validate_sampling_policy(self, multiprocessors, threads_per_sm):
        if not self.is_candidate:
            if (multiprocessors, threads_per_sm) != (128, 1536):
                raise ValueError("Unregistered original sampling execution policy")
        elif (type(multiprocessors) is not int or multiprocessors <= 0 or type(threads_per_sm) is not int
              or threads_per_sm < 256 or threads_per_sm % 32):
            raise ValueError("Invalid observed Spark sampling execution policy")
        return {"multiprocessors": multiprocessors, "threads_per_sm": threads_per_sm,
                "policy_source": "observed_cuda_device", "qualification": self.describe()["qualification"]}


def resolve_execution_target(value=None):
    if value is None:
        return ExecutionTarget()
    if isinstance(value, ExecutionTarget):
        return value
    if not isinstance(value, str):
        raise ValueError("Execution target must be an explicit registered name")
    return ExecutionTarget(value)


def compile_execution_kernel(compile_function, execution_target=None, *, compiler_info,
                             pc_cubin_sha256, **flags):
    """Compile/record a candidate, retaining the old PC hash gate unchanged."""
    target = resolve_execution_target(execution_target)
    if not re.fullmatch(r"[0-9a-f]{64}", pc_cubin_sha256 or ""):
        raise ValueError("Missing pinned PC binary identity")
    before = kernel_source_sha256(execution_target=target,**flags)
    binary = compile_function(target.architecture, compiler_info=compiler_info, **flags)
    if not isinstance(binary, bytes) or not binary or len(binary) > 16 * 1024 * 1024:
        raise ValueError("Invalid native-kernel binary")
    digest = hashlib.sha256(binary).hexdigest()
    if not target.is_candidate:
        if compiler_info.get("nvrtc_version") != [12, 4] or digest != pc_cubin_sha256:
            raise ValueError("Unregistered native compiler/binary; qualification required")
    else:
        options = ["--gpu-architecture=sm_121", "--std=c++17", "--fmad=false", "--ftz=false",
                   "--include-path=<CUDA>/include"]
        if flags.get("sampling"):
            options.append("--include-path=<CUDA>/include/cccl")
        if (compiler_info.get("nvrtc_version") != [13, 0] or compiler_info.get("architecture") != "121"
                or compiler_info.get("options") != options or compiler_info.get("include_path_redacted") is not True
                or compiler_info.get("concatenated_source_sha256") != before
                or kernel_source_sha256(execution_target=target,**flags) != before):
            raise ValueError("Spark candidate compiler, source or options differ from its explicit profile")
    compiler = {key: deepcopy(compiler_info[key]) for key in
                ("nvrtc_version", "architecture", "concatenated_source_sha256", "options", "include_path_redacted")
                if key in compiler_info}
    return binary, {**target.describe(), "compiler": compiler, "cubin_sha256": digest,
                    "source_rounding_adaptations": (["SM121 original complex multiply: imaginary-first fused product"]
                        if target.is_candidate and any(flags.get(k) for k in ('normalization','hc','engram')) else []),
                    "kernel_flags": sorted(key for key, value in flags.items() if value)}
