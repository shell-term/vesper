"""Bounded, separately executed Spark norm/RoPE, window-KV and sampling checks.

The unchanged pinned component oracles run only in reference mode.  Their PC
integration/qualification wrappers are not used.  A pass covers the fixed
generated diagnostic corpus, never learned coefficients or a model forward.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import platform
import re
import time
import uuid
import zipfile

import numpy as np

from . import checkpoint
from .formats import bf16_bits
from .memory import process_memory
from .spark_probe import (AVAILABLE_FLOOR_BYTES, Compiler, Driver, GiB, MiB,
                          Unsupported, host_memory, require_capacity)

SCHEMA = "vesper.spark-operator-probe.v1"
OPERATORS = ("norm", "kv", "sampling")
TARGET = "spark-sm121-candidate"
NATIVE_ADMISSION, REFERENCE_ADMISSION = 256 * MiB, 3 * GiB
DEVICE_CAP, DIAGNOSTIC_CAP = 64 * MiB, 64 * MiB
TORCH_REVISION = "5c4886908584029761b579af026dcfb627c84070"
PROCESS_INSTANCE = uuid.uuid4().hex
_EXECUTION_MODE = None
SCOPES = {
    "norm": "Generated BF16 coefficients; fixed RMS reductions, both original RoPE frequency profiles and rotations",
    "kv": "Post-RoPE per-32 quant/dequant and 128-row window; fixed batch-1/2 prefill, decode and wrap transitions",
    "sampling": "Generated finite complete 129280-token logits; original greedy/stochastic sampling and Philox offsets",
}
EXCLUSIONS = ["learned coefficient provenance", "HC", "attention integration", "compressed/shared KV",
              "rollback/fault recovery", "all model layers", "model forward", "full-model quality", "generation speed"]


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _guard(workspace):
    workspace.check()
    context = workspace.stamp_result({})["workspace_context"]
    if tuple(context[k] for k in ("platform", "profile", "model", "revision")) != (
            "dgx-spark", "native", checkpoint.MODEL, checkpoint.REVISION):
        raise Unsupported("The pinned dgx-spark/native workspace is required")


def _validate(operator, run_id):
    if operator not in OPERATORS or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", run_id):
        raise ValueError("Expected a registered operator and simple lowercase run ID")


def generated_weights():
    """Controls only: these values are not checkpoint-trained coefficients."""
    result = {}
    for width in (128, 512, 1280, 5120):
        rng = np.random.default_rng(71000 + width)
        result[f"control-w{width}"] = bf16_bits(
            (np.float32(1) + rng.standard_normal(width).astype(np.float32) / np.float32(8)))
    return result


def cases(operator):
    """Yield deterministic metadata and one input at a time; no model weights."""
    if operator == "norm":
        from .normalization_validation import inputs
        weights = generated_weights()
        for width in (128, 512, 1280, 5120):
            for rows, pattern in ((1, "near_eps"), (17, "normal")):
                name = f"control-w{width}"
                x = inputs((rows, width), width + rows, pattern)
                yield {"operation": "norm", "weight": name, "pattern": pattern,
                       "weight_sha256": _sha(weights[name].tobytes())}, x
        positions = np.array([0, 1, 127, 128, 65535, 65536, 131071], np.int32)
        for profile in ("window", "compressed-yarn"):
            yield {"operation": "frequencies", "profile": profile}, positions.copy()
            for inverse in (False, True):
                x = inputs((1, 2, 16, 512), 65535 + inverse)
                yield {"operation": "rope", "profile": profile, "start": 65535,
                       "inverse": inverse}, x
    elif operator == "kv":
        from .kv_validation import kv_inputs
        for stream, (batch, length, pattern) in enumerate(((1, 127, "mixed_scales"),
                (1, 129, "random"), (1, 32, "signed_zeros"), (2, 128, "mixed_scales"))):
            position = 0
            for step, sequence in enumerate((length, 1, 1)):
                x = kv_inputs(batch, sequence, 32000 + stream * 10 + step, pattern)
                yield {"operation": "append", "stream": stream, "batch": batch,
                       "position": position, "sequence": sequence, "pattern": pattern}, x
                position += sequence
    elif operator == "sampling":
        from .sampling_reference import sampling_logits
        for seed, batch in ((0, 1), (2026, 2)):
            for pattern, temperature in (("flat", 0.), ("flat", 1.), ("banded", .7),
                                         ("banded", 1e-6), ("large", 2.)):
                meta = {"operation": "sample", "seed": seed, "batch": batch,
                        "pattern": pattern, "temperature": temperature}
                yield meta, sampling_logits(meta)
    else:
        raise ValueError("Unknown operator")


def case_identity(operator):
    return [{**meta, "input_sha256": _sha(x.tobytes()), "input_shape": list(x.shape),
             "input_dtype": x.dtype.str} for meta, x in cases(operator)]


def _layout(meta, shape):
    def item(dimensions, dtype):
        return (tuple(dimensions), np.dtype(dtype))
    operation = meta["operation"]
    if operation == "norm":
        return {name: item((*shape[:-1], 1) if name in ("mean_fp32", "mean_eps_fp32", "rstd_fp32") else shape,
                           np.uint16 if name == "output_bf16" else np.float32)
                for name in ("square_fp32", "mean_fp32", "mean_eps_fp32", "rstd_fp32", "normalized_fp32",
                             "weighted_fp32", "output_bf16")}
    if operation == "frequencies":
        return {"inverse_frequency_fp32": item((32,), np.float32),
                "angle_fp32": item((shape[0], 32), np.float32),
                "frequency_fp32": item((shape[0], 32, 2), np.float32)}
    if operation == "rope":
        return {"frequency_fp32": item((shape[1], 32, 2), np.float32),
                "rotated_fp32": item((*shape[:-1], 64), np.float32),
                "output_bf16": item(shape, np.uint16)}
    if operation == "append":
        batch, sequence, _ = shape
        window = shape if meta["position"] == 0 else (batch, 128, 512)
        indices = (batch, sequence, min(sequence, 128)) if meta["position"] == 0 else (batch, 1, 128)
        return {"rounded_bf16": item(shape, np.uint16), "codes_fp8": item(shape, np.uint8),
                "scales_e8m0": item((batch, sequence, 16), np.uint8),
                "cache_bf16": item((batch, 128, 512), np.uint16),
                "window_bf16": item(window, np.uint16), "indices_i32": item(indices, np.int32)}
    result = {"ids_i64": item((shape[0],), np.int64), "rng_offsets_u64": item((2,), np.uint64)}
    if meta["temperature"]:
        result.update((name, item(shape, np.float32)) for name in
                      ("scaled_fp32", "probabilities_fp32", "exponential_fp32", "scores_fp32"))
    return result


def _normalized(actual, meta, shape):
    actual = dict(actual)
    if meta["operation"] == "sample":
        offsets = [actual.pop(name) for name in ("offset_before", "offset_after")]
        if any(type(x) is not int or not 0 <= x < 2**64 for x in offsets):
            raise ValueError("Invalid original RNG offsets")
        actual["rng_offsets_u64"] = np.array(offsets, np.uint64)
    layout = _layout(meta, shape)
    if set(actual) != set(layout):
        raise ValueError("Incomplete component diagnostics")
    for key, (dimensions, dtype) in layout.items():
        value = actual[key]
        if not isinstance(value, np.ndarray) or value.shape != dimensions or value.dtype != dtype:
            raise ValueError("Diagnostic shape or dtype differs from the fixed corpus")
        if dtype == np.dtype(np.float32) and not np.isfinite(value).all():
            raise ValueError("Non-finite component diagnostic")
    return actual


def _expected_layout(operator):
    return {f"c{i:02d}_{name}": layout for i, (meta, x) in enumerate(cases(operator))
            for name, layout in _layout(meta, x.shape).items()}


def _save(workspace, operator, run_id, arrays):
    expected = _expected_layout(operator)
    if set(arrays) != set(expected) or sum(a.nbytes for a in arrays.values()) > DIAGNOSTIC_CAP - MiB:
        raise ValueError("Incomplete or excessive component diagnostic corpus")
    if any(not isinstance(arrays[key], np.ndarray) or arrays[key].shape != shape or arrays[key].dtype != dtype
           for key, (shape, dtype) in expected.items()):
        raise ValueError("Component archive differs from the fixed diagnostic layout")
    relative = f"s1/operators/{operator}/{run_id}/diagnostics.npz"
    path = workspace.artifact_path(relative, create_parent=True)
    with path.open("xb") as stream:
        np.savez(stream, **arrays)
    if path.stat().st_size > DIAGNOSTIC_CAP:
        raise ValueError("Component diagnostic archive exceeds its cap")
    return {"artifact": relative, "sha256": _sha(path.read_bytes()),
            "file_bytes": path.stat().st_size, "array_bytes": sum(a.nbytes for a in arrays.values())}


def _load(workspace, report):
    operator, receipt = report["operator"], report["diagnostics"]
    path = workspace.artifact_path(receipt["artifact"])
    if (not 0 < path.stat().st_size <= DIAGNOSTIC_CAP or path.stat().st_size != receipt["file_bytes"]
            or _sha(path.read_bytes()) != receipt["sha256"]):
        raise ValueError("Component diagnostic receipt differs")
    expected = _expected_layout(operator)
    with zipfile.ZipFile(path) as archive:
        if sorted(archive.namelist()) != sorted(f"{key}.npy" for key in expected):
            raise ValueError("Missing, duplicate or additional component diagnostic members")
        for key, (shape, dtype) in expected.items():
            entry = archive.getinfo(f"{key}.npy")
            size = int(np.prod(shape)) * dtype.itemsize
            if entry.file_size > size + 1024:
                raise ValueError("Diagnostic member exceeds the fixed shape")
            with archive.open(entry) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    header = np.lib.format.read_array_header_1_0(stream, max_header_size=1024)
                elif version == (2, 0):
                    header = np.lib.format.read_array_header_2_0(stream, max_header_size=1024)
                else:
                    raise ValueError("Unsupported component diagnostic header")
                if header != (shape, False, dtype) or entry.file_size != stream.tell() + size:
                    raise ValueError("Diagnostic header or payload differs from the fixed layout")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in expected}


class NativeRunner:
    def __init__(self, operator):
        self.operator, self.current, self.stream = operator, None, None
        self.phases, self.weights = [], generated_weights() if operator == "norm" else {}

    def __enter__(self):
        if self.operator == "norm":
            from .normalization import NativeNormRoPE
            self.current = NativeNormRoPE(execution_target=TARGET)
        elif self.operator == "sampling":
            from .sampling import NativeSampler
            self.current = NativeSampler(execution_target=TARGET)
        if self.current is not None:
            self.current.__enter__()
        return self

    def _close_current(self):
        if self.current is not None:
            current = self.current
            current.__exit__(None, None, None)
            stats = current.stats()
            self.phases.append(stats)
            if stats["device_owned_bytes"] or stats["host_reserved_bytes"]:
                raise RuntimeError("Native operator retained owned resources")
            if stats["device_peak_owned_bytes"] > DEVICE_CAP:
                raise RuntimeError("Native operator exceeded its device cap")
            self.current = None

    def run(self, meta, x):
        operation = meta["operation"]
        if operation == "norm":
            return self.current.norm(x, self.weights[meta["weight"]])
        if operation == "frequencies":
            return self.current.frequencies(x, meta["profile"])
        if operation == "rope":
            return self.current.rope(x, meta["start"], meta["profile"], meta["inverse"])
        if operation == "append":
            if self.stream != meta["stream"]:
                self._close_current()
                from .kv import NativeWindowKV
                self.current = NativeWindowKV(meta["batch"], execution_target=TARGET)
                self.current.__enter__()
                self.stream = meta["stream"]
            return self.current.append(x, start_pos=meta["position"])
        stream = (meta["seed"], meta["batch"])
        if self.stream != stream:
            self.current.manual_seed(meta["seed"])
            self.stream = stream
        return self.current.sample(x, meta["temperature"])

    def close(self):
        self._close_current()


def _torch_profile(torch, hardware):
    if (torch.__version__ != "2.14.1+cu130" or torch.version.git_version != TORCH_REVISION
            or torch.version.cuda != "13.0" or not torch.cuda.is_available()):
        raise Unsupported("This probe requires the pinned CUDA-13.0 PyTorch reference build")
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    if (torch.cuda.device_count() != 1 or "GB10" not in properties.name
            or (properties.major, properties.minor) != (12, 1)
            or properties.name != hardware["name"]):
        raise Unsupported("The reference selected a different device profile")
    torch.cuda.reset_peak_memory_stats(0)
    return {"torch_version": torch.__version__, "torch_revision": torch.version.git_version,
            "torch_cuda_version": torch.version.cuda, "ordinal": 0,
            "device_name": properties.name, "compute_capability": [properties.major, properties.minor],
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "BLAS": "not used by this corpus; no BLAS version qualification",
            "framework_and_runtime_memory_not_hard_capped": True}


class ReferenceRunner:
    def __init__(self, operator, directory, report, check):
        import torch
        self.torch, self.operator, self.directory = torch, operator, directory
        self.current, self.stream = None, None
        self.stack = ExitStack()
        try:
            self._initialize(report, check)
        except BaseException:
            self.close()
            raise

    def _initialize(self, report, check):
        torch, operator, directory = self.torch, self.operator, self.directory
        report["reference"] = _torch_profile(torch, report["observed_hardware"])
        self.record = report["reference"]
        self.stack.enter_context(torch.inference_mode())
        self.stack.enter_context(torch.cuda.device(0))
        self.stack.enter_context(torch.device("cuda:0"))
        check(REFERENCE_ADMISSION)
        if operator == "norm":
            from .normalization_reference import ReferenceMath, fetch_pytorch_sources
            self.current = ReferenceMath(directory, generated_weights(), 0)
            self.record.update(source=self.current.source, methods_ast_sha256=self.current.method_hashes,
                               pytorch_sources=fetch_pytorch_sources(directory / "pytorch-normalization"),
                               diagnostic_expressions_checked_against_original_method=True)
        elif operator == "kv":
            import tilelang
            if tilelang.__version__ != "0.1.12":
                raise Unsupported("This window oracle requires TileLang 0.1.12")
            self.record["tilelang_version"] = tilelang.__version__
            self.record["boundary"] = "projection, RMS and RoPE identity stand-ins; input is already post-RoPE"
        else:
            from .sampling_reference import reference_sources
            self.record["source"] = checkpoint.fetch_reference(directory, include_model=True)
            self.record["pytorch_sources"] = reference_sources(directory)
            tree = ast.parse((directory / "model.py").read_bytes())
            node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "sample")
            scope = {"torch": torch}
            exec(compile(ast.Module(body=[node], type_ignores=[]), "<pinned-original-sample>", "exec"), scope)
            self.current = scope["sample"]
            self.record["methods_ast_sha256"] = {"sample": _sha(ast.dump(node).encode())}
            self.record["RNG"] = "unmodified original sample; default CUDA generator seed, offset and state advancement"

    def run(self, meta, x):
        operation = meta["operation"]
        if operation == "norm":
            return self.current.norm(x, meta["weight"])
        if operation == "frequencies":
            return self.current.frequencies(x, meta["profile"])
        if operation == "rope":
            return self.current.rope(x, meta["start"], meta["profile"], meta["inverse"])
        if operation == "append":
            if self.stream != meta["stream"]:
                from .kv_reference import WindowOracle
                self.current = None
                self.current = WindowOracle(self.directory, meta["batch"], 0)
                self.record.update(source=self.current.source, methods_ast_sha256=self.current.method_hashes,
                                   runtime_workarounds=self.current.workarounds)
                self.stream = meta["stream"]
            return self.current.step(meta, x)
        from .sampling_reference import original_sample
        stream = (meta["seed"], meta["batch"])
        if self.stream != stream:
            self.torch.cuda.manual_seed(meta["seed"])
            self.stream = stream
        output, before, after = original_sample(self.torch, self.current, x, meta["temperature"])
        return {**output, "offset_before": before, "offset_after": after}

    def close(self):
        self.current = None
        self.stack.close()


def _claim_execution(mode):
    global _EXECUTION_MODE
    if _EXECUTION_MODE is not None:
        raise Unsupported("Run each native/reference operator execution in a fresh process")
    _EXECUTION_MODE = mode


def collect(mode, workspace, *, operator, run_id):
    _validate(operator, run_id)
    if mode not in ("native", "reference"):
        raise ValueError("Unknown operator execution mode")
    _guard(workspace)
    diagnostics = workspace.artifact_path(f"s1/operators/{operator}/{run_id}/diagnostics.npz")
    if diagnostics.parent.exists():
        raise FileExistsError("Refusing to replace operator evidence")
    report = {"schema": SCHEMA, "mode": mode, "operator": operator, "status": "unsupported",
              "model": checkpoint.MODEL, "revision": checkpoint.REVISION, "scope": SCOPES[operator],
              "exclusions": EXCLUSIONS, "original_kernel_parity_qualified": False,
              "full_model_quality_qualified": False, "general_performance_claim": False,
              "execution_process_instance": PROCESS_INSTANCE, "separate_process_mode": True,
              "case_identity": case_identity(operator), "cases": [],
              "limits": {"physical_available_floor_bytes": AVAILABLE_FLOOR_BYTES,
                         "native_phase_admission_bytes": NATIVE_ADMISSION,
                         "reference_phase_admission_bytes": REFERENCE_ADMISSION,
                         "native_explicit_device_cap_bytes": DEVICE_CAP,
                         "diagnostic_cap_bytes": DIAGNOSTIC_CAP,
                         "floor_enforcement": "stage observation, not an OS reservation",
                         "framework_and_context_memory_not_hard_capped": True},
              "math": {"native_fmad": False, "native_ftz": False,
                       "BLAS": "not used by this corpus", "comparison": "exact diagnostic storage/FP32 bits"},
              "memory_observations": [], "process_memory_observations": []}
    runner = None
    try:
        if platform.system() != "Linux" or platform.machine().lower() not in ("aarch64", "arm64"):
            raise Unsupported("Operator execution requires Linux AArch64")
        def check(pending=0):
            workspace.check()
            memory = host_memory()
            report["memory_observations"].append(memory)
            require_capacity(memory, pending)
            report["process_memory_observations"].append(process_memory())
        admission = NATIVE_ADMISSION if mode == "native" else REFERENCE_ADMISSION
        check(admission)
        report["observed_hardware"] = Driver().inventory()
        report["preflight_compiler"] = Compiler().info
        if report["preflight_compiler"]["nvrtc_version"] != [13, 0]:
            raise Unsupported("This qualification targets NVRTC 13.0 exactly")
        _claim_execution(mode)
        diagnostics.parent.mkdir(parents=True, exist_ok=False)
        start = time.perf_counter()
        if mode == "native":
            runner = NativeRunner(operator)
            runner.__enter__()
        else:
            directory = workspace.artifact_path(f"s1/operators/{operator}/{run_id}/reference/source.py", True).parent
            runner = ReferenceRunner(operator, directory, report, check)
        report["initialization_wall_seconds"] = time.perf_counter() - start
        outputs = {}
        for i, (meta, x) in enumerate(cases(operator)):
            check(admission)
            start = time.perf_counter()
            actual = _normalized(runner.run(meta, x), meta, x.shape)
            report["cases"].append({"id": i, "wall_seconds_including_diagnostics": time.perf_counter() - start})
            outputs.update((f"c{i:02d}_{name}", value) for name, value in actual.items())
            check()
        if mode == "reference":
            torch = runner.torch
            torch.cuda.synchronize(0)
            report["reference_framework_memory"] = {
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
                "not_added_to_physical_capacity": True, "process_exit_releases_reference_runtime": True}
        runner.close()
        if mode == "native":
            report["native_phases_after_close"] = runner.phases
            report["owned_native_resources_closed"] = True
        runner = None
        check()
        report["diagnostics"] = _save(workspace, operator, run_id, outputs)
        report["status"] = f"{mode}_execution_pass"
    except (Unsupported, OSError, ValueError, KeyError, AttributeError, RuntimeError, ImportError,
            MemoryError, AssertionError, StopIteration) as error:
        report["failure_type"] = type(error).__name__
        report["reason"] = "Operator qualification could not complete; no parity or model-quality claim"
    finally:
        if runner is not None:
            try:
                runner.close()
                if mode == "native":
                    report["native_phases_after_close"] = runner.phases
                    report["owned_native_resources_closed"] = True
            except Exception as error:
                report.update(status="unsupported", cleanup_failure_type=type(error).__name__,
                              owned_native_resources_closed=False)
    return report


def _read_json(path):
    with path.open("rb") as stream:
        raw = stream.read(2 * MiB + 1)
    if len(raw) > 2 * MiB:
        raise ValueError("Excessive operator report")
    return json.loads(raw)


def compare_runs(workspace, native_result, reference_result, *, operator):
    if operator not in OPERATORS:
        raise ValueError("Unknown operator")
    _guard(workspace)
    reports = []
    for relative, mode in ((native_result, "native"), (reference_result, "reference")):
        workspace.check_result(relative)
        report = _read_json(workspace.result_path(relative))
        if (report.get("schema"), report.get("operator"), report.get("mode"), report.get("status")) != (
                SCHEMA, operator, mode, f"{mode}_execution_pass"):
            raise ValueError("Both independently successful operator executions are required")
        if report.get("case_identity") != case_identity(operator) or len(report.get("cases", [])) != len(report["case_identity"]):
            raise ValueError("Execution corpus differs from the fixed operator inputs")
        reports.append(report)
    native, reference = reports
    if (not all(isinstance(r.get("execution_process_instance"), str) and r["execution_process_instance"] for r in reports)
            or native["execution_process_instance"] == reference["execution_process_instance"]):
        raise ValueError("Native and reference must execute in separate fresh processes")
    for key in ("model", "revision", "scope", "exclusions", "case_identity", "observed_hardware", "preflight_compiler"):
        if native.get(key) != reference.get(key):
            raise ValueError("Operator source, corpus or observed execution identity differs")
    if (native["model"], native["revision"], native["scope"]) != (checkpoint.MODEL, checkpoint.REVISION, SCOPES[operator]):
        raise ValueError("Operator checkpoint or scope differs")
    actual, expected = _load(workspace, native), _load(workspace, reference)
    from .conformance import compare
    checks = [compare(expected[key], actual[key], key) for key in sorted(expected)]
    passed = all(c["status"] == "pass" for c in checks)
    return {"schema": SCHEMA, "mode": "compare", "operator": operator, "status": "pass" if passed else "fail",
            "model": checkpoint.MODEL, "revision": checkpoint.REVISION,
            "scope": SCOPES[operator], "exclusions": EXCLUSIONS, "case_count": len(native["cases"]),
            "checks": checks, "original_kernel_parity_qualified": passed,
            "qualification_scope": "only the fixed generated component diagnostic corpus on observed GB10",
            "full_model_quality_qualified": False, "general_performance_claim": False,
            "observed_hardware": native["observed_hardware"],
            "same_device_evidence": "same bound workspace, ordinal and observed GB10/driver profile; no GPU individual-identity attestation",
            "native_phases_after_close": native.get("native_phases_after_close"),
            "reference": reference.get("reference"),
            "execution_result_sha256": {mode: _sha(workspace.result_path(relative).read_bytes())
                for mode, relative in (("native", native_result), ("reference", reference_result))}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("native", "reference", "compare"))
    parser.add_argument("--operator", choices=OPERATORS, required=True)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--native-result")
    parser.add_argument("--reference-result")
    args = parser.parse_args(argv)
    _validate(args.operator, args.run_id)
    if args.mode == "compare" and not (args.native_result and args.reference_result):
        parser.error("compare requires native-result and reference-result")
    from . import workspace as workspace_module
    workspace = workspace_module.Workspace.open(args.workspace)
    workspace.check_runtime_module(__file__, "src/vesper/spark_operator_probe.py")
    workspace.check_runtime_module(workspace_module.__file__, "src/vesper/workspace.py")
    relative = f"s1/operators/{args.operator}/{args.run_id}.json"
    if workspace.result_path(relative).exists():
        raise FileExistsError("Refusing to overwrite operator evidence")
    initial = workspace.context(args.run_id)
    report = (compare_runs(workspace, args.native_result, args.reference_result, operator=args.operator)
              if args.mode == "compare" else collect(args.mode, workspace, operator=args.operator, run_id=args.run_id))
    path = workspace.write_result(relative, report, run_id=args.run_id, initial_context=initial)
    print(json.dumps({"mode": args.mode, "operator": args.operator, "status": report["status"],
                      "result": path.relative_to(workspace.root).as_posix()}))
    return 0 if report["status"] in ("pass", "native_execution_pass", "reference_execution_pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
