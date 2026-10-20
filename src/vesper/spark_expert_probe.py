"""Bounded Spark native expert execution and separately run reference evidence.

Run native and reference in separate processes. Only compare can establish
one-expert parity; execution alone does not qualify model quality or speed.
Normal imports and native execution do not import a tensor framework.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import re
import time
import uuid
import zipfile

import numpy as np

from . import checkpoint
from .formats import bf16_bits, bf16_values
from .memory import process_memory
from .spark_probe import (AVAILABLE_FLOOR_BYTES, Compiler, Driver, GiB, MiB,
                          Unsupported, host_memory, require_capacity)

DEVICE_CAP = 64 * MiB
NATIVE_ADMISSION = 256 * MiB
REFERENCE_ADMISSION = 3 * GiB
DIAGNOSTIC_CAP = 4 * MiB
FIXTURE_BYTES = 18_800_640
SCHEMA = "vesper.spark-expert-probe.v1"
PROCESS_INSTANCE = uuid.uuid4().hex
_EXECUTION_MODE = None
LAYOUT = {"q": ((5120,), np.uint8), "s": ((160,), np.uint8),
          "w1": ((2304,), np.uint16), "w3": ((2304,), np.uint16),
          "down_input": ((2304,), np.uint16), "trace": ((2304, 5), np.float32),
          "down_q": ((2304,), np.uint8), "down_s": ((72,), np.uint8),
          "w2": ((5120,), np.uint16)}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _json(path, limit=2 * MiB):
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("Expert metadata exceeds its bounded scope")
    return json.loads(raw)


def _guard(workspace):
    workspace.check()
    context = workspace.stamp_result({})["workspace_context"]
    if (context["platform"], context["profile"], context["model"], context["revision"]) != (
            "dgx-spark", "native", checkpoint.MODEL, checkpoint.REVISION):
        raise Unsupported("The pinned dgx-spark/native workspace is required")


def cases():
    """Fixed single-token BF16 inputs, clamp stress, and routing round points."""
    for seed in (20261020, 20261021):
        base = np.random.default_rng(seed).standard_normal(5120).astype(np.float32)
        for scale in (1.0, 32.0):
            x = bf16_values(bf16_bits(base * np.float32(scale)))
            for route in (None, 0.25, 1.5, 0.0):
                yield {"seed": seed, "input_scale": scale, "routing_weight": route,
                       "input_sha256": _sha(x.tobytes())}, x


def _normalized(output, mode):
    if set(output) != set(LAYOUT):
        raise ValueError("Incomplete expert diagnostics")
    result = dict(output)
    if mode == "native":
        value = result["down_input"]
        if value.dtype != np.float32 or value.shape != (2304,) or not np.isfinite(value).all():
            raise ValueError("Invalid native down-projection input")
        result["down_input"] = bf16_bits(value)
    for name, (shape, dtype) in LAYOUT.items():
        value = result[name]
        if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != dtype:
            raise ValueError("Expert diagnostic shape or dtype differs")
        if value.dtype == np.float32 and not np.isfinite(value).all():
            raise ValueError("Non-finite expert trace")
        if value.dtype == np.uint16 and not np.isfinite(bf16_values(value)).all():
            raise ValueError("Non-finite expert BF16 output")
    return result


def _save_diagnostics(workspace, run_id, values):
    relative = f"s1/expert/{run_id}/diagnostics.npz"
    path = workspace.artifact_path(relative, create_parent=True)
    total = sum(v.nbytes for v in values.values())
    if total > DIAGNOSTIC_CAP:
        raise ValueError("Expert diagnostics exceed the bounded cap")
    with path.open("xb") as stream:
        np.savez(stream, **values)
    if path.stat().st_size > DIAGNOSTIC_CAP:
        raise ValueError("Expert diagnostic archive exceeds the bounded cap")
    return {"artifact": relative, "sha256": _sha(path.read_bytes()),
            "file_bytes": path.stat().st_size, "array_bytes": total}


def _load_diagnostics(workspace, report):
    receipt = report["diagnostics"]
    path = workspace.artifact_path(receipt["artifact"])
    if not 0 < path.stat().st_size <= DIAGNOSTIC_CAP or path.stat().st_size != receipt["file_bytes"]:
        raise ValueError("Invalid diagnostic archive size")
    if _sha(path.read_bytes()) != receipt["sha256"]:
        raise ValueError("Diagnostic bytes differ from the execution receipt")
    expected = {f"c{i:02d}_{name}": item for i in range(16) for name, item in LAYOUT.items()}
    with zipfile.ZipFile(path) as archive:
        if sorted(archive.namelist()) != sorted(f"{key}.npy" for key in expected):
            raise ValueError("Diagnostic archive has missing or additional members")
        for key, (shape, dtype) in expected.items():
            entry = archive.getinfo(f"{key}.npy")
            if entry.file_size > int(np.prod(shape)) * np.dtype(dtype).itemsize + 1024:
                raise ValueError("Diagnostic member exceeds its bounded shape")
            with archive.open(entry) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    header = np.lib.format.read_array_header_1_0(stream, max_header_size=1024)
                elif version == (2, 0):
                    header = np.lib.format.read_array_header_2_0(stream, max_header_size=1024)
                else:
                    raise ValueError("Unsupported diagnostic header")
                if header != (shape, False, np.dtype(dtype)):
                    raise ValueError("Diagnostic array header differs from its bounded layout")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in expected}


def _native(arrays, manifest, checkpoint_memory):
    from .expert import ResidentExpert
    return ResidentExpert(arrays, 0, manifest["swiglu_limit"])


def _claim_execution(mode):
    global _EXECUTION_MODE
    if _EXECUTION_MODE is not None:
        raise Unsupported("Run each expert execution in a fresh process")
    _EXECUTION_MODE = mode


def _reference(arrays, manifest, directory, report, checkpoint_memory=lambda *_: None):
    missing = [name for name in ("torch", "tilelang") if importlib.util.find_spec(name) is None]
    if missing:
        report["missing_dependencies"] = missing
        raise Unsupported("The optional official reference dependencies are unavailable")
    import torch
    import tilelang
    checkpoint_memory(REFERENCE_ADMISSION)
    if (not torch.cuda.is_available() or not all(hasattr(torch, name) for name in
            ("float8_e8m0fnu", "float4_e2m1fn_x2"))):
        raise Unsupported("CUDA PyTorch with original FP4/E8M0 dtypes is required")
    if torch.version.cuda is None or not torch.version.cuda.startswith("13."):
        raise Unsupported("The Spark reference profile requires a CUDA 13 PyTorch build")
    if tilelang.__version__ != "0.1.12":
        raise Unsupported("The original expert reference profile requires TileLang 0.1.12")
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)
    checkpoint_memory(REFERENCE_ADMISSION)
    from .expert_reference import OfficialExpert
    reference = OfficialExpert(arrays, manifest, directory, 0)
    report["reference"] = {"source": reference.manifest, "torch_version": torch.__version__,
                           "torch_cuda_version": torch.version.cuda, "tilelang_version": tilelang.__version__,
                           "runtime_workarounds": reference.runtime_workarounds,
                           "method": "unchanged pinned linear and Expert.forward AST bodies"}
    return reference


def collect(mode, workspace, *, fixture="s1/expert-fixture", run_id):
    if mode not in ("fetch", "native", "reference"):
        raise ValueError("Unknown expert execution mode")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", run_id):
        raise ValueError("Expected a simple lowercase run ID")
    _guard(workspace)
    # Resolve every caller-controlled path before network, CUDA or fixture reads.
    fixture_path = workspace.artifact_path(f"{fixture}/native-expert.npz")
    workspace.artifact_path(f"{fixture}/manifest.json")
    diagnostics = workspace.artifact_path(f"s1/expert/{run_id}/diagnostics.npz")
    if diagnostics.parent.exists() or (mode == "fetch" and fixture_path.parent.exists()):
        raise FileExistsError("Refusing to replace existing Spark expert evidence")
    report = {"schema": SCHEMA, "mode": mode, "status": "unsupported",
              "scope": "one original layer-0 expert-0; 16 fixed single-token cases",
              "model": checkpoint.MODEL, "revision": checkpoint.REVISION,
              "original_kernel_parity_qualified": False, "full_model_quality_qualified": False,
              "general_performance_claim": False, "separate_process_mode": True,
              "execution_process_instance": PROCESS_INSTANCE,
              "limits": {"native_explicit_device_cap_bytes": DEVICE_CAP,
                         "native_phase_admission_bytes": NATIVE_ADMISSION,
                         "reference_phase_admission_bytes": REFERENCE_ADMISSION,
                         "diagnostic_cap_bytes": DIAGNOSTIC_CAP,
                         "physical_available_floor_bytes": AVAILABLE_FLOOR_BYTES,
                         "framework_and_context_memory_not_hard_capped": True,
                         "floor_enforcement": "stage_observation_not_OS_reservation"},
              "memory_observations": [], "process_memory_observations": [], "cases": []}
    candidate = None
    try:
        if platform.system() != "Linux" or platform.machine().lower() not in ("aarch64", "arm64"):
            raise Unsupported("Expert execution requires Linux AArch64")
        admission = REFERENCE_ADMISSION if mode == "reference" else NATIVE_ADMISSION
        def check(pending=0):
            workspace.check()
            memory = host_memory()
            report["memory_observations"].append(memory)
            require_capacity(memory, pending)
            report["process_memory_observations"].append(process_memory())
        check(admission)
        if mode == "fetch":
            manifest = checkpoint.fetch_expert_fixture(fixture_path.parent)
            report.update(status="fixture_fetched", weights_fetched_bytes=manifest["weights_fetched_bytes"])
            check()
            return report
        report["observed_hardware"] = Driver().inventory()
        report["preflight_compiler"] = Compiler().info
        if report["preflight_compiler"]["nvrtc_version"] != [13, 0]:
            raise Unsupported("This initial qualification targets NVRTC 13.0 exactly")
        check(admission)
        manifest = _json(fixture_path.parent / "manifest.json", 64 * 1024)
        checkpoint.validate_expert_provenance(manifest)
        if (manifest["layer"], manifest["expert"], manifest["weights_fetched_bytes"]) != (0, 0, FIXTURE_BYTES):
            raise ValueError("This qualification is scoped to original layer-0 expert-0")
        if not FIXTURE_BYTES <= fixture_path.stat().st_size <= FIXTURE_BYTES + 64 * 1024:
            raise ValueError("Expert archive exceeds its bounded native fixture layout")
        arrays, loaded_manifest = checkpoint.load_expert_fixture(fixture_path)
        if loaded_manifest != manifest:
            raise ValueError("Expert provenance changed while loading")
        report["fixture"] = {"layer": 0, "expert": 0, "original_tensor_bytes": FIXTURE_BYTES,
                             "manifest_sha256": _sha((fixture_path.parent / "manifest.json").read_bytes()),
                             "archive_sha256": _sha(fixture_path.read_bytes())}
        check(admission)
        _claim_execution(mode)
        diagnostics.parent.mkdir(parents=True, exist_ok=False)
        start = time.perf_counter()
        if mode == "native":
            candidate = _native(arrays, manifest, check)
            candidate.__enter__()
            if candidate.info.get("compute_capability") != "12.1" or "GB10" not in candidate.info.get("name", ""):
                raise Unsupported("The native executor opened a different CUDA device")
            report["native_kernel"] = {"cubin_sha256": candidate.cubin_sha256,
                                        "actual_device": candidate.info,
                                        "compiler_options": ["fmad=false", "ftz=false"],
                                        "preflight_and_executor_resolvers_are_distinct": True}
        else:
            directory = workspace.artifact_path(f"s1/expert/{run_id}/reference/source.py", create_parent=True).parent
            candidate = _reference(arrays, manifest, directory, report, check)
        report["initialization_wall_seconds"] = time.perf_counter() - start
        check()
        outputs, first = {}, None
        for i, (case, x) in enumerate(cases()):
            check()
            start = time.perf_counter()
            actual = _normalized(candidate.forward(x, case["routing_weight"]), mode)
            report["cases"].append({**case, "wall_seconds_including_diagnostics": time.perf_counter() - start})
            outputs.update({f"c{i:02d}_{name}": value for name, value in actual.items()})
            if i == 0:
                first = (case, x, actual)
            check()
        case, x, expected = first
        repeated = _normalized(candidate.forward(x, case["routing_weight"]), mode)
        from .conformance import compare
        repeat_checks = [compare(expected[name], repeated[name], name) for name in LAYOUT]
        report["repeat_first_case"] = {"checks": repeat_checks,
                                        "status": "pass" if all(c["status"] == "pass" for c in repeat_checks) else "fail"}
        if mode == "native":
            report["peak_explicit_device_allocation_bytes"] = candidate.device.peak
            if candidate.device.peak > DEVICE_CAP:
                raise ValueError("Native expert exceeded its explicit device cap")
            candidate.close()
            candidate = None
            report["owned_native_resources_closed"] = True
        else:
            torch = candidate.torch
            torch.cuda.synchronize(0)
            report["reference_framework_memory"] = {
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
                "not_added_to_physical_capacity": True,
                "native_executor_created_in_this_mode": False,
                "process_exit_releases_reference_runtime": True}
        check()
        report["diagnostics"] = _save_diagnostics(workspace, run_id, outputs)
        report["status"] = f"{mode}_execution_pass" if report["repeat_first_case"]["status"] == "pass" else "fail"
    except (Unsupported, OSError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, MemoryError, AssertionError) as error:
        report["failure_type"] = type(error).__name__
        # Compiler/import failures can contain private filesystem paths.
        report["reason"] = "Expert qualification could not complete; no parity or full-model claim"
    finally:
        if mode == "native" and candidate is not None:
            try:
                candidate.close()
                report["owned_native_resources_closed"] = True
            except Exception as error:
                report.update(status="unsupported", cleanup_failure_type=type(error).__name__,
                              owned_native_resources_closed=False)
    return report


def compare_runs(workspace, native_result, reference_result):
    _guard(workspace)
    reports = []
    for relative, mode in ((native_result, "native"), (reference_result, "reference")):
        workspace.check_result(relative)
        report = _json(workspace.result_path(relative))
        if (report.get("schema"), report.get("mode"), report.get("status")) != (
                SCHEMA, mode, f"{mode}_execution_pass"):
            raise ValueError("Both independently executed successful expert reports are required")
        reports.append(report)
    native, reference = reports
    if (not isinstance(native.get("execution_process_instance"), str)
            or not isinstance(reference.get("execution_process_instance"), str)
            or native["execution_process_instance"] == reference["execution_process_instance"]):
        raise ValueError("Native and reference must come from separate execution processes")
    keys = ("model", "revision", "fixture", "scope")
    inputs = lambda report: [{k: v for k, v in case.items() if k != "wall_seconds_including_diagnostics"}
                             for case in report["cases"]]
    if any(native[k] != reference[k] for k in keys) or inputs(native) != inputs(reference) or len(native["cases"]) != 16:
        raise ValueError("Expert reference identity or case inputs differ")
    a, e = _load_diagnostics(workspace, native), _load_diagnostics(workspace, reference)
    from .conformance import compare
    checks = [compare(e[key], a[key], key) for key in sorted(e)]
    passed = all(check["status"] == "pass" for check in checks)
    return {"schema": SCHEMA, "mode": "compare", "status": "pass" if passed else "fail",
            "scope": native["scope"], "fixture": native["fixture"], "checks": checks,
            "original_kernel_parity_qualified": passed, "full_model_quality_qualified": False,
            "general_performance_claim": False,
            "execution_result_sha256": {mode: _sha(workspace.result_path(relative).read_bytes())
                for mode, relative in (("native", native_result), ("reference", reference_result))}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fetch", "native", "reference", "compare"))
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--fixture", default="s1/expert-fixture")
    parser.add_argument("--native-result")
    parser.add_argument("--reference-result")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", args.run_id):
        parser.error("Expected a simple lowercase run ID")
    if args.mode == "compare" and not (args.native_result and args.reference_result):
        parser.error("compare requires native-result and reference-result")
    from . import workspace as workspace_module
    workspace = workspace_module.Workspace.open(args.workspace)
    workspace.check_runtime_module(__file__, "src/vesper/spark_expert_probe.py")
    workspace.check_runtime_module(workspace_module.__file__, "src/vesper/workspace.py")
    relative = f"s1/expert/{args.run_id}.json"
    if workspace.result_path(relative).exists():
        raise FileExistsError("Refusing to overwrite Spark expert evidence")
    initial = workspace.context(args.run_id)
    report = (compare_runs(workspace, args.native_result, args.reference_result) if args.mode == "compare"
              else collect(args.mode, workspace, fixture=args.fixture, run_id=args.run_id))
    path = workspace.write_result(relative, report, run_id=args.run_id, initial_context=initial)
    print(json.dumps({"mode": args.mode, "status": report["status"],
                      "result": path.relative_to(workspace.root).as_posix()}))
    return 0 if report["status"] in ("pass", "fixture_fetched", "native_execution_pass", "reference_execution_pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
