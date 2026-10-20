"""Native Driver outputs against a current, separately qualified Spark oracle.

Only exact storage hashes are transported from the reference process. Native
execution imports neither Torch nor TileLang, and keeps candidate bundles
immutable and explicitly unqualified. This external receipt covers the fixed
36-case component corpus and two pressure/retry cases, not model quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import sys

import numpy as np

from . import checkpoint
from .attention_inputs import CASES, PROFILES, attention_inputs
from .attention_storage import STAGES, trace_shapes
from .attention_validation import array_digest, exercise_attention
from .execution_target import SPARK_TARGET
from .spark_attention import SparkAttentionCandidate, load_candidate
from .spark_attention_reference import validate_qualification
from .spark_probe import GiB, MiB, Unsupported, host_memory, require_capacity

SCHEMA = "vesper.spark-native-attention-validation.v1"
PROBE = "spark_native_attention_reference_parity"
BOUNDARIES = ("output_bf16", *STAGES)
RESULT_CAP = 2 * MiB
ADMISSION = 128 * MiB
PROFILE_FIELDS = ("heads", "dimension", "diagnostics", "cubin_sha256", "generated_cuda_sha256", "dynamic_shared_bytes")
SCOPE = "Fixed 36-case native sparse Attention outputs and all five FP32 boundaries against the GB10 PTX memory oracle"
EXCLUSIONS = ["untouched 64x512 original kernel resource gate", "model forward", "long-context qualification", "quality", "generation speed"]


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _guard(workspace):
    workspace.check()
    workspace.check_runtime_module(__file__, "src/vesper/spark_attention_validation.py")
    from . import workspace as workspace_module
    workspace.check_runtime_module(workspace_module.__file__, "src/vesper/workspace.py")
    context = workspace.context()
    if tuple(context[k] for k in ("platform", "profile", "model", "revision")) != (
            "dgx-spark", "native", checkpoint.MODEL, checkpoint.REVISION):
        raise ValueError("The pinned dgx-spark/native workspace is required")
    return context


def _result(workspace, relative):
    identity = workspace.check_result(relative)
    with workspace.result_path(relative).open("rb") as stream:
        raw = stream.read(RESULT_CAP + 1)
    if len(raw) > RESULT_CAP or _sha(raw) != identity["result_sha256"]:
        raise ValueError("Attention result bytes changed or exceed the report bound")
    return json.loads(raw), _sha(raw)


def _bundle(workspace, bundle):
    path = Path(bundle)
    if path.is_absolute():
        base = workspace.artifact_path("namespace-check").parent.absolute()
        relative = path.absolute().relative_to(base).as_posix()
    else:
        relative = str(bundle).replace("\\", "/")
    return workspace.artifact_path(relative), relative


def _key(row):
    key = tuple(row.get(name) for name in ("heads", "dimension", "case"))
    if (any(type(value) is not int for value in key) or key[:2] not in PROFILES
            or not 0 <= key[2] < len(CASES)):
        raise ValueError("Foreign Attention case identity")
    return key


def _layouts(heads, dimension, case):
    batch, sequence, topk = case[:3]
    shapes = trace_shapes(batch, sequence, heads, dimension, topk)
    return {"output_bf16": ((batch, sequence, heads, dimension), np.dtype(np.uint16)),
            **{name: (shape, np.dtype(np.float32)) for name, shape in zip(STAGES, shapes)}}


def _case_identity(row, key):
    case = CASES[key[2]]
    _, provenance = attention_inputs(*key[:2], case)
    expected = dict(zip(("batch", "sequence", "topk", "mode", "input_scale"), case))
    if row.get("input_sha256") != provenance or any(row.get(name) != value for name, value in expected.items()):
        raise ValueError("Attention oracle/candidate inputs differ from the fixed corpus")


def reference_goldens(report):
    """Reject a passing-looking report with foreign inputs or diagnostics."""
    validate_qualification(report)
    goldens = {}
    for row in report["cases"]:
        key = _key(row)
        _case_identity(row, key)
        checks = {check["check"]: check for check in row["checks"]}
        stages = {}
        for name, (shape, dtype) in _layouts(*key[:2], CASES[key[2]]).items():
            label = "PTX_original_arithmetic_vs_register_BF16" if name == "output_bf16" else "PTX_vs_register_" + name
            check = checks[label]
            if (check.get("dtype") != str(dtype) or check.get("elements") != int(np.prod(shape))
                    or check.get("mismatches") != 0 or check.get("status") != "pass"
                    or re.fullmatch(r"[0-9a-f]{64}", check.get("expected_sha256", "")) is None
                    or check.get("expected_sha256") != check.get("actual_sha256")):
                raise ValueError("Attention oracle diagnostic dtype, extent or hash differs")
            stages[name] = {"shape": list(shape), "dtype": str(dtype), "elements": int(np.prod(shape)),
                            "sha256": check["expected_sha256"]}
        goldens[key] = {"input_sha256": row["input_sha256"], "stages": stages}
    if len(goldens) != len(PROFILES) * len(CASES):
        raise ValueError("Attention oracle cases are missing or duplicated")
    return goldens


def _profiles(manifest):
    rows = manifest.get("profiles", [])
    result, keys = [], set()
    for row in rows:
        key = tuple(row.get(name) for name in ("heads", "dimension", "diagnostics"))
        if (type(key[0]) is not int or type(key[1]) is not int or type(key[2]) is not bool
                or key[:2] not in PROFILES or key in keys
                or any(re.fullmatch(r"[0-9a-f]{64}", row.get(name, "")) is None for name in
                       ("cubin_sha256", "generated_cuda_sha256"))
                or type(row.get("dynamic_shared_bytes")) is not int or row["dynamic_shared_bytes"] <= 0):
            raise ValueError("Attention bundle profile identity differs")
        keys.add(key)
        result.append({field: row[field] for field in PROFILE_FIELDS})
    if keys != {(h, d, trace) for h, d in PROFILES for trace in (False, True)}:
        raise ValueError("All six native Attention binaries are required")
    return sorted(result, key=lambda row: (row["heads"], row["dimension"], row["diagnostics"]))


def _checker(goldens):
    def check(heads, dimension, index, inputs, provenance, actual):
        key = heads, dimension, index
        expected = goldens[key]
        if provenance != expected["input_sha256"] or provenance != {
                name: array_digest(value) for name, value in zip(("q", "kv", "sink", "indices"), inputs)}:
            raise ValueError("Native Attention input bytes differ from the qualified oracle")
        if set(actual) != set(BOUNDARIES):
            raise ValueError("Native Attention diagnostic boundaries differ")
        checks = []
        for name in BOUNDARIES:
            value, want = actual[name], expected["stages"][name]
            valid_layout = (isinstance(value, np.ndarray) and list(value.shape) == want["shape"]
                            and str(value.dtype) == want["dtype"] and value.flags.c_contiguous)
            actual_hash = array_digest(value) if isinstance(value, np.ndarray) else None
            passed = valid_layout and actual_hash == want["sha256"]
            checks.append({"check": name, "status": "pass" if passed else "fail",
                "shape": want["shape"], "dtype": want["dtype"], "elements": want["elements"],
                "expected_sha256": want["sha256"], "actual_sha256": actual_hash,
                "comparison": "SHA256 of exact storage bits; no tolerance",
                "layout_matches": valid_layout,
                "mismatches": 0 if passed else None})
        return checks
    return check


def _successful_checks(checks, golden):
    if len(checks) != len(BOUNDARIES) or {c.get("check") for c in checks} != set(BOUNDARIES):
        raise ValueError("Native Attention receipt lacks a diagnostic boundary")
    for check in checks:
        want = golden["stages"][check["check"]]
        if (check.get("status") != "pass" or check.get("layout_matches") is not True
                or check.get("mismatches") != 0
                or any(check.get(k) != want[k] for k in ("shape", "dtype", "elements"))
                or check.get("expected_sha256") != want["sha256"]
                or check.get("actual_sha256") != want["sha256"]):
            raise ValueError("Native Attention receipt stage differs from the qualified oracle")


def validate_receipt(report, *, manifest, manifest_sha256, reference, reference_sha256):
    """Coverage/identity checks; caller must authenticate current result files."""
    goldens = reference_goldens(reference)
    required = {"schema": SCHEMA, "probe": PROBE, "status": "pass", "model": checkpoint.MODEL,
                "revision": checkpoint.REVISION, "execution_target": SPARK_TARGET,
                "original_kernel_parity_qualified": True, "full_model_quality_qualified": False,
                "general_performance_claim": False, "native_process_tensor_framework_imported": False,
                "bundle_manifest_sha256": manifest_sha256, "reference_result_sha256": reference_sha256,
                "candidate_bundle_qualification": "unqualified_candidate", "scope": SCOPE, "exclusions": EXCLUSIONS}
    if any(report.get(key) != value for key, value in required.items()):
        raise ValueError("Native Attention receipt identity or claim scope differs")
    if (manifest.get("qualification") != "unqualified_candidate" or manifest.get("original_kernel_parity_qualified") is not False
            or report.get("native_kernel_profiles") != _profiles(manifest)
            or report.get("observed_device") != reference.get("device")
            or report.get("driver_api_version") != reference.get("driver_api_version")):
        raise ValueError("Native Attention receipt targets another bundle, hardware or reference")
    rows, keys = report.get("cases", []), set()
    if len(rows) != 36:
        raise ValueError("Native Attention receipt requires all 36 cases")
    for row in rows:
        key = _key(row)
        if key in keys or row.get("status") != "pass":
            raise ValueError("Duplicated or failing native Attention case")
        keys.add(key)
        _case_identity(row, key)
        checks = row.get("checks", [])
        instrumented = [c for c in checks if c.get("check") == "native_instrumentation_BF16"]
        expected = goldens[key]["stages"]["output_bf16"]
        if (len(instrumented) != 1 or len(checks) != len(BOUNDARIES) + 1
                or instrumented[0].get("status") != "pass" or instrumented[0].get("mismatches") != 0
                or instrumented[0].get("dtype") != expected["dtype"]
                or instrumented[0].get("elements") != expected["elements"]
                or instrumented[0].get("expected_sha256") != expected["sha256"]
                or instrumented[0].get("actual_sha256") != expected["sha256"]):
            raise ValueError("Native plain/traced output does not match the original oracle")
        _successful_checks([c for c in checks if c.get("check") != "native_instrumentation_BF16"], goldens[key])
    faults = report.get("faults", [])
    if len(faults) != 2 or {r.get("condition") for r in faults} != {"host_pressure", "fourth_device_allocation"}:
        raise ValueError("Native Attention receipt lacks both pressure/retry cases")
    for fault in faults:
        attempts = 1 if fault["condition"] == "host_pressure" else 4
        if (fault.get("status") != "pass" or fault.get("caught") != "MemoryError"
                or fault.get("allocation_attempts") != attempts or fault.get("device_owned_bytes_after_failure") != 0
                or fault.get("partial_result_returned") is not False):
            raise ValueError("Native Attention pressure cleanup or retry failed")
        _successful_checks(fault.get("retry_checks", []), goldens[64, 512, 3])
    cleanup = report.get("native_cleanup", {})
    if (any(cleanup.get(k) != 0 for k in ("host_reserved_bytes", "device_owned_bytes", "owned_modules_active"))
            or cleanup.get("owned_context_active") is not False
            or type(cleanup.get("device_peak_owned_bytes")) is not int
            or not 0 < cleanup["device_peak_owned_bytes"] <= 16 * MiB):
        raise ValueError("Native Attention owned resources were not released")
    return report


def compare_native(workspace, bundle, reference_result):
    """Run Driver-only Attention against hashes from a current passing oracle."""
    initial = _guard(workspace)
    reference, reference_sha = _result(workspace, reference_result)
    goldens = reference_goldens(reference)
    bundle_path, bundle_relative = _bundle(workspace, bundle)
    manifest, binaries, manifest_sha = load_candidate(bundle_path)
    profiles = _profiles(manifest)
    del binaries
    if "torch" in sys.modules or "tilelang" in sys.modules:
        raise Unsupported("Run native Attention validation in a fresh framework-free process")
    report = {"schema": SCHEMA, "probe": PROBE, "status": "unsupported", "model": checkpoint.MODEL,
        "revision": checkpoint.REVISION, "execution_target": SPARK_TARGET,
        "scope": SCOPE, "exclusions": EXCLUSIONS,
        "original_kernel_parity_qualified": False, "full_model_quality_qualified": False,
        "general_performance_claim": False, "native_process_tensor_framework_imported": False,
        "bundle": bundle_relative, "candidate_bundle_qualification": manifest["qualification"],
        "bundle_manifest_sha256": manifest_sha, "native_kernel_profiles": profiles,
        "reference_result": reference_result, "reference_result_sha256": reference_sha,
        "cases": [], "faults": [], "memory_observations": []}
    runtime = None
    try:
        if platform.system() != "Linux" or platform.machine().lower() not in ("aarch64", "arm64"):
            raise Unsupported("Native Spark Attention validation requires Linux AArch64")
        def checkpoint_memory(pending=0):
            workspace.check()
            memory = host_memory()
            report["memory_observations"].append(memory)
            require_capacity(memory, pending)
        checkpoint_memory(ADMISSION)
        runtime = SparkAttentionCandidate(bundle_path)
        runtime.__enter__()
        report["observed_device"] = runtime.device_info
        report["driver_api_version"] = runtime.driver.inventory()["driver_api_version"]
        if (runtime.manifest_sha256 != manifest_sha or runtime.device_info != reference["device"]
                or report["driver_api_version"] != reference["driver_api_version"]):
            raise ValueError("Native Attention opened a different bundle/device/driver")
        original_checker = _checker(goldens)
        def checker(*args):
            checkpoint_memory()
            return original_checker(*args)
        rows, faults, minimum = exercise_attention(runtime, checker)
        for row in rows:
            row["input_scale"] = CASES[row["case"]][4]
        report.update(cases=rows, faults=faults, minimum_sampled_host_available_bytes=minimum)
        runtime.__exit__(None, None, None)
        report["native_cleanup"] = runtime.stats()
        runtime = None
        checkpoint_memory()
        current_manifest, current_binaries, current_sha = load_candidate(bundle_path)
        del current_manifest, current_binaries
        _, current_reference_sha = _result(workspace, reference_result)
        if current_sha != manifest_sha or current_reference_sha != reference_sha:
            raise ValueError("Attention candidate or oracle bytes changed during native execution")
        report["native_process_tensor_framework_imported"] = "torch" in sys.modules or "tilelang" in sys.modules
        report["status"] = "pass" if all(row["status"] == "pass" for row in rows + faults) else "fail"
        report["original_kernel_parity_qualified"] = report["status"] == "pass"
        if report["status"] == "pass":
            validate_receipt(report, manifest=manifest, manifest_sha256=manifest_sha,
                             reference=reference, reference_sha256=reference_sha)
    except (Unsupported, OSError, ValueError, KeyError, AttributeError, RuntimeError, MemoryError, AssertionError) as error:
        report.update(status="unsupported", original_kernel_parity_qualified=False,
            failure_type=type(error).__name__, reason="Native Attention/oracle validation could not complete")
    finally:
        if runtime is not None:
            try:
                runtime.__exit__(None, None, None)
                report["native_cleanup"] = runtime.stats()
            except Exception as error:
                report.update(status="unsupported", original_kernel_parity_qualified=False,
                              cleanup_failure_type=type(error).__name__)
        report["native_process_tensor_framework_imported"] = "torch" in sys.modules or "tilelang" in sys.modules
    workspace.stamp_result({}, initial_context=initial)
    return report


def check_receipt(workspace, bundle, reference_result, receipt_result):
    """Reauthenticate persisted evidence and its immutable bundle without CUDA."""
    _guard(workspace)
    reference, reference_sha = _result(workspace, reference_result)
    report, receipt_sha = _result(workspace, receipt_result)
    path, relative = _bundle(workspace, bundle)
    manifest, binaries, manifest_sha = load_candidate(path)
    del binaries
    if report.get("bundle") != relative or report.get("reference_result") != reference_result:
        raise ValueError("Native Attention receipt references different input artifacts")
    validate_receipt(report, manifest=manifest, manifest_sha256=manifest_sha,
                     reference=reference, reference_sha256=reference_sha)
    return {**report, "receipt_result_sha256": receipt_sha}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("native", "check"))
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--bundle", default="s1/attention-candidate")
    parser.add_argument("--reference-result", required=True)
    parser.add_argument("--receipt-result")
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    if args.mode == "native" and not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", args.run_id or ""):
        parser.error("native requires a simple lowercase run ID")
    if args.mode == "check" and not args.receipt_result:
        parser.error("check requires receipt-result")
    from .workspace import Workspace
    workspace = Workspace.open(args.workspace)
    if args.mode == "check":
        report = check_receipt(workspace, args.bundle, args.reference_result, args.receipt_result)
        print(json.dumps({"status": report["status"], "receipt": args.receipt_result}))
        return 0
    _guard(workspace)
    initial = workspace.context(args.run_id)
    relative = f"s1/attention/native-validation-{args.run_id}.json"
    if workspace.result_path(relative).exists():
        raise FileExistsError("Refusing to overwrite native Attention evidence")
    report = compare_native(workspace, args.bundle, args.reference_result)
    path = workspace.write_result(relative, report, run_id=args.run_id, initial_context=initial)
    print(json.dumps({"status": report["status"], "result": path.relative_to(workspace.root).as_posix()}))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
