"""Synthetic CPU/mock evidence checks, never an executable GPU receipt."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from vesper import spark_attention_validation as bridge
from vesper import spark_attention_reference as oracle

DEVICE = {"ordinal": 0, "name": "NVIDIA GB10", "compute_capability": "12.1", "total_bytes": 128 * bridge.GiB}


def passing(check, shape=(8,), dtype=np.dtype(np.float32), digest="a" * 64):
    return {"check": check, "status": "pass", "elements": int(np.prod(shape)), "dtype": str(dtype),
            "mismatches": 0, "expected_sha256": digest, "actual_sha256": digest}


def synthetic_reference():
    """Fixed-layout zero outputs stand in for arithmetic, with no CUDA claim."""
    report = {"probe": oracle.PROBE, "status": "pass", "model": bridge.checkpoint.MODEL,
        "revision": bridge.checkpoint.REVISION, "official_arithmetic_memory_adapter_gate": "pass",
        "negative_control_detected": True, "tilelang_version": "0.1.12", "nvrtc_version": [13, 0],
        "reference": {"files": {"kernel.py": bridge.checkpoint.REFERENCE_HASHES["kernel.py"]}},
        "device": DEVICE.copy(), "driver_api_version": 13000, "execution_target": bridge.SPARK_TARGET,
        "torch_version": "2.14.1+cu130", "torch_git_version": "5c4886908584029761b579af026dcfb627c84070",
        "torch_cuda_version": "13.0", "independent_Spark_qualification": True,
        "negative_control": {"status": "fail", "mismatches": 1}, "profiles": [], "cases": []}
    for digest, (h, d, trace, shared) in oracle.CODEGEN_PROFILES.items():
        report["profiles"].append({"original_generated_cuda_sha256": digest, "heads": h, "dimension": d,
            "diagnostics": trace, "original_shared_bytes": shared, "original_binary_reproduction": True,
            "query_slab_inspection": "pass", "query_output_lifetime_alias": "pass",
            "original_cubin_sha256": oracle.CUBIN_PROFILES[h, d, trace],
            "ptx_audit": {"inverse_ptx_recovery": "pass", "floating_point_instructions_unchanged": True,
                "control_instructions_unchanged": True, "query_output_lifetime_alias": "pass", "ptx_version": "9.0",
                "ptx_target": "sm_121a", "original_ptx_sha256": oracle.PTX_PROFILES[h, d, trace],
                "output_shared_alias_sites": 2 * (h * d * 2 // (256 * 16))}})
    for key in ("load_checks", "store_checks"):
        report[key] = [{**passing(key), "pattern": i} for i in range(4)]
    for h, d in bridge.PROFILES:
        for index, case in enumerate(bridge.CASES):
            _, provenance = bridge.attention_inputs(h, d, case)
            checks = [passing("PTX_instrumentation_BF16"), passing("register_instrumentation_BF16")]
            for name, (shape, dtype) in bridge._layouts(h, d, case).items():
                label = "PTX_original_arithmetic_vs_register_BF16" if name == "output_bf16" else "PTX_vs_register_" + name
                digest = bridge.array_digest(np.zeros(shape, dtype))
                checks.append(passing(label, shape, dtype, digest))
            if (h, d) != (64, 512):
                checks += [passing(name) for name in ("unmodified_instrumentation_BF16", "unmodified_vs_PTX_adapter_BF16",
                                                     *("unmodified_vs_PTX_" + name for name in bridge.STAGES))]
            report["cases"].append({"heads": h, "dimension": d, "case": index,
                **dict(zip(("batch", "sequence", "topk", "mode", "input_scale"), case)),
                "input_sha256": provenance, "checks": checks})
    return report


class Workspace:
    def __init__(self, root):
        self.root, self.epoch = Path(root), 0
        self.platform, self.revision = "dgx-spark", bridge.checkpoint.REVISION
        self.check = Mock()
        self.check_runtime_module = Mock()
        self.check_result = Mock(side_effect=self._check_result)

    def context(self):
        return {"platform": self.platform, "profile": "native", "model": bridge.checkpoint.MODEL,
                "revision": self.revision, "source_epoch": self.epoch}

    def stamp_result(self, value, *, initial_context=None):
        if initial_context != self.context():
            raise ValueError("Source changed")
        return value

    def _path(self, relative, kind):
        if any(part in ("", ".", "..") for part in str(relative).split("/")) or ":" in str(relative):
            raise ValueError("foreign namespace")
        return self.root / kind / "dgx-spark" / relative

    def artifact_path(self, relative):
        return self._path(relative, "artifacts")

    def result_path(self, relative):
        return self._path(relative, "results")

    def save(self, relative, value):
        path = self.result_path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def _check_result(self, relative):
        return {"result_sha256": bridge._sha(self.result_path(relative).read_bytes())}


class Device:
    def __init__(self):
        self.allocations, self.peak, self.serial = {}, 0, 0

    def allocate(self, size):
        self.serial += 1
        self.allocations[self.serial] = size
        self.peak = max(self.peak, sum(self.allocations.values()))
        return self.serial


class Runtime:
    def __init__(self, manifest_sha):
        self.manifest_sha256 = manifest_sha
        self.device_info, self.device = DEVICE.copy(), Device()
        self.driver = Mock(inventory=Mock(return_value={"driver_api_version": 13000}))
        self.budget, self.closed, self.corrupt, self.fail, self.on_call = Mock(), False, None, False, None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.closed = True
        self.device.allocations.clear()

    def __call__(self, q, kv, sink, indices, *, diagnostics=False):
        self.budget.acquire(0)
        try:
            for _ in range(10 if diagnostics else 5):
                self.device.allocate(128)
            if self.on_call:
                self.on_call()
            if self.fail:
                raise RuntimeError("C:/private-person/private-host/driver failure")
            shape = q.shape
            case = (shape[0], shape[1], indices.shape[-1])
            outputs = {name: np.zeros(dimensions, dtype) for name, (dimensions, dtype) in
                       bridge._layouts(shape[2], shape[3], case).items() if diagnostics or name == "output_bf16"}
            if self.corrupt in outputs:
                outputs[self.corrupt].flat[0] = 1
            return outputs
        finally:
            self.device.allocations.clear()

    def stats(self):
        return {"host_reserved_bytes": 0 if self.closed else 32 * bridge.MiB,
                "device_owned_bytes": sum(self.device.allocations.values()), "owned_context_active": not self.closed,
                "owned_modules_active": 0 if self.closed else 6, "device_peak_owned_bytes": self.device.peak}


class NativeAttentionBridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Workspace(self.temp.name)
        self.reference = synthetic_reference()
        self.reference_path = "s1/attention/reference.json"
        self.workspace.save(self.reference_path, self.reference)
        self.manifest = {"qualification": "unqualified_candidate", "original_kernel_parity_qualified": False,
            "profiles": [{"heads": h, "dimension": d, "diagnostics": trace,
                "cubin_sha256": hashlib.sha256(f"CPU-only-{h}-{d}-{trace}".encode()).hexdigest(),
                "generated_cuda_sha256": "b" * 64, "dynamic_shared_bytes": 75776}
                for h, d in bridge.PROFILES for trace in (False, True)]}
        self.manifest_sha = bridge._sha(json.dumps(self.manifest).encode())
        self.runtime = Runtime(self.manifest_sha)
        self.patches = []
        for value in (patch.object(bridge, "load_candidate", return_value=(self.manifest, {}, self.manifest_sha)),
                      patch.object(bridge, "SparkAttentionCandidate", return_value=self.runtime),
                      patch.object(bridge.platform, "system", return_value="Linux"),
                      patch.object(bridge.platform, "machine", return_value="aarch64"),
                      patch.object(bridge, "host_memory", return_value={"physical_total_bytes": 128 * bridge.GiB,
                          "physical_available_bytes": 80 * bridge.GiB, "cgroup_headroom_bytes": None}),
                      patch("vesper.attention_validation.physical_memory", return_value={"available": 80 * bridge.GiB})):
            self.patches.append(value.start())
            self.addCleanup(value.stop)

    def collect(self):
        return bridge.compare_native(self.workspace, "s1/attention-candidate", self.reference_path)

    def validate(self, report):
        return bridge.validate_receipt(report, manifest=self.manifest, manifest_sha256=self.manifest_sha,
            reference=self.reference, reference_sha256=bridge._sha(self.workspace.result_path(self.reference_path).read_bytes()))

    def test_native_import_does_not_import_optional_tensor_frameworks(self):
        code = "import sys; import vesper.spark_attention_validation; assert 'torch' not in sys.modules; assert 'tilelang' not in sys.modules"
        result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_all36_and_fault_retries_compare_to_oracle_without_promoting_bundle(self):
        report = self.collect()
        self.assertEqual(report["status"], "pass")
        self.assertEqual(len(report["cases"]), 36)
        self.assertEqual(len(report["faults"]), 2)
        self.assertEqual(len(report["native_kernel_profiles"]), 6)
        self.assertTrue(report["original_kernel_parity_qualified"])
        self.assertFalse(report["full_model_quality_qualified"])
        self.assertFalse(report["native_process_tensor_framework_imported"])
        self.assertEqual(self.manifest["qualification"], "unqualified_candidate")
        self.assertFalse(self.manifest["original_kernel_parity_qualified"])
        self.assertTrue(self.runtime.closed)
        self.assertIs(self.validate(report), report)

    def test_reference_missing_or_false_gate_is_refused_before_native(self):
        for mutate in (lambda r: r.update(status="fail"), lambda r: r["cases"].pop(),
                       lambda r: r["negative_control"].update(mismatches=0),
                       lambda r: r["profiles"][0].update(original_binary_reproduction=False)):
            reference = copy.deepcopy(self.reference)
            mutate(reference)
            self.workspace.save(self.reference_path, reference)
            with self.assertRaises(ValueError):
                self.collect()
            bridge.SparkAttentionCandidate.assert_not_called()

    def test_reference_inputs_dtype_and_extent_must_be_canonical(self):
        for mutate in (lambda r: r["cases"][0]["input_sha256"].update(q="0" * 64),
                       lambda r: r["cases"][0].update(topk=65),
                       lambda r: r["cases"][0]["checks"][2].update(dtype="float32"),
                       lambda r: r["cases"][0]["checks"][2].update(elements=123)):
            reference = copy.deepcopy(self.reference)
            mutate(reference)
            self.workspace.save(self.reference_path, reference)
            with self.assertRaises(ValueError):
                self.collect()
            bridge.SparkAttentionCandidate.assert_not_called()

    def test_foreign_workspace_stale_result_and_result_byte_race_stop_before_cuda(self):
        self.workspace.platform = "pc-heterogeneous"
        with self.assertRaises(ValueError):
            self.collect()
        self.workspace.platform = "dgx-spark"
        self.workspace.check_result.side_effect = ValueError("source changed")
        with self.assertRaises(ValueError):
            self.collect()
        self.workspace.check_result.side_effect = None
        self.workspace.check_result.return_value = {"result_sha256": "0" * 64}
        with self.assertRaises(ValueError):
            self.collect()
        bridge.SparkAttentionCandidate.assert_not_called()

    def test_native_math_mismatch_cannot_be_qualified_even_if_self_repeat_matches(self):
        self.runtime.corrupt = "qk_fp32"
        report = self.collect()
        self.assertEqual(report["status"], "fail")
        self.assertFalse(report["original_kernel_parity_qualified"])
        check = next(c for c in report["cases"][0]["checks"] if c["check"] == "qk_fp32")
        self.assertEqual(check["status"], "fail")
        self.assertIsNone(check["mismatches"])
        self.assertTrue(self.runtime.closed)

    def test_receipt_rejects_missing_stage_case_fault_profile_or_cleanup(self):
        report = self.collect()
        mutations = (lambda r: r["cases"].pop(), lambda r: r["cases"][0]["checks"].pop(),
                     lambda r: r["faults"].pop(), lambda r: r["native_kernel_profiles"].pop(),
                     lambda r: r["native_cleanup"].update(device_owned_bytes=1),
                     lambda r: r["native_cleanup"].update(owned_context_active=True),
                     lambda r: r["faults"][1].update(allocation_attempts=3),
                     lambda r: r["faults"][0]["retry_checks"][0].update(actual_sha256="0" * 64),
                     lambda r: r["cases"][0]["checks"][0].update(actual_sha256="0" * 64),
                     lambda r: r.update(bundle_manifest_sha256="0" * 64),
                     lambda r: r.update(reference_result_sha256="0" * 64),
                     lambda r: r.update(native_process_tensor_framework_imported=True))
        for mutate in mutations:
            modified = copy.deepcopy(report)
            mutate(modified)
            with self.assertRaises(ValueError):
                self.validate(modified)

    def test_persisted_check_authenticates_current_source_bundle_and_reference_without_cuda(self):
        report = self.collect()
        self.workspace.save("s1/attention/native.json", report)
        bridge.SparkAttentionCandidate.reset_mock()
        checked = bridge.check_receipt(self.workspace, "s1/attention-candidate", self.reference_path, "s1/attention/native.json")
        self.assertEqual(checked["status"], "pass")
        self.assertEqual(checked["receipt_result_sha256"],
                         bridge._sha(self.workspace.result_path("s1/attention/native.json").read_bytes()))
        bridge.SparkAttentionCandidate.assert_not_called()
        with self.assertRaises(ValueError):
            bridge.check_receipt(self.workspace, "s1/other-bundle", self.reference_path, "s1/attention/native.json")
        with patch.object(bridge, "load_candidate", return_value=(self.manifest, {}, "d" * 64)):
            with self.assertRaises(ValueError):
                bridge.check_receipt(self.workspace, "s1/attention-candidate", self.reference_path, "s1/attention/native.json")
        bridge.SparkAttentionCandidate.assert_not_called()

    def test_physical_admission_refuses_before_native_open(self):
        with patch.object(bridge, "host_memory", return_value={"physical_available_bytes": 32 * bridge.GiB,
                "physical_total_bytes": 128 * bridge.GiB, "cgroup_headroom_bytes": None}):
            report = self.collect()
        self.assertEqual(report["status"], "unsupported")
        bridge.SparkAttentionCandidate.assert_not_called()

    def test_existing_tensor_framework_requires_a_separate_native_process(self):
        with patch.dict(sys.modules, {"torch": object()}):
            with self.assertRaises(bridge.Unsupported):
                self.collect()
        bridge.SparkAttentionCandidate.assert_not_called()

    def test_native_exception_is_redacted_and_owned_cleanup_runs(self):
        self.runtime.fail = True
        report = self.collect()
        self.assertEqual(report["status"], "unsupported")
        self.assertFalse(report["original_kernel_parity_qualified"])
        self.assertNotIn("private-person", json.dumps(report))
        self.assertEqual(report["native_cleanup"]["device_owned_bytes"], 0)
        self.assertTrue(self.runtime.closed)

    def test_changed_bundle_or_device_is_not_accepted(self):
        self.runtime.manifest_sha256 = "0" * 64
        report = self.collect()
        self.assertEqual(report["status"], "unsupported")
        self.runtime.manifest_sha256 = self.manifest_sha
        self.runtime.device_info = {**DEVICE, "name": "another device"}
        report = self.collect()
        self.assertEqual(report["status"], "unsupported")

    def test_source_change_during_run_refuses_result_publication(self):
        self.runtime.on_call = lambda: setattr(self.workspace, "epoch", 1)
        with self.assertRaisesRegex(ValueError, "Source changed"):
            self.collect()
        self.assertTrue(self.runtime.closed)


if __name__ == "__main__":
    unittest.main()
