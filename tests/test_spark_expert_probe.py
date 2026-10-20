"""CPU/mock evidence plumbing; these tests do not qualify GB10 arithmetic."""

from contextlib import ExitStack
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile

import numpy as np

from vesper import spark_expert_probe as probe
from vesper.checkpoint import MODEL, REVISION


class Workspace:
    def __init__(self, root):
        self.root = Path(root)
        self.check = Mock()
        self.context_value = {"platform": "dgx-spark", "profile": "native", "model": MODEL,
                              "revision": REVISION, "git_head": "a" * 40, "dirty_diff_sha256": "b" * 64}
        self.check_result = Mock()

    def stamp_result(self, value):
        return {**value, "workspace_context": self.context_value}

    def artifact_path(self, relative, create_parent=False):
        return self._path(relative, "artifacts", create_parent)

    def result_path(self, relative, create_parent=False):
        return self._path(relative, "results", create_parent)

    def _path(self, relative, kind, create_parent):
        if any(part in ("", ".", "..") for part in str(relative).split("/")) or ":" in str(relative):
            raise ValueError("escape")
        path = self.root / kind / "dgx-spark" / relative
        if create_parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def context(self, run_id):
        return {**self.context_value, "run_id": run_id}

    def check_runtime_module(self, module_file, relative):
        from vesper.workspace import Workspace as RealWorkspace
        RealWorkspace.check_runtime_module(self, module_file, relative)


def memory(available=80 * probe.GiB):
    return {"physical_total_bytes": 128 * probe.GiB, "physical_available_bytes": available,
            "cgroup_headroom_bytes": None}


def output(mode):
    values = {name: np.zeros(shape, dtype) for name, (shape, dtype) in probe.LAYOUT.items()}
    if mode == "native":
        values["down_input"] = np.zeros(2304, np.float32)
    return values


class Executor:
    def __init__(self):
        self.info = {"name": "NVIDIA GB10", "compute_capability": "12.1"}
        self.device = Mock(peak=19 * probe.MiB)
        self.cubin_sha256 = "c" * 64
        self.closed = False
        self.calls = 0
        self.fail = False
        self.corrupt_repeat = False

    def __enter__(self):
        return self

    def close(self):
        self.closed = True

    def forward(self, x, route):
        self.calls += 1
        if self.fail:
            raise RuntimeError("private-host/private-path compiler failure")
        value = output("native")
        if self.corrupt_repeat and self.calls == 17:
            value["w2"][0] = 1
        return value


class ExpertProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Workspace(self.temp.name)
        self.fixture = self.workspace.artifact_path("s1/expert-fixture/native-expert.npz", True)
        with self.fixture.open("wb") as stream:
            stream.truncate(probe.FIXTURE_BYTES)
        self.manifest = {"model": MODEL, "revision": REVISION, "kind": "complete_expert",
                         "layer": 0, "expert": 0, "weights_fetched_bytes": probe.FIXTURE_BYTES,
                         "swiglu_limit": 10.0, "tensors": []}
        for name, shape in probe.checkpoint.EXPERT_LAYOUT.items():
            projection, suffix = name.split("_")
            self.manifest["tensors"].append({"name": name, "shape": list(shape),
                "tensor": f"layers.0.ffn.experts.0.{projection}.{suffix}",
                "dtype": "I8" if suffix == "weight" else "F8_E8M0", "bytes": int(np.prod(shape)),
                "sha256": "e" * 64})
        self.fixture.with_name("manifest.json").write_text(json.dumps(self.manifest))

    def collect(self, candidate=None, mode="native", run_id="r01", available=None):
        self.candidate = candidate or Executor()
        with ExitStack() as stack:
            stack.enter_context(patch.object(probe.platform, "system", return_value="Linux"))
            stack.enter_context(patch.object(probe.platform, "machine", return_value="aarch64"))
            stack.enter_context(patch.object(probe, "host_memory", return_value=memory(available or 80 * probe.GiB)))
            stack.enter_context(patch.object(probe, "process_memory", return_value={"resident_bytes": 20 * probe.MiB}))
            self.driver_factory = stack.enter_context(patch.object(probe, "Driver", return_value=Mock(inventory=Mock(return_value={"name": "GB10"}))))
            stack.enter_context(patch.object(probe, "Compiler", return_value=Mock(info={"nvrtc_version": [13, 0]})))
            self.fixture_loader = stack.enter_context(patch.object(probe.checkpoint, "load_expert_fixture", return_value=({}, self.manifest)))
            stack.enter_context(patch.object(probe, "_native", return_value=self.candidate))
            stack.enter_context(patch.object(probe, "_EXECUTION_MODE", None))
            return probe.collect(mode, self.workspace, run_id=run_id)

    def test_native_execution_is_not_parity_or_model_quality(self):
        report = self.collect()
        self.assertEqual(report["status"], "native_execution_pass")
        self.assertEqual(len(report["cases"]), 16)
        self.assertEqual(self.candidate.calls, 17)
        self.assertTrue(self.candidate.closed)
        self.assertFalse(report["original_kernel_parity_qualified"])
        self.assertFalse(report["full_model_quality_qualified"])
        self.assertLess(report["diagnostics"]["array_bytes"], probe.DIAGNOSTIC_CAP)
        values = probe._load_diagnostics(self.workspace, report)
        self.assertEqual(len(values), 16 * len(probe.LAYOUT))
        self.assertEqual(values["c00_down_input"].dtype, np.uint16)

    def test_fixed_cases_are_bf16_and_cover_clamp_stress_and_routes(self):
        cases = list(probe.cases())
        self.assertEqual(len(cases), 16)
        self.assertEqual({c["routing_weight"] for c, _ in cases}, {None, 0, .25, 1.5})
        self.assertEqual({c["input_scale"] for c, _ in cases}, {1, 32})
        for case, x in cases:
            self.assertTrue(np.array_equal(x, probe.bf16_values(probe.bf16_bits(x))))
            self.assertEqual(case["input_sha256"], probe._sha(x.tobytes()))

    def test_normal_import_does_not_import_frameworks(self):
        code = "import sys; import vesper.spark_expert_probe; assert 'torch' not in sys.modules; assert 'tilelang' not in sys.modules"
        result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_foreign_platform_rejected_before_fixture_or_cuda(self):
        self.workspace.context_value["platform"] = "pc-heterogeneous"
        with patch.object(probe.checkpoint, "load_expert_fixture") as load, patch.object(probe, "Driver") as driver:
            with self.assertRaises(probe.Unsupported):
                probe.collect("native", self.workspace, run_id="r01")
            load.assert_not_called()
            driver.assert_not_called()

    def test_foreign_revision_rejected_before_hardware(self):
        self.workspace.context_value["revision"] = "0" * 40
        with patch.object(probe, "Driver") as driver:
            with self.assertRaises(probe.Unsupported):
                probe.collect("native", self.workspace, run_id="r01")
            driver.assert_not_called()

    def test_unsupported_os_does_not_load_cuda(self):
        with patch.object(probe.platform, "system", return_value="Windows"), patch.object(probe, "Driver") as driver:
            report = probe.collect("native", self.workspace, run_id="r01")
        self.assertEqual(report["status"], "unsupported")
        driver.assert_not_called()

    def test_insufficient_physical_floor_prevents_cuda_and_fixture_load(self):
        report = self.collect(available=32 * probe.GiB + 128 * probe.MiB)
        self.assertEqual(report["status"], "unsupported")
        self.assertEqual(self.candidate.calls, 0)
        self.driver_factory.assert_not_called()
        self.fixture_loader.assert_not_called()

    def test_existing_artifact_prevents_reexecution(self):
        self.workspace.artifact_path("s1/expert/r01/reservation", True)
        with patch.object(probe, "Driver") as driver:
            with self.assertRaises(FileExistsError):
                probe.collect("native", self.workspace, run_id="r01")
            driver.assert_not_called()

    def test_mismatching_repeat_is_failure_and_native_closes(self):
        candidate = Executor()
        candidate.corrupt_repeat = True
        report = self.collect(candidate)
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["repeat_first_case"]["status"], "fail")
        self.assertTrue(candidate.closed)

    def test_execution_failure_is_sanitized_and_owned_native_is_closed(self):
        candidate = Executor()
        candidate.fail = True
        report = self.collect(candidate)
        self.assertEqual(report["status"], "unsupported")
        self.assertTrue(candidate.closed)
        self.assertNotIn("private-host", json.dumps(report))
        self.assertNotIn("diagnostics", report)

    def test_wrong_executor_hardware_is_rejected_before_forward_and_closed(self):
        candidate = Executor()
        candidate.info["compute_capability"] = "8.9"
        report = self.collect(candidate)
        self.assertEqual(report["status"], "unsupported")
        self.assertEqual(candidate.calls, 0)
        self.assertTrue(candidate.closed)

    def test_reference_absence_is_blocked_without_loading_native_executor(self):
        with patch.object(probe.importlib.util, "find_spec", return_value=None), \
                patch.object(probe, "_native") as native:
            report = {}
            with self.assertRaises(probe.Unsupported):
                probe._reference({}, self.manifest, self.fixture.parent, report)
        self.assertEqual(report["missing_dependencies"], ["torch", "tilelang"])
        native.assert_not_called()

    def test_second_execution_in_same_process_is_rejected(self):
        with patch.object(probe, "_EXECUTION_MODE", None):
            probe._claim_execution("native")
            with self.assertRaises(probe.Unsupported):
                probe._claim_execution("reference")

    def test_diagnostic_corruption_does_not_reach_numpy_allocation(self):
        report = self.collect()
        path = self.workspace.artifact_path(report["diagnostics"]["artifact"])
        path.write_bytes(b"bad")
        with patch.object(probe.np, "load") as load:
            with self.assertRaises(ValueError):
                probe._load_diagnostics(self.workspace, report)
            load.assert_not_called()

    def test_oversized_declared_npy_shape_rejected_before_numpy_load(self):
        report = self.collect()
        path = self.workspace.artifact_path(report["diagnostics"]["artifact"])
        original = probe._load_diagnostics(self.workspace, report)
        with zipfile.ZipFile(path, "w") as archive:
            for key, (shape, dtype) in {f"c{i:02d}_{n}": item for i in range(16) for n, item in probe.LAYOUT.items()}.items():
                stream = io.BytesIO()
                if key == "c00_q":
                    np.lib.format.write_array_header_1_0(stream, {"shape": (2**40,), "fortran_order": False, "descr": "|u1"})
                else:
                    np.save(stream, original[key], allow_pickle=False)
                archive.writestr(f"{key}.npy", stream.getvalue())
        report["diagnostics"].update(file_bytes=path.stat().st_size, sha256=probe._sha(path.read_bytes()))
        with patch.object(probe.np, "load") as load:
            with self.assertRaises(ValueError):
                probe._load_diagnostics(self.workspace, report)
            load.assert_not_called()

    def _comparison_reports(self):
        native = self.collect()
        reference = {**native, "mode": "reference", "status": "reference_execution_pass", "execution_process_instance": "different-process"}
        for name, value in (("native", native), ("reference", reference)):
            self.workspace.result_path(f"{name}.json", True).write_text(json.dumps(value))
        return native, reference

    def test_comparison_requires_successful_distinct_reference_execution(self):
        native, reference = self._comparison_reports()
        result = probe.compare_runs(self.workspace, "native.json", "reference.json")
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["original_kernel_parity_qualified"])
        self.assertFalse(result["full_model_quality_qualified"])
        self.assertEqual(self.workspace.check_result.call_count, 2)
        reference["execution_process_instance"] = native["execution_process_instance"]
        self.workspace.result_path("reference.json").write_text(json.dumps(reference))
        with self.assertRaises(ValueError):
            probe.compare_runs(self.workspace, "native.json", "reference.json")

    def test_case_fixture_or_unsuccessful_reference_mismatch_is_rejected(self):
        native, reference = self._comparison_reports()
        for update in ({"status": "unsupported"}, {"fixture": {"layer": 1}}, {"cases": []}):
            self.workspace.result_path("reference.json").write_text(json.dumps({**reference, **update}))
            with self.assertRaises(ValueError):
                probe.compare_runs(self.workspace, "native.json", "reference.json")

    def test_cli_foreign_module_is_rejected_before_execution(self):
        ws = self.workspace
        ws.root = Path(probe.__file__).resolve().parents[2].parent / "other-checkout"
        with patch("vesper.workspace.Workspace.open", return_value=ws), patch.object(probe, "collect") as collect:
            with self.assertRaises(ValueError):
                probe.main(["native", "--run-id", "r01"])
            collect.assert_not_called()

    def test_cli_records_source_context_and_writes_via_workspace(self):
        ws = self.workspace
        ws.root = Path(probe.__file__).resolve().parents[2]
        ws.check_runtime_module = Mock()
        ws.result_path = Mock(return_value=Mock(exists=Mock(return_value=False)))
        ws.write_result = Mock(return_value=ws.root / "results/dgx-spark/s1/expert/r01.json")
        with patch("vesper.workspace.Workspace.open", return_value=ws), \
                patch.object(probe, "collect", return_value={"status": "native_execution_pass"}), \
                patch("sys.stdout", new_callable=io.StringIO) as stdout:
            self.assertEqual(probe.main(["native", "--run-id", "r01"]), 0)
        ws.write_result.assert_called_once_with("s1/expert/r01.json", {"status": "native_execution_pass"},
                                              run_id="r01", initial_context=ws.context("r01"))
        self.assertEqual(json.loads(stdout.getvalue())["status"], "native_execution_pass")


if __name__ == "__main__":
    unittest.main()
