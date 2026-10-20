"""CPU/mock coverage of bounded evidence; no GPU arithmetic qualification."""

from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch
import zipfile

import numpy as np

from vesper import spark_operator_probe as probe


class Workspace:
    def __init__(self, root):
        self.root = Path(root)
        self.check, self.check_result = Mock(), Mock()
        self.identity = {"platform": "dgx-spark", "profile": "native",
                         "model": probe.checkpoint.MODEL, "revision": probe.checkpoint.REVISION}

    def stamp_result(self, value):
        return {**value, "workspace_context": self.identity}

    def _path(self, relative, kind, create_parent):
        if any(part in ("", ".", "..") for part in str(relative).split("/")) or ":" in str(relative):
            raise ValueError("escape")
        path = self.root / kind / "dgx-spark" / relative
        if create_parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def artifact_path(self, relative, create_parent=False):
        return self._path(relative, "artifacts", create_parent)

    def result_path(self, relative, create_parent=False):
        return self._path(relative, "results", create_parent)


def memory(available=80 * probe.GiB):
    return {"physical_total_bytes": 128 * probe.GiB, "physical_available_bytes": available,
            "cgroup_headroom_bytes": None}


def zero_outputs(meta, x):
    result = {name: np.zeros(shape, dtype) for name, (shape, dtype) in probe._layout(meta, x.shape).items()}
    if meta["operation"] == "sample":
        del result["rng_offsets_u64"]
        result.update(offset_before=0, offset_after=0 if meta["temperature"] == 0 else 4)
    return result


class Runner:
    def __init__(self):
        self.closed, self.calls, self.fail = False, 0, False
        self.phases = [{"device_owned_bytes": 0, "host_reserved_bytes": 0,
                        "device_peak_owned_bytes": probe.MiB,
                        "execution": {"execution_target": probe.TARGET}}]
        self.torch = Mock()

    def __enter__(self):
        return self

    def run(self, meta, x):
        self.calls += 1
        if self.fail:
            raise RuntimeError("C:/private-user/private-host/compiler-error")
        return zero_outputs(meta, x)

    def close(self):
        self.closed = True


class OperatorProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Workspace(self.temp.name)

    def collect(self, operator="norm", run_id="r01", runner=None, available=None):
        self.runner = runner or Runner()
        with ExitStack() as stack:
            stack.enter_context(patch.object(probe.platform, "system", return_value="Linux"))
            stack.enter_context(patch.object(probe.platform, "machine", return_value="aarch64"))
            stack.enter_context(patch.object(probe, "host_memory", return_value=memory(available or 80 * probe.GiB)))
            stack.enter_context(patch.object(probe, "process_memory", return_value={"resident_bytes": probe.MiB}))
            self.driver = stack.enter_context(patch.object(probe, "Driver", return_value=Mock(
                inventory=Mock(return_value={"name": "NVIDIA GB10", "compute_capability": [12, 1], "ordinal": 0}))))
            self.compiler = stack.enter_context(patch.object(probe, "Compiler", return_value=Mock(
                info={"nvrtc_version": [13, 0]})))
            self.factory = stack.enter_context(patch.object(probe, "NativeRunner", return_value=self.runner))
            stack.enter_context(patch.object(probe, "_EXECUTION_MODE", None))
            return probe.collect("native", self.workspace, operator=operator, run_id=run_id)

    def results(self, operator="norm"):
        native = self.collect(operator)
        reference = json.loads(json.dumps(native))
        reference.update(mode="reference", status="reference_execution_pass", execution_process_instance="separate-reference")
        paths = []
        for mode, report in (("native", native), ("reference", reference)):
            relative = f"s1/{mode}.json"
            self.workspace.result_path(relative, True).write_text(json.dumps(report), encoding="utf-8")
            paths.append(relative)
        return native, reference, paths

    def write_reference(self, reference, paths):
        self.workspace.result_path(paths[1]).write_text(json.dumps(reference), encoding="utf-8")

    def test_fixed_corpora_are_small_deterministic_and_cover_boundaries(self):
        counts = {operator: len(list(probe.cases(operator))) for operator in probe.OPERATORS}
        self.assertEqual(counts, {"norm": 14, "kv": 12, "sampling": 10})
        for operator in probe.OPERATORS:
            self.assertEqual(probe.case_identity(operator), probe.case_identity(operator))
            total = sum(int(np.prod(shape)) * dtype.itemsize for shape, dtype in probe._expected_layout(operator).values())
            self.assertLess(total, probe.DIAGNOSTIC_CAP - probe.MiB)
        kv = list(probe.cases("kv"))
        self.assertEqual({meta["batch"] for meta, _ in kv}, {1, 2})
        self.assertTrue({127, 128, 129, 130}.issubset({meta["position"] for meta, _ in kv}))
        norm = list(probe.cases("norm"))
        self.assertEqual({m["profile"] for m, _ in norm if "profile" in m}, {"window", "compressed-yarn"})
        self.assertEqual({m["temperature"] for m, _ in probe.cases("sampling")}, {0, 1, .7, 1e-6, 2})

    def test_import_is_framework_free(self):
        code = "import sys; import vesper.spark_operator_probe; assert 'torch' not in sys.modules; assert 'tilelang' not in sys.modules"
        result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_normal_execution_is_not_reference_or_model_qualification(self):
        report = self.collect()
        self.assertEqual(report["status"], "native_execution_pass")
        self.assertFalse(report["original_kernel_parity_qualified"])
        self.assertFalse(report["full_model_quality_qualified"])
        self.assertTrue(report["owned_native_resources_closed"])
        self.assertTrue(self.runner.closed)
        self.assertEqual(self.runner.calls, 14)
        self.assertEqual(len(probe._load(self.workspace, report)), len(probe._expected_layout("norm")))
        self.assertIn("learned coefficient provenance", report["exclusions"])

    def test_failure_redacts_private_reason_and_closes_owned_runner(self):
        runner = Runner()
        runner.fail = True
        report = self.collect(runner=runner)
        self.assertEqual(report["status"], "unsupported")
        self.assertEqual(report["failure_type"], "RuntimeError")
        self.assertNotIn("private-user", json.dumps(report))
        self.assertTrue(runner.closed)
        self.assertTrue(report["owned_native_resources_closed"])

    def test_foreign_workspace_or_revision_refused_before_cuda(self):
        for field, value in (("platform", "pc-heterogeneous"), ("revision", "0" * 40)):
            original = self.workspace.identity[field]
            self.workspace.identity[field] = value
            with patch.object(probe, "Driver") as driver:
                with self.assertRaises(probe.Unsupported):
                    probe.collect("native", self.workspace, operator="norm", run_id="r01")
                driver.assert_not_called()
            self.workspace.identity[field] = original

    def test_physical_memory_admission_precedes_cuda(self):
        report = self.collect(available=32 * probe.GiB + 128 * probe.MiB)
        self.assertEqual(report["status"], "unsupported")
        self.driver.assert_not_called()
        self.factory.assert_not_called()

    def test_wrong_os_precedes_cuda(self):
        with patch.object(probe.platform, "system", return_value="Windows"), patch.object(probe, "Driver") as driver:
            report = probe.collect("native", self.workspace, operator="norm", run_id="r01")
        self.assertEqual(report["status"], "unsupported")
        driver.assert_not_called()

    def test_second_execution_in_one_process_refused(self):
        with patch.object(probe, "_EXECUTION_MODE", None):
            probe._claim_execution("native")
            with self.assertRaises(probe.Unsupported):
                probe._claim_execution("reference")

    def test_existing_artifact_and_invalid_run_rejected_before_cuda(self):
        self.workspace.artifact_path("s1/operators/norm/r01/existing", True)
        with patch.object(probe, "Driver") as driver:
            with self.assertRaises(FileExistsError):
                probe.collect("native", self.workspace, operator="norm", run_id="r01")
            with self.assertRaises(ValueError):
                probe.collect("native", self.workspace, operator="norm", run_id="../escape")
            driver.assert_not_called()

    def test_sampling_normalization_preserves_integer_offsets(self):
        meta, x = next(probe.cases("sampling"))
        output = zero_outputs(meta, x)
        output.update(offset_before=2**40, offset_after=2**40)
        actual = probe._normalized(output, meta, x.shape)
        self.assertEqual(actual["rng_offsets_u64"].dtype, np.uint64)
        self.assertEqual(actual["rng_offsets_u64"].tolist(), [2**40, 2**40])
        for invalid in (-1, 2**64, .5):
            output["offset_after"] = invalid
            with self.assertRaises(ValueError):
                probe._normalized(output, meta, x.shape)

    def test_diagnostic_missing_stage_wrong_dtype_or_nonfinite_rejected(self):
        meta, x = next(probe.cases("norm"))
        output = zero_outputs(meta, x)
        del output["rstd_fp32"]
        with self.assertRaises(ValueError):
            probe._normalized(output, meta, x.shape)
        output = zero_outputs(meta, x)
        output["rstd_fp32"] = output["rstd_fp32"].astype(np.float64)
        with self.assertRaises(ValueError):
            probe._normalized(output, meta, x.shape)
        output = zero_outputs(meta, x)
        output["rstd_fp32"].fill(np.nan)
        with self.assertRaises(ValueError):
            probe._normalized(output, meta, x.shape)

    def test_compare_qualifies_only_fixed_component_scope(self):
        _, _, paths = self.results()
        report = probe.compare_runs(self.workspace, *paths, operator="norm")
        self.assertEqual(report["status"], "pass")
        self.assertTrue(report["original_kernel_parity_qualified"])
        self.assertFalse(report["full_model_quality_qualified"])
        self.assertIn("fixed generated", report["qualification_scope"])

    def test_compare_rejects_same_process_changed_device_or_inputs(self):
        native, reference, paths = self.results()
        for key, value in (("execution_process_instance", native["execution_process_instance"]),
                           ("observed_hardware", {"name": "other"}), ("revision", "0" * 40),
                           ("preflight_compiler", {"nvrtc_version": [12, 4]}), ("case_identity", [])):
            changed = {**reference, key: value}
            self.write_reference(changed, paths)
            with self.assertRaises(ValueError):
                probe.compare_runs(self.workspace, *paths, operator="norm")

    def test_stage_mismatch_is_diagnostic_failure(self):
        native, reference, paths = self.results()
        arrays = probe._load(self.workspace, native)
        arrays["c00_rstd_fp32"].flat[0] = np.float32(1)
        reference["diagnostics"] = probe._save(self.workspace, "norm", "ref", arrays)
        self.write_reference(reference, paths)
        report = probe.compare_runs(self.workspace, *paths, operator="norm")
        self.assertEqual(report["status"], "fail")
        self.assertFalse(report["original_kernel_parity_qualified"])
        failures = [c for c in report["checks"] if c["status"] == "fail"]
        self.assertEqual([c["check"] for c in failures], ["c00_rstd_fp32"])

    def test_kv_and_sampling_archives_round_trip_and_rng_mismatch_fails(self):
        for operator in ("kv", "sampling"):
            native = self.collect(operator, run_id=f"{operator}-native")
            self.assertEqual(native["status"], "native_execution_pass")
            arrays = probe._load(self.workspace, native)
            self.assertEqual(set(arrays), set(probe._expected_layout(operator)))
            reference = {**native, "mode": "reference", "status": "reference_execution_pass",
                         "execution_process_instance": "independent-reference"}
            paths = [f"s1/{operator}-native.json", f"s1/{operator}-reference.json"]
            if operator == "sampling":
                arrays["c01_rng_offsets_u64"][1] += np.uint64(4)
                reference["diagnostics"] = probe._save(self.workspace, operator, "sampling-ref", arrays)
            for path, report in zip(paths, (native, reference)):
                self.workspace.result_path(path, True).write_text(json.dumps(report), encoding="utf-8")
            compared = probe.compare_runs(self.workspace, *paths, operator=operator)
            self.assertEqual(compared["status"], "pass" if operator == "kv" else "fail")
            if operator == "sampling":
                self.assertEqual([c["check"] for c in compared["checks"] if c["status"] == "fail"],
                                 ["c01_rng_offsets_u64"])

    def test_archive_digest_and_member_allowlist_rejected(self):
        native = self.collect()
        path = self.workspace.artifact_path(native["diagnostics"]["artifact"])
        with path.open("ab") as stream:
            stream.write(b"tampered")
        with self.assertRaises(ValueError):
            probe._load(self.workspace, native)
        with zipfile.ZipFile(path, "a") as archive:
            archive.writestr("foreign.npy", b"x")
        native["diagnostics"].update(sha256=probe._sha(path.read_bytes()), file_bytes=path.stat().st_size)
        with self.assertRaises(ValueError):
            probe._load(self.workspace, native)

    def test_archive_header_shape_rejected_before_numpy_allocation(self):
        native = self.collect()
        path = self.workspace.artifact_path(native["diagnostics"]["artifact"])
        arrays = probe._load(self.workspace, native)
        arrays["c00_rstd_fp32"] = np.zeros((1,), np.float32)
        with path.open("wb") as stream:
            np.savez(stream, **arrays)
        native["diagnostics"].update(sha256=probe._sha(path.read_bytes()), file_bytes=path.stat().st_size)
        with patch.object(probe.np, "load") as load:
            with self.assertRaises(ValueError):
                probe._load(self.workspace, native)
            load.assert_not_called()

    def test_native_runner_explicitly_selects_spark(self):
        with patch("vesper.normalization.NativeNormRoPE") as factory:
            operator = MagicMock()
            operator.stats.return_value = {"device_owned_bytes": 0, "host_reserved_bytes": 0,
                                           "device_peak_owned_bytes": probe.MiB}
            factory.return_value = operator
            runner = probe.NativeRunner("norm")
            runner.__enter__()
            runner.close()
            factory.assert_called_once_with(execution_target="spark-sm121-candidate")
            operator.__exit__.assert_called_once_with(None, None, None)

    def test_native_retained_resource_receipt_is_a_failure(self):
        runner = probe.NativeRunner("norm")
        runner.current = MagicMock()
        runner.current.stats.return_value = {"device_owned_bytes": 1024, "host_reserved_bytes": 0,
                                             "device_peak_owned_bytes": probe.MiB}
        with self.assertRaises(RuntimeError):
            runner.close()

    def test_reference_constructor_failure_unwinds_owned_contexts(self):
        torch = Mock()
        with patch.dict(sys.modules, {"torch": torch}), patch.object(probe.ReferenceRunner, "_initialize", side_effect=RuntimeError), \
                patch.object(probe.ReferenceRunner, "close", autospec=True) as close:
            with self.assertRaises(RuntimeError):
                probe.ReferenceRunner("norm", Path(self.temp.name), {}, Mock())
            close.assert_called_once()

    def test_reference_sampling_delegates_original_rng_without_reseeding_each_call(self):
        runner = probe.ReferenceRunner.__new__(probe.ReferenceRunner)
        runner.torch, runner.current, runner.stream = Mock(), Mock(), None
        with patch("vesper.sampling_reference.original_sample", return_value=({"ids_i64": np.array([7], np.int64)}, 8, 12)) as original:
            samples = list(probe.cases("sampling"))
            for meta, x in samples:
                output = runner.run(meta, x)
                self.assertEqual((output["offset_before"], output["offset_after"]), (8, 12))
            self.assertEqual(original.call_count, 10)
            self.assertEqual(runner.torch.cuda.manual_seed.call_count, 2)
            self.assertEqual([c.args[0] for c in runner.torch.cuda.manual_seed.call_args_list], [0, 2026])
            self.assertEqual([c.args[3] for c in original.call_args_list], [m["temperature"] for m, _ in samples])

    def test_unpinned_reference_framework_refused_before_device_selection(self):
        torch = Mock(__version__="2.14.1+cu128")
        torch.version.git_version, torch.version.cuda = probe.TORCH_REVISION, "12.8"
        with self.assertRaises(probe.Unsupported):
            probe._torch_profile(torch, {"name": "NVIDIA GB10"})
        torch.cuda.set_device.assert_not_called()


if __name__ == "__main__":
    unittest.main()
