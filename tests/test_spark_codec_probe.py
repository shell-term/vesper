"""CPU codec checks and authenticated-reader seams; no model qualification."""

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import zlib

import numpy as np

from vesper import spark_codec_probe as probe
from vesper.memory import RuntimeMemory, RuntimeCancelled, current_policy
from vesper.resources import MemoryBudget, MiB, GiB


class Workspace:
    def __init__(self, root):
        self.root = Path(root); self.changed = False
        self.check = Mock(); self.check_runtime_module = Mock()
    def context(self, run_id=None):
        return {"platform": "dgx-spark", "profile": "native", "model": probe.MODEL,
            "revision": probe.REVISION, "git_head": "a" * 40, "run_id": run_id}
    def artifact_path(self, relative, create_parent=False):
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts: raise ValueError("foreign artifact path")
        result = self.root / "artifacts/dgx-spark" / path
        if create_parent: result.parent.mkdir(parents=True, exist_ok=True)
        return result
    def result_path(self, relative, create_parent=False):
        result = self.root / "results/dgx-spark" / relative
        if create_parent: result.parent.mkdir(parents=True, exist_ok=True)
        return result
    def stamp_result(self, payload, run_id=None, *, initial_context=None):
        if self.changed: raise ValueError("source changed")
        return {**payload, "workspace_context": initial_context or self.context(run_id)}
    def write_result(self, relative, payload, run_id=None, *, initial_context=None):
        path = self.result_path(relative, create_parent=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(self.stamp_result(payload, run_id, initial_context=initial_context), stream)
        return path


class Reader:
    failure = False
    instances = []
    def __init__(self, metadata, originals, integrity, budget, *, tensor_names, workspace):
        self.budget, self.workspace = budget, workspace
        self.reads = self.logical = 0; self.closed = False
        self.instances.append(self)
        self.tensors = {}
        for layer, expert in probe.SAMPLES:
            for short, shape in probe.EXPERT_LAYOUT.items():
                self.tensors[probe._name(layer, expert, short)] = {"dtype": "I8" if short.endswith("_weight") else "F8_E8M0",
                    "shape": list(shape), "bytes": int(np.prod(shape))}
        if set(tensor_names) != set(self.tensors): raise AssertionError("Expected only the fixed 72 tensors")
    def __enter__(self):
        plans = list((self.workspace.root / "artifacts/dgx-spark/s2/codec").glob("*/plan.json"))
        if len(plans) != 1: raise AssertionError("Plan must exist before metadata reads")
        self.budget.acquire(16 * MiB, category="metadata")
        return self
    def read_into(self, name, offset, target):
        self.budget.acquire(0)
        if self.failure: raise ValueError("authenticated source block SHA mismatch")
        target[:] = np.arange(target.size, dtype=np.uint8) if name.endswith(".weight") else 127
        self.reads += 1; self.logical += target.size
    def stats(self): return {"logical_reads": self.reads, "logical_bytes": self.logical,
        "physical_data_bytes_read": self.reads * probe.BLOCK, "integrity_index_bytes_read": self.reads * 4096}
    def __exit__(self, *_):
        self.budget.release(16 * MiB, category="metadata"); self.closed = True


class SparkCodecProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.workspace = Workspace(self.temp.name)
        self.patches = []
        for name in ("vesper.resources.physical_memory", "vesper.memory.physical_memory"):
            self.use(patch(name, return_value={"available": 128 * GiB}))
        self.use(patch("vesper.memory.process_memory", return_value={"resident_bytes": MiB}))
        self.use(patch.object(RuntimeMemory, "ENABLE_WATCHDOG", False))
        self.layout = {}; shapes = {}
        offset = 0
        for projection in ("w1", "w3", "w2"):
            for suffix, shape in (("weight", (64, 64)), ("scale", (64, 2))):
                name = projection + "_" + suffix; size = int(np.prod(shape))
                shapes[name] = shape; self.layout[name] = {"offset": offset, "bytes": size}
                offset += size
        self.use(patch.object(probe, "EXPERT_LAYOUT", shapes))
        self.use(patch.object(probe, "EXPERT_BYTES", offset))
        self.use(patch.object(probe, "tensor_layout", return_value=self.layout))
        self.use(patch.object(probe, "OriginalTensorReader", Reader))
        Reader.failure = False; Reader.instances = []

    def use(self, patcher):
        result = patcher.start(); self.addCleanup(patcher.stop); return result

    def test_fixed_plan_precedes_reads_and_success_is_only_lossless_sampling(self):
        report = probe.collect(self.workspace, run_id="codec-r01")
        self.assertEqual(report["status"], "sampled_exact_lossless_frames")
        self.assertEqual(len(report["samples"]), 12)
        self.assertEqual(report["plan"]["samples"], [{"layer": l, "expert": e} for l, e in probe.SAMPLES])
        self.assertEqual(report["reader_before_close"]["logical_reads"], 72)
        for row in report["samples"]:
            self.assertEqual(len(row["codecs"]), 2)
            self.assertEqual(row["fp4_packed_code_bytes"] + row["e8m0_scale_bytes"], probe.EXPERT_BYTES)
            for codec in row["codecs"]:
                self.assertTrue(codec["exact_reconstruction"])
                self.assertEqual(codec["original_sha256"], codec["reconstructed_sha256"])
                self.assertEqual(codec["modeled_storage_bytes"] % 4096, 0)
        self.assertLessEqual(report["sample_budget_peak_bytes"], probe.SAMPLE_CAP)
        self.assertTrue(report["owned_resources_closed"]); self.assertTrue(report["memory_runtime_closed"])
        self.assertTrue(all(report["terminal_runtime_checks"].values()))
        self.assertEqual(report["sample_budget_reserved_after"], 0)
        self.assertFalse(report["full_model_quality_qualified"]); self.assertFalse(report["gpu_requested"])
        self.assertFalse(report["general_speed_claim"])
        self.assertTrue(Reader.instances[0].closed)
        self.assertIsNone(current_policy())
        self.assertEqual(report["plan"]["limits"]["catalog_transient_admission_bytes_separate"], 512 * MiB)

    def test_real_multiframe_zlib_reconstruction_and_padding_accounting(self):
        budget = MemoryBudget(cap=32 * MiB)
        rng = np.random.default_rng(20261021)
        data = rng.integers(0, 256, 2 * probe.BLOCK + 17, dtype=np.uint8)
        result = probe._codec(data, 1, budget)
        self.assertEqual(result["frames"], 3)
        self.assertEqual(result["original_sha256"], hashlib.sha256(data).hexdigest())
        self.assertEqual(result["modeled_storage_bytes"], result["aligned_payload_bytes"] + 4096 + result["aligned_index_bytes"])
        self.assertGreater(result["modeled_storage_bytes"], data.nbytes)
        self.assertEqual(budget.reserved, 0)

    def test_codec_identity_records_optional_extension_hash_without_private_path(self):
        path = Path(self.temp.name) / "codec-extension.bin"
        path.write_bytes(b"bounded synthetic extension identity")
        with patch.object(probe.zlib, "__file__", str(path), create=True):
            identity = probe._codec_identity()
        self.assertEqual(identity["extension_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertNotIn(self.temp.name, str(identity))

    def test_corrupt_or_wrong_output_frame_never_passes_and_releases_scratch(self):
        budget = MemoryBudget(cap=32 * MiB)
        data = np.arange(255, dtype=np.uint8)
        for replacement in (b"bad-zlib", zlib.compress(bytes(data.size)), zlib.compress(bytes(data.size + 1))):
            with self.subTest(replacement=replacement[:8]), patch.object(probe.zlib, "compress", return_value=replacement):
                with self.assertRaises((ValueError, zlib.error)): probe._codec(data, 1, budget)
            self.assertEqual(budget.reserved, 0)

    def test_authenticated_source_failure_reports_no_success_and_cleans_up(self):
        Reader.failure = True
        report = probe.collect(self.workspace, run_id="bad-source-r01")
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["samples"], [])
        self.assertEqual(report["failure_type"], "ValueError")
        self.assertTrue(report["private_failure"]["saved"])
        self.assertEqual(report["sample_budget_reserved_after"], 0)
        self.assertTrue(Reader.instances[0].closed)
        self.assertTrue(report["memory_runtime_closed"])

    def test_keyboard_interrupt_during_codec_publishes_immutable_cancelled_evidence(self):
        with patch.object(probe.zlib, "compress", side_effect=KeyboardInterrupt("cancel bounded sample")) as compress, \
                patch("vesper.workspace.Workspace.open", return_value=self.workspace), redirect_stdout(io.StringIO()):
            self.assertEqual(probe.main(["--run-id", "cancel-r01"]), 2)
            with self.assertRaises(FileExistsError): probe.main(["--run-id", "cancel-r01"])
        compress.assert_called_once()
        report = json.loads(self.workspace.result_path("s2/codec/cancel-r01.json").read_text())
        self.assertEqual(report["status"], "cancelled")
        self.assertEqual(report["failure_type"], "KeyboardInterrupt")
        self.assertEqual(report["failure_stage"], "expert-0-0")
        self.assertEqual(report["workspace_context"]["git_head"], "a" * 40)
        self.assertTrue(report["owned_resources_closed"])
        self.assertTrue(report["memory_runtime_closed"])
        self.assertEqual(report["sample_budget_reserved_after"], 0)
        self.assertEqual(report["samples"][0]["codecs"], [])
        self.assertTrue(Reader.instances[0].closed)
        self.assertIsNone(current_policy())
        raw = self.workspace.artifact_path(report["private_failure"]["artifact"]).read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), report["private_failure"]["sha256"])
        self.assertEqual(json.loads(raw)["failure_type"], "KeyboardInterrupt")
        self.assertNotIn(self.temp.name, json.dumps(report))

    def test_runtime_cancellation_preserves_first_failure_when_reader_cleanup_also_fails(self):
        close = Reader.__exit__
        def dirty_close(reader, *args):
            close(reader, *args)
            raise RuntimeError("secondary reader cleanup error")
        with patch.object(Reader, "read_into", side_effect=RuntimeCancelled("requested cancellation")), \
                patch.object(Reader, "__exit__", dirty_close):
            report = probe.collect(self.workspace, run_id="cancel-cleanup-r01")
        self.assertEqual(report["status"], "cancelled")
        self.assertEqual(report["failure_type"], "RuntimeCancelled")
        self.assertTrue(report["owned_resources_closed"])
        self.assertTrue(report["memory_runtime_closed"])
        private = json.loads(self.workspace.artifact_path(report["private_failure"]["artifact"]).read_bytes())
        self.assertEqual(private["failure_type"], "RuntimeCancelled")
        self.assertNotIn("secondary reader cleanup error", private["traceback"])
        self.assertIsNone(current_policy())

    def test_terminal_monitor_error_refuses_success_after_all_samples_and_cleanup(self):
        close = Reader.__exit__
        def dirty_close(reader, *args):
            close(reader, *args)
            current_policy().monitor_error = "injected terminal monitor failure"
        with patch.object(Reader, "__exit__", dirty_close):
            report = probe.collect(self.workspace, run_id="monitor-error-r01")
        self.assertEqual(len(report["samples"]), 12)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["failure_stage"], "terminal_memory_checkpoint")
        self.assertFalse(report["terminal_runtime_checks"]["monitor_observation_healthy"])
        self.assertFalse(report["owned_resources_closed"])
        self.assertTrue(report["memory_runtime_closed"])
        self.assertEqual(report["memory_after"]["charged_host_bytes"], 0)
        self.assertIsNone(current_policy())

    def test_monitor_that_does_not_stop_cannot_pass_terminal_gate(self):
        close = Reader.__exit__
        monitor = Mock()
        monitor.is_alive.return_value = True
        def dirty_close(reader, *args):
            close(reader, *args)
            current_policy().monitor = monitor
        with patch.object(Reader, "__exit__", dirty_close):
            report = probe.collect(self.workspace, run_id="monitor-live-r01")
        monitor.join.assert_called_once_with(timeout=2)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["failure_stage"], "cleanup")
        self.assertFalse(report["terminal_runtime_checks"]["monitor_thread_stopped"])
        self.assertFalse(report["owned_resources_closed"])
        self.assertTrue(report["memory_runtime_closed"])
        self.assertIsNone(current_policy())

    def test_final_sample_pressure_is_checked_before_success(self):
        sample = RuntimeMemory.sample
        def final_pressure(memory, label):
            if label == "expert-39-383": memory.available = lambda: 31 * GiB
            return sample(memory, label)
        with patch.object(RuntimeMemory, "sample", final_pressure):
            report = probe.collect(self.workspace, run_id="final-floor-r01")
        self.assertEqual(len(report["samples"]), 12)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["failure_type"], "MemoryError")
        self.assertEqual(report["failure_stage"], "expert-39-383")
        self.assertEqual(report["memory_after"]["minimum_observed_available_bytes"], 31 * GiB)
        self.assertFalse(report["terminal_runtime_checks"]["observed_physical_floor_preserved"])
        self.assertEqual(report["sample_budget_reserved_after"], 0)
        self.assertTrue(report["memory_runtime_closed"])
        self.assertIsNone(current_policy())

    def test_physical_reserve_refuses_before_metadata_or_payload(self):
        with patch("vesper.memory.physical_memory", return_value={"available": 31 * GiB}):
            report = probe.collect(self.workspace, run_id="floor-r01")
        self.assertEqual(report["failure_stage"], "memory_runtime")
        self.assertEqual(report["failure_type"], "MemoryError")
        self.assertEqual(Reader.instances, [])
        self.assertTrue(report["memory_runtime_closed"])

    def test_scratch_admission_pressure_refuses_before_codec_work(self):
        budget = MemoryBudget(cap=MiB)
        with patch.object(probe.zlib, "compress", side_effect=AssertionError("No codec before admission")):
            with self.assertRaises(MemoryError): probe._codec(np.zeros(16, np.uint8), 1, budget)
        self.assertEqual(budget.reserved, 0)

    def test_foreign_scope_and_changed_source_refuse_publication(self):
        with patch.object(self.workspace, "context", return_value={**self.workspace.context(), "profile": "other"}):
            with self.assertRaises(ValueError): probe.collect(self.workspace, run_id="foreign-r01")
        self.assertFalse((self.workspace.root / "artifacts").exists())
        self.workspace.changed = True
        with self.assertRaisesRegex(ValueError, "source changed"):
            probe.collect(self.workspace, run_id="changed-r01")
        self.assertIsNone(current_policy())

    def test_plan_and_result_are_exclusive_and_source_bound(self):
        with patch("vesper.workspace.Workspace.open", return_value=self.workspace), redirect_stdout(io.StringIO()):
            self.assertEqual(probe.main(["--run-id", "cli-r01"]), 0)
            with self.assertRaises(FileExistsError): probe.main(["--run-id", "cli-r01"])
        report = json.loads(self.workspace.result_path("s2/codec/cli-r01.json").read_text())
        raw = self.workspace.artifact_path(report["plan_artifact"]).read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), report["plan_sha256"])
        self.assertEqual(report["workspace_context"]["git_head"], "a" * 40)
        with self.assertRaises(FileExistsError): probe.collect(self.workspace, run_id="cli-r01")


if __name__ == "__main__": unittest.main()
