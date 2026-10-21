"""Synthetic CPU-only metadata census/guard tests; not model qualification."""

from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from vesper import spark_inventory_probe as probe
from vesper.memory import RuntimeMemory, current_policy
from vesper.resources import MemoryBudget, GiB


class Workspace:
    def __init__(self, root):
        self.root, self.changed = Path(root), False
        self.check, self.check_runtime_module = Mock(), Mock()
        self.identity = {"platform": "dgx-spark", "profile": "native", "model": probe.MODEL,
                         "revision": probe.REVISION, "git_head": "a" * 40}

    def context(self, run_id=None):
        return {**self.identity, "run_id": run_id}

    def _path(self, root, relative, create_parent):
        if not isinstance(relative, str) or "\\" in relative or ":" in relative or any(
                p in ("", ".", "..") for p in relative.split("/")) or Path(relative).is_absolute():
            raise ValueError("Invalid relative namespace")
        result = self.root / root / relative
        if create_parent: result.parent.mkdir(parents=True, exist_ok=True)
        return result

    def artifact_path(self, relative, create_parent=False):
        return self._path("artifacts/dgx-spark", relative, create_parent)

    def result_path(self, relative, create_parent=False):
        return self._path("results/dgx-spark", relative, create_parent)

    def stamp_result(self, value, run_id=None, *, initial_context=None):
        if self.changed: raise ValueError("Source changed before publication")
        if initial_context != self.context(run_id): raise ValueError("Source identity differs")
        return {**deepcopy(value), "workspace_context": deepcopy(initial_context)}

    def write_result(self, relative, value, run_id=None, *, initial_context=None):
        stamped = self.stamp_result(value, run_id, initial_context=initial_context)
        path = self.result_path(relative, create_parent=True)
        with path.open("x", encoding="utf-8") as stream: json.dump(stamped, stream)
        return path


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        temp = self.stack.enter_context(tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]))
        self.workspace = Workspace(temp)
        for target in ("vesper.resources.physical_memory", "vesper.memory.physical_memory"):
            self.stack.enter_context(patch(target, return_value={"available": 128 * GiB}))
        self.stack.enter_context(patch.object(RuntimeMemory, "ENABLE_WATCHDOG", False))
        self.stack.enter_context(patch("vesper.memory.process_memory", return_value={"resident_bytes": 0}))
        self.stack.enter_context(patch("urllib.request.urlopen", side_effect=AssertionError("Inventory is offline")))
        root = self.workspace.artifact_path("s1/import-source")
        names = [*(v[0] for v in probe.model_artifacts.FILES.values()), "tree.json"]
        names += [f"headers/model-{i:05d}-of-00048.safetensors.json" for i in range(1, 49)]
        for name in names:
            path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"{}")
        rows = [
            ("layers.0.ffn.experts.0.w1.weight", "I8"),
            ("layers.0.ffn.experts.0.w1.scale", "F8_E8M0"),
            ("layers.1.engram.embed.weight", "F8_E4M3"),
            ("layers.1.engram.embed.scale", "F8_E8M0"),
            ("layers.1.engram.proj.weight", "BF16"),
            ("layers.0.ffn.shared_experts.w1.weight", "F8_E4M3"),
            ("layers.0.ffn.shared_experts.w1.scale", "F8_E8M0"),
            ("layers.0.hc_attn_scale", "F32"),
            ("embed.weight", "BF16"), ("head.weight", "BF16"), ("norm.weight", "BF16"),
            ("mtp.0.ffn.experts.0.w1.weight", "I8"),
            ("vision.blocks.0.weight", "BF16"), ("aligner.0.weight", "BF16"),
            ("image_start.weight", "BF16"), ("image_end.weight", "BF16"),
            ("image_newline.weight", "BF16"), ("unclassified_control", "F32")]
        self.catalog, offset = {}, 8
        shard = "model-00001-of-00048.safetensors"
        for name, dtype in rows:
            size = 4 * probe.model_artifacts.DTYPE_BYTES[dtype]
            self.catalog[name] = {"name": name, "dtype": dtype, "shape": [4], "bytes": size,
                                  "shard_byte_offset": offset, "shard": shard}
            offset += size
        self.identity = {"model": probe.MODEL, "revision": probe.REVISION,
            "catalog_sha256": probe.model_artifacts.CATALOG_SHA256,
            "tensor_count": len(rows), "original_tensor_bytes": offset - 8,
            "shards": [{"shard": f"model-{i:05d}-of-00048.safetensors", "file_bytes": offset if i == 1 else 8}
                       for i in range(1, 49)]}
        self.runtime = None
        self.real_loader = probe.model_artifacts.load_catalog
        self.loader = self.stack.enter_context(patch.object(probe.model_artifacts, "load_catalog", side_effect=self.load))

    def load(self, path):
        self.runtime = current_policy()
        self.assertIsNotNone(self.runtime)
        transient = MemoryBudget(cap=probe.CATALOG_ADMISSION_BYTES, owner="synthetic_loader")
        transient.acquire(probe.CATALOG_ADMISSION_BYTES)
        try: return deepcopy(self.catalog), deepcopy(self.identity)
        finally: transient.release(probe.CATALOG_ADMISSION_BYTES)

    def collect(self, run_id="inventory-r01", **kwargs):
        return probe.collect(self.workspace, run_id=run_id, **kwargs)

    def assert_clean(self):
        self.assertIsNone(current_policy())
        if self.runtime is not None:
            final = self.runtime.snapshot()
            self.assertEqual(final["charged_host_bytes"], 0)
            self.assertEqual(final["live_leases"], 0)
            self.assertEqual(final["live_reclaimers"], 0)
            self.assertTrue(final["monitor_thread_stopped"])

    def test_inventory_partitions_real_named_storage_and_records_scope_and_cleanup(self):
        report = self.collect()
        self.assertEqual(report["tensor_count"], len(self.catalog))
        self.assertEqual(sum(g["bytes"] for g in report["groups"].values()), report["tensor_bytes"])
        self.assertEqual(sum(g["tensor_count"] for g in report["groups"].values()), len(self.catalog))
        self.assertEqual(report["groups"]["routed_experts"]["tensor_count"], 2)
        self.assertEqual(report["groups"]["mtp"]["tensor_count"], 1)
        self.assertEqual(report["groups"]["image_markers"]["tensor_count"], 3)
        self.assertEqual(report["groups"]["routed_experts"]["roles"]["quantization_scale_storage"]["bytes"], 4)
        self.assertEqual(report["groups"]["core_layers"]["roles"]["quantization_scale_storage"]["bytes"], 0)
        self.assertEqual(report["shard_header_and_container_overhead_bytes"], 48 * 8)
        for group in report["groups"].values():
            self.assertEqual(sum(r["bytes"] for r in group["roles"].values()), group["bytes"])
            self.assertEqual(sum(r["tensor_count"] for r in group["dtypes"].values()), group["tensor_count"])
        self.assertFalse(report["classification_is_runtime_dependency_proof"])
        self.assertFalse(report["model_data_removal_authorized"])
        self.assertFalse(report["image_input_qualified_by_this_probe"])
        self.assertIn("image_input", report["desired_capabilities"])
        self.assertEqual(report["tensor_payload_bytes_read"], 0)
        self.assertFalse(report["gpu_work"])
        self.assertGreaterEqual(report["memory_after_runtime_close"]["charged_host_peak_bytes"], 2 * probe.CATALOG_ADMISSION_BYTES)
        self.assertLess(len(json.dumps(report).encode()), probe.MAX_REPORT_BYTES)
        self.assertNotIn(str(self.workspace.root), json.dumps(report))
        self.assert_clean()

    def test_run_writes_exclusively_and_duplicate_refuses_before_catalog(self):
        path = probe.run(self.workspace, run_id="inventory-r01")
        prior = path.read_bytes(); self.loader.reset_mock()
        with self.assertRaises(FileExistsError): probe.run(self.workspace, run_id="inventory-r01")
        self.loader.assert_not_called()
        self.assertEqual(path.read_bytes(), prior)
        self.assert_clean()

    def test_foreign_identity_module_failure_and_bad_names_refuse_before_read(self):
        for key, value in (("platform", "pc-heterogeneous"), ("profile", "approximate"), ("model", "foreign"), ("revision", "bad")):
            original = self.workspace.identity[key]; self.workspace.identity[key] = value
            with self.assertRaises(ValueError): self.collect()
            self.workspace.identity[key] = original
        for run_id in ("../escape", "Upper", "", "a" * 65):
            with self.assertRaises(ValueError): self.collect(run_id)
        with patch.object(self.workspace, "check_runtime_module", side_effect=ValueError("module changed")):
            with self.assertRaisesRegex(ValueError, "module changed"): self.collect()
        self.loader.assert_not_called(); self.assert_clean()

    def test_missing_or_escaping_metadata_refuses_before_authentication(self):
        for relative in ("../outside", "s1/missing", "C:/outside", "/outside"):
            with self.assertRaises(ValueError): self.collect(metadata=relative)
        path = self.workspace.artifact_path("s1/import-source/headers/model-00048-of-00048.safetensors.json")
        path.unlink()
        with self.assertRaises(FileNotFoundError): self.collect()
        self.loader.assert_not_called(); self.assert_clean()

    def test_actual_loader_corrupt_metadata_refuses_and_releases_both_admissions(self):
        def real(path):
            self.runtime = current_policy()
            return self.real_loader(path)
        self.loader.side_effect = real
        with self.assertRaisesRegex(ValueError, "byte count"):
            self.collect()
        self.assert_clean()
        self.assertFalse(self.workspace.result_path("s2/inventory/inventory-r01.json").exists())

    def test_malformed_catalog_or_totals_refuse_without_result(self):
        changes = [("catalog_sha256", "bad"), ("tensor_count", len(self.catalog) + 1), ("original_tensor_bytes", 1)]
        for key, changed in changes:
            original = self.identity[key]; self.identity[key] = changed
            with self.assertRaises(ValueError): self.collect()
            self.identity[key] = original; self.assert_clean()
        self.catalog["head.weight"]["bytes"] += 1
        with self.assertRaises(ValueError): self.collect()
        self.assert_clean()

    def test_floor_refuses_before_loader_and_midrun_pressure_cleans_up(self):
        with patch("vesper.memory.physical_memory", return_value={"available": 32 * GiB - 1}):
            with self.assertRaises(MemoryError): self.collect()
        self.loader.assert_not_called(); self.assert_clean()
        original = self.load
        def pressure(path):
            result = original(path)
            self.runtime.available = lambda: 32 * GiB - 1
            return result
        self.loader.side_effect = pressure
        with self.assertRaises(MemoryError): self.collect()
        self.assert_clean()

    def test_cancellation_and_source_drift_do_not_publish_and_cleanup(self):
        def cancel(path):
            self.runtime = current_policy()
            raise KeyboardInterrupt()
        self.loader.side_effect = cancel
        with self.assertRaises(KeyboardInterrupt): self.collect()
        self.assert_clean()
        def change(path):
            result = self.load(path); self.workspace.changed = True; return result
        self.loader.side_effect = change
        with self.assertRaisesRegex(ValueError, "Source changed"): self.collect()
        self.assert_clean()

    def test_report_bound_and_cli_failure_do_not_publish_private_exception_text(self):
        with patch.object(probe, "MAX_REPORT_BYTES", 64):
            with self.assertRaisesRegex(ValueError, "one MiB"): self.collect()
        self.assert_clean()
        self.loader.side_effect = ValueError("C:/private/hostname 192.168.0.1")
        stream = io.StringIO()
        with patch.object(probe.workspace_module.Workspace, "open", return_value=self.workspace), redirect_stdout(stream):
            code = probe.main(["--run-id", "inventory-r01"])
        self.assertEqual(code, 2)
        value = json.loads(stream.getvalue()); self.assertEqual(value["failure_type"], "ValueError")
        self.assertNotIn("private", stream.getvalue()); self.assertFalse(value["result_written"])
        self.assert_clean()


if __name__ == "__main__": unittest.main()
