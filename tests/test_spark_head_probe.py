"""Synthetic component integrity/driver tests, never official head qualification."""

from contextlib import redirect_stdout
import gc
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from vesper import spark_head_probe as probe
from vesper.formats import bf16_bits, bf16_values
from vesper.memory import current_policy, RuntimeMemory
from vesper.resources import MiB, GiB


class Workspace:
    def __init__(self, root):
        self.root = Path(root)
        self.check, self.check_runtime_module = Mock(), Mock()

    def context(self, run_id=None):
        return {"platform": "dgx-spark", "profile": "native", "model": probe.MODEL, "revision": probe.REVISION,
                "run_id": run_id, "git_head": "b" * 40}

    def artifact_path(self, relative, create_parent=False):
        relative = Path(relative)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("outside own artifact namespace")
        path = self.root / "artifacts/dgx-spark" / relative
        if create_parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def result_path(self, relative, create_parent=False):
        path = self.root / "results/dgx-spark" / relative
        if create_parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def stamp_result(self, payload, run_id=None, *, initial_context=None):
        return {**payload, "workspace_context": initial_context or self.context(run_id)}

    def write_result(self, relative, payload, run_id=None, *, initial_context=None):
        path = self.result_path(relative, create_parent=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(self.stamp_result(payload, run_id, initial_context=initial_context), stream)
        return path

    def check_result(self, relative):
        data = json.loads(self.result_path(relative).read_text())
        if data["workspace_context"]["model"] != probe.MODEL:
            raise ValueError("foreign result")


class Native:
    """Small CPU arithmetic seam for orchestration tests, not the candidate."""
    def __init__(self, rows, *, block_rows):
        self.rows, self.block_rows = rows, block_rows
        self.blas_provenance = {"version": 130101, "math_mode": 0, "TF32_enabled": False}
        self.calls = 0

    def __enter__(self):
        if current_policy() is None:
            raise AssertionError("shared memory policy must already exist")
        return self

    def project(self, x):
        raw = self.rows(0, probe.VOCAB)
        weight = bf16_values(raw)
        promoted = bf16_values(x)
        logits = np.ascontiguousarray(promoted @ weight.T, dtype=np.float32)
        self.calls += 1
        return {"promoted_fp32": promoted, "logits_fp32": logits,
                "greedy_ids_i64": np.argmax(logits, axis=-1).astype(np.int64)}

    def stats(self):
        return {"device_owned_bytes": 0, "host_reserved_bytes": 0, "device_peak_owned_bytes": MiB,
                "calls": self.calls, "rows_read": self.calls * probe.VOCAB, "block_rows": probe.ROWS,
                "full_vocabulary_rows": probe.VOCAB, "head_compute_dtype": "float32", "TF32_enabled": False,
                "qualification": "synthetic orchestration seam only"}

    def __exit__(self, *_):
        pass


class Reference(Native):
    def __init__(self, rows, source, check):
        super().__init__(rows, block_rows=probe.ROWS)
        self.source, self.check = source, check

    def __enter__(self):
        super().__enter__()
        _, self.source_identity = probe.load_head_method(self.source)
        self.blas = {**self.blas_provenance, "handle_source": "torch.cuda.current_blas_handle", "preferred_backend": "Cublas"}
        self.torch = SimpleNamespace(__version__="2.14.1+cu130", version=SimpleNamespace(cuda="13.0"), cuda=SimpleNamespace(
            max_memory_allocated=lambda n: 0, max_memory_reserved=lambda n: 0,
            memory_allocated=lambda n: 0, memory_reserved=lambda n: 0))
        return self

    def run(self, x):
        self.check()
        return self.project(x[:, -1].copy())

    def close(self):
        pass


class HeadProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.addCleanup(self.temp.cleanup)
        self.ws = Workspace(self.temp.name)
        self.metadata, self.originals, self.integrity = [self.ws.artifact_path("s1/head-component/" + part)
                                                       for part in ("metadata", "originals", "integrity")]
        self.metadata.mkdir(parents=True); self.originals.mkdir(parents=True)
        self.weight = bf16_bits(np.arange(128, dtype=np.float32).reshape(16, 8) / np.float32(128))
        raw = bytes(4096) + self.weight.tobytes() + bytes(16)
        self.source = self.originals / "model-00043-of-00048.safetensors"
        self.source.write_bytes(raw)
        self.expected = {"file": self.source.name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        self.tensor = {"name": "head.weight", "dtype": "BF16", "shape": [16, 8], "bytes": self.weight.nbytes,
                       "shard": self.source.name, "shard_byte_offset": 4096}
        inventory = [{"file": f"model-{i:05d}-of-00048.safetensors", "bytes": 32, "sha256": "a" * 64}
                     if i != 43 else self.expected for i in range(1, 49)]
        self.reference_source = self.ws.artifact_path("s1/head-component/reference/model.py", create_parent=True)
        self.reference_source.write_text("class ParallelHead:\n    def forward(self, x):\n        return F.linear(x[:, -1].float(), self.weight)\n", encoding="utf-8")
        license = self.reference_source.parent / "LICENSE"; license.write_bytes(b"synthetic license seam")
        hashes = {"model.py": hashlib.sha256(self.reference_source.read_bytes()).hexdigest(),
                  "LICENSE": hashlib.sha256(license.read_bytes()).hexdigest()}
        self.inventory = Mock(return_value={"device_count": 1, "name": "NVIDIA GB10", "compute_capability": [12, 1],
            "ordinal": 0, "driver_api_version": 13000, "cuda_capacity_added_to_physical_ram": False,
            "attributes": {"unified_addressing": 1, "managed_memory": 1, "integrated": 1,
                           "pageable_memory_access": 1, "host_page_tables": 1}})
        patches = [patch.object(probe, "VOCAB", 16), patch.object(probe, "WIDTH", 8),
            patch("vesper.spark_output.WIDTH", 8), patch.object(probe, "HEAD_SHA256", hashlib.sha256(self.weight.tobytes()).hexdigest()),
            patch.object(probe, "_metadata_paths"), patch.object(probe, "load_catalog", return_value=({"head.weight": self.tensor}, {})),
            patch.object(probe, "shard_inventory", return_value=inventory), patch.object(probe, "REFERENCE_HASHES", hashes),
            patch("vesper.resources.physical_memory", return_value={"available": 100 * GiB}),
            patch("vesper.memory.physical_memory", return_value={"available": 100 * GiB}),
            patch("vesper.memory.process_memory", return_value={"resident_bytes": MiB}),
            patch.object(probe, "process_memory", return_value={"resident_bytes": MiB}),
            patch.object(probe, "host_memory", return_value={"physical_available_bytes": 100 * GiB, "cgroup_headroom_bytes": None}),
            patch.object(probe.shutil, "disk_usage", return_value=SimpleNamespace(free=4 * 1024**4)),
            patch.object(probe.platform, "system", return_value="Linux"), patch.object(probe.platform, "machine", return_value="aarch64"),
            patch.object(probe, "Driver", return_value=SimpleNamespace(inventory=self.inventory)),
            patch.object(RuntimeMemory, "ENABLE_WATCHDOG", False),
            patch.object(probe, "_EXECUTION_MODE", None),
            patch("urllib.request.urlopen", side_effect=AssertionError("component must stay offline"))]
        for item in patches:
            item.start(); self.addCleanup(item.stop)
        self.addCleanup(self.assert_closed)

    def assert_closed(self):
        gc.collect()
        self.assertIsNone(current_policy())

    def register(self):
        return probe.register_component(self.metadata, self.originals, self.integrity, workspace=self.ws)

    def reader(self):
        return probe.HeadRows(self.metadata, self.originals, self.integrity, workspace=self.ws)

    def test_head_only_plan_has_no_payload_registration_or_network(self):
        with patch.object(probe, "register_blob") as register:
            report = probe.component_plan(self.metadata, self.originals, self.integrity, workspace=self.ws)
        self.assertEqual(report["required_complete_original_shards"], 1)
        self.assertFalse(report["full_checkpoint_available_or_registered"])
        self.assertFalse(report["source_payload_opened"])
        self.assertTrue(report["selected_shard_size_matches"])
        register.assert_not_called()
        self.assertFalse(self.integrity.exists())

    def test_register_real_integrity_then_read_exact_native_rows_without_other47(self):
        report = self.register()
        self.assertEqual(len(list(self.originals.iterdir())), 1)
        self.assertTrue(report["whole_selected_shard_sha256_verified"])
        self.assertTrue(report["head_tensor_sha256_verified"])
        self.assertFalse(report["full_checkpoint_available_or_registered"])
        with self.reader() as rows:
            actual = rows(7, 3)
            np.testing.assert_array_equal(actual, self.weight[7:10])
            self.assertFalse(actual.flags.writeable)
            self.assertFalse(rows.stats()["whole_host_head_materialized"])
            del actual
        self.assertEqual(rows.budget.reserved, 0)
        with self.assertRaises(FileExistsError):
            self.register()

    def test_readonly_component_allows_multiple_source_hardlinks(self):
        original_stat = Path.stat
        def linked(path, *args, **kwargs):
            value = original_stat(path, *args, **kwargs)
            if path == self.source:
                attrs = {n: getattr(value, n) for n in dir(value) if n.startswith("st_")}
                attrs["st_nlink"] = 2
                return SimpleNamespace(**attrs)
            return value
        with patch.object(Path, "stat", new=linked):
            report = self.register()
            with self.reader() as rows:
                value = rows(0, 1); del value
        self.assertEqual(report["source_link_count_observed"], 2)
        self.assertTrue(report["source_hardlinks_allowed"])

    def test_partial_or_wrong_whole_sha_never_registers_original(self):
        raw = self.source.read_bytes()
        self.source.write_bytes(raw[:-1])
        with self.assertRaises(ValueError):
            self.register()
        self.assertFalse(self.integrity.exists())
        changed = bytearray(raw); changed[-1] ^= 1; self.source.write_bytes(changed)
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.register()
        self.assertFalse((self.integrity / "manifest.json").exists())

    def test_pinned_tensor_sha_cannot_be_replaced_by_a_manifest_claim(self):
        with patch.object(probe, "HEAD_SHA256", "0" * 64):
            with self.assertRaisesRegex(ValueError, "head tensor SHA"):
                self.register()
        self.assertFalse((self.integrity / "manifest.json").exists())

    def test_block_corruption_or_receipt_tamper_refuses_and_releases_metadata(self):
        self.register()
        with self.source.open("r+b") as stream:
            stream.seek(4096); stream.write(b"tamper")
        rows = self.reader()
        with self.assertRaises(ValueError):
            rows.__enter__()
        self.assertEqual(rows.budget.reserved, 0)
        self.assertIsNone(rows.blob)

    def test_lease_follows_actual_array_owner_after_reader_close(self):
        self.register()
        rows = self.reader().__enter__()
        owner = rows(2, 2); borrowed = owner[:, 1:3]
        size = owner.nbytes
        del owner
        rows.__exit__()
        gc.collect()
        self.assertEqual(rows.budget.reserved, size)
        np.testing.assert_array_equal(borrowed, self.weight[2:4, 1:3])
        del borrowed
        gc.collect()
        self.assertEqual(rows.budget.reserved, 0)

    def test_rows_reject_out_of_scope_requests_without_publication(self):
        self.register()
        with self.reader() as rows:
            for start, count in ((-1, 1), (0, 513), (16, 1), (0, True)):
                with self.subTest(start=start, count=count), self.assertRaises(ValueError):
                    rows(start, count)
            self.assertEqual(rows.stats()["row_calls"], 0)

    def test_original_head_ast_and_license_are_pinned_without_source_fetch(self):
        node, source = probe.load_head_method(self.reference_source)
        self.assertEqual(node.name, "forward")
        self.assertFalse(source["method_body_modified"])
        self.assertEqual(source["world_size"], 1)
        self.reference_source.write_text("class ParallelHead: pass\n")
        with self.assertRaisesRegex(ValueError, "pinned release"):
            probe.load_head_method(self.reference_source)

    def test_corpus_covers_both_seeds_patterns_batches_with_finite_bf16(self):
        cases = list(probe.cases())
        self.assertEqual(len(cases), 8)
        self.assertEqual({meta["batch"] for meta, _ in cases}, {1, 2})
        self.assertEqual({meta["seed"] for meta, _ in cases if meta["pattern"] == "random"}, {17, 2026})
        for meta, x in cases:
            self.assertEqual(x.shape, (meta["batch"], 1, 8))
            self.assertEqual(x.dtype, np.uint16)
            self.assertTrue(np.isfinite(bf16_values(x)).all())

    def executions(self):
        self.register()
        reports = []
        for mode, instance in (("native", "native-instance"), ("reference", "reference-instance")):
            with patch.object(probe, "SparkOutputCandidate", Native), patch.object(probe, "ReferenceHead", Reference), \
                 patch.object(probe, "_EXECUTION_MODE", None), patch.object(probe, "PROCESS_INSTANCE", instance):
                report = probe.collect(mode, self.ws, run_id=mode + "-r01")
            self.assertEqual(report["status"], mode + "_execution_pass", report)
            relative = mode + ".json"; self.ws.write_result(relative, report, run_id=mode)
            reports.append((relative, report))
        return reports

    def test_separate_execution_reports_compare_all_logits_exact_and_do_not_claim_model_quality(self):
        (_, native), (_, reference) = self.executions()
        result = probe.compare_runs(self.ws, "native.json", "reference.json")
        self.assertEqual(result["status"], "pass")
        self.assertEqual(len(result["checks"]), 24)
        self.assertTrue(result["original_kernel_parity_qualified"])
        self.assertFalse(result["full_model_quality_qualified"])
        self.assertFalse(result["general_performance_claim"])
        self.assertTrue(native["memory_runtime_closed"])
        self.assertTrue(reference["memory_runtime_closed"])
        self.assertEqual(reference["reference_admission_bytes"], 6 * GiB)
        self.assertEqual(native["memory_after_runtime_close"]["charged_host_bytes"], 0)
        self.assertEqual(reference["memory_after_runtime_close"]["charged_host_bytes"], 0)

    def test_single_fp32_bit_change_is_not_accepted_as_tolerance(self):
        (_, native), _ = self.executions()
        arrays = probe._load(self.ws, native)
        arrays["c00_logits_fp32"].view(np.uint32)[0, 0] ^= 1
        native["diagnostics"] = probe._save(self.ws, "changed-logit", arrays)
        self.ws.result_path("native.json").write_text(json.dumps(self.ws.stamp_result(native)), encoding="utf-8")
        result = probe.compare_runs(self.ws, "native.json", "reference.json")
        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["original_kernel_parity_qualified"])
        self.assertTrue(any(c.get("mismatches") == 1 for c in result["checks"]))

    def test_compare_refuses_same_process_foreign_source_or_cleanup_failure(self):
        (_, native), (_, reference) = self.executions()
        original = json.loads(self.ws.result_path("reference.json").read_text())
        for change in ({"execution_process_instance": native["execution_process_instance"]}, {"model": "foreign/model"},
                       {"owned_resources_closed": False}, {"reference": {"method_body_modified": True}},
                       {"observed_hardware": None}, {"reference": {**original["reference"], "torch_version": "unknown"}},
                       {"reference": {**original["reference"], "torch_revision": "0" * 40}},
                       {"reference": {**original["reference"], "torch_cuda_version": "12.4"}},
                       {"blas": {**original["blas"], "handle_source": "own unrelated handle"}},
                       {"blas": {**original["blas"], "preferred_backend": "CublasLt"}}):
            with self.subTest(change=change):
                self.ws.result_path("reference.json").write_text(json.dumps({**original, **change}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    probe.compare_runs(self.ws, "native.json", "reference.json")
        self.ws.result_path("reference.json").write_text(json.dumps(original), encoding="utf-8")

    def test_native_claim_refuses_framework_contamination_and_second_execution(self):
        with patch.dict("sys.modules", {"torch": SimpleNamespace()}):
            with self.assertRaises(Unsupported := probe.Unsupported):
                probe._claim("native")
        probe._claim("native")
        with self.assertRaises(Unsupported):
            probe._claim("reference")

    def test_resource_guard_and_scope_refuse_before_driver_or_artifact_registration(self):
        self.ws.check_runtime_module.side_effect = ValueError("foreign imported module")
        with self.assertRaisesRegex(ValueError, "foreign imported module"):
            probe.collect("native", self.ws, run_id="blocked")
        self.inventory.assert_not_called()
        self.assertFalse(self.integrity.exists())

    def test_cli_plan_writes_relative_result_without_claiming_execution(self):
        with patch("vesper.workspace.Workspace.open", return_value=self.ws), redirect_stdout(io.StringIO()) as output:
            code = probe.main(["plan", "--run-id", "plan-r01"])
        self.assertEqual(code, 0)
        status = json.loads(output.getvalue())
        self.assertEqual(status["result"], "results/dgx-spark/s1/head/plan-r01.json")
        self.assertFalse(self.integrity.exists())
        self.inventory.assert_not_called()
        report = json.loads(self.ws.result_path("s1/head/plan-r01.json").read_text())
        self.assertFalse(report["original_kernel_parity_qualified"])
        self.assertFalse(report["plan"]["source_payload_opened"])


if __name__ == "__main__":
    unittest.main()
