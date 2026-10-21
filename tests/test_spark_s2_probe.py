"""CPU/mock evidence for the bounded native S2 component probe."""

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from vesper import spark_s2_probe as probe
from vesper.execution_target import _current_spark_kernel_cache
from vesper.memory import RuntimeMemory, current_policy
from vesper.resources import GiB


class Workspace:
    def __init__(self, root):
        self.root = Path(root)
        self.check, self.check_runtime_module = Mock(), Mock()
        self.changed = False
    def context(self, run_id=None):
        return {"platform": "dgx-spark", "profile": "native", "model": probe.MODEL,
                "revision": probe.REVISION, "run_id": run_id, "git_head": "a" * 40}
    def artifact_path(self, relative, create_parent=False):
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("outside own namespace")
        result = self.root / "artifacts/dgx-spark" / path
        if create_parent:
            result.parent.mkdir(parents=True, exist_ok=True)
        return result
    def result_path(self, relative, create_parent=False):
        result = self.root / "results/dgx-spark" / relative
        if create_parent:
            result.parent.mkdir(parents=True, exist_ok=True)
        return result
    def stamp_result(self, payload, run_id=None, *, initial_context=None):
        if self.changed:
            raise ValueError("source changed")
        return {**payload, "workspace_context": initial_context or self.context(run_id)}
    def write_result(self, relative, payload, run_id=None, *, initial_context=None):
        value = self.stamp_result(payload, run_id, initial_context=initial_context)
        path = self.result_path(relative, create_parent=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream)
        return path


class Catalog:
    def __init__(self, metadata, originals, integrity, budget, workspace):
        self.budget, self.opened = budget, False
    def __enter__(self):
        self.budget.acquire(64, category="metadata")
        self.opened = True
        return self
    def __exit__(self, *_):
        self.opened = False
        self.budget.release(64, category="metadata")
    def stats(self):
        return {"opened": self.opened, "charged_retained_bytes": self.budget.reserved}


class Backends:
    def __init__(self, workspace):
        self.opened = False
    def __enter__(self):
        self.opened = True
        Assets.events.append("backends_open")
        return self
    def __exit__(self, *_):
        self.opened = False
        Assets.events.append("backends_close")
    def stats(self):
        return {"opened": self.opened, "entries": int(self.opened)}


class Assets:
    events = []
    def __init__(self, metadata, originals, integrity, *, workspace, fixtures, budget, catalog=None):
        self.workspace, self.budget, self.catalog = workspace, budget, catalog
        self.reader, self.reads, self.leases = None, 0, {"bytes": 0}
    def __enter__(self):
        if current_policy() is None:
            raise AssertionError("memory runtime must precede assets")
        if self.catalog is not None and not self.catalog.opened:
            raise AssertionError("catalog must precede assets")
        self.budget.acquire(64, category="metadata")
        self.reader = SimpleNamespace(stats=lambda: {"logical_reads": self.reads})
        self.events.append("assets_open")
        return self
    def __exit__(self, *_):
        self.events.append("assets_close")
        self.reader = None
        self.budget.release(64, category="metadata")
    def moe_paths(self, layer):
        return (Path("gate"), Path("router"), Path("shared"))
    def stats(self):
        return {"live_array_bytes": 0, "live_arrays": 0}


class MoE:
    def __init__(self, layer):
        self.layer, self.last_attempt = layer, None
        self.experts = SimpleNamespace(store=SimpleNamespace(stats=lambda: {
            "logical_expert_reads": 6, "logical_expert_bytes": 112803840,
            "physical_data_bytes_read": 113000448}))
    def __enter__(self):
        return self
    def __exit__(self, *_):
        Assets.events.append("moe_close")
    def forward(self, values, image_span):
        if Pool.failure is not None:
            raise Pool.failure("injected native failure")
        route = np.zeros(6, np.int32) if Pool.bad_route else np.arange(6, dtype=np.int32)
        self.last_attempt = {"status": "pass", "completed_expert_ids": route.tolist()}
        return {"routing": {"indices": route, "weights": np.ones(6, np.float32)},
                "output_bf16": np.array(values, copy=True)}


class Proxy:
    def __init__(self, pool, kind, layer):
        self.pool, self.kind, self.layer = pool, kind, layer
    def _open(self):
        cache = _current_spark_kernel_cache()
        key = ("mock", self.kind)
        if cache.get(key) is None:
            cache.put(key, b"synthetic unqualified test binary", {})
        self.pool.assets.reads += 1
        self.pool.active[self.kind] = (self.layer, SimpleNamespace(stats=lambda: {"owned": True}), None)
    def mixes(self, values, part):
        self._open()
        return {"pre_fp32": np.full((1, 1, 4), 1., np.float32)}
    def pre(self, values, weights):
        return {"output_bf16": np.full((1, 1, 5120), 0x3f80, np.uint16)}
    def fp8(self, name, values):
        self._open()
        if Pool.change_source:
            self.pool.assets.workspace.changed = True
        delta = int(Pool.mismatch and self.pool.backend_pool is not None)
        return {"output_bf16": np.full((1, 1, 1280), self.layer + delta, np.uint16)}
    def grouped(self, values):
        return {"output_bf16": np.full((1, 1, 8, 1024), self.layer, np.uint16)}
    def forward(self, values, image_span):
        with self.pool.moe_factory(self.layer) as owner:
            return owner.forward(values, image_span)


class Pool:
    failure = None
    mismatch = change_source = bad_route = False
    def __init__(self, assets, backend_pool=None):
        self.assets, self.backend_pool, self.moe_factory, self.active = assets, backend_pool, MoE, {}
    def __enter__(self):
        if self.backend_pool is not None and not self.backend_pool.opened:
            raise AssertionError("backends must precede operators")
        Assets.events.append("pool_open")
        return self
    def __exit__(self, *_):
        Assets.events.append("pool_close")
        self.active.clear()
    def proxy(self, kind, layer):
        return Proxy(self, kind, layer)
    def stats(self):
        return {"active_layers": list(self.active), "moe_cleanup_owner_retained": False}


class SparkS2ProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Workspace(self.temp.name)
        Pool.failure = None
        Pool.mismatch = Pool.change_source = Pool.bad_route = False
        Assets.events = []
        for target, value in (("platform.system", "Linux"), ("platform.machine", "aarch64"),
                ("vesper.resources.physical_memory", {"available": 128 * GiB}),
                ("vesper.memory.physical_memory", {"available": 128 * GiB}),
                ("vesper.memory.process_memory", {"resident_bytes": 1024})):
            patcher = patch(target, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        for name, value in (("SparkModelAssets", Assets), ("LayerOperatorPool", Pool),
                            ("_catalog", Catalog), ("_backends", Backends)):
            patcher = patch.object(probe, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        for name, value in (("host_memory", {"physical_available_bytes": 128 * GiB}), ("require_capacity", None)):
            patcher = patch.object(probe, name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(probe, "Driver")
        self.driver = patcher.start()
        self.driver.return_value.inventory.return_value = {"devices": [{"name": "NVIDIA GB10"}]}
        self.addCleanup(patcher.stop)
        patcher = patch.object(RuntimeMemory, "ENABLE_WATCHDOG", False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_exact_components_layer_revisit_plan_and_separate_cache_lifetimes(self):
        report = probe.collect(self.workspace, run_id="components-r01")
        self.assertEqual(report["status"], "equivalent_s2_components")
        self.assertTrue(report["array_bits_exact"])
        self.assertTrue(report["owned_resources_closed"])
        self.assertEqual(report["layer_sequence_per_mode"], [0, 1, 0])
        for mode in report["modes"]:
            self.assertTrue(mode["revisited_layer_array_bits_exact"])
            self.assertEqual(mode["kernel_cache_before"]["entries"], 0)
            self.assertEqual(mode["kernel_cache_before"]["hits"], 0)
            self.assertTrue(mode["kernel_cache_after_close"]["closed"])
            self.assertEqual(mode["memory_after_close"]["charged_host_bytes"], 0)
            self.assertTrue(mode["memory_after_close"]["monitor_thread_stopped"])
            self.assertEqual(mode["diagnostic_reserved_bytes_after"], 0)
            self.assertEqual([row["layer"] for row in mode["moe"]], [0, 1, 0])
            self.assertIn("visit0-layer0/projection_grouped/output_bf16", mode["array_hashes"])
        baseline, candidate = report["modes"]
        self.assertNotIn("catalog_before_close", baseline)
        self.assertEqual(candidate["catalog_after_close"]["charged_retained_bytes"], 0)
        self.assertFalse(candidate["backends_after_close"]["opened"])
        self.assertEqual(Assets.events[-3:], ["pool_close", "backends_close", "assets_close"])
        plan = report["plan"]
        raw = self.workspace.artifact_path(plan["artifact"]).read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), plan["sha256"])
        self.assertEqual(json.loads(raw)["worker_timeout_seconds"], 300)
        self.assertEqual(json.loads(raw)["timeout_enforced_by"], "external worker")
        self.assertFalse(report["general_performance_claim"])
        self.assertFalse(report["full_model_functional_acceptance"])
        self.assertIsNone(current_policy())
        self.assertIsNone(_current_spark_kernel_cache())

    def test_plan_is_exclusive_and_written_before_hardware(self):
        plan_path = self.workspace.artifact_path("s2/components/plan-r01/plan.json")
        self.driver.side_effect = lambda: self.assertTrue(plan_path.exists()) or Mock(inventory=lambda: {})
        probe.collect(self.workspace, run_id="plan-r01")
        calls = self.driver.call_count
        with self.assertRaises(FileExistsError):
            probe.collect(self.workspace, run_id="plan-r01")
        self.assertEqual(self.driver.call_count, calls)

    def test_exact_array_mismatch_is_failure_with_cleanup(self):
        Pool.mismatch = True
        report = probe.collect(self.workspace, run_id="mismatch-r01")
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["array_bits_exact"])
        self.assertTrue(report["owned_resources_closed"])
        self.assertIn("visit0-layer0/projection_wq_a/output_bf16", report["mismatched_arrays"])

    def test_native_exception_and_keyboard_interrupt_cleanup_and_private_failure(self):
        for error, status in ((ValueError, "failed"), (KeyboardInterrupt, "cancelled")):
            with self.subTest(error=error):
                Pool.failure = error
                report = probe.collect(self.workspace, run_id=f"failure-{status}")
                self.assertEqual(report["status"], status)
                self.assertEqual(len(report["modes"]), 1)
                self.assertIn("visit0-layer0/moe", report["modes"][0]["timings_seconds"])
                self.assertTrue(report["owned_resources_closed"])
                raw = self.workspace.artifact_path(report["private_failure"]["artifact"]).read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), report["private_failure"]["sha256"])
                self.assertIsNone(current_policy())

    def test_invalid_native_six_route_does_not_qualify(self):
        Pool.bad_route = True
        report = probe.collect(self.workspace, run_id="route-r01")
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["failure_type"], "ValueError")
        self.assertTrue(report["owned_resources_closed"])

    def test_workspace_and_run_identity_rejected_before_plan_or_hardware(self):
        for run_id in ("../escape", "", "A", None):
            with self.assertRaises(ValueError):
                probe.collect(self.workspace, run_id=run_id)
        with patch.object(self.workspace, "context", return_value={**self.workspace.context(), "platform": "pc-heterogeneous"}):
            with self.assertRaises(ValueError):
                probe.collect(self.workspace, run_id="foreign-r01")
        self.driver.assert_not_called()
        self.assertEqual(Assets.events, [])

    def test_physical_floor_refuses_before_assets_and_cleans_memory(self):
        with patch("vesper.memory.physical_memory", return_value={"available": 31 * GiB}):
            report = probe.collect(self.workspace, run_id="floor-r01")
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["failure_type"], "MemoryError")
        self.assertEqual(Assets.events, [])
        self.assertTrue(report["owned_resources_closed"])
        self.assertIsNone(current_policy())

    def test_source_change_prevents_publication_after_all_cleanup(self):
        Pool.change_source = True
        with self.assertRaisesRegex(ValueError, "source changed"):
            probe.collect(self.workspace, run_id="changed-r01")
        self.assertIsNone(current_policy())
        self.assertIsNone(_current_spark_kernel_cache())
        self.assertEqual(Assets.events[-1], "assets_close")

    def test_hash_storage_bits_and_contiguity_and_admission_bounds(self):
        original = np.array([0., -0.], np.float32)
        self.assertNotEqual(probe._hash_arrays(original), probe._hash_arrays(original[::-1].copy()))
        with self.assertRaises(ValueError):
            probe._hash_arrays(np.zeros((3, 3), np.float32).T)
        with patch.object(probe, "DIAGNOSTIC_ALLOWANCE", 4):
            with self.assertRaises(MemoryError):
                probe._hash_arrays(original)

    def test_cli_writes_exclusive_source_bound_public_result(self):
        with patch("vesper.workspace.Workspace.open", return_value=self.workspace), redirect_stdout(io.StringIO()):
            self.assertEqual(probe.main(["--run-id", "cli-r01"]), 0)
            with self.assertRaises(FileExistsError):
                probe.main(["--run-id", "cli-r01"])
        value = json.loads(self.workspace.result_path("s2/components/cli-r01.json").read_text())
        self.assertEqual(value["schema"], probe.SCHEMA)
        self.assertEqual(value["workspace_context"]["git_head"], "a" * 40)


if __name__ == "__main__":
    unittest.main()
