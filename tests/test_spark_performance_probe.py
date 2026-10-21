"""CPU-only scope, exact-bit comparison and ownership tests with mock operators."""

from collections.abc import Mapping
from contextlib import contextmanager, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from vesper import spark_performance_probe as probe
from vesper.execution_target import _current_spark_kernel_cache
from vesper.memory import RuntimeMemory, current_policy
from vesper.resources import GiB


class Workspace:
    def __init__(self, root):
        self.root = Path(root)
        self.check = Mock(); self.check_runtime_module = Mock()
        self.changed = False
    def context(self, run_id=None):
        return {"platform": "dgx-spark", "profile": "native", "model": probe.MODEL,
                "revision": probe.REVISION, "run_id": run_id, "git_head": "a" * 40}
    def artifact_path(self, relative, create_parent=False):
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts: raise ValueError("outside own namespace")
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
        value = self.stamp_result(payload, run_id, initial_context=initial_context)
        path = self.result_path(relative, create_parent=True)
        with path.open("x", encoding="utf-8") as stream: json.dump(value, stream)
        return path


class Arrays(Mapping):
    def __init__(self, assets, cached=False): self.assets, self.cached, self.values = assets, cached, {}
    def __len__(self): return 1
    def __iter__(self): return iter(("weight",))
    def __getitem__(self, name):
        if name != "weight": raise KeyError(name)
        if self.cached and name in self.values: return self.values[name]
        self.assets.reads += 1
        value = np.ones(8, np.uint8)
        if self.cached: self.values[name] = value
        return value


class Assets:
    events = []
    def __init__(self, metadata, originals, integrity, *, workspace, fixtures, budget):
        self.workspace, self.budget = workspace, budget
        self.reader = None; self.reads = 0; self.leases = {"bytes": 0}
    def __enter__(self):
        if current_policy() is None: raise AssertionError("Runtime must precede assets")
        if self.budget.cap != GiB: raise AssertionError("Unexpected asset cap")
        self.budget.acquire(64, category="metadata")
        self.reader = SimpleNamespace(stats=lambda: {"logical_reads": self.reads,
            "physical_data_bytes_read": self.reads * 65536, "logical_bytes": self.reads * 8})
        self.events.append("assets_open")
        return self
    def __exit__(self, *_):
        self.events.append("assets_close"); self.reader = None
        self.budget.release(64, category="metadata")
    def arrays_for_layer(self, layer): return Arrays(self)
    @contextmanager
    def upload_arrays(self, layer): yield Arrays(self, cached=True)
    def moe_paths(self, layer): return (Path("gate"), Path("router"), Path("shared"))
    def stats(self): return {"live_array_bytes": 0, "live_arrays": 0}


class MoE:
    def __init__(self, layer):
        self.layer, self.last_attempt = layer, None
        self.experts = SimpleNamespace(store=SimpleNamespace(stats=lambda: {
            "logical_expert_reads": 6, "logical_expert_bytes": 112803840,
            "physical_data_bytes_read": 113000448}))
    def __enter__(self): return self
    def __exit__(self, *_): Assets.events.append("moe_close")
    def forward(self, values, image_span):
        if Pool.failure: raise ValueError("injected native failure")
        self.last_attempt = {"status": "pass", "completed_expert_ids": list(range(6))}
        return {"routing": {"indices": np.arange(6, dtype=np.int32),
            "weights": np.ones(6, np.float32) / np.float32(6)},
            "output_bf16": np.array(values, copy=True),
            "stages": {"accum_fp32": np.ones(5120, np.float32)}}


class Proxy:
    def __init__(self, pool, kind): self.pool, self.kind = pool, kind
    def _open(self):
        if self.kind in self.pool.active: return
        cache = _current_spark_kernel_cache()
        key = ("mock", self.kind)
        if cache.get(key) is None: cache.put(key, b"synthetic unqualified test binary", {})
        with self.pool.assets.upload_arrays(0) as arrays:
            arrays["weight"]; arrays["weight"]
        self.pool.active[self.kind] = (0, SimpleNamespace(stats=lambda: {"owned": True}), None)
    def mixes(self, values, part):
        self._open()
        delta = .5 if Pool.mismatch and not isinstance(self.pool.assets, probe._UncachedUploadAssets) else 0.
        return {"pre_fp32": np.full((1, 1, 4), 1. + delta, np.float32)}
    def pre(self, values, weights): return {"output_bf16": np.full((1, 1, 5120), 0x3f80, np.uint16)}
    def fp8(self, name, values):
        self._open()
        if Pool.change_source: self.pool.assets.workspace.changed = True
        return {"output_bf16": np.zeros((1, 1, 1280), np.uint16)}
    def forward(self, values, image_span):
        with self.pool.moe_factory(0) as owner: return owner.forward(values, image_span)


class Pool:
    failure = mismatch = change_source = False
    def __init__(self, assets): self.assets, self.moe_factory, self.active = assets, MoE, {}
    def __enter__(self): Assets.events.append("pool_open"); return self
    def __exit__(self, *_): Assets.events.append("pool_close"); self.active.clear()
    def proxy(self, kind, layer): return Proxy(self, kind)
    def stats(self): return {"active_layers": list(self.active), "moe_cleanup_owner_retained": False}


class SparkPerformanceProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.workspace = Workspace(self.temp.name)
        Pool.failure = Pool.mismatch = Pool.change_source = False; Assets.events = []
        for target, value in (("platform.system", "Linux"), ("platform.machine", "aarch64"),
                ("vesper.resources.physical_memory", {"available": 128 * GiB}),
                ("vesper.memory.process_memory", {"resident_bytes": 1024})):
            patcher = patch(target, return_value=value); patcher.start(); self.addCleanup(patcher.stop)
        for name, value in (("SparkModelAssets", Assets), ("LayerOperatorPool", Pool)):
            patcher = patch.object(probe, name, value); patcher.start(); self.addCleanup(patcher.stop)
        for name, value in (("host_memory", {"physical_available_bytes": 128 * GiB}),
                ("require_capacity", None)):
            patcher = patch.object(probe, name, return_value=value); patcher.start(); self.addCleanup(patcher.stop)
        self.driver = patch.object(probe, "Driver"); self.inventory = self.driver.start()
        self.inventory.return_value.inventory.return_value = {"devices": [{"name": "NVIDIA GB10"}]}
        self.addCleanup(self.driver.stop)
        watchdog = patch.object(RuntimeMemory, "ENABLE_WATCHDOG", False)
        watchdog.start(); self.addCleanup(watchdog.stop)

    def test_exact_arrays_scope_read_reduction_and_all_cleanup(self):
        report = probe.collect(self.workspace, run_id="layer-r01")
        self.assertEqual(report["status"], "equivalent_layer_components")
        self.assertTrue(report["array_bits_exact"])
        self.assertEqual(report["layers"], [0])
        first, second = report["modes"]
        self.assertEqual(first["reader_after"]["logical_reads"] - first["reader_before"]["logical_reads"], 4)
        self.assertEqual(second["reader_after"]["logical_reads"] - second["reader_before"]["logical_reads"], 2)
        self.assertEqual(first["moe"][0]["source"]["logical_expert_reads"], 6)
        self.assertIn("moe/routing/weights", first["array_hashes"])
        self.assertGreater(second["kernel_cache_after"]["hits"], first["kernel_cache_after"]["hits"])
        self.assertTrue(report["kernel_cache_after_close"]["closed"])
        self.assertEqual(report["kernel_cache_after_close"]["charged_bytes"], 0)
        self.assertTrue(report["owned_resources_closed"]); self.assertTrue(report["memory_runtime_closed"])
        self.assertEqual(report["memory_after"]["charged_host_bytes"], 0)
        self.assertEqual(report["diagnostic_reserved_bytes_after"], 0)
        self.assertFalse(report["full_model_functional_acceptance"])
        self.assertFalse(report["general_performance_claim"])
        self.assertTrue(report["both_modes_use_current_verified_bulk_storage"])
        self.assertIn("empty scoped kernel cache", report["timing_order_limitation"])
        self.assertIsNone(current_policy()); self.assertIsNone(_current_spark_kernel_cache())

    def test_mismatch_does_not_qualify_and_still_closes(self):
        Pool.mismatch = True
        report = probe.collect(self.workspace, run_id="mismatch-r01")
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["array_bits_exact"])
        self.assertIn("hc_mixes/pre_fp32", report["mismatched_arrays"])
        self.assertTrue(report["owned_resources_closed"]); self.assertTrue(report["memory_runtime_closed"])

    def test_native_failure_preserves_partial_timing_private_diagnostic_and_cleanup(self):
        Pool.failure = True
        report = probe.collect(self.workspace, run_id="failure-r01")
        self.assertEqual(report["failure_type"], "ValueError")
        self.assertEqual(report["failure_stage"], "uncached_upload")
        self.assertIn("moe_forward_including_open", report["modes"][0]["timings_seconds"])
        self.assertEqual(report["modes"][0]["pool_after"]["active_layers"], [])
        self.assertEqual(Assets.events[-2:], ["pool_close", "assets_close"])
        self.assertTrue(report["private_failure"]["saved"])
        raw = self.workspace.artifact_path(report["private_failure"]["artifact"]).read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), report["private_failure"]["sha256"])
        self.assertEqual(report["memory_after"]["live_leases"], 0)
        self.assertIsNone(current_policy())

    def test_non_spark_workspace_fails_before_hardware_or_assets(self):
        for context in ({**self.workspace.context(), "platform": "pc-heterogeneous"},
                {**self.workspace.context(), "revision": "0" * 40}):
            with patch.object(self.workspace, "context", return_value=context):
                with self.assertRaises(ValueError): probe.collect(self.workspace, run_id="guard-r01")
        self.inventory.assert_not_called(); self.assertEqual(Assets.events, [])

    def test_source_change_rejects_publication_after_cleanup(self):
        Pool.change_source = True
        with self.assertRaisesRegex(ValueError, "source changed"):
            probe.collect(self.workspace, run_id="changed-r01")
        self.assertIsNone(current_policy()); self.assertEqual(Assets.events[-1], "assets_close")

    def test_hashes_preserve_fp32_bits_and_reject_unbounded_or_noncontiguous_arrays(self):
        first = np.array([0., -0.], np.float32)
        self.assertNotEqual(probe._hash_arrays(first), probe._hash_arrays(first[::-1].copy()))
        with self.assertRaises(ValueError): probe._hash_arrays(np.zeros((3, 3), np.float32).T)
        with patch.object(probe, "DIAGNOSTIC_ALLOWANCE", 4):
            with self.assertRaises(MemoryError): probe._hash_arrays(first)

    def test_cli_writes_bound_exclusive_result(self):
        with patch("vesper.workspace.Workspace.open", return_value=self.workspace), redirect_stdout(io.StringIO()):
            self.assertEqual(probe.main(["--run-id", "cli-r01"]), 0)
            with self.assertRaises(FileExistsError): probe.main(["--run-id", "cli-r01"])
        path = self.workspace.result_path("s1/performance/cli-r01.json")
        result = json.loads(path.read_text())
        self.assertEqual(result["workspace_context"]["git_head"], "a" * 40)
        self.assertEqual(result["schema"], probe.SCHEMA)


if __name__ == "__main__": unittest.main()
