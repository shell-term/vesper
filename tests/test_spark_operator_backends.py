"""CPU/mock ownership tests; no real compiler or GPU numerical qualification."""

from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from vesper import spark_operator_backends as backends
from vesper.execution_target import SPARK_TARGET
from vesper.hc import NativeHC
from vesper.memory import RuntimeMemory, MemoryPolicy, RuntimeCancelled, current_policy
from vesper.projections import NativeProjections, FP8_LAYOUT
from vesper.resources import MiB, GiB
from tests.test_execution_target import mock_compile


class Device:
    fail_load = fail_free = False
    def __init__(self, driver, ordinal, *, cap, free_floor):
        self.driver, self.ordinal, self.cap, self.free_floor = driver, ordinal, cap, free_floor
        self.allocations, self.modules = {}, []
        self.context, self.peak, self.next_pointer = True, 0, 1
    def current(self): pass
    def allocate(self, size):
        current_policy().reserve_device(self.ordinal, size)
        pointer = self.next_pointer; self.next_pointer += 1
        self.allocations[pointer] = size
        self.peak = max(self.peak, sum(self.allocations.values()))
        return pointer
    def free(self, pointer):
        if self.fail_free:
            Device.fail_free = False
            raise RuntimeError("injected failed free")
        current_policy().release_device(self.ordinal, self.allocations.pop(pointer))
    def load(self, binary):
        if self.fail_load: raise ValueError("injected failed module load")
        self.modules.append(7); return 7
    def close(self):
        if self.allocations or self.modules: raise AssertionError("Backend must clean its owned resources first")
        self.context = False
    def download(self, pointer, shape, dtype): return np.zeros(shape, dtype)
    def launch(self, *_args, **_kwargs): pass


class Blas:
    def __init__(self): self.version, self.math_mode, self.handle = 130101, 0, 1
    def workspace(self, *_): pass
    def close(self): self.handle = 0


class SparkBackendTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(RuntimeMemory, "ENABLE_WATCHDOG", False))
        self.stack.enter_context(patch.object(RuntimeMemory, "CONTROL_METADATA_BYTES", 0))
        for target in ("vesper.resources.physical_memory", "vesper.memory.physical_memory"):
            self.stack.enter_context(patch(target, return_value={"available": 128 * GiB}))
        self.stack.enter_context(patch("vesper.execution_target.platform.system", return_value="Linux"))
        self.stack.enter_context(patch("vesper.execution_target.platform.machine", return_value="aarch64"))
        self.workspace = SimpleNamespace(check=Mock(), check_runtime_module=Mock(), context=lambda: {
            "platform": "dgx-spark", "profile": "native", "model": backends.MODEL,
            "revision": backends.REVISION})
        self.inventory = {"driver_api_version": 13000, "devices": [{"ordinal": 0, "name": "NVIDIA GB10",
            "compute_capability": "12.1", "total_bytes": 128 * GiB}]}
        self.driver = Mock()
        self.driver.inventory.return_value = self.inventory
        self.driver.check.side_effect = lambda code: None
        self.compiler = Mock(side_effect=mock_compile)
        Device.fail_load = Device.fail_free = False
        for name, value in (("Driver", Mock(return_value=self.driver)), ("Device", Device),
                ("Cublas", Blas), ("GroupedBF16", Blas), ("compile_probe", self.compiler),
                ("load_toolkit_library", Mock(return_value=object())),
                ("_compiler_library_identity", Mock(return_value=(1, "/private/compiler", 2)))):
            self.stack.enter_context(patch.object(backends, name, value))

    def runtime(self, cap=2 * GiB):
        return RuntimeMemory(MemoryPolicy(host_cap_bytes=cap, device_caps={0:GiB},
            pinned_cap_bytes=128 * MiB, unified_device_memory=True),
            available=lambda:128 * GiB, process=lambda:{"resident_bytes": 0})

    def test_two_kinds_reuse_exact_backend_and_release_all_retention(self):
        with self.runtime() as memory:
            with backends.SparkOperatorBackends(self.workspace) as pool:
                for kind in ("hc", "projection"):
                    first = None
                    for _ in range(3):
                        with pool.acquire(kind, 0, SPARK_TARGET) as backend:
                            if first is None: first = backend
                            self.assertIs(backend, first)
                            pointer = backend.device.allocate(1024)
                            backend.device.free(pointer)
                stats = pool.stats()
                self.assertEqual((stats["opens"], stats["reuse_hits"], stats["borrows"]), (2, 4, 6))
                self.assertEqual(stats["host_admission_bytes"], 640 * MiB)
                self.assertEqual(stats["device_owned_bytes"], 2 * backends.GATE_BLAS_WORKSPACE)
                self.assertEqual(self.compiler.call_count, 2)
                self.assertNotIn("/private/compiler", str(stats))
                self.assertFalse(stats["qualification_receipt_issued"])
            self.assertEqual(memory.host, 0); self.assertFalse(memory.reclaimers)
            self.assertEqual(pool.stats()["entries"], 0)

    def test_pressure_evicts_idle_but_never_borrowed_backend(self):
        with self.runtime(cap=GiB) as memory:
            with backends.SparkOperatorBackends(self.workspace) as pool:
                with pool.acquire("hc", 0, SPARK_TARGET):
                    with self.assertRaises(MemoryError): memory.reserve(950 * MiB, "scratch", "probe")
                    self.assertEqual(pool.stats()["entries"], 1)
                lease = memory.reserve(950 * MiB, "scratch", "probe")
                self.assertEqual(pool.stats()["entries"], 0)
                lease.release()
                with pool.acquire("hc", 0, SPARK_TARGET): pass
                self.assertEqual(pool.stats()["opens"], 2)

    def test_identity_change_refuses_reuse_and_cleanup_still_succeeds(self):
        with self.runtime() as memory:
            with backends.SparkOperatorBackends(self.workspace) as pool:
                with pool.acquire("hc", 0, SPARK_TARGET): pass
                with patch.object(backends, "_compiler_library_identity", return_value=(2, "changed")):
                    with self.assertRaisesRegex(ValueError, "identity changed"):
                        with pool.acquire("hc", 0, SPARK_TARGET): pass
                self.assertTrue(pool.stats()["poisoned"])
            self.assertEqual(memory.host, 0)

    def test_partial_initialization_closes_and_allows_fresh_retry(self):
        with self.runtime() as memory:
            with backends.SparkOperatorBackends(self.workspace) as pool:
                Device.fail_load = True
                with self.assertRaisesRegex(ValueError, "module load"):
                    with pool.acquire("hc", 0, SPARK_TARGET): pass
                self.assertEqual(memory.host, 0)
                self.assertFalse(pool.stats()["poisoned"])
                Device.fail_load = False
                with pool.acquire("hc", 0, SPARK_TARGET): pass

    def test_cancellation_refuses_new_borrow_and_preserves_cleanup(self):
        with self.runtime() as memory:
            with backends.SparkOperatorBackends(self.workspace) as pool:
                with pool.acquire("hc", 0, SPARK_TARGET): pass
                before = pool.stats()["borrows"]
                memory.cancel()
                with self.assertRaises(RuntimeCancelled):
                    with pool.acquire("hc", 0, SPARK_TARGET): pass
                self.assertEqual(pool.stats()["borrows"], before)
            self.assertEqual(memory.host, 0)
            self.assertFalse(memory.reclaimers)

    def test_failed_layer_cleanup_poison_prevents_reuse_until_owner_closes(self):
        with self.runtime() as memory:
            pool = backends.SparkOperatorBackends(self.workspace).__enter__()
            with self.assertRaisesRegex(RuntimeError, "live allocations"):
                with pool.acquire("hc", 0, SPARK_TARGET) as backend:
                    backend.device.allocate(1024)
            with self.assertRaisesRegex(RuntimeError, "requires cleanup"):
                with pool.acquire("hc", 0, SPARK_TARGET): pass
            pool.close()
            self.assertEqual(memory.host, 0)

    def test_failed_backend_free_retains_context_and_explicit_close_retries(self):
        with self.runtime() as memory:
            pool = backends.SparkOperatorBackends(self.workspace).__enter__()
            with pool.acquire("hc", 0, SPARK_TARGET): pass
            Device.fail_free = True
            with self.assertRaisesRegex(RuntimeError, "explicit retry"): pool.close()
            self.assertEqual(pool.stats()["entries"], 1)
            self.assertTrue(pool.entries["hc"].device.context)
            pool.close()
            self.assertEqual(memory.host, 0)

    def test_failed_reclaim_after_workspace_free_cannot_publish_another_borrow(self):
        with self.runtime() as memory:
            pool = backends.SparkOperatorBackends(self.workspace).__enter__()
            with pool.acquire("hc", 0, SPARK_TARGET) as backend: pass
            original_close = backend.blas.close
            def fail_destroy():
                backend.blas.handle = 0
                raise RuntimeError("injected BLAS destruction failure")
            with patch.object(backend.blas, "close", side_effect=fail_destroy):
                with self.assertRaisesRegex(RuntimeError, "BLAS destruction failure"):
                    pool.reclaim(1)
            self.assertEqual(backend.device.allocations, {})
            self.assertTrue(backend.device.context)
            self.assertTrue(backend.poisoned); self.assertTrue(pool.poisoned)
            self.assertIs(pool.entries["hc"], backend)
            self.assertEqual(backend.blas.handle, 1)
            entered = []
            with self.assertRaisesRegex(RuntimeError, "requires cleanup"):
                with pool.acquire("hc", 0, SPARK_TARGET): entered.append(True)
            self.assertEqual(entered, [])
            # The original owner remains reachable for a healthy cleanup retry.
            backend.blas.close = original_close
            pool.close()
            self.assertEqual(memory.host, 0)
            self.assertFalse(memory.reclaimers)

    def test_incomplete_idle_backend_is_rejected_before_borrow_body(self):
        with self.runtime() as memory:
            pool = backends.SparkOperatorBackends(self.workspace).__enter__()
            with pool.acquire("hc", 0, SPARK_TARGET) as backend: pass
            backend.device.free(next(iter(backend.device.allocations)))
            entered = []
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                with pool.acquire("hc", 0, SPARK_TARGET): entered.append(True)
            self.assertEqual(entered, [])
            self.assertTrue(pool.poisoned)
            pool.close()
            self.assertEqual(memory.host, 0)

    def test_real_hc_entry_wiring_reuses_backend_but_not_layer_weights(self):
        arrays = {}
        for layer in (0, 1):
            for part in ("attn", "ffn"):
                for suffix, shape in (("fn", (24, 20480)), ("scale", (3,)), ("base", (24,))):
                    arrays[f"layers.{layer}.hc_{part}_{suffix}"] = np.zeros(shape, np.float32)
        with self.runtime() as memory:
            with backends.SparkOperatorBackends(self.workspace) as pool:
                for layer in (0, 1):
                    with NativeHC(arrays, layer=layer, profile="backbone_probe", execution_target=SPARK_TARGET,
                            backend_pool=pool) as operator:
                        self.assertEqual(len(operator.weights), 6)
                        self.assertGreater(len(operator.device.allocations), 1)
                    self.assertEqual(len(pool.entries["hc"].device.allocations), 1)
                self.assertEqual(self.compiler.call_count, 1)
            self.assertEqual(memory.host, 0)

    def test_projection_entry_wiring_keeps_original_conversion_path(self):
        arrays = {f"layers.0.attn.{name}.{suffix}": np.zeros(1, np.uint8)
            for name in (*FP8_LAYOUT, "wo_a") for suffix in ("weight", "scale")}
        with patch("vesper.projections.validate_projection_weights"):
            with self.runtime() as memory:
                with backends.SparkOperatorBackends(self.workspace) as pool:
                    with NativeProjections(arrays, profile="backbone_probe", execution_target=SPARK_TARGET,
                            backend_pool=pool) as operator:
                        self.assertEqual(len(operator.weights), 8)
                        self.assertEqual(operator.device.allocations[operator.wo_a], 8192 * 4096 * 2)
                    self.assertEqual(len(pool.entries["projection"].device.allocations), 1)
                self.assertEqual(memory.host, 0)

    def test_explicit_target_runtime_and_single_borrow_rules(self):
        with self.assertRaises(ValueError): backends.SparkOperatorBackends(self.workspace).__enter__()
        pool = backends.SparkOperatorBackends(self.workspace)
        with self.assertRaises(RuntimeError):
            with pool.acquire("hc", 0, SPARK_TARGET): pass
        with self.runtime():
            with pool:
                for kind, target in (("bad", SPARK_TARGET), ("hc", None)):
                    with self.assertRaises(ValueError):
                        with pool.acquire(kind, 0, target): pass
                with pool.acquire("hc", 0, SPARK_TARGET):
                    with self.assertRaisesRegex(RuntimeError, "simultaneous"):
                        with pool.acquire("hc", 0, SPARK_TARGET): pass


if __name__ == "__main__": unittest.main()
