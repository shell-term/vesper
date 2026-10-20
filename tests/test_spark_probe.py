"""CPU-only safety and plumbing checks; no CUDA libraries or hardware needed."""

import ctypes as C
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import Mock, patch

from vesper import spark_probe as spark


def memory(available=80 * spark.GiB, cgroup=None):
    return {"physical_total_bytes": 128 * spark.GiB, "physical_available_bytes": available,
            "shared_pool_budget_bytes": 96 * spark.GiB, "available_floor_bytes": 32 * spark.GiB,
            "ram_plus_vram_is_not_capacity": True, "cgroup_headroom_bytes": cgroup}


def device_info():
    return {"device_count": 1, "name": "NVIDIA GB10", "compute_capability": [12, 1],
            "attributes": {"integrated": 1, "unified_addressing": 1, "managed_memory": 1,
                           "pageable_memory_access": 1, "concurrent_managed_access": 1, "host_page_tables": 1}}


class Workspace:
    def __init__(self, platform="dgx-spark", profile="native"):
        self.check = Mock()
        self.root = Path(spark.__file__).resolve().parents[2]
        self.context_value = {"platform": platform, "profile": profile, "git_head": "a" * 40,
                        "dirty_diff_sha256": "b" * 64, "binding_id": "test-binding",
                        "model": "test-only", "revision": "c" * 40, "run_id": "test"}

    def stamp_result(self, value):
        return {**value, "workspace_context": self.context_value}

    def context(self, run_id=None):
        return {**self.context_value, "run_id": run_id or "test"}

    def check_runtime_module(self, module_file, relative):
        from vesper.workspace import Workspace as BoundWorkspace
        return BoundWorkspace.check_runtime_module(self, module_file, relative)


class Driver:
    """Emulate only the owned buffer operations with tiny CPU allocations."""
    def __init__(self):
        self.events = []
        self.buffers = []
        self.corrupt = False
        self.fail_host = False
        self.fail_free = False
        self.cuda_free = 80 * spark.GiB
        self.cuda_total = 128 * spark.GiB
        self.cuda_memory_status = 0

    def inventory(self):
        return device_info()

    def cuCtxGetCurrent(self, output):
        output._obj.value = 987
        return 0

    def cuCtxCreate_v2(self, output, flags, ordinal):
        self.events.append("context_create")
        output._obj.value = 123
        return 0

    def cuMemGetInfo_v2(self, free, total):
        free._obj.value, total._obj.value = self.cuda_free, self.cuda_total
        return self.cuda_memory_status

    def cuMemAlloc_v2(self, output, size):
        self.events.append("device_allocate")
        data = C.create_string_buffer(size)
        self.buffers.append(data)
        output._obj.value = C.addressof(data)
        self.device_address, self.size = C.addressof(data), size
        return 0

    def cuMemHostAlloc(self, output, size, flags):
        self.events.append("pinned_allocate")
        if self.fail_host:
            return 2
        data = C.create_string_buffer(size)
        self.buffers.append(data)
        output._obj.value = C.addressof(data)
        return 0

    def cuMemcpyHtoD_v2(self, dst, src, size):
        C.memmove(dst.value, src.value, size)
        return 0

    def cuMemcpyDtoH_v2(self, dst, src, size):
        C.memmove(dst.value, src.value, size)
        return 0

    def cuModuleLoadData(self, output, image):
        self.events.append("module_load")
        output._obj.value = 456
        return 0

    def cuModuleGetFunction(self, output, module, name):
        output._obj.value = 789
        return 0

    def cuLaunchKernel(self, *_):
        self.events.append("kernel_launch")
        view = (C.c_uint32 * (self.size // 4)).from_address(self.device_address)
        for i in range(len(view)):
            view[i] ^= 0x5A5A5A5A
        if self.corrupt:
            view[0] ^= 1
        return 0

    def cuCtxSynchronize(self):
        return 0

    def cuModuleUnload(self, _):
        self.events.append("module_unload")
        return 0

    def cuMemFree_v2(self, _):
        self.events.append("device_free")
        return 2 if self.fail_free else 0

    def cuMemFreeHost(self, _):
        self.events.append("pinned_free")
        return 0

    def cuCtxDestroy_v2(self, _):
        self.events.append("context_destroy")
        return 0

    def cuCtxSetCurrent(self, pointer):
        self.events.append(("restore", pointer.value))
        return 0


class Compiler:
    def __init__(self):
        self.info = {"nvrtc_version": [13, 0], "architecture": "sm_121", "library": {"elf_architecture": "aarch64"}}
        self.compile = Mock(return_value=b"cpu-test-only-not-real-cubin")


class SparkProbeTests(unittest.TestCase):
    def collect(self, mode="probe", driver=None, compiler=None, workspace=None):
        driver, compiler = driver or Driver(), compiler or Compiler()
        with patch.object(spark.platform, "system", return_value="Linux"), \
             patch.object(spark.platform, "machine", return_value="aarch64"), \
             patch.object(spark, "host_memory", return_value=memory()), \
             patch.object(spark, "Driver", return_value=driver), \
             patch.object(spark, "Compiler", return_value=compiler):
            return spark.collect(mode, workspace or Workspace(), buffer_bytes=64)

    def test_buffers_rejected_before_workspace_or_cuda_for_overflow_and_invalid_values(self):
        ws = Workspace()
        for value in (0, -4, 3, True, 8 * spark.MiB + 4):
            with self.subTest(value=value), patch.object(spark, "Driver") as driver:
                with self.assertRaises(ValueError):
                    spark.collect("probe", ws, buffer_bytes=value)
                driver.assert_not_called()
        ws.check.assert_not_called()

    def test_wrong_workspace_rejected_before_loading_cuda(self):
        with patch.object(spark, "Driver") as driver:
            with self.assertRaises(spark.Unsupported):
                spark.collect("probe", Workspace("pc"))
            driver.assert_not_called()

    def test_guard_failure_is_not_caught_and_does_not_load_cuda(self):
        ws = Workspace()
        ws.check.side_effect = RuntimeError("binding failed")
        with patch.object(spark, "Driver") as driver:
            with self.assertRaisesRegex(RuntimeError, "binding failed"):
                spark.collect("probe", ws)
            driver.assert_not_called()

    def test_wrong_host_returns_unsupported_before_library_load(self):
        with patch.object(spark.platform, "system", return_value="Windows"), patch.object(spark, "Driver") as driver:
            report = spark.collect("probe", Workspace())
            self.assertEqual(report["status"], "unsupported")
            driver.assert_not_called()

    def test_device_identity_and_shared_memory_attributes_are_required(self):
        for key, value in (("device_count", 2), ("name", "NVIDIA RTX 4090"), ("compute_capability", [12, 0])):
            record = device_info()
            record[key] = value
            with self.subTest(key=key), self.assertRaises(spark.Unsupported):
                spark.validate_device(record)
        record = device_info()
        record["attributes"]["integrated"] = record["attributes"]["host_page_tables"] = 0
        with self.assertRaises(spark.Unsupported):
            spark.validate_device(record)

    def test_physical_floor_uses_sum_of_owned_buffers_not_extra_vram(self):
        needed = 16 * spark.MiB + spark.RUNTIME_HEADROOM_BYTES
        spark.require_capacity(memory(spark.AVAILABLE_FLOOR_BYTES + needed), 16 * spark.MiB)
        with self.assertRaises(spark.Unsupported):
            spark.require_capacity(memory(spark.AVAILABLE_FLOOR_BYTES + needed - 1), 16 * spark.MiB)
        with self.assertRaises(spark.Unsupported):
            spark.require_capacity(memory(cgroup=spark.RUNTIME_HEADROOM_BYTES - 1))

    def test_memory_parser_respects_parent_cgroup_and_uses_physical_pool_once(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            root = Path(directory)
            proc, cgroup = root / "proc", root / "cgroup"
            (proc / "self").mkdir(parents=True)
            (cgroup / "child").mkdir(parents=True)
            (proc / "meminfo").write_text("MemTotal: 134217728 kB\nMemAvailable: 83886080 kB\n")
            (proc / "self/cgroup").write_text("0::/child\n")
            (cgroup / "memory.max").write_text(str(4 * spark.GiB))
            (cgroup / "memory.current").write_text(str(3 * spark.GiB))
            (cgroup / "child/memory.max").write_text("max")
            result = spark.host_memory(proc, cgroup)
            self.assertEqual(result["cgroup_headroom_bytes"], spark.GiB)
            self.assertEqual(result["shared_pool_budget_bytes"], 96 * spark.GiB)
            self.assertEqual(result["physical_total_bytes"], 128 * spark.GiB)
            (proc / "self/cgroup").write_text("0::/../escape\n")
            with self.assertRaises(spark.Unsupported):
                spark.host_memory(proc, cgroup)

    def test_elf_architecture_rejects_foreign_libraries_before_loading(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            path = Path(directory) / "libnvrtc.so"
            header = bytearray(64)
            header[:6] = b"\x7fELF\x02\x01"
            struct.pack_into("<H", header, 18, 62)
            path.write_bytes(header)
            with patch.object(spark, "library_candidates", return_value=[(path, "foreign")]), patch.object(spark.C, "CDLL") as loader:
                with self.assertRaises(spark.Unsupported):
                    spark.load_library("nvrtc")
                loader.assert_not_called()
            struct.pack_into("<H", header, 18, 183)
            path.write_bytes(header)
            self.assertEqual(spark.elf_architecture(path), "aarch64")

    def test_inventory_does_not_compile_or_allocate(self):
        driver, compiler = Driver(), Compiler()
        report = self.collect("inventory", driver, compiler)
        self.assertEqual(report["status"], "supported_inventory")
        self.assertEqual(driver.events, [])
        compiler.compile.assert_not_called()
        self.assertNotIn("integer_roundtrip", report)

    def test_complete_small_roundtrip_and_owned_cleanup(self):
        driver = Driver()
        report = self.collect(driver=driver)
        self.assertEqual(report["status"], "bounded_roundtrip_pass")
        result = report["integer_roundtrip"]
        self.assertEqual(result["mismatches"], 0)
        self.assertEqual(result["checked_uint32_elements"], 16)
        self.assertEqual(result["device_peak_explicit_bytes"], 64)
        self.assertEqual(result["pinned_peak_explicit_bytes"], 64)
        self.assertEqual(driver.events[-5:], ["module_unload", "device_free", "pinned_free", "context_destroy", ("restore", 987)])
        self.assertTrue(result["cleanup"]["owned_resources_release_calls_succeeded"])
        self.assertFalse(report["full_model_quality_qualified"])
        self.assertFalse(report["general_performance_claim"])

    def test_pinned_failure_frees_owned_device_and_context(self):
        driver = Driver()
        driver.fail_host = True
        report = self.collect(driver=driver)
        self.assertEqual(report["status"], "unsupported")
        self.assertIn("device_free", driver.events)
        self.assertNotIn("pinned_free", driver.events)
        self.assertEqual(driver.events[-1], ("restore", 987))
        self.assertTrue(report["integer_roundtrip"]["cleanup"]["owned_resources_release_calls_succeeded"])

    def test_cuda_heap_below_32gib_is_allowed_when_physical_reserve_and_small_buffer_fit(self):
        driver = Driver()
        driver.cuda_free = spark.MiB
        report = self.collect(driver=driver)
        self.assertEqual(report["status"], "bounded_roundtrip_pass")
        observation = report["integer_roundtrip"]["cuda_memory_before"]
        self.assertEqual(observation["free_bytes"], spark.MiB)
        self.assertFalse(observation["physical_floor_applied_again_to_cuda_free"])
        self.assertTrue(observation["physical_available_floor_enforced_by_host_checkpoints"])

    def test_cuda_buffer_shortage_preserves_numbers_and_does_not_allocate(self):
        driver = Driver()
        driver.cuda_free = 63
        report = self.collect(driver=driver)
        self.assertEqual(report["status"], "unsupported")
        observation = report["integer_roundtrip"]["cuda_memory_before"]
        self.assertEqual(observation["free_bytes"], 63)
        self.assertEqual(observation["required_owned_device_buffer_bytes"], 64)
        self.assertEqual(observation["query_status"], 0)
        self.assertNotIn("device_allocate", driver.events)
        self.assertEqual(driver.events[-1], ("restore", 987))

    def test_cuda_memory_query_failure_preserves_query_status_and_does_not_allocate(self):
        driver = Driver()
        driver.cuda_free = driver.cuda_total = 0
        driver.cuda_memory_status = 999
        report = self.collect(driver=driver)
        self.assertEqual(report["status"], "unsupported")
        self.assertEqual(report["integer_roundtrip"]["cuda_memory_before"]["query_status"], 999)
        self.assertNotIn("device_allocate", driver.events)

    def test_large_cuda_heap_cannot_override_insufficient_physical_memory(self):
        driver = Driver()
        with patch.object(spark.platform, "system", return_value="Linux"), \
             patch.object(spark.platform, "machine", return_value="aarch64"), \
             patch.object(spark, "host_memory", return_value=memory(32 * spark.GiB - 1)), \
             patch.object(spark, "Driver", return_value=driver) as factory:
            report = spark.collect("probe", Workspace(), buffer_bytes=64)
        self.assertEqual(report["status"], "unsupported")
        factory.assert_not_called()
        self.assertEqual(driver.events, [])

    def test_mismatch_is_not_reported_as_pass_and_cleanup_survives(self):
        driver = Driver()
        driver.corrupt = True
        report = self.collect(driver=driver)
        self.assertEqual(report["status"], "unsupported")
        self.assertEqual(report["integer_roundtrip"]["mismatches"], 1)
        self.assertIn("pinned_free", driver.events)

    def test_cleanup_failure_is_explicit_and_other_owned_cleanup_still_runs(self):
        driver = Driver()
        driver.fail_free = True
        report = self.collect(driver=driver)
        self.assertEqual(report["status"], "unsupported")
        self.assertFalse(report["integer_roundtrip"]["cleanup"]["owned_resources_release_calls_succeeded"])
        self.assertIn("pinned_free", driver.events)
        self.assertIn("context_destroy", driver.events)

    def test_floor_drop_after_device_allocation_prevents_pinned_allocation_and_cleans_up(self):
        driver, count = Driver(), 0
        def checkpoint(*_):
            nonlocal count
            count += 1
            if count == 3:
                raise spark.Unsupported("floor dropped")
        with self.assertRaises(spark.Unsupported) as failure:
            spark.integer_probe(driver, b"test", 64, checkpoint)
        self.assertNotIn("pinned_allocate", driver.events)
        self.assertIn("device_free", driver.events)
        self.assertTrue(failure.exception.probe_evidence["cleanup"]["owned_resources_release_calls_succeeded"])

    def test_compile_failure_prevents_any_cuda_context_or_allocation(self):
        driver, compiler = Driver(), Compiler()
        compiler.compile.side_effect = spark.Unsupported("compile rejected")
        report = self.collect(driver=driver, compiler=compiler)
        self.assertEqual(report["status"], "unsupported")
        self.assertEqual(driver.events, [])

    def test_os_metadata_errors_do_not_publish_private_paths(self):
        with patch.object(spark.platform, "system", return_value="Linux"), \
             patch.object(spark.platform, "machine", return_value="aarch64"), \
             patch.object(spark, "host_memory", side_effect=OSError("private-host-path/credential-directory")), \
             patch.object(spark, "Driver") as driver:
            report = spark.collect("inventory", Workspace())
        self.assertEqual(report["status"], "unsupported")
        self.assertEqual(report["failure_type"], "OSError")
        self.assertNotIn("private-host-path", json.dumps(report))
        driver.assert_not_called()

    def test_cli_other_checkout_root_is_rejected_before_collect_or_result(self):
        ws = Workspace()
        ws.root = ws.root.parent / "another-checkout"
        ws.write_result = Mock()
        with patch("vesper.workspace.Workspace.open", return_value=ws), patch.object(spark, "collect") as collect:
            with self.assertRaises(ValueError):
                spark.main(["probe", "--workspace", str(ws.root), "--run-id", "r01"])
            collect.assert_not_called()
            ws.write_result.assert_not_called()

    def test_cli_foreign_imported_probe_module_is_rejected_before_collect(self):
        ws = Workspace()
        with patch("vesper.workspace.Workspace.open", return_value=ws), \
             patch.object(spark, "__file__", str(ws.root.parent / "other/src/vesper/spark_probe.py")), \
             patch.object(spark, "collect") as collect:
            with self.assertRaises(ValueError):
                spark.main(["probe", "--run-id", "r01"])
            collect.assert_not_called()

    def test_cli_foreign_imported_workspace_module_is_rejected_before_collect(self):
        ws = Workspace()
        with patch("vesper.workspace.Workspace.open", return_value=ws), \
             patch("vesper.workspace.__file__", str(ws.root.parent / "other/src/vesper/workspace.py")), \
             patch.object(spark, "collect") as collect:
            with self.assertRaises(ValueError):
                spark.main(["probe", "--run-id", "r01"])
            collect.assert_not_called()

    def test_cli_refuses_existing_evidence_before_probe(self):
        ws = Workspace()
        ws.result_path = Mock(return_value=Mock(exists=Mock(return_value=True)))
        with patch("vesper.workspace.Workspace.open", return_value=ws), patch.object(spark, "collect") as collect:
            with self.assertRaises(FileExistsError):
                spark.main(["probe", "--run-id", "r01"])
            collect.assert_not_called()

    def test_cli_writes_only_through_workspace_and_preserves_status(self):
        ws = Workspace()
        ws.result_path = Mock(return_value=Mock(exists=Mock(return_value=False)))
        ws.write_result = Mock(return_value=ws.root / "results/dgx-spark/s0/r01.json")
        with patch("vesper.workspace.Workspace.open", return_value=ws), patch.object(spark, "collect", return_value={"status": "unsupported"}), patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(spark.main(["inventory", "--run-id", "r01"]), 2)
            ws.write_result.assert_called_once_with("s0/r01.json", {"status": "unsupported"}, run_id="r01",
                                                   initial_context=ws.context("r01"))
            self.assertEqual(json.loads(output.getvalue())["status"], "unsupported")
            self.assertEqual(json.loads(output.getvalue())["result"], "results/dgx-spark/s0/r01.json")


if __name__ == "__main__":
    unittest.main()
