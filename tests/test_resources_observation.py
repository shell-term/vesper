"""Fresh physical-memory observations, strict Linux fields and floor behavior."""

import io
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from vesper import resources


class EarlyStop(io.StringIO):
    def __next__(self):
        result = super().__next__()
        if result.startswith("NeverRead"):
            raise AssertionError("Parser continued after both required fields")
        return result


class PhysicalMemoryObservationTests(unittest.TestCase):
    def linux(self, *texts, stream=io.StringIO):
        paths = Mock()
        paths.return_value.open.side_effect = [stream(text) for text in texts]
        os_patch = patch.object(resources, "os", SimpleNamespace(name="posix"))
        path_patch = patch.object(resources, "Path", paths)
        os_patch.start(); self.addCleanup(os_patch.stop)
        path_patch.start(); self.addCleanup(path_patch.stop)
        return paths

    def test_schema_and_early_stop_without_parsing_unneeded_fields(self):
        paths = self.linux("MemTotal:  1000 kB\nUnused: words with no numeric value\n"
            "MemAvailable:\t250 kB\nNeverRead: malformed\n", stream=EarlyStop)
        self.assertEqual(resources.physical_memory(), {"total": 1024000, "available": 256000})
        paths.assert_called_once_with("/proc/meminfo")
        paths.return_value.open.assert_called_once_with("r", encoding="ascii")

    def test_each_call_opens_and_observes_fresh_value_without_coalescing(self):
        paths = self.linux("MemTotal: 1000 kB\nMemAvailable: 800 kB\n",
                           "MemAvailable: 2 kB\nMemTotal: 1000 kB\n")
        self.assertEqual(resources.physical_memory()["available"], 800 * 1024)
        self.assertEqual(resources.physical_memory()["available"], 2 * 1024)
        self.assertEqual(paths.return_value.open.call_count, 2)

    def test_missing_duplicate_malformed_and_wrong_units_fail_closed(self):
        cases = ("", "MemTotal: 1000 kB\n", "MemAvailable: 100 kB\n",
            "MemTotal: 1000 kB\nMemTotal: 1000 kB\nMemAvailable: 100 kB\n",
            "MemTotal: 1000 MB\nMemAvailable: 100 kB\n",
            "MemTotal: 1000 kB\nMemAvailable: 100 KB\n",
            "MemTotal: 1000\nMemAvailable: 100 kB\n",
            "MemTotal 1000 kB\nMemAvailable: 100 kB\n",
            "MemTotal: one kB\nMemAvailable: 100 kB\n",
            "MemTotal: 1000 kB extra\nMemAvailable: 100 kB\n",
            "MemTotal: 0 kB\nMemAvailable: 0 kB\n",
            "MemTotal: -1000 kB\nMemAvailable: 100 kB\n",
            "MemTotal: 1000 kB\nMemAvailable: -1 kB\n",
            "MemTotal: 1000 kB\nMemAvailable: 1001 kB\n")
        for text in cases:
            with self.subTest(text=text), patch.object(resources, "os", SimpleNamespace(name="posix")), \
                    patch.object(resources, "Path") as paths:
                paths.return_value.open.return_value = io.StringIO(text)
                with self.assertRaises(ValueError):
                    resources.physical_memory()

    def test_zero_available_is_observed_as_zero_and_read_errors_propagate(self):
        self.linux("MemTotal: 1000 kB\nMemAvailable: 0 kB\n")
        self.assertEqual(resources.physical_memory()["available"], 0)
        with patch.object(resources, "Path") as paths:
            paths.return_value.open.side_effect = OSError("unavailable observation")
            with self.assertRaises(OSError):
                resources.physical_memory()

    def test_available_change_still_controls_real_admission_floor(self):
        gib, mib = resources.GiB, resources.MiB
        self.linux(f"MemTotal: {128*gib//1024} kB\nMemAvailable: {(32*gib+2*mib)//1024} kB\n",
            f"MemTotal: {128*gib//1024} kB\nMemAvailable: {(32*gib-1024)//1024} kB\n")
        with patch("vesper.memory.current_policy", return_value=None):
            budget = resources.MemoryBudget(cap=2*mib, available_floor=32*gib)
            budget.acquire(mib)
            budget.release(mib)
            with self.assertRaises(MemoryError):
                budget.acquire(mib)
        self.assertEqual(budget.reserved, 0)

    def test_windows_schema_and_native_failure_path_are_unchanged(self):
        def observe(pointer):
            pointer._obj.total = 128 * resources.GiB
            pointer._obj.available = 40 * resources.GiB
            return 1
        native = SimpleNamespace(kernel32=SimpleNamespace(GlobalMemoryStatusEx=Mock(side_effect=observe)))
        with patch.object(resources, "os", SimpleNamespace(name="nt")), \
                patch.object(resources.ctypes, "windll", native, create=True), \
                patch.object(resources, "Path", side_effect=AssertionError("No proc read on Windows")):
            self.assertEqual(resources.physical_memory(), {"total": 128*resources.GiB, "available": 40*resources.GiB})
            native.kernel32.GlobalMemoryStatusEx.side_effect = None
            native.kernel32.GlobalMemoryStatusEx.return_value = 0
            with self.assertRaises(OSError):
                resources.physical_memory()


if __name__ == "__main__":
    unittest.main()
