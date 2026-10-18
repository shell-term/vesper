import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from vesper.checkpoint import LIMIT, MODEL, REVISION, fetch_reference, load_fixture, read_range
from vesper.resources import MemoryBudget


class BoundaryTests(unittest.TestCase):
    def test_owned_cap_and_system_floor_both_apply(self):
        budget = MemoryBudget(cap=100, available_floor=30)
        budget.acquire(60, available=200)
        with self.assertRaises(MemoryError):
            budget.acquire(41, available=200)
        with self.assertRaises(MemoryError):
            budget.acquire(20, available=49)
        self.assertEqual(budget.reserved, 60)
        budget.release(60)
        self.assertEqual(budget.reserved, 0)
        self.assertEqual(budget.peak, 60)

    def test_invalid_memory_sizes_do_not_change_reservations(self):
        budget = MemoryBudget(cap=100, available_floor=0)
        with self.assertRaises(ValueError):
            budget.acquire(-1, available=200)
        with self.assertRaises(ValueError):
            budget.release(1)
        self.assertEqual(budget.reserved, 0)

    def test_range_ignoring_server_is_rejected_before_read(self):
        class Response:
            status = 200
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def read(self, *_): raise AssertionError("Must never consume a full shard")
        with patch("urllib.request.urlopen", return_value=Response()):
            with self.assertRaisesRegex(ValueError, "ignored Range"):
                read_range("model-00003-of-00048.safetensors", 0, 8)

    def test_wrong_range_is_rejected_before_read(self):
        class Response:
            status = 206
            headers = {"Content-Range": "bytes 0-15/999999999"}
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def read(self, *_): raise AssertionError("Wrong range must not be consumed")
        with patch("urllib.request.urlopen", return_value=Response()):
            with self.assertRaisesRegex(ValueError, "different byte range"):
                read_range("model-00003-of-00048.safetensors", 16, 8)

    def test_invalid_range_never_opens_network(self):
        with patch("urllib.request.urlopen") as opener:
            for shard, offset, size in (("../config.json", 0, 8), ("model-00003-of-00048.safetensors", -1, 8),
                                        ("model-00003-of-00048.safetensors", 0, LIMIT + 1)):
                with self.assertRaises(ValueError):
                    read_range(shard, offset, size)
            opener.assert_not_called()

    def test_tampered_fixture_and_revision_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weight, scales = np.zeros((1, 16), np.uint8), np.full((1, 1), 127, np.uint8)
            entries = [{"selected_shape": list(a.shape), "sha256": hashlib.sha256(a.tobytes()).hexdigest()}
                       for a in (weight, scales)]
            manifest = {"model": MODEL, "revision": REVISION, "tensors": entries}
            path = root / "fixture.npz"
            np.savez(path, weight=weight, scales=scales)
            (root / "manifest.json").write_text(json.dumps(manifest))
            np.testing.assert_array_equal(load_fixture(path)["weight"], weight)
            weight[0, 0] = 1
            np.savez(path, weight=weight, scales=scales)
            with self.assertRaisesRegex(ValueError, "source hash"):
                load_fixture(path)
            manifest["revision"] = "another_checkpoint"
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "pinned checkpoint"):
                load_fixture(path)

    def test_tampered_cached_reference_is_rejected_without_execution_or_fetch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "kernel.py").write_text("raise AssertionError('must not execute')", encoding="utf-8")
            with patch("vesper.checkpoint.read_small") as fetch:
                with self.assertRaisesRegex(ValueError, "source hash"):
                    fetch_reference(root)
                fetch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
