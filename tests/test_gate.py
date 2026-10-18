import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import zipfile

import numpy as np

from vesper.blas import Cublas, GATE_BLAS_WORKSPACE
from vesper.checkpoint import MODEL, REVISION
from vesper.formats import bf16_values
from vesper.gate import (GATE_NATIVE_BYTES, GATE_SHAPE, GateProjection, fetch_gate_fixture,
                         load_gate_fixture)
from vesper.gate_reference import compare_gate, fetch_gate_reference
from vesper.resources import MemoryBudget, MiB


class GateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.weight = np.zeros(GATE_SHAPE, np.uint16)
        cls.weight.setflags(write=False)

    def fixture(self, root, weight=None):
        weight = self.weight if weight is None else weight
        manifest = {"kind": "gate_projection_fixture", "model": MODEL, "revision": REVISION, "layer": 0,
                    "tensor": {"name": "layers.0.ffn.gate.weight", "dtype": "BF16", "shape": list(GATE_SHAPE),
                               "bytes": GATE_NATIVE_BYTES, "sha256": hashlib.sha256(memoryview(weight)).hexdigest()}}
        np.savez(root / "native-gate.npz", weight=weight)
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return manifest

    def test_original_bf16_bits_and_hash_are_preserved(self):
        weight = self.weight.copy()
        weight[0, 0], weight[383, 5119] = 0x8000, 0x3f80
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root, weight)
            loaded, _ = load_gate_fixture(root / "native-gate.npz")
            self.assertTrue(np.array_equal(loaded, weight))
            self.assertEqual(int(bf16_values(loaded)[0, 0].view(np.uint32)), 0x80000000)
            self.assertEqual(float(bf16_values(loaded)[383, 5119]), 1.0)
            weight[0, 1] = 1
            np.savez(root / "native-gate.npz", weight=weight)
            with self.assertRaisesRegex(ValueError, "original source hash"):
                load_gate_fixture(root / "native-gate.npz")

    def test_fixture_identity_is_rejected_before_numpy_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.fixture(root)
            manifest["layer"] = 1
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with patch("vesper.gate.np.load") as loader:
                with self.assertRaisesRegex(ValueError, "identity/layout"):
                    load_gate_fixture(root / "native-gate.npz")
                loader.assert_not_called()

    def test_archive_shape_bomb_and_size_are_bounded_before_numpy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            target = root / "native-gate.npz"
            header = io.BytesIO()
            np.lib.format.write_array_header_1_0(header, {"descr": "<u2", "fortran_order": False, "shape": (384, 2**40)})
            with zipfile.ZipFile(target, "w") as archive:
                archive.writestr("weight.npy", header.getvalue())
            with patch("vesper.gate.np.load") as loader:
                with self.assertRaisesRegex(ValueError, "bounded BF16"):
                    load_gate_fixture(target)
                loader.assert_not_called()
            with target.open("ab") as stream:
                stream.write(bytes(4 * MiB))
            with patch("vesper.gate.zipfile.ZipFile") as archive:
                with self.assertRaisesRegex(ValueError, "bounded fixture size"):
                    load_gate_fixture(target)
                archive.assert_not_called()

    def test_bad_weight_and_reentry_are_rejected_before_cuda(self):
        invalid = self.weight.copy()
        invalid[0, 0] = 0x7f80
        with patch("vesper.gate.Driver") as driver:
            for weight in (np.zeros((384, 5119), np.uint16), self.weight.astype(np.float32), invalid):
                with self.assertRaises(ValueError):
                    GateProjection(weight)
            projection = GateProjection(self.weight)
            projection.device = object()
            with self.assertRaisesRegex(RuntimeError, "already open"):
                projection.__enter__()
            driver.assert_not_called()

    def test_invalid_activation_is_rejected_before_a_device_call(self):
        projection = GateProjection(self.weight)
        with self.assertRaisesRegex(RuntimeError, "context manager"):
            projection.project(np.zeros(5120, np.float32))
        projection.device = MagicMock()
        for x in (np.zeros(5119, np.float32), np.full(5120, np.nan, np.float32), np.full(5120, np.inf, np.float32)):
            with self.assertRaises(ValueError):
                projection.project(x)
        projection.device.current.assert_not_called()

    def test_host_cap_refuses_entry_before_cuda(self):
        budget = MemoryBudget(cap=MiB, available_floor=0)
        projection = GateProjection(self.weight, host_budget=budget)
        with patch("vesper.gate.Driver") as driver:
            with self.assertRaises(MemoryError):
                projection.__enter__()
            driver.assert_not_called()
        self.assertEqual(budget.reserved, 0)
        self.assertIsNone(projection.device)

    def test_cuda_discovery_failure_releases_host_reservation(self):
        budget = MemoryBudget(cap=32 * MiB, available_floor=0)
        projection = GateProjection(self.weight, host_budget=budget)
        with patch("vesper.gate.Driver", side_effect=RuntimeError("unavailable")):
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                projection.__enter__()
        self.assertEqual(budget.reserved, 0)
        self.assertIsNone(projection.device)
        projection.__exit__()

    def test_workspace_allocation_failure_cleans_handle_device_and_host(self):
        budget = MemoryBudget(cap=32 * MiB, available_floor=0)
        projection = GateProjection(self.weight, host_budget=budget)
        device = MagicMock()
        device.allocate.side_effect = [100, 200, MemoryError("workspace cap")]
        blas = MagicMock()
        with patch("vesper.gate.Driver") as driver, patch("vesper.gate.Device", return_value=device), patch("vesper.gate.Cublas", return_value=blas):
            driver.return_value.inventory.return_value = {"devices": [{"ordinal": 0}]}
            with self.assertRaisesRegex(MemoryError, "workspace cap"):
                projection.__enter__()
        blas.close.assert_called_once()
        blas.workspace.assert_not_called()
        device.close.assert_called_once()
        self.assertEqual(budget.reserved, 0)
        self.assertIsNone(projection.device)
        self.assertIsNone(projection.blas)

    def test_forced_tf32_is_refused_before_library_loading(self):
        with patch.dict("os.environ", {"NVIDIA_TF32_OVERRIDE": "1"}), patch("vesper.blas.find_cublas") as library:
            with self.assertRaisesRegex(ValueError, "forced TF32"):
                Cublas()
            library.assert_not_called()

    def test_invalid_layer_and_reference_tampering_make_no_network_calls(self):
        with patch("vesper.gate.read_small") as read:
            for layer in (-1, 40, True):
                with self.assertRaises(ValueError):
                    fetch_gate_fixture(Path("unused"), layer)
            read.assert_not_called()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "CUDABlas.cpp").write_bytes(b"changed source")
            with patch("vesper.gate_reference.urllib.request.urlopen") as request:
                with self.assertRaisesRegex(ValueError, "pinned hash"):
                    fetch_gate_reference(root)
                request.assert_not_called()

    def test_incompatible_reference_is_blocked_before_fixture_or_cuda_loading(self):
        fake = SimpleNamespace(version=SimpleNamespace(git_version="other-version"),
                               get_float32_matmul_precision=lambda: "highest",
                               cuda=SimpleNamespace(is_available=lambda: True, tunable=SimpleNamespace(is_enabled=lambda: False)),
                               backends=SimpleNamespace(cuda=SimpleNamespace(preferred_blas_library=lambda: "_BlasBackend.Cublas")))
        with patch.dict("sys.modules", {"torch": fake}), patch("vesper.gate_reference.importlib.util.find_spec", return_value=object()), \
             patch("vesper.gate_reference.load_gate_fixture") as fixture, patch("vesper.gate_reference.Driver") as driver:
            for revision, precision, backend, tuning in (
                ("other-version", "highest", "_BlasBackend.Cublas", False),
                ("5c4886908584029761b579af026dcfb627c84070", "high", "_BlasBackend.Cublas", False),
                ("5c4886908584029761b579af026dcfb627c84070", "highest", "_BlasBackend.Cublaslt", False),
                ("5c4886908584029761b579af026dcfb627c84070", "highest", "_BlasBackend.Cublas", True),
            ):
                fake.version.git_version = revision
                fake.get_float32_matmul_precision = lambda: precision
                fake.backends.cuda.preferred_blas_library = lambda: backend
                fake.cuda.tunable.is_enabled = lambda: tuning
                result = compare_gate(Path("unused"), Path("unused"), Path("unused"))
                self.assertEqual(result["status"], "blocked")
            fixture.assert_not_called()
            driver.assert_not_called()
