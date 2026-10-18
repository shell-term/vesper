import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from vesper.checkpoint import MODEL, REVISION
from vesper.router import (ROUTER_CONFIG, ROUTER_INFERENCE_CONFIG_HASH, RouterProbe, UnsupportedRoutingTie,
                           load_router_fixture, select_scores)


class RouterTests(unittest.TestCase):
    def fixture(self):
        arrays = {name: np.zeros(384, np.float32) for name in ("bias", "bias_vl")}
        manifest = {"kind": "router_bias_fixture", "model": MODEL, "revision": REVISION, "layer": 0,
                    "config": dict(ROUTER_CONFIG), "inference_config_sha256": ROUTER_INFERENCE_CONFIG_HASH, "tensors": [
                        {"name": name, "tensor": f"layers.0.ffn.gate.{name}", "dtype": "F32", "shape": [384],
                         "bytes": 1536, "sha256": hashlib.sha256(memoryview(array)).hexdigest()}
                        for name, array in arrays.items()]}
        return arrays, manifest

    def test_bias_selects_experts_without_scaling_their_raw_weights(self):
        scores = np.linspace(1, 2, 384, dtype=np.float32)
        bias = np.zeros(384, np.float32)
        bias[0] = 100
        weights, indices = select_scores(scores, bias)
        self.assertEqual(indices.tolist(), [0, 383, 382, 381, 380, 379])
        self.assertLess(weights[0], weights[1])
        self.assertAlmostEqual(float(weights.sum()), 1.5, places=6)
        self.assertLess(float(weights[0]), 0.2)

    def test_selected_ties_and_boundary_ties_are_refused(self):
        scores = np.arange(384, dtype=np.float32)
        bias = np.zeros(384, np.float32)
        for positions in ((382, 383), (377, 378)):
            changed = scores.copy()
            changed[positions[0]] = changed[positions[1]]
            with self.assertRaises(UnsupportedRoutingTie):
                select_scores(changed, bias)
        scores[0] = scores[1]  # An unselected tie does not affect top-six routing.
        _, indices = select_scores(scores, bias)
        self.assertEqual(indices.tolist(), [383, 382, 381, 380, 379, 378])

    def test_nonfinite_wrong_shape_and_missing_bias_are_rejected_before_cuda(self):
        arrays, _ = self.fixture()
        with patch("vesper.router.Driver") as driver:
            for invalid in ({"bias": arrays["bias"]}, {**arrays, "bias_vl": np.zeros(383, np.float32)},
                            {**arrays, "bias": np.full(384, np.nan, np.float32)}):
                with self.assertRaises(ValueError):
                    RouterProbe(invalid)
            probe = RouterProbe(arrays)
            probe.device = object()
            with self.assertRaisesRegex(RuntimeError, "already open"):
                probe.__enter__()
            driver.assert_not_called()
        for scores in (np.ones(383, np.float32), np.full(384, np.inf, np.float32), np.full(384, -1, np.float32)):
            with self.assertRaises(ValueError):
                select_scores(scores, arrays["bias"])

    def test_diagnostic_tie_cannot_be_admitted_as_a_decision(self):
        probe = object.__new__(RouterProbe)
        probe.inspect = lambda *args, **kwargs: {"tied": np.array([1], np.int32)}
        with self.assertRaises(UnsupportedRoutingTie):
            probe.decision(np.zeros(384, np.float32))

    def test_fixture_bytes_identity_and_configuration_are_checked(self):
        arrays, manifest = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "native-router.npz"
            np.savez(target, **arrays)
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            loaded, _ = load_router_fixture(target)
            self.assertEqual(loaded["bias"].shape, (384,))
            arrays["bias"][0] = 1
            np.savez(target, **arrays)
            with self.assertRaisesRegex(ValueError, "original source hash"):
                load_router_fixture(target)
            manifest["layer"] = 1
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identity/layout"):
                load_router_fixture(target)
            manifest["layer"] = 0
            manifest["config"]["route_scale"] = 2.0
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "pinned source/configuration"):
                load_router_fixture(target)

    def test_archive_shape_bomb_is_rejected_before_numpy_allocation(self):
        _, manifest = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            target = root / "native-router.npz"
            with zipfile.ZipFile(target, "w") as archive:
                for name in ("bias", "bias_vl"):
                    header = io.BytesIO()
                    np.lib.format.write_array_header_1_0(header, {"descr": "<f4", "fortran_order": False, "shape": (2**40,)})
                    archive.writestr(name + ".npy", header.getvalue())
            with patch("vesper.router.np.load") as loader:
                with self.assertRaisesRegex(ValueError, "bounded FP32"):
                    load_router_fixture(target)
                loader.assert_not_called()
            with target.open("ab") as stream:
                stream.write(bytes(16384))
            with patch("vesper.router.zipfile.ZipFile") as archive:
                with self.assertRaisesRegex(ValueError, "bounded fixture size"):
                    load_router_fixture(target)
                archive.assert_not_called()
