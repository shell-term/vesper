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
from vesper.router import (ROUTER_CONFIG, ROUTER_INFERENCE_CONFIG_HASH, RouterProbe,
                           load_router_fixture, select_scores, top6_indices)
from vesper.router_reference import TOPK_SOURCE_HASHES, fetch_topk_reference


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

    def test_selected_and_boundary_ties_keep_the_selected_expert_set(self):
        scores = np.arange(384, dtype=np.float32)
        bias = np.zeros(384, np.float32)
        for positions in ((382, 383), (377, 378)):
            changed = scores.copy()
            changed[positions[0]] = changed[positions[1]]
            weights, indices = select_scores(changed, bias)
            self.assertEqual(set(indices.tolist()), set(range(378, 384)) if positions[0] == 382 else {377, 379, 380, 381, 382, 383})
            self.assertAlmostEqual(float(weights.sum()), 1.5, places=6)
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

    def test_pinned_cuda_tie_order_goldens_and_padding(self):
        # CUDA zero/signed-zero goldens; equal finite levels must exclude padding.
        for level in (0, -1, 1):
            indices = top6_indices(np.full(384, level, np.float32))
            self.assertEqual(indices.tolist(), [1, 0, 4, 5, 2, 3])
        signed = np.resize(np.array([-0.0, 0.0], np.float32), 384)
        self.assertEqual(top6_indices(signed).tolist(), [3, 1, 9, 11, 5, 7])
        for invalid in (np.zeros(383, np.float32), np.full(384, np.nan, np.float32), np.full(384, np.inf, np.float32)):
            with self.assertRaises(ValueError):
                top6_indices(invalid)

    def test_synthetic_selection_rejects_invalid_inputs_before_cuda(self):
        probe = object.__new__(RouterProbe)
        probe.device = object()
        valid = np.ones(384, np.float32)
        for invalid in (np.zeros(383, np.float32), np.full(384, -1, np.float32),
                        np.full(384, np.inf, np.float32), np.full(384, np.finfo(np.float32).max, np.float32)):
            with self.assertRaises(ValueError):
                probe.inspect_selection(invalid, valid)
        with self.assertRaises(ValueError):
            probe.inspect_selection(valid, np.full(384, np.nan, np.float32))

    def test_pinned_topk_source_refuses_tampering_and_oversized_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'TensorTopK.cpp').write_bytes(b'changed source')
            with patch('vesper.router_reference.urllib.request.urlopen') as request:
                with self.assertRaisesRegex(ValueError, 'pinned hash'):
                    fetch_topk_reference(root)
                request.assert_not_called()
            (root / 'TensorTopK.cpp').write_bytes(bytes(262145))
            with self.assertRaisesRegex(ValueError, 'exceeds its bound'):
                fetch_topk_reference(root)
        self.assertEqual(len(TOPK_SOURCE_HASHES), 7)

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
