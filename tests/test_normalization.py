import hashlib
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from vesper.normalization import NativeNormRoPE, mean_threads, validate_norm, validate_positions, validate_rope
from vesper.normalization_fixture import NORM_WEIGHTS, load_norm_weights
from vesper.normalization_reference import fetch_pytorch_sources
from vesper.normalization_validation import inputs, validate_goldens
from vesper.checkpoint import MODEL, REVISION


class NormalizationTests(unittest.TestCase):
    def test_mean_geometry_matches_qualified_vectorized_reduce_profiles(self):
        expected = {512: (128, 128, 128, 64, 32), 1280: (256, 256, 128, 64, 32), 5120: (512, 256, 128, 64, 32)}
        for width, threads in expected.items():
            self.assertEqual(tuple(mean_threads(rows, width) for rows in (1, 2, 4, 8, 16)), threads)
            self.assertEqual(mean_threads(514, width), 32)
        for args in ((0, 512), (515, 512), (True, 512), (1, 64), (1, 512.)):
            with self.assertRaises(ValueError):
                mean_threads(*args)

    def test_normalization_inputs_are_bounded_before_numerical_allocation(self):
        value, weight = inputs((2, 1280), 1), np.full(1280, 0x3F80, np.uint16)
        self.assertEqual(validate_norm(value, weight), (2, 1280))
        for bad in (value.astype(np.float32), value[:, ::-1], value[0], np.zeros((515, 1280), np.uint16),
                    np.zeros((2, 1024), np.uint16), np.full_like(value, 0x7F80)):
            with self.assertRaises(ValueError):
                validate_norm(bad, weight)
        for bad in (weight.astype(np.float32), weight[:-1], np.full_like(weight, 0x7FC0)):
            with self.assertRaises(ValueError):
                validate_norm(value, bad)

    def test_rotary_layout_position_head_and_profile_contracts(self):
        value = inputs((2, 3, 64, 512), 1)
        self.assertEqual(validate_rope(value, 131069, "window", True), (2, 3, 64, 512))
        for args in ((value, 131070, "window", False), (value, -1, "window", False), (value, True, "window", False),
                     (value, 0, "unknown", False), (value, 0, "window", 1),
                     (value[..., ::-1], 0, "window", False), (value.astype(np.float32), 0, "window", False),
                     (np.zeros((2, 257, 64, 512), np.uint16), 0, "window", False),
                     (np.zeros((1, 1, 3, 512), np.uint16), 0, "window", False)):
            with self.assertRaises(ValueError):
                validate_rope(*args)
        for bad in (np.array([-1], np.int32), np.array([131072], np.int32), np.arange(4097, dtype=np.int32),
                    np.arange(2, dtype=np.int64), np.zeros((1, 1), np.int32)):
            with self.assertRaises(ValueError):
                validate_positions(bad, "window")

    def test_stateless_calls_refuse_closed_runtime_and_low_memory_entry(self):
        native = NativeNormRoPE()
        for operation in (lambda: native.norm(None, None), lambda: native.rope(None), lambda: native.frequencies(None)):
            with self.assertRaisesRegex(RuntimeError, "not open"):
                operation()
        with patch("vesper.resources.physical_memory", return_value={"available": 16 * 1024**3}):
            with self.assertRaises(MemoryError):
                native.__enter__()
        self.assertEqual(native.budget.reserved, 0)
        native.opened = True
        with self.assertRaisesRegex(RuntimeError, "already open"):
            native.__enter__()

    def test_only_selected_norm_vectors_are_loaded_and_tampering_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arrays = {name: np.full(width, 0x3F80, np.uint16) for name, width in NORM_WEIGHTS.items()}
            np.savez(root / "norms.npz", **arrays)
            entries = [{"name": n, "dtype": "BF16", "shape": list(a.shape), "bytes": a.nbytes,
                        "sha256": hashlib.sha256(a.tobytes()).hexdigest()} for n, a in arrays.items()]
            manifest = {"kind": "original_layer0_prefix_fixture", "model": MODEL, "revision": REVISION, "layer": 0, "tensors": entries}
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            loaded, provenance = load_norm_weights(root / "norms.npz")
            self.assertEqual(set(loaded), set(NORM_WEIGHTS))
            self.assertEqual(provenance["original_weight_bytes_reused"], 24064)
            self.assertFalse(provenance["other_prefix_matrices_loaded"])
            entries[0]["sha256"] = "0" * 64
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "bytes differ"):
                load_norm_weights(root / "norms.npz")
            entries.append(entries[0])
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                load_norm_weights(root / "norms.npz")
            manifest["tensors"] = entries[:4]
            path.write_text(json.dumps(manifest), encoding="utf-8")
            arrays[next(iter(NORM_WEIGHTS))] = np.zeros((1280, 1), np.uint16)
            np.savez(root / "norms.npz", **arrays)
            with patch("numpy.load", side_effect=AssertionError("Header must be checked before NumPy allocation")):
                with self.assertRaisesRegex(ValueError, "header differs"):
                    load_norm_weights(root / "norms.npz")

    def test_bad_cached_pytorch_source_is_rejected_before_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Reduce.cuh").write_bytes(b"changed reduction")
            with patch("urllib.request.urlopen", side_effect=AssertionError("No network expected")):
                with self.assertRaisesRegex(ValueError, "pinned hash"):
                    fetch_pytorch_sources(root)

    def test_published_source_and_complete_reference_evidence_are_required(self):
        root = Path(__file__).resolve().parents[1]
        report = json.loads((root / "results/2026-10-18-normalization-rope/normalization.json").read_bytes())
        self.assertEqual(len(validate_goldens(report)), 535)
        source = (root / "src/vesper/kernels/format_probe.cu").read_bytes() + b"\n" + (root / "src/vesper/kernels/normalization_probe.cu").read_bytes()
        self.assertEqual(hashlib.sha256(source.replace(b"\r\n", b"\n")).hexdigest(),
                         report["native_profile"]["compiler"]["concatenated_source_sha256"])
        for change in (lambda r: r["component_cases"].pop(), lambda r: r["chain_cases"].pop(),
                       lambda r: r["component_cases"][0]["checks"][0].update(actual_sha256="0" * 64),
                       lambda r: r["pytorch_source"]["files"]["Reduce.cuh"].update(sha256="0" * 64),
                       lambda r: r["negative_controls"].pop(), lambda r: r["faults"][0].update(partial_result_returned=True)):
            broken = copy.deepcopy(report)
            change(broken)
            with self.assertRaises(ValueError):
                validate_goldens(broken)


if __name__ == "__main__":
    unittest.main()
