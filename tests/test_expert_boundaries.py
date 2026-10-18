import json
import io
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest.mock import patch

import numpy as np

from vesper.checkpoint import EXPERT_LAYOUT, MODEL, REVISION, load_expert_fixture, validate_expert_arrays
from vesper.expert import ResidentExpert


class ExpertBoundaryTests(unittest.TestCase):
    def tensors(self):
        # Layout validation only; broadcast views avoid allocating a real expert.
        return {name: np.broadcast_to(np.array(0, np.uint8), shape) for name, shape in EXPERT_LAYOUT.items()}

    def test_missing_or_transposed_tensors_are_rejected_before_cuda(self):
        arrays = self.tensors()
        validate_expert_arrays(arrays)
        with patch("vesper.expert.Driver") as driver:
            invalid = dict(arrays)
            invalid["w2_weight"] = arrays["w2_weight"].T
            with self.assertRaisesRegex(ValueError, "pinned shape"):
                ResidentExpert(invalid)
            invalid = dict(arrays)
            del invalid["w3_scale"]
            with self.assertRaisesRegex(ValueError, "exactly six"):
                ResidentExpert(invalid)
            driver.assert_not_called()

    def test_invalid_clamp_limit_is_rejected_before_cuda(self):
        for limit in (-1, float("inf"), float("nan")):
            with self.assertRaisesRegex(ValueError, "SwiGLU limit"):
                ResidentExpert(self.tensors(), swiglu_limit=limit)

    def test_wrong_checkpoint_and_incomplete_provenance_do_not_open_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {"model": MODEL, "revision": "wrong", "kind": "complete_expert",
                        "layer": 0, "expert": 0, "swiglu_limit": 10.0}
            target = root / "native-expert.npz"
            with patch("vesper.checkpoint.np.load") as loader:
                (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "pinned checkpoint"):
                    load_expert_fixture(target)
                manifest["revision"] = REVISION
                (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Incomplete"):
                    load_expert_fixture(target)
                loader.assert_not_called()

    def test_excessive_declared_shape_is_rejected_before_numpy_allocation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {"model": MODEL, "revision": REVISION, "kind": "complete_expert",
                        "layer": 0, "expert": 0, "swiglu_limit": 10.0,
                        "tensors": [{"name": name, "shape": list(shape), "sha256": "unused"}
                                    for name, shape in EXPERT_LAYOUT.items()]}
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            path = root / "native-expert.npz"
            with zipfile.ZipFile(path, "w") as archive:
                for name in EXPERT_LAYOUT:
                    header = io.BytesIO()
                    np.lib.format.write_array_header_1_0(header, {"descr": "|u1", "fortran_order": False, "shape": (2**40,)})
                    archive.writestr(f"{name}.npy", header.getvalue())
            with patch("vesper.checkpoint.np.load") as loader:
                with self.assertRaisesRegex(ValueError, "bounded native layout"):
                    load_expert_fixture(path)
                loader.assert_not_called()
