import base64
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
from vesper.prefix import PREFIX_IDS, PREFIX_LAYOUT, STORAGE_DTYPES, load_moe_inputs, load_prefix_fixture
from vesper.prefix_reference import required_shared_bytes


class PrefixTests(unittest.TestCase):
    def test_generated_launch_resource_requirement_is_read_without_execution(self):
        for name in ("cuKernelSetAttribute", "cuda.cuKernelSetAttribute"):
            source = f"{name}(CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 141312, kernel, device)"
            self.assertEqual(required_shared_bytes(source), 141312)
        with self.assertRaisesRegex(RuntimeError, "Unable to inspect"):
            required_shared_bytes("unrelated_value = 141312")

    def test_captured_inputs_require_exact_bf16_storage_and_hashes(self):
        raw = np.zeros(5120, np.uint16).tobytes()
        case = {"activation_bf16_base64": base64.b64encode(raw).decode(),
                "activation_bf16_sha256": hashlib.sha256(raw).hexdigest()}
        report = {"probe": "original_prefix_moe_inputs", "status": "pass", "model": MODEL, "revision": REVISION,
                  "cases": [dict(case), dict(case)]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inputs.json"
            path.write_text(json.dumps(report))
            self.assertEqual(len(load_moe_inputs(path)[0]), 2)
            report["cases"][0]["activation_bf16_sha256"] = "0" * 64
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "source hash"):
                load_moe_inputs(path)
            report["status"] = "blocked"
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "passing original-prefix"):
                load_moe_inputs(path)

    def test_prefix_array_headers_are_bounded_before_numpy_loading(self):
        entries = [{"name": name, "dtype": dtype, "shape": list(shape),
                    "bytes": int(np.prod(shape)) * STORAGE_DTYPES[dtype].itemsize, "sha256": "0" * 64}
                   for name, (dtype, shape) in PREFIX_LAYOUT.items()]
        manifest = {"kind": "original_layer0_prefix_fixture", "model": MODEL, "revision": REVISION, "layer": 0,
                    "tensors": entries, "embedding_rows": [{"token_id": i, "sha256": "0" * 64} for i in PREFIX_IDS]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps(manifest))
            path = root / "native-prefix.npz"
            with zipfile.ZipFile(path, "w") as archive:
                for name in (*PREFIX_LAYOUT, "embedding_rows"):
                    buffer = io.BytesIO()
                    np.lib.format.write_array_header_1_0(buffer, {"descr": "|u1", "fortran_order": False, "shape": (2**40,)})
                    archive.writestr(name + ".npy", buffer.getvalue())
            with patch("vesper.prefix.np.load") as loader:
                with self.assertRaisesRegex(ValueError, "bounded layout"):
                    load_prefix_fixture(path)
                loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
