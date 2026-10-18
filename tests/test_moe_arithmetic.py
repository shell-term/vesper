import copy
import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import zipfile

import numpy as np

from vesper.checkpoint import LIMIT, MODEL, REVISION
from vesper.moe import HIDDEN, MoEAccumulator, participation
from vesper.resources import GiB
from vesper.shared import (SHARED_BYTES, SHARED_LAYOUT, ResidentSharedExpert,
                           fetch_shared_fixture, load_shared_fixture, validate_shared_provenance)


class SharedTests(unittest.TestCase):
    def arrays(self):
        return {n: np.broadcast_to(np.array(0, np.uint8), shape) for n, shape in SHARED_LAYOUT.items()}

    def provenance(self):
        return {"kind": "shared_expert_fp8", "model": MODEL, "revision": REVISION,
                "layer": 0, "swiglu_limit": 10.0,
                "tensors": [{"name": name, "tensor": "layers.0.ffn.shared_experts." + name.replace("_", "."),
                             "dtype": "F8_E4M3" if name.endswith("weight") else "F8_E8M0",
                             "shape": list(shape), "bytes": int(np.prod(shape)), "sha256": "0" * 64}
                            for name, shape in SHARED_LAYOUT.items()]}

    def test_shared_scales_are_blocks_not_routed_expert_rows(self):
        with patch("vesper.expert.Driver") as driver:
            ResidentSharedExpert(self.arrays())
            arrays = self.arrays()
            arrays["w1_scale"] = np.broadcast_to(np.array(0, np.uint8), (2304, 160))
            with self.assertRaisesRegex(ValueError, "FP8/block-32"):
                ResidentSharedExpert(arrays)
            del arrays["w1_scale"]
            with self.assertRaisesRegex(ValueError, "exactly six"):
                ResidentSharedExpert(arrays)
            driver.assert_not_called()

    def test_native_dtype_and_tensor_identity_cannot_be_relabelled(self):
        manifest = self.provenance()
        validate_shared_provenance(manifest)
        for field, value in (("dtype", "I8"), ("shape", [2304, 2560]),
                             ("tensor", "layers.1.ffn.shared_experts.w1.weight")):
            changed = copy.deepcopy(manifest)
            changed["tensors"][0][field] = value
            with self.assertRaisesRegex(ValueError, "identity/layout"):
                validate_shared_provenance(changed)
        for layer in (-1, 40, True):
            changed = copy.deepcopy(manifest)
            changed["layer"] = layer
            with self.assertRaisesRegex(ValueError, "backbone"):
                validate_shared_provenance(changed)

    def test_bad_numpy_shape_is_rejected_before_allocation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps(self.provenance()), encoding="utf-8")
            path = root / "native-shared.npz"
            with zipfile.ZipFile(path, "w") as archive:
                for name in SHARED_LAYOUT:
                    buffer = io.BytesIO()
                    np.lib.format.write_array_header_1_0(buffer, {"descr": "|u1", "fortran_order": False, "shape": (2**40,)})
                    archive.writestr(f"{name}.npy", buffer.getvalue())
            with patch("vesper.shared.np.load") as loader:
                with self.assertRaisesRegex(ValueError, "bounded native layout"):
                    load_shared_fixture(path)
                loader.assert_not_called()

    def test_changed_original_bytes_fail_hash_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arrays = {n: np.zeros(s, np.uint8) for n, s in SHARED_LAYOUT.items()}
            manifest = self.provenance()
            for entry in manifest["tensors"]:
                entry["sha256"] = hashlib.sha256(arrays[entry["name"]].tobytes()).hexdigest()
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            np.savez(root / "native-shared.npz", **arrays)
            loaded, _ = load_shared_fixture(root / "native-shared.npz")
            self.assertEqual(sum(a.nbytes for a in loaded.values()), SHARED_BYTES)
            del loaded
            arrays["w1_weight"][0, 0] = 1
            np.savez(root / "native-shared.npz", **arrays)
            with self.assertRaisesRegex(ValueError, "source hash"):
                load_shared_fixture(root / "native-shared.npz")

    def test_weight_reads_above_eight_mib_are_split_without_changing_bytes(self):
        manifest, header, index = self.provenance(), {}, {"weight_map": {}}
        offset = 0
        for entry in manifest["tensors"]:
            header[entry["tensor"]] = {"dtype": entry["dtype"], "shape": entry["shape"],
                                       "data_offsets": [offset, offset + entry["bytes"]]}
            index["weight_map"][entry["tensor"]] = "model-00003-of-00048.safetensors"
            offset += entry["bytes"]
        encoded = json.dumps(header).encode()
        config = {"text_config": {"hidden_size": 5120, "moe_intermediate_size": 2304, "swiglu_limit": 10.0}}
        def small(path):
            return json.dumps(index if path.startswith("model.") else config).encode()
        ranges = []
        def ranged(shard, start, size):
            if start == 0:
                return struct.pack("<Q", len(encoded))
            if start == 8:
                return encoded
            self.assertLessEqual(size, LIMIT)
            ranges.append((start, size))
            return bytes([0x38]) * size
        with tempfile.TemporaryDirectory() as directory, \
                patch("vesper.shared.read_small", side_effect=small), \
                patch("vesper.shared.read_range", side_effect=ranged), \
                patch("vesper.resources.physical_memory", return_value={"available": 128 * GiB}), \
                patch("vesper.shared.np.savez") as save:
            result = fetch_shared_fixture(Path(directory))
            arrays = save.call_args.kwargs
            self.assertEqual(result["weights_fetched_bytes"], SHARED_BYTES)
            self.assertEqual(sum(size for _, size in ranges), SHARED_BYTES)
            self.assertEqual(len(ranges), 9)  # two ranges per weight, one per scale
            self.assertEqual(sum(a.nbytes for a in arrays.values()), SHARED_BYTES)
            self.assertTrue(all(np.all(a == 0x38) for a in arrays.values()))


class AggregationTests(unittest.TestCase):
    def test_sparse_participation_uses_expert_id_order_and_token_order(self):
        ids = np.array([[383, 1, 4, 2, 0, 3], [5, 4, 3, 2, 1, 0]], np.int64)
        rows = participation(ids)
        self.assertEqual(list(rows), [0, 1, 2, 3, 4, 5, 383])
        self.assertEqual(rows[0].tolist(), [0, 1])
        self.assertEqual(rows[5].tolist(), [1])
        self.assertEqual(rows[383].tolist(), [0])
        self.assertEqual(participation(np.empty((0, 6), np.int64)), {})

    def test_invalid_routes_and_overlapping_same_expert_writes_are_rejected(self):
        valid = np.arange(6, dtype=np.int64).reshape(1, 6)
        invalid = [valid.astype(np.float32), valid[:, :5], np.tile(valid, (33, 1)),
                   np.array([[-1, 1, 2, 3, 4, 5]], np.int64), np.array([[0, 1, 2, 3, 4, 384]], np.int64),
                   np.array([[0, 0, 2, 3, 4, 5]], np.int64)]
        for ids in invalid:
            with self.assertRaises(ValueError):
                participation(ids)

    def test_out_of_order_and_early_shared_calls_do_not_touch_device(self):
        accumulator = MoEAccumulator()
        accumulator.device = MagicMock()
        accumulator.state, accumulator.order, accumulator.position = "routed", [0, 2], 0
        with patch.object(accumulator, "_upload") as upload, patch.object(accumulator, "_launch") as launch:
            with self.assertRaisesRegex(ValueError, "ascending"):
                accumulator.add(2, np.zeros((1, HIDDEN), np.uint16))
            with self.assertRaisesRegex(ValueError, "precede"):
                accumulator.finish(np.zeros((1, HIDDEN), np.uint16))
            upload.assert_not_called()
            launch.assert_not_called()

    def test_missing_contributions_and_nonfinite_bf16_fail_before_begin(self):
        accumulator = MoEAccumulator()
        ids = np.arange(6, dtype=np.int64).reshape(1, 6)
        outputs = {e: np.zeros((1, HIDDEN), np.uint16) for e in range(6)}
        with patch.object(accumulator, "begin") as begin:
            with self.assertRaisesRegex(ValueError, "exactly the selected"):
                accumulator.run(ids, {}, outputs[0])
            outputs[0][0, 0] = 0x7f80
            with self.assertRaisesRegex(ValueError, "finite"):
                accumulator.run(ids, outputs, outputs[1])
            outputs[0][0, 0] = 0
            with self.assertRaisesRegex(ValueError, "BF16 storage"):
                accumulator.run(ids, outputs, np.zeros((1, HIDDEN), np.float32))
            begin.assert_not_called()

    def test_closed_runtime_and_invalid_workspace_are_rejected(self):
        for size in (0, 33, True):
            with self.assertRaises(ValueError):
                MoEAccumulator(max_tokens=size)
        with self.assertRaisesRegex(RuntimeError, "context manager"):
            MoEAccumulator().begin(np.arange(6, dtype=np.int64).reshape(1, 6))


if __name__ == "__main__":
    unittest.main()
