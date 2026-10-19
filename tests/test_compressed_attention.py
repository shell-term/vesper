import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from vesper.checkpoint import MODEL, REVISION
from vesper.compressed_attention import NativeCompressedAttention, CompressedAttentionSnapshot, validate_frame, initial_state
from vesper.compressed_attention_fixture import LAYOUT, FIXTURE_BYTES, fetch_compressed_attention_fixture, load_compressed_attention_fixture
from vesper.compressed_attention_validation import expected_metadata, validate_goldens, boundary_names
from vesper.prefix import STORAGE_DTYPES
from vesper.projections import validate_projection_weights
from vesper.resources import GiB


def frame_args():
    # Identity stubs are for publication contracts, not numerical comparison.
    ops = {l: SimpleNamespace(opened=True, layer=l, ordinal=0) for l in (2, 20, 24)}
    other = [SimpleNamespace(opened=True, ordinal=0) for _ in range(4)]
    return {}, ops, *other


def xs(sequence=1):
    return {l: np.zeros((1, sequence, 5120), np.uint16) for l in (2, 20, 24)}


class CompressedAttentionTests(unittest.TestCase):
    def test_full_frame_layout_and_context_bounds(self):
        self.assertEqual(validate_frame(xs(3), 1, 0), 3)
        self.assertEqual(validate_frame(xs(), 1, 63), 1)
        for values, batch, pos in ((xs(3), 1, 1), (xs(), 1, 64), (xs(), 1, True), (xs(), 2, 0),
                ({2: xs()[2]}, 1, 0), ({l: a.astype(np.float32) for l, a in xs().items()}, 1, 0),
                ({l: np.full_like(a, 0x7fc0) for l, a in xs().items()}, 1, 0)):
            with self.assertRaises(ValueError): validate_frame(values, batch, pos)

    def test_operator_layer_and_device_identity_before_allocation(self):
        args = frame_args()
        args[1][24].layer = 20
        with patch("vesper.compressed_attention.initial_state", side_effect=AssertionError("No allocation expected")):
            with self.assertRaises(ValueError): NativeCompressedAttention(*args).__enter__()
        args = frame_args()
        args[1][20].ordinal = 1
        with self.assertRaises(ValueError): NativeCompressedAttention(*args).__enter__()
        with self.assertRaises(ValueError): validate_projection_weights({}, layer=3)

    def test_pressure_refuses_before_frame_allocation(self):
        frame = NativeCompressedAttention(*frame_args())
        with patch("vesper.resources.physical_memory", return_value={"available": 16 * GiB}), patch("vesper.compressed_attention.initial_state", side_effect=AssertionError("No allocation expected")):
            with self.assertRaises(MemoryError): frame.__enter__()
        self.assertFalse(frame.opened)
        self.assertEqual(frame.budget.reserved, 0)

    def test_finalizer_failure_keeps_all_committed_state(self):
        with patch("vesper.resources.physical_memory", return_value={"available": 64 * GiB}):
            with NativeCompressedAttention(*frame_args()) as frame:
                candidate = initial_state(1)
                candidate["windows"][24].fill(0x3f80)
                def fail(_): raise OSError("Injected finalizer failure")
                with patch("vesper.compressed_attention.stage_frame", return_value=({}, candidate)):
                    with self.assertRaises(OSError): frame.step(xs(), finalize=fail)
                    self.assertEqual(frame.position, 0)
                    self.assertFalse(frame.inspect()["state"]["windows"][24].any())
                    frame.step(xs())
                    self.assertEqual(frame.position, 1)
                    self.assertTrue((frame.inspect()["state"]["windows"][24] == 0x3f80).all())

    def test_late_headroom_failure_refuses_candidate_publication(self):
        with patch("vesper.resources.physical_memory", return_value={"available": 64 * GiB}):
            with NativeCompressedAttention(*frame_args()) as frame:
                original, count = frame.budget.acquire, 0
                def fail_late(size):
                    nonlocal count
                    count += 1
                    if count == 2: raise MemoryError("Injected final publication pressure")
                    original(size)
                with patch.object(frame.budget, "acquire", side_effect=fail_late), patch("vesper.compressed_attention.stage_frame", return_value=({}, initial_state(1))):
                    with self.assertRaises(MemoryError): frame.step(xs())
                self.assertEqual(frame.position, 0)
            self.assertEqual(frame.budget.reserved, 0)

    def test_snapshot_reset_failure_identity_and_copies(self):
        with patch("vesper.resources.physical_memory", return_value={"available": 64 * GiB}):
            with NativeCompressedAttention(*frame_args()) as frame:
                handle = frame.snapshot()
                with self.assertRaises(ValueError): frame.rollback(CompressedAttentionSnapshot(0))
                with self.assertRaises(RuntimeError): frame.snapshot()
                with patch("vesper.compressed_attention.initial_state", side_effect=MemoryError("Injected allocation failure")):
                    with self.assertRaises(MemoryError): frame.reset()
                self.assertIs(frame.handle, handle)
                restored = frame.rollback(handle)
                restored["state"]["windows"][2].fill(0x3f80)
                self.assertFalse(frame.inspect()["state"]["windows"][2].any())
                with self.assertRaises(ValueError): frame.discard(handle)
                handle = frame.snapshot()
                frame.reset()
                with self.assertRaises(ValueError): frame.rollback(handle)

    def test_existing_fixture_destination_refuses_before_network(self):
        with tempfile.TemporaryDirectory() as tmp, patch("vesper.compressed_attention_fixture.read_small", side_effect=AssertionError("No network expected")):
            with self.assertRaises(FileExistsError): fetch_compressed_attention_fixture(tmp)

    def test_forged_fixture_shape_refuses_before_array_allocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tensors = [{"name": n, "dtype": d, "shape": list(s), "bytes": int(np.prod(s)) * STORAGE_DTYPES[d].itemsize, "sha256": "0" * 64}
                       for n, (d, s) in LAYOUT.items()]
            manifest = {"kind": "original_compressed_attention_fixture", "model": MODEL, "revision": REVISION,
                        "layers": [2, 20, 24], "tensors": tensors, "weights_fetched_bytes": FIXTURE_BYTES}
            (root / "manifest.json").write_text(json.dumps(manifest))
            path = root / "native-compressed-attention.npz"
            with zipfile.ZipFile(path, "w") as archive:
                for n in LAYOUT:
                    with archive.open(n + ".npy", "w") as stream:
                        np.lib.format.write_array_header_1_0(stream, {"shape": (2**50,), "fortran_order": False, "descr": "<u2"})
            with patch("vesper.compressed_attention_fixture.np.load", side_effect=AssertionError("No array allocation expected")):
                with self.assertRaises(ValueError): load_compressed_attention_fixture(path)

    def test_published_complete_goldens_and_tamper_refusal(self):
        root = Path(__file__).resolve().parents[1]
        report = json.loads((root / "results/2026-10-19-compressed-attention/compressed-attention.json").read_bytes())
        self.assertEqual(len(list(expected_metadata())), 103)
        self.assertEqual(len(validate_goldens(report)), 103)
        self.assertEqual({len(boundary_names(m)) for m in expected_metadata()}, {289, 323})
        for change in (lambda r: r["frame_cases"].pop(), lambda r: r["reference_goldens"][0]["stage_sha256"].popitem(),
                lambda r: r["frame_cases"][0].update(position=1), lambda r: r["original_methods_ast_sha256"].update({"Attention.forward": "0" * 64}),
                lambda r: r["original_index_quant_cuda_sha256"].update({"True": "0" * 64}),
                lambda r: r["native_profiles"]["projections"]["24"].update(converted_wo_a_bf16_sha256="0" * 64),
                lambda r: r["unchanged_complete_Attention_controls"][0][0].update(mismatches=1),
                lambda r: r["faults"][-1].update(position_preserved=False), lambda r: r["snapshot_events"][0].update(restored_position=4),
                lambda r: r["memory"]["cleanup"]["projection_24"].update(device_owned_bytes=1), lambda r: r["negative_controls"].pop()):
            changed = copy.deepcopy(report)
            change(changed)
            with self.assertRaises(ValueError): validate_goldens(changed)


if __name__ == "__main__": unittest.main()
