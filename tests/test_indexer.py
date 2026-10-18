import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from vesper.checkpoint import MODEL, REVISION
from vesper.indexer import MAX_INDEX_POSITIONS, NativeIndexer, finite_bf16, validate_query, validate_selection
from vesper.indexer_fixture import DTYPES, INDEXER_BYTES, INDEXER_LAYOUT, fetch_indexer_fixture, load_indexer_fixture
from vesper.indexer_state import IndexSnapshot, NativeIndexSession
from vesper.indexer_validation import component_cases, expected_metadata, validate_goldens
from vesper.normalization import mean_threads, validate_norm, validate_rope
from vesper.resources import GiB


def toy_backend():
    """Controlled publication outputs; this is not a numerical reference."""
    def queries(x, qr, layer, position):
        return {"quant_rounded_bf16": np.zeros((*x.shape[:2], 32, 128), np.uint16),
                "weights_scaled_bf16": np.zeros((*x.shape[:2], 32), np.uint16)}

    def select(scores, lengths, offset, *, candidate_source=False, candidates=None):
        ids = np.broadcast_to(np.arange(min(512, scores.shape[-1]), dtype=np.int32) + offset,
                              (*scores.shape[:2], min(512, scores.shape[-1]))).copy()
        result = {"indices_i32": ids}
        if candidate_source: result["candidates_bool"] = np.ones(scores.shape, np.bool_)
        return result

    return SimpleNamespace(opened=True,
        key=lambda latent, *args: {"quant_rounded_bf16": latent[..., :128].copy()}, queries=queries,
        score=lambda q, k, w: {"score_bf16": np.zeros((*q.shape[:2], k.shape[1]), np.uint16)}, select=select)


def frame_inputs(sequence, position=0):
    xs = {layer: np.full((1, sequence, 5120), 0x3f80, np.uint16) for layer in (2, 20, 24)}
    qrs = {layer: np.zeros((1, sequence, 1280), np.uint16) for layer in xs}
    count2 = sequence // 2 if position == 0 else (position + 1) // 2 - position // 2
    latents = {2: np.full((1, count2, 512), 0x4000, np.uint16) if count2 else None,
               20: np.full((1, sequence, 512), 0x4040, np.uint16)}
    return xs, qrs, latents


class IndexerTests(unittest.TestCase):
    def test_query_layout_and_finite_bf16_contract(self):
        x, qr = np.zeros((2, 8, 5120), np.uint16), np.zeros((2, 8, 1280), np.uint16)
        validate_query(x, qr, 24, 131064)
        for a, b, layer, pos in ((x.astype(np.float32), qr, 2, 0), (x[..., ::-1], qr, 2, 0),
                (x, qr[:, :1], 2, 0), (x, qr, 3, 0), (x, qr, True, 0), (x, qr, 2, True),
                (x, qr, 20, 131065), (np.full_like(x, 0x7f80), qr, 2, 0)):
            with self.assertRaises(ValueError): validate_query(a, b, layer, pos)
        with self.assertRaises(ValueError): finite_bf16(np.zeros((1, 1025, 128), np.uint16), 128, 1024)

    def test_selection_bound_and_candidate_ownership(self):
        scores = np.zeros((2, 8, MAX_INDEX_POSITIONS), np.uint16)
        lengths = np.full((2, 8), MAX_INDEX_POSITIONS, np.int32)
        validate_selection(scores, lengths, 128, True, None)
        validate_selection(scores, lengths, 128, False, np.ones(scores.shape, np.bool_))
        for s, lens, offset, source, candidates in (
                (np.zeros((1, 1, MAX_INDEX_POSITIONS + 1), np.uint16), lengths[:1, :1], 128, False, None),
                (scores, lengths.astype(np.int64), 128, False, None),
                (scores, lengths + 1, 128, False, None), (scores, lengths, 2**31 - 1, False, None),
                (scores, lengths, 128, True, np.ones(scores.shape, np.bool_)),
                (scores, lengths, 128, False, np.ones(scores.shape, np.uint8)),
                (np.full_like(scores, 0x7fc0), lengths, 128, False, None)):
            with self.assertRaises(ValueError): validate_selection(s, lens, offset, source, candidates)

    def test_index_key_position_refusal_precedes_gemm(self):
        native = object.__new__(NativeIndexer)
        import threading
        native.lock, native._ready = threading.RLock(), lambda: None
        native._bf16_linear = lambda *args: self.fail("Invalid positions reached GEMM")
        latent = np.zeros((1, 2, 512), np.uint16)
        for positions in (np.array([0], np.int32), np.array([0, 2], np.int64), np.array([0, 131072], np.int32)):
            with self.assertRaises(ValueError): native.key(latent, 2, positions)

    def test_new_norm_and_rotary_width_preserves_admission_bounds(self):
        for rows in (1, 2, 7, 16, 257, 514): self.assertEqual(mean_threads(rows, 128), 32)
        self.assertEqual(validate_norm(np.zeros((2, 257, 128), np.uint16), np.zeros(128, np.uint16)), (514, 128))
        self.assertEqual(validate_rope(np.zeros((2, 8, 32, 128), np.uint16), 131064, "compressed-yarn", False), (2, 8, 32, 128))
        with self.assertRaises(ValueError): validate_norm(np.zeros((515, 128), np.uint16), np.zeros(128, np.uint16))

    def test_missing_weights_and_existing_destination_refuse_before_io(self):
        with patch("vesper.indexer.Driver", side_effect=AssertionError("No driver access expected")):
            with self.assertRaises(ValueError): NativeIndexer({})
        with tempfile.TemporaryDirectory() as tmp, patch("vesper.indexer_fixture.read_small", side_effect=AssertionError("No network expected")):
            with self.assertRaises(FileExistsError): fetch_indexer_fixture(Path(tmp))

    def test_pressure_refuses_before_allocating_caches(self):
        state = NativeIndexSession(toy_backend())
        with patch("vesper.resources.physical_memory", return_value={"available": 16 * GiB}), patch.object(state, "_clear", side_effect=AssertionError("No allocation expected")):
            with self.assertRaises(MemoryError): state.__enter__()
        self.assertFalse(state.opened)
        self.assertEqual(state.budget.reserved, 0)

    def test_publication_copies_all_shared_outputs(self):
        with patch("vesper.resources.physical_memory", return_value={"available": 64 * GiB}):
            with NativeIndexSession(toy_backend()) as state:
                result = state.step(*frame_inputs(3))
                before = state.inspect()
                for a in result.values(): a.fill(0)
                ids = state.shared_indices(3, 0, 3)
                ids.fill(-1)
                after = state.inspect()
                self.assertTrue(np.array_equal(before["cache2_bf16"], after["cache2_bf16"]))
                self.assertTrue(np.array_equal(before["cache20_bf16"], after["cache20_bf16"]))
                for name in ("keys20", "candidates20", "indices3", "indices25"):
                    self.assertTrue(np.array_equal(before["shared"][name], after["shared"][name]))
                with self.assertRaises(ValueError): state.shared_indices(25, 1, 3)
                with self.assertRaises(ValueError): state.shared_indices(24, 0, 3)
            state.__exit__()
            self.assertEqual(state.budget.reserved, 0)

    def test_consumer_failure_does_not_publish_staged_owner_keys(self):
        runtime = toy_backend()
        with patch("vesper.resources.physical_memory", return_value={"available": 64 * GiB}):
            with NativeIndexSession(runtime) as state:
                state.step(*frame_inputs(3))
                before, original = state.inspect(), runtime.queries
                def fail_consumer(x, qr, layer, position):
                    if layer == 24: raise RuntimeError("Injected consumer failure")
                    return original(x, qr, layer, position)
                with patch.object(runtime, "queries", side_effect=fail_consumer):
                    with self.assertRaises(RuntimeError): state.step(*frame_inputs(1, 3))
                self.assertEqual(state.position, 3)
                after = state.inspect()
                for n in ("cache2_bf16", "cache20_bf16"): self.assertTrue(np.array_equal(before[n], after[n]))
                state.step(*frame_inputs(1, 3))
                self.assertEqual(state.position, 4)

    def test_invalid_frame_refuses_before_backend(self):
        runtime = toy_backend()
        runtime.queries = lambda *args: self.fail("Invalid frame reached backend")
        with patch("vesper.resources.physical_memory", return_value={"available": 64 * GiB}):
            with NativeIndexSession(runtime) as state:
                xs, qrs, latents = frame_inputs(1)
                latents[2] = np.zeros((1, 1, 512), np.uint16)
                with self.assertRaises(ValueError): state.step(xs, qrs, latents)
                with self.assertRaises(ValueError): state.step(*frame_inputs(1), start_pos=1)
                self.assertEqual(state.position, 0)

    def test_snapshot_identity_and_reset_allocation_failure(self):
        with patch("vesper.resources.physical_memory", return_value={"available": 64 * GiB}):
            with NativeIndexSession(toy_backend()) as state:
                state.step(*frame_inputs(3))
                handle = state.snapshot()
                with self.assertRaises(ValueError): state.rollback(IndexSnapshot(3))
                with self.assertRaises(RuntimeError): state.snapshot()
                with patch("vesper.indexer_state.np.zeros", side_effect=MemoryError("Injected allocation failure")):
                    with self.assertRaises(MemoryError): state.reset()
                self.assertIs(state.handle, handle)
                self.assertEqual(state.position, 3)
                state.step(*frame_inputs(1, 3))
                restored = state.rollback(handle)
                self.assertEqual(restored["position"], 3)
                restored["cache20_bf16"].fill(0)
                self.assertTrue((state.inspect()["cache20_bf16"][:, :3] == 0x4040).all())
                with self.assertRaises(ValueError): state.discard(handle)
                handle = state.snapshot()
                state.reset()
                with self.assertRaises(ValueError): state.rollback(handle)

    def test_priming_is_explicit_bounded_and_copies_inputs(self):
        with patch("vesper.resources.physical_memory", return_value={"available": 64 * GiB}):
            with NativeIndexSession(toy_backend(), capacity=8) as state:
                keys = np.full((1, 3, 128), 0x3f80, np.uint16)
                with self.assertRaises(ValueError): state.prime(keys, keys, 3)
                state.prime(keys[:, :1].copy(), keys, 3)
                keys.fill(0)
                self.assertTrue((state.inspect()["cache20_bf16"][:, :3] == 0x3f80).all())
                with self.assertRaises(ValueError): state.prime(keys[:, :1].copy(), keys, 3)

    def test_forged_npy_shape_refuses_before_numpy_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entries = [{"name": n, "dtype": d, "shape": list(s), "bytes": int(np.prod(s)) * DTYPES[d].itemsize, "sha256": "0" * 64}
                       for n, (d, s) in INDEXER_LAYOUT.items()]
            manifest = {"kind": "original_indexer_fixture", "model": MODEL, "revision": REVISION,
                        "layers": [2, 20, 24], "weights_fetched_bytes": INDEXER_BYTES, "tensors": entries}
            (root / "manifest.json").write_text(json.dumps(manifest))
            path = root / "native-indexer.npz"
            with zipfile.ZipFile(path, "w") as archive:
                for name in INDEXER_LAYOUT:
                    with archive.open(name + ".npy", "w") as stream:
                        np.lib.format.write_array_header_1_0(stream, {"shape": (2**50,), "fortran_order": False, "descr": "<u2"})
            with patch("vesper.indexer_fixture.np.load", side_effect=AssertionError("No array allocation expected")):
                with self.assertRaises(ValueError): load_indexer_fixture(path)

    def test_published_goldens_source_and_tamper_refusal(self):
        root = Path(__file__).resolve().parents[1]
        report = json.loads((root / "results/2026-10-18-indexer/indexer.json").read_bytes())
        self.assertEqual(len(list(component_cases())), 261)
        self.assertEqual(len(list(expected_metadata())), 360)
        self.assertEqual(len(validate_goldens(report)), 360)
        source = b"\n".join((root / "src/vesper/kernels" / n).read_bytes() for n in ("format_probe.cu", "projection_probe.cu", "indexer_probe.cu"))
        self.assertEqual(hashlib.sha256(source.replace(b"\r\n", b"\n")).hexdigest(), report["indexer_profile"]["compiler"]["concatenated_source_sha256"])
        for change in (lambda r: r["component_cases"].pop(), lambda r: r["frame_cases"].pop(),
                lambda r: r["frame_cases"][0].update(position=1), lambda r: r["reference_goldens"][0]["stage_sha256"].popitem(),
                lambda r: r["original_quant_cuda_sha256"].update({"True": "0" * 64}),
                lambda r: r["unchanged_original_controls"]["forward_sharing"][0][0].update(mismatches=1),
                lambda r: r["faults"][-1].update(position_preserved=False),
                lambda r: r["snapshot_events"][0].update(restored_position=4),
                lambda r: r["memory"]["indexer_cleanup"].update(device_owned_bytes=1),
                lambda r: r["negative_controls"].pop()):
            changed = copy.deepcopy(report)
            change(changed)
            with self.assertRaises(ValueError): validate_goldens(changed)


if __name__ == "__main__": unittest.main()
