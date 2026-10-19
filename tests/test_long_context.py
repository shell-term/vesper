import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from vesper.attention import NativeAttention, validate_inputs
from vesper.attention_storage import trace_shapes
from vesper.compressed_attention import NativeCompressedAttention, initial_state, validate_diagnostic_seed, validate_frame
from vesper.compression import validate_compression_input
from vesper.kv import window_indices
from vesper.long_context_validation import compact_comparison, expected_metadata, replay_comparison, seed_specs, validate_goldens
from vesper.resources import GiB, MiB


def inputs(sequence=1):
    return {l: np.zeros((1, sequence, 5120), np.uint16) for l in (2, 20, 24)}


def frame():
    projections = {l: SimpleNamespace(opened=True, layer=l, ordinal=0) for l in (2, 20, 24)}
    ops = [SimpleNamespace(opened=True, ordinal=0) for _ in range(4)]
    ops[-1].profile = "long_context"
    return NativeCompressedAttention({}, projections, *ops, profile="long_context", capacity=1152)


class LongContextTests(unittest.TestCase):
    def test_large_attention_requires_explicit_registered_shape(self):
        q, kv = np.zeros((1, 1, 64, 512), np.uint16), np.zeros((1, 16520, 512), np.uint16)
        sink, idx = np.zeros(64, np.float32), np.zeros((1, 1, 640), np.int32)
        with self.assertRaises(ValueError): validate_inputs(q, kv, sink, idx)
        shapes = validate_inputs(q, kv, sink, idx, True, profile="long_context")
        self.assertEqual(shapes[0], (1, 1, 10, 64, 64))
        for s, h, d, k in ((2, 64, 512, 640), (1, 16, 512, 640), (1, 64, 256, 640), (1, 64, 512, 641)):
            with self.assertRaises(ValueError): trace_shapes(1, s, h, d, k, profile="long_context")
        with self.assertRaises(ValueError): validate_inputs(q, np.zeros((1, 16521, 512), np.uint16), sink, idx, profile="long_context")
        self.assertEqual(NativeAttention(Path("unused")).device_cap, 16 * MiB)
        self.assertEqual(NativeAttention(Path("unused"), profile="long_context").device_cap, 64 * MiB)
        with self.assertRaises(ValueError): NativeAttention(Path("unused"), profile="unbounded")

    def test_context_limits_and_default_admission_are_retained(self):
        with self.assertRaises(ValueError): validate_frame(inputs(), 1, 64)
        for capacity in (1152, 16392):
            self.assertEqual(validate_frame(inputs(), 1, capacity - 1, profile="long_context", capacity=capacity), 1)
            with self.assertRaises(ValueError): validate_frame(inputs(), 1, capacity, profile="long_context", capacity=capacity)
        with self.assertRaises(ValueError): initial_state(1, profile="long_context", capacity=20000)
        with self.assertRaises(ValueError): validate_compression_input(inputs()[2], 1, 1024, 1024)
        validate_compression_input(inputs()[2], 1, 16391, 16391, max_position=16392)
        with self.assertRaises(ValueError): window_indices(1, 1, 16384)
        np.testing.assert_array_equal(window_indices(1, 1, 16384, max_position=16392)[0, 0], np.r_[1:128, 0])
        with self.assertRaises(ValueError): window_indices(1, 1, 16392, max_position=16392)

    def test_diagnostic_seed_rejects_nonfinite_future_and_wrong_layout(self):
        source = initial_state(1, profile="long_context", capacity=1152)
        validate_diagnostic_seed(source, 1, 1152, 1024)
        for mutate in (
                lambda s: s["windows"][2].fill(0x7f80),
                lambda s: s["windows"].update({2: s["windows"][2].astype(np.float32)}),
                lambda s: s["compressors"][2][0].fill(np.nan),
                lambda s: s["compressors"][2][1].fill(np.inf),
                lambda s: s["compressors"][2][2].fill(0x4529),
                lambda s: s["compressors"][2][2].__setitem__((0, 512, 0), 1),
                lambda s: s["index_keys"][20].__setitem__((0, 1024, 0), 1),
                lambda s: s.update(sharing={}),
                lambda s: s["index_keys"].pop(2)):
            state = copy.deepcopy(source)
            mutate(state)
            with self.assertRaises(ValueError): validate_diagnostic_seed(state, 1, 1152, 1024)
        for pos in (0, 1023, 1152, True):
            with self.assertRaises(ValueError): validate_diagnostic_seed(source, 1, 1152, pos)

    def test_seed_copy_failure_pressure_and_snapshot_are_atomic(self):
        source = initial_state(1, profile="long_context", capacity=1152)
        source["windows"][2].fill(0x3f80)
        with patch("vesper.resources.physical_memory", return_value={"available": 64 * GiB}):
            with frame() as f:
                with patch("vesper.compressed_attention.copy.deepcopy", side_effect=MemoryError("copy")):
                    with self.assertRaises(MemoryError): f.prime_diagnostic(source, 1024)
                self.assertEqual(f.position, 0)
                self.assertFalse(f.state["windows"][2].any())
                original, count = f.budget.acquire, 0
                def late(size):
                    nonlocal count
                    count += 1
                    if count == 2: raise MemoryError("late pressure")
                    return original(size)
                with patch.object(f.budget, "acquire", side_effect=late):
                    with self.assertRaises(MemoryError): f.prime_diagnostic(source, 1024)
                self.assertEqual(f.position, 0)
                handle = f.snapshot()
                with self.assertRaises(ValueError): f.prime_diagnostic(source, 1024)
                f.discard(handle)
                f.prime_diagnostic(source, 1024)
                source["windows"][2].fill(0)
                self.assertTrue((f.state["windows"][2] == 0x3f80).all())
                with self.assertRaises(ValueError): f.prime_diagnostic(source, 1024)
                handle = f.snapshot()
                f.position = 1025
                f.state["windows"][2].fill(0)
                self.assertEqual(f.rollback(handle)["position"], 1024)
                self.assertTrue((f.state["windows"][2] == 0x3f80).all())
                f.reset()
                self.assertEqual(f.position, 0)
                self.assertEqual(f.state["index_keys"][20].shape, (1, 1152, 128))
            self.assertEqual(f.budget.reserved, 0)

    def test_long_frame_refuses_unqualified_operator_before_allocation(self):
        f = frame()
        f.attention.profile = "baseline"
        with patch("vesper.compressed_attention.initial_state", side_effect=AssertionError("no allocation")):
            with self.assertRaises(ValueError): f.__enter__()
        self.assertEqual(f.budget.reserved, 0)

    def test_compact_replay_detects_bits_shape_dtype_size_and_missing_boundaries(self):
        arrays = {"a": np.zeros((2, 3), np.uint16), "b": np.ones(4, np.float32)}
        golden = compact_comparison(arrays, arrays, ["a", "b"])
        self.assertEqual(replay_comparison(golden, arrays, ["a", "b"])["status"], "pass")
        for a in (np.ones_like(arrays["a"]), arrays["a"].reshape(3, 2), arrays["a"].view(np.int16)):
            self.assertEqual(replay_comparison(golden, {**arrays, "a": a}, ["a", "b"])["status"], "fail")
        with self.assertRaises(ValueError): replay_comparison(golden, {"b": arrays["b"]}, ["a", "b"])
        changed = copy.deepcopy(golden)
        changed["boundaries"][0][1] += 1
        self.assertEqual(replay_comparison(changed, arrays, ["a", "b"])["status"], "fail")

    def test_published_complete_goldens_refuse_coverage_and_provenance_tampering(self):
        root = Path(__file__).resolve().parents[1] / "results/2026-10-19-long-context"
        report = json.loads((root / "long-context.json").read_bytes())
        self.assertEqual(len(list(expected_metadata())), 1441)
        self.assertEqual(len(list(seed_specs())), 9)
        self.assertEqual(len(validate_goldens(report)), 1441)
        changed = {**report, "frame_cases": report["frame_cases"][:-1]}
        with self.assertRaises(ValueError): validate_goldens(changed)
        for mutate in (lambda r: r["boundaries"][0].__setitem__(1, r["boundaries"][0][1] + 1),
                       lambda r: r.update(actual_layout_sha256="0" * 64),
                       lambda r: r["observations"].update(consumer_respects_candidates=False)):
            row = copy.deepcopy(report["frame_cases"][0])
            mutate(row)
            changed = {**report, "frame_cases": [row, *report["frame_cases"][1:]]}
            with self.assertRaises(ValueError): validate_goldens(changed)
        for key, mutate in (("original_methods_ast_sha256", lambda v: v.update({"Attention.forward": "0" * 64})),
                            ("original_seed_controls", lambda v: v[0]["checks"].pop()),
                            ("faults", lambda v: v[0].update(position_preserved=False)),
                            ("snapshot_events", lambda v: v[0].update(restored_position=1024)),
                            ("instrumentation_counts", lambda v: v.update(index_sum=0)),
                            ("negative_controls", lambda v: v.pop())):
            value = copy.deepcopy(report[key])
            mutate(value)
            with self.assertRaises(ValueError): validate_goldens({**report, key: value})

    def test_published_summary_binds_complete_paired_and_native_evidence(self):
        root = Path(__file__).resolve().parents[1] / "results/2026-10-19-long-context"
        summary = json.loads((root / "summary.json").read_bytes())
        for name, digest in summary["artifacts_sha256"].items():
            self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), digest)
        native = json.loads((root / "native-only-recheck.json").read_bytes())
        self.assertEqual(native["reference_report_sha256"], summary["artifacts_sha256"]["long-context.json"])
        self.assertEqual(native["status"], "pass")
        self.assertEqual(len(native["frame_cases"]), 1441)
        self.assertEqual(sum(r["boundary_comparisons"] for r in native["frame_cases"]), summary["boundary_comparisons"])
        self.assertEqual(sum(r["compared_storage_elements"] for r in native["frame_cases"]), summary["compared_storage_elements"])
        self.assertFalse(any(native[n + suffix] for n in ("torch", "tilelang") for suffix in ("_imported", "_installed")))


if __name__ == "__main__": unittest.main()
