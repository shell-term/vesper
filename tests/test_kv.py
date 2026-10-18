import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from vesper.kv import NativeWindowKV, WindowSnapshot, validate_kv_input, window_indices
from vesper.kv_validation import kv_inputs, validate_goldens


class WindowKVTests(unittest.TestCase):
    def test_prefill_indices_match_causal_token_windows(self):
        for length in (1, 2, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257):
            expected = np.array([[token if token <= row else -1 for token in
                                 range(max(0, row - 127), max(0, row - 127) + min(length, 128))]
                                 for row in range(length)], np.int32)
            actual = window_indices(2, length, 0)
            self.assertTrue(actual.flags.c_contiguous)
            np.testing.assert_array_equal(actual[0], expected)
            np.testing.assert_array_equal(actual[1], expected)
            actual[0, 0, 0] = 99
            self.assertNotEqual(actual[1, 0, 0], 99)

    def test_decode_indices_preserve_order_and_leading_padding_across_wraps(self):
        for position in range(1, 1026):
            expected = np.array([token % 128 if token >= 0 else -1 for token in range(position - 127, position + 1)], np.int32)
            np.testing.assert_array_equal(window_indices(1, 1, position)[0, 0], expected)

    def test_unsupported_chunk_batch_and_position_are_refused(self):
        for args in ((0, 1, 0), (3, 1, 0), (True, 1, 0), (1, 0, 0), (1, 258, 0),
                     (1, 2, 1), (1, 1, -1), (1, 1, 4096), (1, 1, 1.), (1, 1, True)):
            with self.assertRaises(ValueError):
                window_indices(*args)

    def test_invalid_bf16_layout_position_and_nonfinite_refused(self):
        valid = kv_inputs(2, 2, 1)
        validate_kv_input(valid, 2, 0, 0)
        for value in (valid.astype(np.float32), valid[:, :, ::-1], valid[0], valid[:, :0],
                      np.zeros((2, 258, 512), np.uint16), np.zeros((2, 2, 448), np.uint16),
                      np.full_like(valid, 0x7F80), np.full_like(valid, 0xFFC1)):
            with self.assertRaises(ValueError):
                validate_kv_input(value, 2, 0, 0)
        with self.assertRaises(ValueError):
            validate_kv_input(valid[:, :1].copy(), 2, 129, 128)

    def test_finite_encoding_fixture_covers_every_finite_bf16_bit_pattern(self):
        values = kv_inputs(1, 128, 1, "finite_bf16_encodings")
        expected = np.arange(65536, dtype=np.uint16)
        expected = expected[(expected & 0x7F80) != 0x7F80]
        np.testing.assert_array_equal(np.unique(values), expected)

    def test_snapshot_identity_and_failed_publication_do_not_change_active_state(self):
        native = NativeWindowKV()
        native.position, native.active, native.buffers = 129, 0, [1, 2, 3]
        with patch.object(native, "_require_open"), patch.object(native, "_copy") as transfer:
            transfer.side_effect = OSError("copy failed")
            with self.assertRaises(OSError):
                native.snapshot()
            self.assertIsNone(native.handle)
            transfer.side_effect = None
            handle = native.snapshot()
            with self.assertRaisesRegex(RuntimeError, "one bounded"):
                native.snapshot()
            for foreign in (None, WindowSnapshot(129), copy.copy(handle)):
                with self.assertRaisesRegex(ValueError, "Foreign"):
                    native.rollback(foreign)
            native.position, native.active = 131, 1
            with patch.object(native, "_cache", side_effect=OSError("publication failed")):
                with self.assertRaises(OSError):
                    native.rollback(handle)
            self.assertEqual((native.position, native.active), (131, 1))
            self.assertIs(native.handle, handle)
            with patch.object(native, "_cache", return_value=np.zeros((1, 128, 512), np.uint16)):
                self.assertEqual(native.rollback(handle)["position"], 129)
            self.assertIsNone(native.handle)
            with self.assertRaises(ValueError):
                native.rollback(handle)

    def test_closed_operations_and_failed_entry_keep_allowance_zero(self):
        native = NativeWindowKV()
        for operation in (native.inspect, native.reset, native.snapshot, lambda: native.rollback(None),
                          lambda: native.append(None), lambda: native.discard(None)):
            with self.assertRaisesRegex(RuntimeError, "not open"):
                operation()
        with patch("vesper.resources.physical_memory", return_value={"available": 16 * 1024**3}):
            with self.assertRaises(MemoryError):
                native.__enter__()
        self.assertEqual(native.budget.reserved, 0)
        native.opened = True
        with self.assertRaisesRegex(RuntimeError, "already open"):
            native.__enter__()

    def test_published_source_bindings_coverage_and_failure_proofs_are_required(self):
        root = Path(__file__).resolve().parents[1]
        report = json.loads((root / "results/2026-10-18-window-kv/window-kv.json").read_bytes())
        self.assertEqual(len(validate_goldens(report)), 398)
        source = (root / "src/vesper/kernels/format_probe.cu").read_bytes() + b"\n" + (root / "src/vesper/kernels/kv_probe.cu").read_bytes()
        self.assertEqual(hashlib.sha256(source.replace(b"\r\n", b"\n")).hexdigest(),
                         report["native_profile"]["compiler"]["concatenated_source_sha256"])
        for change in (lambda r: r["reference_goldens"].pop(), lambda r: r["cases"][0]["checks"].pop(),
                       lambda r: r["cases"][0]["checks"][0].update(actual_sha256="0" * 64),
                       lambda r: r["negative_controls"].pop(), lambda r: r["faults"][0].update(active_state_unchanged=False),
                       lambda r: r["native_profile"].update(cubin_sha256="0" * 64)):
            broken = copy.deepcopy(report)
            change(broken)
            with self.assertRaises(ValueError):
                validate_goldens(broken)


if __name__ == "__main__":
    unittest.main()
