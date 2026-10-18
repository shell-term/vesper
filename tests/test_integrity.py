import hashlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from vesper.integrity import DeviceIntegrity, digest_shape, hash_chunks
from vesper.timing import Timings


class IntegrityTests(unittest.TestCase):
    def test_known_sha256_and_every_chunk_boundary(self):
        abc = np.frombuffer(b"abc", np.uint8)
        actual = np.empty(digest_shape(3), np.uint8)
        hash_chunks(abc, actual)
        self.assertEqual(actual.tobytes().hex(), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        for size in (1, 55, 56, 63, 64, 65, 4095, 4096, 4097, 8192):
            data = np.arange(size, dtype=np.int64).astype(np.uint8)
            actual = np.empty(digest_shape(size), np.uint8)
            hash_chunks(data, actual)
            for chunk, digest in enumerate(actual):
                self.assertEqual(digest.tobytes(), hashlib.sha256(data[chunk * 4096:(chunk + 1) * 4096]).digest())

    def test_partial_or_excessive_buffers_are_rejected(self):
        for size in (0, -1, 2**30 + 1, True):
            with self.assertRaises(ValueError):
                digest_shape(size)
        data = np.zeros(8192, np.uint8)
        for invalid in (data[::2], data.view(np.uint16), data.reshape(2, -1)):
            with self.assertRaises(ValueError):
                hash_chunks(invalid, np.zeros((2, 32), np.uint8))
        with self.assertRaises(ValueError):
            hash_chunks(data, np.zeros((1, 32), np.uint8))
        with self.assertRaisesRegex(ValueError, "overwrite"):
            hash_chunks(data, data[:64].reshape(2, 32))

    def test_audit_readback_cannot_overwrite_its_own_expected_digest(self):
        checker = object.__new__(DeviceIntegrity)
        checker.shape = (2, 32)
        expected = np.zeros(checker.shape, np.uint8)
        with self.assertRaisesRegex(ValueError, "original digest"):
            checker.verify(0, expected, target=expected)
        with self.assertRaisesRegex(ValueError, "comparison"):
            checker.verify(0, expected, comparison=expected.view(np.bool_))

    def test_closed_or_unowned_input_is_rejected_before_launching_cuda(self):
        checker = object.__new__(DeviceIntegrity)
        checker.pointer, checker.size = None, 4096
        with self.assertRaisesRegex(RuntimeError, "closed"):
            checker.digests(123)
        checker.pointer = 456
        checker.device = SimpleNamespace(allocations={123: 32})
        with self.assertRaisesRegex(ValueError, "owned source"):
            checker.digests(123, size=33)
        with self.assertRaisesRegex(ValueError, "owned source"):
            checker.digests(999, size=1)

    def test_timing_records_failures_and_returns_per_call_deltas(self):
        timings = Timings(enabled=True)
        with patch("vesper.timing.time.perf_counter_ns", side_effect=(100, 300, 400, 900)):
            with timings.section("copy"):
                pass
            before = timings.snapshot()
            with self.assertRaises(ValueError):
                with timings.section("copy"):
                    raise ValueError("injected")
        self.assertEqual(timings.since(before), {"copy": {"calls": 1, "total_ms": 0.0005}})
        disabled = Timings()
        with patch("vesper.timing.time.perf_counter_ns") as clock:
            with disabled.section("copy"):
                pass
            clock.assert_not_called()
        self.assertEqual(disabled.snapshot(), {})
