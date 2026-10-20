"""Tiny CPU SGEMM mocks check control flow, not real head/kernel parity."""

from contextlib import ExitStack
import ctypes as C
import unittest
from unittest.mock import Mock, patch

import numpy as np

from vesper import output as shared_output
from vesper import spark_output as spark
from vesper.formats import bf16_bits, bf16_values


class Device:
    def __init__(self, driver, ordinal, cap, free_floor):
        self.driver, self.cap = driver, cap
        self.allocations, self.buffers = {}, {}
        self.peak, self.context, self.closed = 0, True, False
        self.next_pointer = 0x100000
        driver.device = self

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.closed, self.context = True, False
        self.allocations.clear()
        self.buffers.clear()

    def current(self):
        if not self.context:
            raise RuntimeError("Closed fake device")

    def allocate(self, size):
        if sum(self.allocations.values()) + size > self.cap:
            raise MemoryError("Fake device allocation cap")
        pointer = self.next_pointer
        self.next_pointer += 0x1000000
        self.allocations[pointer] = size
        self.peak = max(self.peak, sum(self.allocations.values()))
        return pointer

    def download(self, pointer, shape, dtype):
        return np.frombuffer(self.buffers[pointer], dtype=dtype).reshape(shape).copy()


class Driver:
    def __init__(self, cc="12.1"):
        self.info = {"ordinal": 0, "name": "NVIDIA GB10", "compute_capability": cc}
        self.uploads = []

    def inventory(self):
        return {"driver_api_version": 13000, "devices": [self.info]}

    def check(self, status):
        if status:
            raise RuntimeError("Injected CUDA status")

    def cuMemcpyHtoD_v2(self, pointer, source, size):
        if size > self.device.allocations[pointer]:
            raise AssertionError("Upload exceeds the allocated staging block")
        self.device.buffers[pointer] = C.string_at(source, size)
        self.uploads.append((pointer, size))
        return 0

    def cuCtxSynchronize(self):
        return 0


class Blas:
    version, math_mode, handle = 130000, 0, 1

    def __init__(self, driver):
        self.driver, self.calls, self.closed = driver, [], False
        self.fail_at = None

    def close(self):
        self.closed = True

    def workspace(self, pointer, size):
        self.workspace_receipt = pointer, size

    def check(self, status):
        if status:
            raise RuntimeError("Injected SGEMM failure")

    def cublasSgemm_v2(self, handle, trans_a, trans_b, rows, count, width, alpha,
                       weight, lda, x_pointer, ldb, beta, destination, ldc):
        self.calls.append((trans_a, trans_b, rows, count, width, lda, ldb, ldc))
        if self.fail_at == len(self.calls):
            return 1
        device = self.driver.device
        weights = np.frombuffer(device.buffers[weight], np.float32).reshape(rows, width)
        x = np.frombuffer(device.buffers[x_pointer], np.float32).reshape(count, width)
        device.buffers[destination] = np.ascontiguousarray(x @ weights.T).tobytes()
        return 0


class SparkOutputTests(unittest.TestCase):
    def setUp(self):
        # Deliberately tiny dimensions exercise a non-divisible final chunk.
        # None of this mock evidence transfers numerical qualification to GB10.
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for module in (spark, shared_output):
            self.stack.enter_context(patch.object(module, "VOCAB", 11))
            self.stack.enter_context(patch.object(module, "WIDTH", 4))
        self.stack.enter_context(patch("vesper.execution_target.platform.system", return_value="Linux"))
        self.stack.enter_context(patch("vesper.execution_target.platform.machine", return_value="aarch64"))
        self.stack.enter_context(patch("vesper.resources.physical_memory", return_value={"available": 80 * spark.GiB}))
        self.driver = Driver()
        self.blas = Blas(self.driver)
        self.device_factory = self.stack.enter_context(patch.object(spark, "Device", side_effect=Device))
        self.stack.enter_context(patch.object(spark, "Driver", return_value=self.driver))
        self.stack.enter_context(patch.object(spark, "Cublas", return_value=self.blas))
        weights = np.arange(44, dtype=np.float32).reshape(11, 4) / np.float32(4) - np.float32(5)
        self.weights = bf16_bits(weights)
        self.reads = []

    def reader(self, start, count):
        self.reads.append((start, count))
        return self.weights[start:start + count].copy()

    def test_blocks_cover_exact_original_vocabulary_once_with_bounded_tail(self):
        self.assertEqual(spark.row_blocks(4), ((0, 4), (4, 4), (8, 3)))
        self.assertEqual(spark.row_blocks(1024), ((0, 11),))
        for bad in (0, -1, 1025, True, 1.0):
            with self.assertRaises(ValueError):
                spark.row_blocks(bad)

    def test_full_logits_use_exact_requested_row_spans_and_fp32_lift(self):
        x = bf16_bits(np.array([[1, -2, 3, 4], [-1, 2, -3, -4]], np.float32))
        head = spark.SparkOutputCandidate(self.reader, block_rows=4)
        with head:
            actual = head.project(x)
            np.testing.assert_array_equal(actual["promoted_fp32"], bf16_values(x))
            np.testing.assert_array_equal(actual["logits_fp32"], bf16_values(x) @ bf16_values(self.weights).T)
            np.testing.assert_array_equal(actual["greedy_ids_i64"], np.argmax(actual["logits_fp32"], axis=1))
            self.assertEqual(actual["logits_fp32"].shape, (2, 11))
            self.assertEqual(self.reads, [(0, 4), (4, 4), (8, 3)])
            self.assertEqual(self.blas.calls, [(1, 0, rows, 2, 4, 4, 4, rows) for rows in (4, 4, 3)])
            self.assertEqual((head.calls, head.rows_read), (1, 11))
            self.assertEqual(head.stats()["qualification"], "unqualified_candidate")
            self.assertFalse(head.stats()["TF32_enabled"])
            self.assertLessEqual(head.device.peak, 64 * spark.MiB)
        self.assertTrue(self.blas.closed)
        self.assertTrue(self.driver.device.closed)
        self.assertEqual(head.budget.reserved, 0)
        self.assertEqual(head.stats()["device_owned_bytes"], 0)

    def test_mid_read_failure_publishes_no_call_and_retry_starts_from_zero(self):
        fail_once = True
        def reader(start, count):
            nonlocal fail_once
            if start == 4 and fail_once:
                fail_once = False
                raise OSError("Injected bounded row read failure")
            return self.reader(start, count)
        x = bf16_bits(np.ones((1, 4), np.float32))
        with spark.SparkOutputCandidate(reader, block_rows=4) as head:
            with self.assertRaises(OSError):
                head.project(x)
            self.assertEqual((head.calls, head.rows_read), (0, 0))
            self.assertTrue(head.opened)
            self.reads.clear()
            result = head.project(x)
            self.assertEqual(self.reads, [(0, 4), (4, 4), (8, 3)])
            np.testing.assert_array_equal(result["logits_fp32"], bf16_values(x) @ bf16_values(self.weights).T)
            self.assertEqual((head.calls, head.rows_read), (1, 11))
        self.assertEqual(head.budget.reserved, 0)

    def test_mid_sgemm_failure_does_not_publish_partial_logits_and_recovers(self):
        self.blas.fail_at = 2
        x = bf16_bits(np.ones((1, 4), np.float32))
        with spark.SparkOutputCandidate(self.reader, block_rows=4) as head:
            with self.assertRaises(RuntimeError):
                head.project(x)
            self.assertEqual((head.calls, head.rows_read), (0, 0))
            self.blas.fail_at = None
            actual = head.project(x)
            np.testing.assert_array_equal(actual["logits_fp32"], bf16_values(x) @ bf16_values(self.weights).T)
            self.assertEqual((head.calls, head.rows_read), (1, 11))

    def test_bad_bf16_rows_rejected_before_sgemm(self):
        bad_rows = (np.zeros((4, 4), np.float32), np.zeros((3, 4), np.uint16),
                    np.full((4, 4), 0x7f80, np.uint16), np.full((4, 4), 0xff80, np.uint16),
                    np.full((4, 4), 0x7fc1, np.uint16), np.zeros((4, 8), np.uint16)[:, ::2])
        x = np.zeros((1, 4), np.uint16)
        for bad in bad_rows:
            with self.subTest(shape=bad.shape, dtype=bad.dtype):
                with spark.SparkOutputCandidate(lambda *_: bad, block_rows=4) as head:
                    with self.assertRaises(ValueError):
                        head.project(x)
                    self.assertEqual(head.calls, 0)
                self.assertEqual(head.budget.reserved, 0)
        self.assertEqual(self.blas.calls, [])

    def test_bad_hidden_input_rejected_before_row_read_or_upload(self):
        with spark.SparkOutputCandidate(self.reader, block_rows=4) as head:
            for bad in (np.zeros((1, 4), np.float32), np.zeros((3, 4), np.uint16),
                        np.zeros((1, 3), np.uint16), np.full((1, 4), 0x7f80, np.uint16)):
                with self.assertRaises(ValueError):
                    head.project(bad)
            self.assertEqual((head.calls, head.rows_read), (0, 0))
            self.assertEqual(self.reads, [])
            self.assertEqual(self.driver.uploads, [])

    def test_foreign_device_refused_before_owned_device_allocation(self):
        self.driver.info["compute_capability"] = "8.9"
        head = spark.SparkOutputCandidate(self.reader, block_rows=4)
        with self.assertRaises(ValueError):
            head.__enter__()
        self.device_factory.assert_not_called()
        self.assertEqual(head.budget.reserved, 0)

    def test_blas_profile_failure_closes_owned_device_and_budget(self):
        self.blas.math_mode = 1
        head = spark.SparkOutputCandidate(self.reader, block_rows=4)
        with self.assertRaises(ValueError):
            head.__enter__()
        self.assertTrue(self.driver.device.closed)
        self.assertTrue(self.blas.closed)
        self.assertEqual(head.budget.reserved, 0)

    def test_closed_or_reentered_operator_cannot_publish_calls(self):
        head = spark.SparkOutputCandidate(self.reader, block_rows=4)
        with self.assertRaises(RuntimeError):
            head.project(np.zeros((1, 4), np.uint16))
        with head:
            with self.assertRaises(RuntimeError):
                head.__enter__()
        with self.assertRaises(RuntimeError):
            head.__enter__()
        self.assertEqual(head.calls, 0)


if __name__ == "__main__":
    unittest.main()
