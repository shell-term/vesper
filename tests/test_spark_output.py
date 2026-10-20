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
        self.upload_attempts = 0
        self.fail_upload_at = None

    def inventory(self):
        return {"driver_api_version": 13000, "devices": [self.info]}

    def check(self, status):
        if status:
            raise RuntimeError("Injected CUDA status")

    def cuMemcpyHtoD_v2(self, pointer, source, size):
        self.upload_attempts += 1
        if self.fail_upload_at == self.upload_attempts:
            return 1
        owners = [(base, capacity) for base, capacity in self.device.allocations.items()
                  if base <= pointer < base + capacity]
        if len(owners) != 1:
            raise AssertionError("Upload does not address one owned allocation")
        base, capacity = owners[0]
        offset = pointer - base
        if offset + size > capacity:
            raise AssertionError("Upload exceeds the allocated matrix")
        buffer = self.device.buffers.setdefault(base, bytearray(capacity))
        buffer[offset:offset + size] = C.string_at(source, size)
        self.uploads.append((pointer, size))
        return 0

    def cuCtxSynchronize(self):
        return 0


class Blas:
    version, math_mode, handle = 130101, 0, 1

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
            # A failed launch may have touched output. No caller may receive it.
            self.driver.device.buffers[destination] = np.full((count, rows), np.nan, np.float32).tobytes()
            return 1
        device = self.driver.device
        weights = np.frombuffer(device.buffers[weight], np.float32).reshape(rows, width)
        x = np.frombuffer(device.buffers[x_pointer], np.float32, count=count * width).reshape(count, width)
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
        self.stack.enter_context(patch.object(spark, "physical_memory", return_value={"available": 80 * spark.GiB}))
        self.driver = Driver()
        self.blas = Blas(self.driver)
        self.device_factory = self.stack.enter_context(patch.object(spark, "Device", side_effect=Device))
        self.stack.enter_context(patch.object(spark, "Driver", return_value=self.driver))
        self.stack.enter_context(patch.object(spark, "Cublas", return_value=self.blas))
        weights = np.arange(44, dtype=np.float32).reshape(11, 4) / np.float32(4) - np.float32(5)
        weights[3, 1] = np.float32(-0.)
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
            self.assertEqual(self.blas.calls, [])
            self.assertEqual(self.reads, [(0, 4), (4, 4), (8, 3)])
            self.assertEqual(self.driver.uploads, [(head.weight_pointer, 64),
                (head.weight_pointer + 64, 64), (head.weight_pointer + 128, 48)])
            staged = np.frombuffer(head.device.buffers[head.weight_pointer], np.float32).reshape(11, 4)
            np.testing.assert_array_equal(staged.view(np.uint32), bf16_values(self.weights).view(np.uint32))
            actual = head.project(x)
            np.testing.assert_array_equal(actual["promoted_fp32"], bf16_values(x))
            np.testing.assert_array_equal(actual["logits_fp32"], bf16_values(x) @ bf16_values(self.weights).T)
            np.testing.assert_array_equal(actual["greedy_ids_i64"], np.argmax(actual["logits_fp32"], axis=1))
            self.assertEqual(actual["logits_fp32"].shape, (2, 11))
            self.assertEqual(self.reads, [(0, 4), (4, 4), (8, 3)])
            self.assertEqual(self.blas.calls, [(1, 0, 11, 2, 4, 4, 4, 11)])
            self.assertEqual((head.calls, head.rows_read), (1, 11))
            self.assertEqual(head.stats()["qualification"], "unqualified_candidate")
            self.assertFalse(head.stats()["TF32_enabled"])
            self.assertTrue(head.stats()["retained_head"])
            self.assertEqual(head.stats()["head_device_bytes"], 11 * 4 * 4)
            self.assertEqual(head.stats()["device_cap_bytes"], 3 * spark.GiB)
            self.assertEqual(head.device.cap, spark.HEAD_DEVICE_CAP)
            self.assertLessEqual(head.device.peak, spark.HEAD_DEVICE_CAP)
        self.assertTrue(self.blas.closed)
        self.assertTrue(self.driver.device.closed)
        self.assertEqual(head.budget.reserved, 0)
        self.assertEqual(head.stats()["device_owned_bytes"], 0)
        self.assertFalse(head.stats()["retained_head"])
        self.assertEqual(head.stats()["head_device_bytes"], 0)

    def test_mid_build_read_failure_closes_every_owner_and_new_operator_starts_from_zero(self):
        fail_once = True
        def reader(start, count):
            nonlocal fail_once
            if start == 4 and fail_once:
                fail_once = False
                raise OSError("Injected bounded row read failure")
            return self.reader(start, count)
        x = bf16_bits(np.ones((1, 4), np.float32))
        head = spark.SparkOutputCandidate(reader, block_rows=4)
        with self.assertRaises(OSError):
            head.__enter__()
        self.assertFalse(head.opened)
        self.assertEqual((head.calls, head.rows_read), (0, 4))
        self.assertEqual(head.budget.reserved, 0)
        self.assertTrue(self.driver.device.closed)
        self.assertTrue(self.blas.closed)
        self.assertEqual(head.stats()["device_owned_bytes"], 0)
        self.assertEqual(self.blas.calls, [])
        with self.assertRaises(RuntimeError):
            head.project(x)
        with self.assertRaises(RuntimeError):
            head.__enter__()
        self.reads.clear()
        with spark.SparkOutputCandidate(reader, block_rows=4) as head:
            result = head.project(x)
            self.assertEqual(self.reads, [(0, 4), (4, 4), (8, 3)])
            np.testing.assert_array_equal(result["logits_fp32"], bf16_values(x) @ bf16_values(self.weights).T)
            self.assertEqual((head.calls, head.rows_read), (1, 11))
        self.assertEqual(head.budget.reserved, 0)

    def test_single_sgemm_failure_does_not_publish_partial_logits_and_recovers_without_weight_read(self):
        self.blas.fail_at = 1
        x = bf16_bits(np.ones((1, 4), np.float32))
        with spark.SparkOutputCandidate(self.reader, block_rows=4) as head:
            with self.assertRaises(RuntimeError):
                head.project(x)
            self.assertEqual((head.calls, head.rows_read), (0, 11))
            self.assertTrue(head.stats()["retained_head"])
            self.blas.fail_at = None
            actual = head.project(x)
            np.testing.assert_array_equal(actual["logits_fp32"], bf16_values(x) @ bf16_values(self.weights).T)
            self.assertEqual((head.calls, head.rows_read), (1, 11))
            self.assertEqual(self.reads, [(0, 4), (4, 4), (8, 3)])
            self.assertEqual(self.blas.calls, [(1, 0, 11, 1, 4, 4, 4, 11)] * 2)

    def test_retained_cache_reads_weights_once_across_batch_one_and_two_calls(self):
        x = bf16_bits(np.ones((1, 4), np.float32))
        expected = bf16_values(x) @ bf16_values(self.weights).T
        with spark.SparkOutputCandidate(self.reader, block_rows=4) as head:
            self.weights.fill(0)  # Forward must use its already verified device copy.
            for count in (1, 2, 1):
                actual = head.project(np.repeat(x, count, axis=0))
                np.testing.assert_array_equal(actual["logits_fp32"], np.repeat(expected, count, axis=0))
            self.assertEqual(head.calls, 3)
            self.assertEqual(head.rows_read, 11)
            self.assertEqual(self.reads, [(0, 4), (4, 4), (8, 3)])
            self.assertEqual([call[2:4] for call in self.blas.calls], [(11, 1), (11, 2), (11, 1)])

    def test_mid_build_upload_failure_never_opens_or_publishes_and_releases_all_resources(self):
        self.driver.fail_upload_at = 2
        head = spark.SparkOutputCandidate(self.reader, block_rows=4)
        with self.assertRaises(RuntimeError):
            head.__enter__()
        self.assertFalse(head.opened)
        self.assertEqual((head.calls, head.rows_read), (0, 4))
        self.assertTrue(self.blas.closed)
        self.assertTrue(self.driver.device.closed)
        self.assertEqual(head.budget.reserved, 0)
        self.assertEqual(head.stats()["device_owned_bytes"], 0)
        self.assertFalse(head.stats()["retained_head"])
        self.assertEqual(self.blas.calls, [])

    def test_device_allocation_failure_precedes_source_reads_and_closes_context(self):
        def too_small(driver, ordinal, cap, free_floor):
            return Device(driver, ordinal, cap=64, free_floor=free_floor)
        with patch.object(spark, "Device", side_effect=too_small):
            head = spark.SparkOutputCandidate(self.reader, block_rows=4)
            with self.assertRaises(MemoryError):
                head.__enter__()
        self.assertEqual(self.reads, [])
        self.assertEqual(self.driver.uploads, [])
        self.assertTrue(self.driver.device.closed)
        self.assertTrue(self.blas.closed)
        self.assertEqual(head.budget.reserved, 0)

    def test_full_head_physical_admission_refuses_before_driver_and_original_reads(self):
        with patch.object(spark, "physical_memory", return_value={"available": 33 * spark.GiB}), \
                patch.object(spark, "Driver") as driver:
            head = spark.SparkOutputCandidate(self.reader, block_rows=4)
            with self.assertRaises(MemoryError):
                head.__enter__()
        driver.assert_not_called()
        self.assertEqual(head.budget.reserved, 0)
        self.assertEqual(self.reads, [])

    def test_forward_upload_failure_can_retry_without_rebuilding_device_head(self):
        x = np.zeros((1, 4), np.uint16)
        with spark.SparkOutputCandidate(self.reader, block_rows=4) as head:
            self.driver.fail_upload_at = self.driver.upload_attempts + 1
            with self.assertRaises(RuntimeError):
                head.project(x)
            self.assertEqual(head.calls, 0)
            self.assertEqual(self.blas.calls, [])
            self.driver.fail_upload_at = None
            actual = head.project(x)
            np.testing.assert_array_equal(actual["logits_fp32"], np.zeros((1, 11), np.float32))
            self.assertEqual((head.calls, head.rows_read), (1, 11))
            self.assertEqual(self.reads, [(0, 4), (4, 4), (8, 3)])

    def test_bad_bf16_rows_rejected_before_sgemm(self):
        bad_rows = (np.zeros((4, 4), np.float32), np.zeros((3, 4), np.uint16),
                    np.full((4, 4), 0x7f80, np.uint16), np.full((4, 4), 0xff80, np.uint16),
                    np.full((4, 4), 0x7fc1, np.uint16), np.zeros((4, 8), np.uint16)[:, ::2])
        x = np.zeros((1, 4), np.uint16)
        for bad in bad_rows:
            with self.subTest(shape=bad.shape, dtype=bad.dtype):
                head = spark.SparkOutputCandidate(lambda *_: bad, block_rows=4)
                with self.assertRaises(ValueError):
                    head.__enter__()
                self.assertEqual(head.calls, 0)
                self.assertFalse(head.opened)
                self.assertTrue(self.driver.device.closed)
                self.assertEqual(head.budget.reserved, 0)
        self.assertEqual(self.blas.calls, [])

    def test_bad_hidden_input_rejected_before_row_read_or_upload(self):
        with spark.SparkOutputCandidate(self.reader, block_rows=4) as head:
            reads, uploads = self.reads.copy(), self.driver.uploads.copy()
            for bad in (np.zeros((1, 4), np.float32), np.zeros((3, 4), np.uint16),
                        np.zeros((1, 3), np.uint16), np.full((1, 4), 0x7f80, np.uint16)):
                with self.assertRaises(ValueError):
                    head.project(bad)
            self.assertEqual((head.calls, head.rows_read), (0, 11))
            self.assertEqual(self.reads, reads)
            self.assertEqual(self.driver.uploads, uploads)

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
