import hashlib
import threading
import unittest
from unittest.mock import patch

import numpy as np

from vesper.cache import TieredCache
from vesper.resources import MemoryBudget


class Source:
    size = 4

    def __init__(self):
        self.entries = {k: np.frombuffer(k.encode() * self.size, np.uint8).copy() for k in "ABCD"}
        self.reads, self.bad = [], False

    def validate(self, key, data):
        if hashlib.sha256(data).digest() != hashlib.sha256(self.entries[key]).digest():
            raise ValueError("Corrupt bytes")

    def read_into(self, key, target):
        self.reads.append(key)
        np.copyto(target, self.entries[key])
        if self.bad:
            target[0] ^= 1
        self.validate(key, target)


class Backend:
    def __init__(self, primary=1, helper=1):
        self.primary_slots, self.helper_slots = primary, helper
        self.staging = [np.zeros(4, np.uint8), np.zeros(4, np.uint8)]
        self.slots = {"primary": [np.zeros(4, np.uint8) for _ in range(primary)],
                      "helper": [np.zeros(4, np.uint8) for _ in range(helper)]}
        self.corrupt_write, self.failed_write = False, False
        self.verify_calls = []

    def read(self, tier, slot, target):
        np.copyto(target, self.slots[tier][slot])

    def write(self, tier, slot, data):
        np.copyto(self.slots[tier][slot], data)
        if self.corrupt_write:
            self.slots[tier][slot][0] ^= 1
        if self.failed_write:
            raise OSError("Injected transfer failure")

    def verify(self, tier, slot, key):
        self.verify_calls.append((tier, key))
        self.source.validate(key, self.slots[tier][slot])
        return 32


class CacheTests(unittest.TestCase):
    def make(self, ram=4, cap=12, helper=1, gpu_verify=False):
        source, backend = Source(), Backend(helper=helper)
        backend.source = source
        budget = MemoryBudget(cap=cap, available_floor=100)
        budget.acquire(8, available=1000)  # Model the two fixed staging buffers.
        self.base_available = 1000
        cache = TieredCache(source, backend, budget, ram,
                            available=lambda: self.base_available - budget.reserved, gpu_verify=gpu_verify)
        self.addCleanup(cache.close)
        return cache, source, backend, budget

    def request(self, cache, key):
        with cache.acquire(key) as (slot, origin):
            np.testing.assert_array_equal(cache.backend.slots["primary"][slot], cache.source.entries[key])
            snapshot = cache.snapshot()
            self.assertLessEqual(snapshot["host_reserved_bytes"], cache.budget.cap)
            return origin

    def test_all_tiers_are_exclusive_and_warm_trace_never_rereads_store(self):
        cache, source, _, budget = self.make()
        sequence = "ABACCBACABC"
        origins = [self.request(cache, key) for key in sequence]
        self.assertEqual(set(origins), {"store", "primary_gpu", "helper_gpu", "ram"})
        self.assertEqual(source.reads, ["A", "B", "C"])
        self.assertEqual(cache.stats["requests_completed"], len(sequence))
        self.assertEqual(budget.peak, 12)
        cache.close()
        self.assertEqual(budget.reserved, 8)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.request(cache, "A")

    def test_capacity_eviction_and_zero_ram_reread_correct_expert(self):
        for ram in (0, 4):
            cache, source, _, _ = self.make(ram=ram)
            for key in "ABCDACBD":
                self.request(cache, key)
            self.assertGreater(len(source.reads), 4)
            if ram:
                self.assertGreater(cache.stats["ram_capacity_evictions"], 0)

    def test_pressure_releases_ram_then_refuses_without_changing_slots(self):
        cache, _, _, budget = self.make()
        for key in "ABC":
            self.request(cache, key)
        self.base_available = 108
        self.request(cache, "C")
        self.assertFalse(cache.ram)
        self.assertEqual(cache.stats["ram_pressure_evictions"], 1)
        before = cache.snapshot()
        self.base_available = 107
        with self.assertRaises(MemoryError):
            self.request(cache, "A")
        self.assertEqual(cache.snapshot()["primary_keys"], before["primary_keys"])
        self.assertEqual(cache.snapshot()["helper_keys"], before["helper_keys"])
        self.assertEqual(budget.reserved, 8)
        self.base_available = 1000
        self.request(cache, "A")

    def test_host_cap_refuses_ram_admission_without_overcommit(self):
        cache, _, _, budget = self.make(cap=8)
        for key in "ABCAB":
            self.request(cache, key)
        self.assertFalse(cache.ram)
        self.assertEqual(budget.peak, 8)
        self.assertGreater(cache.stats["ram_admission_refused"], 0)

    def test_pressure_after_array_allocation_releases_reservation(self):
        cache, _, _, budget = self.make()
        empty = np.empty
        def allocation(*args, **kwargs):
            array = empty(*args, **kwargs)
            self.base_available = 107
            return array
        for key in "AB":
            self.request(cache, key)
        with patch("vesper.cache.np.empty", side_effect=allocation):
            with self.assertRaises(MemoryError):
                self.request(cache, "C")
        self.assertEqual(budget.reserved, 8)
        self.assertFalse(cache.ram)
        self.base_available = 1000
        self.request(cache, "C")

    def test_failed_allocation_releases_reservation_and_preserves_bytes(self):
        cache, _, _, budget = self.make()
        for key in "AB":
            self.request(cache, key)
        with patch("vesper.cache.np.empty", side_effect=MemoryError("Injected allocation failure")):
            with self.assertRaises(MemoryError):
                self.request(cache, "C")
        self.assertEqual(budget.reserved, 8)
        for key in "ABC":
            self.request(cache, key)

    def test_corrupt_store_does_not_publish_or_displace_valid_slots(self):
        cache, source, _, _ = self.make()
        self.request(cache, "A")
        source.bad = True
        with self.assertRaisesRegex(ValueError, "Corrupt"):
            self.request(cache, "B")
        self.assertEqual(cache.snapshot()["primary_keys"], ["A"])
        self.assertFalse(cache.helper.entries)
        source.bad = False
        self.request(cache, "B")
        self.assertEqual(cache.stats["integrity_failures"], 1)

    def test_corrupt_or_failed_upload_never_publishes_partial_slot(self):
        for fault in ("corrupt_write", "failed_write"):
            cache, _, backend, _ = self.make()
            setattr(backend, fault, True)
            with self.assertRaises((ValueError, OSError)):
                self.request(cache, "A")
            self.assertFalse(cache.primary.entries)
            setattr(backend, fault, False)
            self.request(cache, "B")
            self.request(cache, "A")

    def test_no_helper_preserves_ram_and_direct_store_paths(self):
        cache, _, _, _ = self.make(helper=0)
        self.assertEqual([self.request(cache, k) for k in "ABA"], ["store", "store", "ram"])

    def test_unknown_identity_and_nested_requests_cannot_replace_live_slot(self):
        cache, source, _, _ = self.make()
        with cache.acquire("A") as (slot, _):
            with self.assertRaisesRegex(RuntimeError, "Nested"):
                self.request(cache, "B")
            with self.assertRaisesRegex(RuntimeError, "live"):
                cache.close()
            np.testing.assert_array_equal(cache.backend.slots["primary"][slot], source.entries["A"])
        with self.assertRaises(KeyError):
            self.request(cache, "missing")

    def test_another_thread_cannot_replace_slot_until_compute_releases_it(self):
        cache, _, _, _ = self.make()
        started, acquired = threading.Event(), threading.Event()
        errors = []
        def other():
            started.set()
            try:
                self.request(cache, "B")
                acquired.set()
            except BaseException as exc:
                errors.append(exc)
        with cache.acquire("A"):
            worker = threading.Thread(target=other)
            worker.start()
            self.assertTrue(started.wait(2))
            self.assertFalse(acquired.wait(0.03))
            self.assertEqual(cache.stats["requests_started"], 1)
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertFalse(errors)
        self.assertTrue(acquired.is_set())

    def test_device_audit_rejects_corrupted_ram_before_publishing_compute_slot(self):
        cache, _, backend, _ = self.make(gpu_verify=True)
        for key in "ABC":
            self.request(cache, key)
        cache.ram["A"][0] ^= 1
        with self.assertRaisesRegex(ValueError, "Corrupt"):
            self.request(cache, "A")
        self.assertNotIn("A", cache.primary.entries)
        self.assertEqual(cache.stats["integrity_failures"], 1)
        self.request(cache, "A")
        self.assertGreater(cache.stats["cpu_checks_deferred"], 0)
        self.assertIn(("primary", "A"), backend.verify_calls)
        self.assertEqual(cache.stats["primary_audit_readback_bytes"], 0)

    def test_device_audit_rejects_corrupt_and_failed_uploads(self):
        for fault in ("corrupt_write", "failed_write"):
            cache, _, backend, _ = self.make(gpu_verify=True)
            setattr(backend, fault, True)
            with self.assertRaises((ValueError, OSError)):
                self.request(cache, "A")
            self.assertFalse(cache.primary.entries)
            setattr(backend, fault, False)
            self.request(cache, "A")

    def test_device_verification_cannot_disable_its_upload_audit(self):
        cache, source, backend, budget = self.make()
        with self.assertRaises(ValueError):
            TieredCache(source, backend, budget, 4, audit=False, gpu_verify=True)


if __name__ == "__main__":
    unittest.main()
