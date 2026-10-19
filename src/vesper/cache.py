"""Synchronous, exclusive cache tiers for immutable original byte blocks.

No routing decisions are made here. Requests are serialized until computation
finishes, so a referenced GPU slot cannot be replaced by another request.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
import threading

import numpy as np

from .resources import MemoryBudget, physical_memory
from .timing import Timings


class SlotIndex:
    def __init__(self, capacity: int):
        if capacity < 0:
            raise ValueError("Negative slot capacity")
        self.capacity = capacity
        self.free = list(range(capacity))
        self.entries = OrderedDict()

    def lookup(self, key):
        slot = self.entries.get(key)
        if slot is not None:
            self.entries.move_to_end(key)
        return slot

    def remove(self, key):
        slot = self.entries.pop(key)
        self.free.append(slot)

    def publish(self, key, slot):
        if key in self.entries or slot not in self.free:
            raise RuntimeError("Cache slot is already published")
        self.free.remove(slot)
        self.entries[key] = slot

    def oldest(self):
        return next(iter(self.entries.items()))


class TieredCache:
    """Storage-independent byte cache; backend owns fixed GPU/staging buffers.

    Source provides size, entries, read_into and validate. Backend provides two
    pinned staging arrays and read/write for primary/helper slots. The RAM tier
    owns its arrays and accounts them in the shared host budget.

    GPU-SHA mode defers host-copy checks until admission to a GPU. The backend
    must hash every uploaded byte against the verified original digest table
    before a slot is published. A failed audit never permits model execution.
    """

    def __init__(self, source, backend, budget: MemoryBudget, ram_capacity_bytes: int,
                 available=None, audit=True, timings=None, gpu_verify=False):
        if ram_capacity_bytes < 0:
            raise ValueError("Negative RAM cache capacity")
        self.source, self.backend, self.budget = source, backend, budget
        self.timings = timings or Timings()
        if gpu_verify and (not audit or not callable(getattr(backend, "verify", None))):
            raise ValueError("GPU verification requires a complete upload audit backend")
        self.gpu_verify = gpu_verify
        self.size, self.ram_capacity = source.size, ram_capacity_bytes
        self.available = available or (lambda: physical_memory()["available"])
        self.primary, self.helper = SlotIndex(backend.primary_slots), SlotIndex(backend.helper_slots)
        if not self.primary.capacity:
            raise ValueError("A compute cache requires at least one primary slot")
        self.staging, self.audit = backend.staging, audit
        if any(a.dtype != np.uint8 or a.shape != (self.size,) or not a.flags.c_contiguous for a in self.staging):
            raise ValueError("Staging must contain two complete contiguous byte blocks")
        if len(self.staging) != 2 or np.shares_memory(*self.staging):
            raise ValueError("Promotion and demotion need two distinct staging buffers")
        self.ram = OrderedDict()
        self.lock, self.closed, self.active = threading.RLock(), False, False
        names = ("requests_started", "requests_completed", "requests_failed", "primary_hits", "helper_hits", "ram_hits",
                 "store_read_attempts", "store_reads_completed", "store_bytes_requested", "primary_evictions", "helper_evictions",
                 "ram_capacity_evictions", "ram_pressure_evictions", "ram_admission_refused", "pressure_refusals", "integrity_failures",
                 "primary_h2d_bytes", "helper_h2d_bytes", "primary_d2h_bytes", "helper_d2h_bytes",
                 "primary_audit_readback_bytes", "helper_audit_readback_bytes", "cpu_checks_deferred",
                 "primary_gpu_audit_payload_bytes", "helper_gpu_audit_payload_bytes",
                 "primary_gpu_audit_digest_d2h_bytes", "helper_gpu_audit_digest_d2h_bytes")
        self.stats = dict.fromkeys(names, 0)
        self._memory_reclaimer=budget.register_reclaimer(self.reclaim)

    def _drop_ram(self, key, reason=None):
        del self.ram[key]
        self.budget.release(self.size,category='cache')
        if reason:
            self.stats[reason] += 1

    def enforce_pressure(self):
        while self.available() < self.budget.available_floor and self.ram:
            self._drop_ram(next(iter(self.ram)), "ram_pressure_evictions")
        if self.available() < self.budget.available_floor:
            self.stats["pressure_refusals"] += 1
            raise MemoryError("System memory remains below the reserve after releasing the RAM cache")
        if self.budget._policy is not None: self.budget.acquire(0,available=self.available())

    def reclaim(self,requested):
        if not self.lock.acquire(blocking=False): return 0
        try:
            freed=0
            while self.ram and freed<requested:
                self._drop_ram(next(iter(self.ram)),"ram_pressure_evictions"); freed+=self.size
            return freed
        finally: self.lock.release()

    def _verify(self, key, data):
        if self.gpu_verify:
            # Host/intermediate copies may be unverified. Every GPU upload is
            # checked against the immutable original digest table before the
            # slot becomes visible, including before any model computation.
            self.stats["cpu_checks_deferred"] += 1
            return
        try:
            self.source.validate(key, data)
        except Exception:
            self.stats["integrity_failures"] += 1
            raise

    def _read_gpu(self, tier, slot, key, target):
        self.backend.read(tier, slot, target)
        self.stats[f"{tier}_d2h_bytes"] += self.size
        self._verify(key, target)

    def _write_gpu(self, tier, slot, key, data):
        self.backend.write(tier, slot, data)
        self.stats[f"{tier}_h2d_bytes"] += self.size
        if self.audit:
            if self.gpu_verify:
                try:
                    digest_bytes = self.backend.verify(tier, slot, key)
                except ValueError:
                    self.stats["integrity_failures"] += 1
                    raise
                self.stats[f"{tier}_gpu_audit_payload_bytes"] += self.size
                self.stats[f"{tier}_gpu_audit_digest_d2h_bytes"] += digest_bytes
            else:
                self.backend.read(tier, slot, self.staging[1])
                self.stats[f"{tier}_audit_readback_bytes"] += self.size
                self._verify(key, self.staging[1])

    def _store_ram(self, key, data):
        if self.ram_capacity < self.size:
            self.stats["ram_admission_refused"] += 1
            return False
        self.enforce_pressure()
        def fits():
            return ((len(self.ram) + 1) * self.size <= self.ram_capacity
                    and self.budget.reserved + self.size <= self.budget.cap
                    and self.available() - self.size >= self.budget.available_floor)
        while not fits() and self.ram:
            self._drop_ram(next(iter(self.ram)), "ram_capacity_evictions")
        if not fits():
            self.stats["ram_admission_refused"] += 1
            return False
        self.budget.acquire(self.size, available=self.available(),category='cache')
        array, reserved = None, True
        try:
            with self.timings.section("host.ram_allocate"):
                array = np.empty(self.size, np.uint8)
            with self.timings.section("host.ram_copy"):
                np.copyto(array, data)
            self._verify(key, array)
            if self.available() < self.budget.available_floor:
                del array
                array = None
                self.budget.release(self.size,category='cache')
                reserved = False
                self.enforce_pressure()
                self.stats["ram_admission_refused"] += 1
                return False
            self.ram[key] = array
            return True
        except BaseException:
            array = None
            # If pressure cleanup released the reservation already, do not
            # release a different live entry's allocation on exception.
            if reserved:
                self.budget.release(self.size,category='cache')
            raise

    def _demote_primary(self):
        key, slot = self.primary.oldest()
        if self.helper.capacity:
            if not self.helper.free:
                old_key, old_slot = self.helper.oldest()
                if self.ram_capacity >= self.size:
                    self._read_gpu("helper", old_slot, old_key, self.staging[1])
                    self._store_ram(old_key, self.staging[1])
                self.helper.remove(old_key)
                self.stats["helper_evictions"] += 1
            self._read_gpu("primary", slot, key, self.staging[1])
            destination = self.helper.free[0]
            self._write_gpu("helper", destination, key, self.staging[1])
            self.helper.publish(key, destination)
        elif self.ram_capacity >= self.size:
            self._read_gpu("primary", slot, key, self.staging[1])
            self._store_ram(key, self.staging[1])
        self.primary.remove(key)
        self.stats["primary_evictions"] += 1

    def _ensure_primary(self, key):
        if key not in self.source.entries:
            raise KeyError("Requested expert is absent from the pinned native store")
        slot = self.primary.lookup(key)
        if slot is not None:
            self.stats["primary_hits"] += 1
            return slot, "primary_gpu"
        slot = self.helper.lookup(key)
        if slot is not None:
            self._read_gpu("helper", slot, key, self.staging[0])
            self.helper.remove(key)
            self.stats["helper_hits"] += 1
            origin = "helper_gpu"
        elif key in self.ram:
            with self.timings.section("host.ram_copy"):
                np.copyto(self.staging[0], self.ram[key])
            self._verify(key, self.staging[0])
            self._drop_ram(key)
            self.stats["ram_hits"] += 1
            origin = "ram"
        else:
            self.stats["store_read_attempts"] += 1
            self.stats["store_bytes_requested"] += self.size
            try:
                self.source.read_into(key, self.staging[0])
            except ValueError:
                self.stats["integrity_failures"] += 1
                raise
            self.stats["store_reads_completed"] += 1
            origin = "store"
        if not self.primary.free:
            self._demote_primary()
        destination = self.primary.free[0]
        self._write_gpu("primary", destination, key, self.staging[0])
        self.primary.publish(key, destination)
        return destination, origin

    @contextmanager
    def acquire(self, key):
        with self.lock:
            if self.closed:
                raise RuntimeError("Cache is closed")
            if self.active:
                raise RuntimeError("Nested requests could replace a live compute slot")
            self.active = True
            self.stats["requests_started"] += 1
            try:
                self.enforce_pressure()
                slot, origin = self._ensure_primary(key)
                yield slot, origin
                self.enforce_pressure()
                self.stats["requests_completed"] += 1
            except BaseException:
                self.stats["requests_failed"] += 1
                raise
            finally:
                self.active = False

    def snapshot(self):
        with self.lock:
            return self._snapshot()

    def _snapshot(self):
        keys = list(self.primary.entries) + list(self.helper.entries) + list(self.ram)
        if len(keys) != len(set(keys)):
            raise RuntimeError("Exclusive cache tiers contain duplicate expert identities")
        return {"primary_keys": list(self.primary.entries), "helper_keys": list(self.helper.entries),
                "ram_keys": list(self.ram), "ram_cache_bytes": len(self.ram) * self.size,
                "host_reserved_bytes": self.budget.reserved, "host_peak_reserved_bytes": self.budget.peak,
                "integrity_mode": "gpu-sha256" if self.gpu_verify else "readback",
                "counters": dict(self.stats)}

    def close(self):
        with self.lock:
            if self.active:
                raise RuntimeError("Cannot close a cache with a live compute slot")
            for key in list(self.ram):
                self._drop_ram(key)
            self.closed = True
            self.budget.unregister_reclaimer(self._memory_reclaimer)
