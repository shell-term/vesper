"""Bounded direct-store/RAM/helper-GPU/compute-GPU expert execution."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
import time

from .cache import TieredCache
from .cuda import Device, Pinned
from .expert import ExpertExecutor
from .resources import GiB, MiB, MemoryBudget
from .storage import NativeStore


@dataclass(frozen=True)
class CacheConfig:
    host_cap_bytes: int = 96 * GiB
    ram_cache_bytes: int = 80 * GiB
    available_floor: int = 32 * GiB
    primary_device: int = 0
    primary_slots: int = 1
    helper_device: int | None = 1
    helper_slots: int = 1
    device_cap_bytes: int = 64 * MiB
    metadata_allowance_bytes: int = MiB

    def __post_init__(self):
        if self.primary_slots < 1 or self.helper_slots < 0 or self.ram_cache_bytes < 0:
            raise ValueError("Invalid expert cache capacity")
        if self.helper_device == self.primary_device:
            raise ValueError("The helper must be a distinct storage GPU")
        if self.helper_device is None and self.helper_slots:
            raise ValueError("Helper slots require a helper device")
        if self.metadata_allowance_bytes < MiB:
            raise ValueError("The bounded store requires at least 1 MiB metadata allowance")


class CudaCacheBackend:
    """Two portable pinned buffers and fixed slots; helper performs no arithmetic."""

    def __init__(self, executor, size, config, budget):
        self.executor, self.size, self.config, self.budget = executor, size, config, budget
        self.primary_slots, self.helper_slots = config.primary_slots, config.helper_slots
        self.stack, self.helper_device = None, None

    def __enter__(self):
        if self.stack is not None:
            raise RuntimeError("Cache backend is already open")
        self.stack = ExitStack()
        try:
            if self.helper_slots:
                self.helper_device = self.stack.enter_context(Device(self.executor.driver, self.config.helper_device,
                                                                    cap=self.config.device_cap_bytes))
            self.pinned = [self.stack.enter_context(Pinned(self.executor.driver, self.size, self.budget)) for _ in range(2)]
            self.staging = [p.array() for p in self.pinned]
            self.pointers = {"primary": [], "helper": []}
            for _ in range(self.primary_slots):
                pointer = self.executor.device.allocate(self.size)
                self.stack.callback(self.executor.device.free, pointer)
                self.pointers["primary"].append(pointer)
            if self.helper_device:
                self.pointers["helper"] = [self.helper_device.allocate(self.size) for _ in range(self.helper_slots)]
            return self
        except BaseException:
            self.stack.close()
            raise

    def _device(self, tier):
        return self.executor.device if tier == "primary" else self.helper_device

    def read(self, tier, slot, target):
        device = self._device(tier)
        device.current()
        device.driver.check(device.driver.cuMemcpyDtoH_v2(target.ctypes.data, self.pointers[tier][slot], self.size))
        device.driver.check(device.driver.cuCtxSynchronize())

    def write(self, tier, slot, data):
        device = self._device(tier)
        device.current()
        device.driver.check(device.driver.cuMemcpyHtoD_v2(self.pointers[tier][slot], data.ctypes.data, self.size))
        device.driver.check(device.driver.cuCtxSynchronize())

    def __exit__(self, *_):
        if self.stack is not None:
            self.stack.close()
            self.stack = None


class TieredExpertRuntime:
    """Serial single-token execution; all persistent cache/staging buffers bounded.

    The host ledger includes cache arrays, pinned staging, direct-I/O scratch and
    an explicit metadata allowance. Python, CUDA contexts, compilation, returned
    diagnostic arrays and an optional oracle are outside this buffer ledger.
    """

    def __init__(self, store: Path, config: CacheConfig | None = None):
        self.store_path, self.config = Path(store), config or CacheConfig()
        self.stack, self.cache = None, None

    def __enter__(self):
        if self.stack is not None:
            raise RuntimeError("Tiered runtime is already open")
        c = self.config
        self.stack = ExitStack()
        self.budget = MemoryBudget(c.host_cap_bytes, c.available_floor)
        try:
            self.budget.acquire(c.metadata_allowance_bytes)
            self.stack.callback(self.budget.release, c.metadata_allowance_bytes)
            self.store = self.stack.enter_context(NativeStore(self.store_path, self.budget))
            fixed = c.metadata_allowance_bytes + self.store.size * 3
            if fixed > c.host_cap_bytes:
                raise MemoryError("Host cap cannot fit direct-I/O scratch and two pinned staging buffers")
            self.executor = self.stack.enter_context(ExpertExecutor(c.primary_device, device_cap=c.device_cap_bytes))
            self.backend = self.stack.enter_context(CudaCacheBackend(self.executor, self.store.size, c, self.budget))
            self.primary_device_object = self.executor.device
            self.helper_device_object = self.backend.helper_device
            self.cache = TieredCache(self.store, self.backend, self.budget, c.ram_cache_bytes)
            self.stack.callback(self.cache.close)
            self.last_event = None
            return self
        except BaseException:
            self.stack.close()
            self.stack = None
            raise

    def forward(self, key, x, routing_weight=None):
        if self.stack is None:
            raise RuntimeError("Use the tiered runtime as a context manager")
        started = time.perf_counter()
        with self.cache.acquire(key) as (slot, origin):
            for device in (self.executor.device, self.backend.helper_device):
                if device is not None and device.memory()["free"] < device.free_floor:
                    raise MemoryError("GPU headroom fell below the reserve; refusing further work")
            base = self.backend.pointers["primary"][slot]
            weights = {name: base + region["offset"] for name, region in self.store.manifest["layout"].items()}
            result = self.executor.execute(x, weights, routing_weight)
        self.last_event = {"key": key, "source_tier": origin, "wall_ms": (time.perf_counter() - started) * 1000}
        return result

    def snapshot(self):
        result = self.cache.snapshot()
        result.update({"primary_peak_explicit_bytes": self.primary_device_object.peak,
                       "helper_peak_explicit_bytes": 0 if self.helper_device_object is None else self.helper_device_object.peak,
                       "store_direct_read_calls": self.store.read_calls, "store_direct_read_bytes": self.store.bytes_read,
                       "host_cap_bytes": self.config.host_cap_bytes, "host_available_floor_bytes": self.config.available_floor,
                       "ram_capacity_bytes": self.config.ram_cache_bytes,
                       "timing_scope": "Synchronous calls including hashes, audit readbacks and diagnostic copies; no token-rate claim"})
        return result

    def __exit__(self, *_):
        if self.stack is not None:
            self.stack.close()
            self.stack = None
        self.cleanup = {"host_buffer_reservations_bytes": self.budget.reserved,
                        "primary_explicit_live_bytes": sum(self.primary_device_object.allocations.values()),
                        "helper_explicit_live_bytes": 0 if self.helper_device_object is None else sum(self.helper_device_object.allocations.values())}
