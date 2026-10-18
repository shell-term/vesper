"""Bounded, synchronous transfer probes; numbers include call and synchronization overhead."""

from __future__ import annotations

import ctypes as C
import statistics
import time

import numpy as np

from .cuda import Device, Driver, Pinned
from .resources import GiB, MiB, MemoryBudget

EXPERT_BYTES = 18_800_640


def _time(operation, byte_count, repeats=12):
    operation()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - start)
    median = statistics.median(samples)
    return {"bytes_per_operation": byte_count, "repeats": repeats, "median_ms": median * 1000,
            "min_ms": min(samples) * 1000, "p90_ms": sorted(samples)[int(.9 * (len(samples) - 1))] * 1000,
            "effective_GBps": byte_count / median / 1e9}


def benchmark(devices: list[int] | None = None) -> dict:
    driver = Driver()
    inventory = driver.inventory()
    chosen = devices if devices is not None else [d["ordinal"] for d in inventory["devices"]]
    budget = MemoryBudget(cap=256 * MiB, available_floor=32 * GiB)
    results = []
    for ordinal in chosen:
        with Device(driver, ordinal) as device, Pinned(driver, EXPERT_BYTES, budget) as pinned:
            source = pinned.array()
            source[:] = np.arange(EXPERT_BYTES, dtype=np.uint8)
            allocation = device.allocate(EXPERT_BYTES)
            for size in (20_480, 65_536, EXPERT_BYTES):
                def htod():
                    driver.check(driver.cuMemcpyHtoD_v2(allocation, pinned.pointer, size))
                    driver.check(driver.cuCtxSynchronize())
                def dtoh():
                    driver.check(driver.cuMemcpyDtoH_v2(pinned.pointer, allocation, size))
                    driver.check(driver.cuCtxSynchronize())
                up = _time(htod, size)
                source[:size] = 0
                down = _time(dtoh, size)
                if not np.array_equal(source[:size], np.arange(size, dtype=np.uint8)):
                    raise AssertionError("Transfer did not preserve data bytes")
                results.append({"device": ordinal, "direction": "pinned_host_to_device", **up, "bytes_verified": True})
                results.append({"device": ordinal, "direction": "device_to_pinned_host", **down, "bytes_verified": True})
    staged = []
    if len(chosen) >= 2:
        # Portable pinned host memory supports the documented explicit staging route.
        # No peer access is enabled and no runtime's hidden fallback is assumed.
        with Device(driver, chosen[0]) as first, Device(driver, chosen[1]) as second, Pinned(driver, EXPERT_BYTES, budget) as pinned:
            source = pinned.array()
            source[:] = np.arange(EXPERT_BYTES, dtype=np.uint8)
            a, b = first.allocate(EXPERT_BYTES), second.allocate(EXPERT_BYTES)
            first.current()
            driver.check(driver.cuMemcpyHtoD_v2(a, pinned.pointer, EXPERT_BYTES))
            for src, dst, psrc, pdst in ((first, second, a, b), (second, first, b, a)):
                # Both source devices must be initialized before either direction.
                if src is second:
                    second.current()
                    source[:] = np.arange(EXPERT_BYTES, dtype=np.uint8)
                    driver.check(driver.cuMemcpyHtoD_v2(b, pinned.pointer, EXPERT_BYTES))
                for size in (20_480, EXPERT_BYTES):
                    def copy():
                        src.current()
                        driver.check(driver.cuMemcpyDtoH_v2(pinned.pointer, psrc, size))
                        driver.check(driver.cuCtxSynchronize())
                        dst.current()
                        driver.check(driver.cuMemcpyHtoD_v2(pdst, pinned.pointer, size))
                        driver.check(driver.cuCtxSynchronize())
                    measurement = _time(copy, size)
                    dst.current()
                    source[:size] = 0
                    driver.check(driver.cuMemcpyDtoH_v2(pinned.pointer, pdst, size))
                    if not np.array_equal(source[:size], np.arange(size, dtype=np.uint8)):
                        raise AssertionError("Staged device transfer corrupted data")
                    staged.append({"source_device": src.ordinal, "destination_device": dst.ordinal,
                                   "route": "explicit_device_to_pinned_host_to_device", **measurement,
                                   "bytes_verified": True})
    return {"schema_version": 1, "probe": "transfer_bandwidth", "status": "pass", "inventory": inventory,
            "transfers": results, "staged_transfers": staged,
            "peak_reserved_pinned_bytes": budget.peak,
            "limitations": ["Other workloads remain active; measurements include contention and host-call overhead.",
                            "Synchronous end-to-end transfer timing, not PCIe peak bandwidth or full-model throughput.",
                            "GPU contexts and module allocations are additional to explicit device-buffer accounting."]}
