"""Bounded GPU SHA-256 checks against hashlib, including deliberate corruption."""

from contextlib import ExitStack
import ctypes as C
import time

import numpy as np

from .bandwidth import EXPERT_BYTES
from .cuda import Device, Driver, Pinned
from .integrity import DeviceIntegrity, digest_shape, hash_chunks
from .resources import MemoryBudget, MiB


def validate_integrity(ordinals=None):
    driver = Driver()
    inventory = driver.inventory()
    ordinals = [d["ordinal"] for d in inventory["devices"]] if ordinals is None else ordinals
    if not ordinals or len(ordinals) != len(set(ordinals)):
        raise ValueError("Select distinct available GPU ordinals")
    if any(n not in [d["ordinal"] for d in inventory["devices"]] for n in ordinals):
        raise ValueError("Selected integrity GPU does not exist")
    budget = MemoryBudget(cap=64 * MiB)
    shape = digest_shape(EXPERT_BYTES)
    host_array_bytes = EXPERT_BYTES * 2 + shape[0] * shape[1] * 2
    records = []
    with ExitStack() as stack:
        budget.acquire(host_array_bytes)
        stack.callback(budget.release, host_array_bytes)
        original = np.random.default_rng(20261018).integers(0, 256, EXPERT_BYTES, dtype=np.uint8)
        work = np.empty_like(original)
        expected = np.empty(shape, np.uint8)
        comparison = np.empty(shape, np.bool_)
        lengths = (1, 3, 55, 56, 63, 64, 65, 119, 120, 127, 128, 4095, 4096, 4097, 8192, EXPERT_BYTES)
        for ordinal in ordinals:
            info = next(d for d in inventory["devices"] if d["ordinal"] == ordinal)
            cases, faults = [], []
            with Device(driver, ordinal, cap=32 * MiB) as device:
                with Pinned(driver, shape[0] * shape[1], budget) as pinned:
                    readback = pinned.array().reshape(shape)
                    checker = DeviceIntegrity(device, info["compute_capability"], EXPERT_BYTES)
                    try:
                        pointer = device.allocate(EXPERT_BYTES)
                        for pattern in ("random", "zeros", "ones"):
                            if pattern == "random":
                                np.copyto(work, original)
                            else:
                                work.fill(0 if pattern == "zeros" else 255)
                            for size in lengths:
                                subshape = digest_shape(size)
                                target = expected[:subshape[0]]
                                hash_chunks(work[:size], target)
                                device.current()
                                driver.check(driver.cuMemcpyHtoD_v2(pointer, work.ctypes.data, size))
                                started = time.perf_counter()
                                actual = checker.digests(pointer, size=size, target=readback[:subshape[0]])
                                elapsed = (time.perf_counter() - started) * 1000
                                np.not_equal(actual, target, out=comparison[:subshape[0]])
                                differences = int(np.count_nonzero(comparison[:subshape[0]]))
                                cases.append({"pattern": pattern, "bytes": size, "chunks": subshape[0],
                                              "digest_byte_mismatches": differences, "wall_ms": elapsed})
                        # A known, nonrandom message also checks byte order and padding.
                        work[:3] = np.frombuffer(b"abc", np.uint8)
                        driver.check(driver.cuMemcpyHtoD_v2(pointer, work.ctypes.data, 3))
                        abc = checker.digests(pointer, size=3, target=readback[:1]).tobytes().hex()
                        known_ok = abc == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
                        np.copyto(work, original)
                        hash_chunks(work, expected)
                        driver.check(driver.cuMemcpyHtoD_v2(pointer, work.ctypes.data, EXPERT_BYTES))
                        checker.verify(pointer, expected, readback, comparison)
                        for offset in (0, 63, 4095, 4096, EXPERT_BYTES - 1):
                            altered = C.c_uint8(int(work[offset]) ^ 1)
                            driver.check(driver.cuMemcpyHtoD_v2(pointer + offset, C.byref(altered), 1))
                            refused = False
                            try:
                                checker.verify(pointer, expected, readback, comparison)
                            except ValueError:
                                refused = True
                            restored = C.c_uint8(int(work[offset]))
                            driver.check(driver.cuMemcpyHtoD_v2(pointer + offset, C.byref(restored), 1))
                            checker.verify(pointer, expected, readback, comparison)
                            faults.append({"offset": offset, "changed_bits": 1, "corruption_rejected": refused,
                                           "restored_bytes_accepted": True})
                        peak = device.peak
                    finally:
                        checker.close()
            records.append({"device": info, "cases": cases, "known_abc_sha256_matches": known_ok,
                            "faults": faults, "compiled_cubin_sha256": checker.cubin_sha256,
                            "peak_explicit_device_bytes": peak,
                            "cleanup_explicit_device_bytes": sum(device.allocations.values())})
        del original, work, expected, comparison
    ok = all(r["known_abc_sha256_matches"] and all(c["digest_byte_mismatches"] == 0 for c in r["cases"])
             and all(f["corruption_rejected"] for f in r["faults"]) and not r["cleanup_explicit_device_bytes"] for r in records)
    return {"probe": "gpu_chunk_sha256", "status": "pass" if ok and budget.reserved == 0 else "fail",
            "cuda": inventory, "chunk_bytes": 4096, "digest_bytes_per_chunk": 32, "records": records,
            "host_peak_buffer_reservations_bytes": budget.peak, "host_cleanup_reserved_bytes": budget.reserved,
            "scope": "Nonempty bounded payloads, SHA padding/chunk boundaries, three byte patterns, one known message and single-bit faults; not cryptographic certification",
            "full_model_quality": "not_run"}
