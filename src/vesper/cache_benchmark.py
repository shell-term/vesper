"""Paired, warm-cache microbenchmark. No router or model-token rate is inferred."""

from collections import Counter
import hashlib
from pathlib import Path
import statistics
import time

import numpy as np

from .bandwidth import EXPERT_BYTES
from .resources import MemoryBudget, MiB, physical_memory
from .storage import NativeStore
from .tiered import CacheConfig, TieredExpertRuntime


def output_digest(output):
    digest = hashlib.sha256()
    for name, array in sorted(output.items()):
        digest.update(name.encode())
        digest.update(array.dtype.str.encode())
        digest.update(str(array.shape).encode())
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def benchmark_cache(store: Path, ordinal=0, helper_ordinal=1, trials=6, repeats=8):
    if not 2 <= trials <= 8 or not 1 <= repeats <= 16:
        raise ValueError("Paired probe requires 2..8 trials and 1..16 pattern repetitions")
    metadata = NativeStore(store, MemoryBudget(cap=96 * MiB))
    keys = list(metadata.entries)
    if len(keys) != 3:
        raise ValueError("This paired benchmark requires exactly three original experts")
    pattern = [keys[i] for i in (0, 1, 0, 2, 2, 1, 0)]
    x = np.random.default_rng(20261018).standard_normal(5120).astype(np.float32)
    expected, runs = {}, []
    before_memory, minimum_available = physical_memory(), physical_memory()["available"]
    for trial in range(trials):
        order = ("readback", "gpu-sha256") if trial % 2 == 0 else ("gpu-sha256", "readback")
        for order_index, mode in enumerate(order):
            setup_start = time.perf_counter()
            runtime = TieredExpertRuntime(store, CacheConfig(host_cap_bytes=96 * MiB, ram_cache_bytes=EXPERT_BYTES,
                                         primary_device=ordinal, helper_device=helper_ordinal, profile=True, integrity_mode=mode))
            records, mismatches = [], 0
            with runtime:
                setup_ms = (time.perf_counter() - setup_start) * 1000
                for key in pattern:
                    result = runtime.forward(key, x, 0.25)
                    digest = output_digest(result)
                    if key not in expected:
                        expected[key] = digest
                    if expected[key] != digest:
                        mismatches += 1
                after_warm = runtime.snapshot()
                before_timings = runtime.timings.snapshot()
                for _ in range(repeats):
                    for key in pattern:
                        result = runtime.forward(key, x, 0.25)
                        # Timing stops inside forward; fingerprint comparison is outside it.
                        if output_digest(result) != expected[key]:
                            mismatches += 1
                        records.append(dict(runtime.last_event))
                        minimum_available = min(minimum_available, physical_memory()["available"])
                snapshot = runtime.snapshot()
                stage_timings = runtime.timings.since(before_timings)
                if snapshot["store_direct_read_calls"] != after_warm["store_direct_read_calls"]:
                    raise AssertionError("Warm benchmark unexpectedly reloaded storage; results would not be comparable")
                info = runtime.executor.driver.inventory()
            runs.append({"trial": trial, "order_index": order_index, "integrity_mode": mode,
                         "setup_ms": setup_ms, "warmup_calls": len(pattern), "after_warmup": after_warm,
                         "calls": records, "timed_stage_totals": stage_timings, "snapshot": snapshot,
                         "cleanup": runtime.cleanup, "output_fingerprint_mismatches": mismatches,
                         "timed_call_wall_ms": sum(c["wall_ms"] for c in records)})
    ratios = []
    for trial in range(trials):
        pair = {r["integrity_mode"]: r["timed_call_wall_ms"] for r in runs if r["trial"] == trial}
        ratios.append(pair["readback"] / pair["gpu-sha256"])
    modes = {}
    for mode in ("readback", "gpu-sha256"):
        calls = [c for r in runs if r["integrity_mode"] == mode for c in r["calls"]]
        modes[mode] = {"timed_calls": len(calls), "source_counts": dict(Counter(c["source_tier"] for c in calls)),
                       "source_median_ms": {tier: statistics.median(c["wall_ms"] for c in calls if c["source_tier"] == tier)
                                            for tier in sorted({c["source_tier"] for c in calls})}}
    ok = all(not r["output_fingerprint_mismatches"] and not any(r["cleanup"].values()) for r in runs)
    return {"probe": "paired_cache_benchmark", "status": "pass" if ok else "fail", "cuda": info,
            "trials": trials, "repeats": repeats, "seed": 20261018, "routing_weight": 0.25,
            "pattern": pattern, "reference_output_fingerprints": expected, "modes": modes, "runs": runs,
            "paired_wall_time_ratios_readback_over_gpu_sha256": ratios,
            "paired_ratio_median": statistics.median(ratios), "paired_ratio_min": min(ratios), "paired_ratio_max": max(ratios),
            "minimum_observed_system_available_bytes": minimum_available, "host_memory_before": before_memory,
            "host_memory_after": physical_memory(),
            "scope": "Alternating mode order; identical warm synthetic request traces and capacities; no timed store reads",
            "timing_scope": "Host wall time inside forward includes synchronous transfers, hashing, compute and diagnostics; excludes setup, oracle and fingerprint comparisons",
            "memory_scope": "Runtime buffer ledger includes original digest tables, pinned readback and comparison scratch; Python/driver/compiler and returned diagnostics are outside it",
            "full_model_quality": "not_run", "generation_speed": "not_measured"}
