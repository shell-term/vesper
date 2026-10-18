"""Controlled, real-weight cache traces compared with the unchanged official expert.

The trace is synthetic. It does not estimate router locality or generation speed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import time

import numpy as np

from .bandwidth import EXPERT_BYTES
from .checkpoint import MODEL, REVISION, load_expert_fixture
from .conformance import compare
from .cuda import Device, Driver
from .expert_reference import OfficialExpert
from .formats import bf16_bits, bf16_values
from .resources import GiB, MiB, physical_memory
from .storage import expert_key
from .tiered import CacheConfig, TieredExpertRuntime


def pipeline_checks(expected, actual):
    actual["down_input"] = bf16_bits(actual["down_input"])
    checks = [compare(expected[name], actual[name], name) for name in
              ("q", "s", "w1", "w3", "down_input", "down_q", "down_s", "w2")]
    checks += [compare(expected["trace"][:, i], actual["trace"][:, i], name) for i, name in
               enumerate(("clamped_gate", "clamped_up", "silu_fp32", "product_fp32", "routed_fp32"))]
    return checks


def compare_cache(store: Path, fixtures: list[Path], reference_directory: Path, ordinal=0, helper_ordinal=1,
                  reference_ordinal=0, seeds=(20261018, 0, 1, 42), input_scales=(1.0, 16.0),
                  routing_weights=(None, 0.25, 1.5, 0.0), integrity_mode="readback", profile=False) -> dict:
    missing = [name for name in ("torch", "tilelang") if importlib.util.find_spec(name) is None]
    if missing:
        return {"probe": "tiered_expert_parity", "status": "blocked", "missing_dependencies": missing}
    if len(fixtures) != 3:
        raise ValueError("This controlled cache probe requires exactly three distinct real experts")
    import torch
    import tilelang
    if not torch.cuda.is_available() or not all(hasattr(torch, name) for name in ("float8_e8m0fnu", "float4_e2m1fn_x2")):
        return {"probe": "tiered_expert_parity", "status": "blocked", "reason": "Native FP4/E8M0 CUDA PyTorch is required"}
    with Device(Driver(), reference_ordinal) as device:
        if device.memory()["free"] < 1536 * MiB:
            return {"probe": "tiered_expert_parity", "status": "blocked", "reason": "Insufficient oracle VRAM headroom"}
    before_memory = physical_memory()
    references, provenance = {}, {}
    for fixture in fixtures:
        arrays, manifest = load_expert_fixture(fixture)
        key = expert_key(manifest["layer"], manifest["expert"])
        if key in references:
            raise ValueError("Cache validation needs distinct original expert identities")
        references[key] = OfficialExpert(arrays, manifest, reference_directory, reference_ordinal)
        provenance[key] = manifest
        del arrays
    keys = list(references)
    pattern = [keys[i] for i in (0, 1, 0, 2, 2, 1, 0)]
    records, phases, minimum_available = [], [], before_memory["available"]

    def check(runtime, phase, key, x, seed, scale, route):
        nonlocal minimum_available
        expected = references[key].forward(x, route)
        actual = runtime.forward(key, x, route)
        checks = pipeline_checks(expected, actual)
        minimum_available = min(minimum_available, physical_memory()["available"])
        records.append({"phase": phase, "key": key, "seed": seed, "input_scale": scale,
                        "routing_weight": route, **runtime.last_event, "checks": checks,
                        "status": "pass" if all(c["status"] == "pass" for c in checks) else "fail"})

    def runtime_for(ram):
        return TieredExpertRuntime(store, CacheConfig(host_cap_bytes=96 * MiB, ram_cache_bytes=ram,
                                   available_floor=32 * GiB, primary_device=ordinal, helper_device=helper_ordinal,
                                   integrity_mode=integrity_mode, profile=profile))

    runtime = runtime_for(EXPERT_BYTES)
    with runtime:
        if set(runtime.store.entries) != set(keys):
            raise ValueError("Store and oracle fixtures must identify the same three experts")
        for key in keys:
            if runtime.store.entries[key]["provenance"] != provenance[key]:
                raise ValueError("Store and oracle fixture provenance differ")
        for seed in seeds:
            base = np.random.default_rng(seed).standard_normal(5120).astype(np.float32)
            for scale in input_scales:
                x = bf16_values(bf16_bits(base * np.float32(scale)))
                for route in routing_weights:
                    for key in pattern:
                        check(runtime, "three_tier_warm_trace", key, x, seed, scale, route)
        warm_snapshot = runtime.snapshot()
        if warm_snapshot["store_direct_read_calls"] != 3 or len(runtime.cache.ram) != 1:
            raise AssertionError("Three available cache slots did not retain the three requested experts")
        # Inject only the observation; do not consume real system memory.
        reload_key = next(iter(runtime.cache.ram))
        saved_provider = runtime.cache.available
        refused = False
        try:
            runtime.cache.available = lambda: runtime.config.available_floor - 1
            try:
                runtime.forward(reload_key, x, route)
            except MemoryError:
                refused = True
        finally:
            runtime.cache.available = saved_provider
        pressure_snapshot = runtime.snapshot()
        if not refused or pressure_snapshot["ram_cache_bytes"] or pressure_snapshot["store_direct_read_calls"] != 3:
            raise AssertionError("Injected pressure did not clear RAM and refuse work before I/O")
        check(runtime, "pressure_recovery", reload_key, x, seed, scale, route)
        recovered_snapshot = runtime.snapshot()
        if recovered_snapshot["store_direct_read_calls"] != 4:
            raise AssertionError("The released expert was not reloaded from the original store")
        primary_info, cubin = runtime.executor.info, runtime.executor.cubin_sha256
        helper_info = next(d for d in runtime.executor.driver.inventory()["devices"] if d["ordinal"] == helper_ordinal)
    phases.append({"name": "ram_one_expert", "warm": warm_snapshot,
                   "injected_pressure": {"method": "Available-memory provider returns floor minus one; no real allocation",
                                         "request_refused": refused, "snapshot": pressure_snapshot},
                   "after_recovery": recovered_snapshot, "cleanup": runtime.cleanup})

    runtime = runtime_for(0)
    with runtime:
        for key in pattern:
            check(runtime, "zero_ram_forced_reload", key, x, seed, scale, route)
        forced_snapshot = runtime.snapshot()
        if forced_snapshot["store_direct_read_calls"] != 5 or forced_snapshot["ram_cache_bytes"]:
            raise AssertionError("Two GPU slots with zero RAM did not force the expected original-byte reloads")
    phases.append({"name": "ram_zero", "snapshot": forced_snapshot, "cleanup": runtime.cleanup})
    cleanup_ok = all(not any(p["cleanup"].values()) for p in phases)
    stages_ok = bool(records) and all(r["status"] == "pass" for r in records)
    first_reference = references[keys[0]]
    return {"probe": "tiered_expert_parity", "status": "pass" if stages_ok and cleanup_ok else "fail",
            "model": MODEL, "revision": REVISION, "primary_device": primary_info, "helper_device": helper_info,
            "helper_role": "Original byte storage only; no model arithmetic", "reference_device_ordinal": reference_ordinal,
            "integrity_mode": integrity_mode,
            "reference": first_reference.manifest, "reference_runtime_workarounds": first_reference.runtime_workarounds,
            "fixtures": list(provenance.values()), "request_pattern": pattern, "cases": records, "phases": phases,
            "final_bf16_values_checked": len(records) * 5120, "checks_per_case": 13,
            "minimum_observed_system_available_bytes": minimum_available, "host_memory_before_oracle": before_memory,
            "host_memory_after_probe": physical_memory(), "compiled_cubin_sha256": cubin,
            "torch_version": torch.__version__, "torch_cuda_version": torch.version.cuda,
            "tilelang_version": tilelang.__version__, "numpy_version": np.__version__,
            "compiler_options": ["fmad=false", "ftz=false"],
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "scope": "Three original experts; synthetic single-token request order; synchronous transfers; all uploads audited",
            "memory_scope": "Owned cache/staging/I-O buffers plus metadata allowance only; excludes Python, CUDA contexts, compiler, returned diagnostics and the three resident official oracles",
            "router_locality": "not_measured", "full_layer_quality": "not_run", "full_model_quality": "not_run",
            "generation_speed": "not_measured"}
