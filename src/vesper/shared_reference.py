"""Original FP8 shared-expert parity, with complete per-boundary evidence."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import time

import numpy as np

from .conformance import compare
from .expert_reference import OfficialExpert
from .formats import bf16_bits, bf16_values
from .resources import MemoryBudget, MiB, physical_memory
from .shared import ResidentSharedExpert, load_shared_fixture


def shared_inputs():
    """Deterministic finite BF16 inputs; no captured full-model activations."""
    for seed in (20261018, 0, 1, 42, 127, 512, 4096, 65535):
        base = np.random.default_rng(seed).standard_normal(5120).astype(np.float32)
        for scale in (2.0**-20, 2.0**-8, 0.0625, 1.0, 16.0, 128.0):
            yield f"normal_seed{seed}_scale{scale:g}", bf16_values(bf16_bits(base * np.float32(scale)))
    yield "zero", np.zeros(5120, np.float32)
    zero = np.zeros(5120, np.float32)
    zero[1::2] = -0.0
    yield "signed_zero", zero
    for value in (1.0, -1.0, 1e-30, 128.0):
        yield f"constant_{value:g}", bf16_values(bf16_bits(np.full(5120, value, np.float32)))


def expert_stages(output, *, native=False):
    stages = {name: output[name] for name in ("q", "s", "w1", "w3", "down_input", "down_q", "down_s", "w2")}
    if native:
        stages["down_input"] = bf16_bits(stages["down_input"])
    stages.update({name: output["trace"][:, i] for i, name in enumerate(
        ("clamped_gate", "clamped_up", "silu_fp32", "product_fp32", "routed_fp32"))})
    return stages


def digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def compare_shared(fixture: Path, reference_directory: Path, ordinal=0, reference_ordinal=0):
    missing = [name for name in ("torch", "tilelang") if importlib.util.find_spec(name) is None]
    if missing:
        return {"probe": "shared_expert_parity", "status": "blocked", "missing_dependencies": missing}
    import torch
    import tilelang

    budget = MemoryBudget(cap=192 * MiB)
    budget.acquire(192 * MiB)
    try:
        available_samples = [physical_memory()["available"]]
        arrays, manifest = load_shared_fixture(fixture)
        reference = OfficialExpert(arrays, manifest, reference_directory, reference_ordinal,
                                   weight_dtype=torch.float8_e4m3fn)
        records = []
        with ResidentSharedExpert(arrays, ordinal) as candidate:
            for label, x in shared_inputs():
                available_samples.append(physical_memory()["available"])
                budget.acquire(0, available=available_samples[-1])
                expected = expert_stages(reference.forward(x))
                actual = expert_stages(candidate.forward(x), native=True)
                checks = [compare(expected[name], actual[name], name) for name in expected]
                records.append({"input": label, "input_bf16_sha256": digest(bf16_bits(x)), "checks": checks,
                                "expected_sha256": {name: digest(array) for name, array in expected.items()},
                                "status": "pass" if all(c["status"] == "pass" for c in checks) else "fail"})
            info, peak = candidate.info, candidate.device.peak
            device_object = candidate.device
            cubin, fp8_cubin = candidate.cubin_sha256, candidate.fp8_cubin_sha256
        return {"probe": "shared_expert_parity", "schema_version": 1,
                "status": "pass" if records and all(r["status"] == "pass" for r in records) else "fail",
                "device": info, "reference_device_ordinal": reference_ordinal,
                "reference": reference.manifest, "fixture": manifest, "cases": records,
                "compiled_expert_cubin_sha256": cubin, "compiled_fp8_cubin_sha256": fp8_cubin,
                "peak_explicit_device_allocation_bytes": peak,
                "cleanup_explicit_device_allocation_bytes": sum(device_object.allocations.values()),
                "minimum_observed_host_available_bytes": min(available_samples),
                "host_probe_reservation_bytes": budget.peak, "host_available_floor_bytes": budget.available_floor,
                "resource_exclusions": ["Python/NumPy runtime and metadata", "CUDA modules/contexts/compiler",
                                        "PyTorch/TileLang oracle allocations"],
                "torch_version": torch.__version__, "torch_git_version": torch.version.git_version,
                "torch_cuda_version": torch.version.cuda, "tilelang_version": tilelang.__version__,
                "numpy_version": np.__version__, "compiler_options": ["fmad=false", "ftz=false"],
                "reference_method": "Unchanged linear and Expert.forward AST bodies with original FP8 shared weights",
                "reference_runtime_workarounds": reference.runtime_workarounds,
                "scope": f"One layer-{manifest['layer']} shared expert; finite BF16 single-token inputs; 13 arithmetic boundaries",
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "full_moe_quality": "not_run", "full_model_quality": "not_run"}
    finally:
        budget.release(192 * MiB)
