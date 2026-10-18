"""Component checks against independent CPU specifications, with no tolerances."""

from __future__ import annotations

import ctypes as C
import hashlib

import numpy as np

from .cuda import Device, Driver, compile_probe
from .formats import (bf16_bits, bf16_values, encode_e4m3, fp4_gemv_spec,
                      quantize_activation, unpack_fp4)


def compare(expected: np.ndarray, actual: np.ndarray, label: str) -> dict:
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        return {"check": label, "status": "fail", "reason": "shape_or_dtype_mismatch"}
    if expected.dtype == np.float32:
        nan_expected, nan_actual = np.isnan(expected), np.isnan(actual)
        same = (expected.view(np.uint32) == actual.view(np.uint32)) | (nan_expected & nan_actual)
        semantics = "exact_float32_bits_with_NaN_classification; signed_zero_preserved"
    else:
        same = expected == actual
        semantics = "exact_storage_bits"
    mismatches = int(np.count_nonzero(~same))
    return {"check": label, "status": "pass" if not mismatches else "fail", "elements": int(expected.size),
            "mismatches": mismatches, "comparison": semantics,
            "first_mismatch_indices": np.flatnonzero(~same)[:8].tolist()}


def _run(device, module, name, arrays, outputs, trailing, items):
    inputs, destinations = [], []
    try:
        inputs = [device.upload(array) for array in arrays]
        destinations = [device.allocate(int(np.prod(shape)) * np.dtype(dtype).itemsize) for shape, dtype in outputs]
        args = [C.c_uint64(p) for p in inputs + destinations] + [C.c_int(v) for v in trailing]
        device.launch(module, name, args, items)
        return [device.download(pointer, shape, dtype) for pointer, (shape, dtype) in zip(destinations, outputs)]
    finally:
        # Device context cleanup also frees any allocations if an upload failed.
        for pointer in inputs + destinations:
            device.free(pointer)


def validate(devices: list[int] | None = None, fixture=None) -> dict:
    driver = Driver()
    inventory = driver.inventory()
    chosen = devices if devices is not None else [d["ordinal"] for d in inventory["devices"]]
    rng = np.random.default_rng(20261018)
    packed = np.tile(np.array([0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE], np.uint8), (256, 2))
    scales = np.arange(256, dtype=np.uint8).reshape(256, 1)
    expected_fp4 = unpack_fp4(packed, scales)
    bf16_space = bf16_values(np.arange(65536, dtype=np.uint16))
    bf16_space = bf16_space[np.isfinite(bf16_space)]
    # Include FP32 values immediately around every E4M3 midpoint too.
    from .formats import e4m3_values
    levels = e4m3_values(np.arange(127, dtype=np.uint8))
    midpoints = (levels[:-1] + levels[1:]) / np.float32(2)
    conversion = np.concatenate([bf16_space, midpoints, -midpoints,
                                 np.nextafter(midpoints, np.float32(np.inf)),
                                 np.nextafter(midpoints, np.float32(-np.inf))]).astype(np.float32)
    expected_fp8 = encode_e4m3(conversion)
    activation = rng.standard_normal((64, 512)).astype(np.float32)
    activation *= np.ldexp(np.ones((64, 1), np.float32), rng.integers(-20, 16, (64, 1)))
    activation[0] = 0
    activation[1, 0::2], activation[1, 1::2] = 0.0, -0.0
    activation[2] = np.float32(1e-30)
    activation = bf16_values(bf16_bits(activation))
    q_expected, s_expected = quantize_activation(activation)
    weight = rng.integers(0, 256, (128, 256), dtype=np.uint8)
    weight_scales = rng.integers(115, 129, (128, 16), dtype=np.uint8)
    gemv_activation = bf16_values(bf16_bits(rng.standard_normal(512).astype(np.float32)))
    q_gemv, s_gemv = quantize_activation(gemv_activation)
    gemv_expected = fp4_gemv_spec(q_gemv, s_gemv, weight, weight_scales)
    fixture_data = None
    if fixture is not None:
        from .checkpoint import load_fixture
        fixture_data = load_fixture(fixture)
        fw, fs = fixture_data["weight"], fixture_data["scales"]
        fx = bf16_values(bf16_bits(rng.standard_normal(fw.shape[1] * 2).astype(np.float32)))
        fq, fqs = quantize_activation(fx)
        fixture_expected = fp4_gemv_spec(fq, fqs, fw, fs)
    results = []
    for ordinal in chosen:
        info = next(d for d in inventory["devices"] if d["ordinal"] == ordinal)
        architecture = info["compute_capability"].replace(".", "")
        cubin = compile_probe(architecture)
        with Device(driver, ordinal) as device:
            memory_before = device.memory()
            module = device.load(cubin)
            checks = []
            actual, = _run(device, module, "decode_fp4", [packed, scales],
                           [(expected_fp4.shape, np.float32)], [expected_fp4.size], expected_fp4.size)
            checks.append(compare(expected_fp4, actual, "E2M1_all_codes_E8M0_all_codes"))
            actual, = _run(device, module, "encode_fp8", [conversion],
                           [(conversion.shape, np.uint8)], [conversion.size], conversion.size)
            checks.append(compare(expected_fp8, actual, "E4M3_BF16_finite_space_and_rounding_midpoints"))
            q_actual, s_actual = _run(device, module, "activation32", [activation],
                                     [(activation.shape, np.uint8), (s_expected.shape, np.uint8)],
                                     [activation.size // 32], activation.size // 32)
            checks.append(compare(q_expected, q_actual, "per32_activation_E4M3_bytes"))
            checks.append(compare(s_expected, s_actual, "per32_activation_E8M0_scale_bytes"))
            actual, = _run(device, module, "gemv_fp4", [q_gemv, s_gemv, weight, weight_scales],
                           [(gemv_expected.shape, np.uint16)], [weight.shape[0], weight.shape[1] * 2], weight.shape[0])
            checks.append(compare(gemv_expected, actual, "synthetic_FP8xFP4_block_GEMV_BF16_output"))
            if fixture_data is not None:
                actual, = _run(device, module, "gemv_fp4", [fq, fqs, fw, fs],
                               [(fixture_expected.shape, np.uint16)], [fw.shape[0], fw.shape[1] * 2], fw.shape[0])
                checks.append(compare(fixture_expected, actual, "official_weight_slice_FP8xFP4_GEMV_specification"))
            results.append({"device": info, "checks": checks, "memory_after_context_before_probes": memory_before,
                            "peak_explicit_device_allocation_bytes": device.peak,
                            "compiled_cubin_sha256": hashlib.sha256(cubin).hexdigest(),
                            "status": "pass" if all(c["status"] == "pass" for c in checks) else "fail"})
    return {"schema_version": 1, "probe": "component_conformance", "backend": "CUDA Driver + NVRTC",
            "compiler_options": ["fmad=false", "ftz=false"], "inventory": inventory, "devices": results,
            "status": "pass" if results and all(r["status"] == "pass" for r in results) else "fail",
            "oracle": "Independent NumPy format/block arithmetic specification; no tolerances",
            "official_TileLang_kernel_parity": "not_run",
            "full_model_quality": "not_run",
            "limitations": ["Scalar GEMV is a correctness probe, not an optimized inference kernel.",
                            "Passing component checks is not proof of official-kernel parity or unchanged full-model quality.",
                            "NaNs are compared by classification, finite values and signed zero by exact bits."]}
