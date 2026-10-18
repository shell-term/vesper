"""Localize native-FP8/BF16 differences without treating either as exact parity."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .checkpoint import MODEL, REVISION, load_fixture
from .conformance import _run, compare
from .cuda import Device, Driver, compile_probe
from .formats import bf16_bits, bf16_values, e4m3_values, fp4_values, quantize_activation


def trace_reduction(fixture: Path, ordinal=1, reference_ordinal=0, seed=20261018) -> dict:
    arrays = load_fixture(fixture)
    w, sw = arrays["weight"], arrays["scales"]
    rows, width = w.shape[0], w.shape[1] * 2
    groups = width // 32
    x = bf16_values(bf16_bits(np.random.default_rng(seed).standard_normal(width).astype(np.float32)))
    q, sa = quantize_activation(x)
    codes = np.empty((rows, width), np.uint8)
    codes[:, 0::2], codes[:, 1::2] = w & 15, w >> 4
    dots = np.einsum("gk,ngk->ng", e4m3_values(q).reshape(groups, 32).astype(np.float64),
                     fp4_values(codes).reshape(rows, groups, 32).astype(np.float64)).astype(np.float32)
    driver = Driver()
    inventory = driver.inventory()
    ref_info = next(d for d in inventory["devices"] if d["ordinal"] == reference_ordinal)
    info = next(d for d in inventory["devices"] if d["ordinal"] == ordinal)
    if tuple(map(int, ref_info["compute_capability"].split("."))) < (8, 9):
        raise ValueError("The diagnostic reference device requires native FP8 MMA")
    with Device(driver, reference_ordinal) as ref, Device(driver, ordinal) as candidate:
        ref_cubin = compile_probe(ref_info["compute_capability"].replace(".", ""), reduction_trace=True)
        candidate_cubin = compile_probe(info["compute_capability"].replace(".", ""), reduction_trace=True)
        rm, cm = ref.load(ref_cubin), candidate.load(candidate_cubin)
        def trace(device, module, kernel):
            return _run(device, module, kernel, [q, sa, w, sw],
                        [((rows, groups, 3), np.float32)], [rows, width], rows * 4)[0]
        def projection(device, module, kernel):
            return _run(device, module, kernel, [q, sa, w, sw],
                        [((rows,), np.uint16)], [rows, width], rows * 4)[0]
        native = trace(ref, rm, "trace_fp4_native")
        bf16 = trace(candidate, cm, "trace_fp4_bf16")
        bf16_on_ref = trace(ref, rm, "trace_fp4_bf16")
        integrity = [compare(projection(ref, rm, "gemv_fp4_mma"), bf16_bits(native[:, -1, 2]), "native_trace_final"),
                     compare(projection(candidate, cm, "gemv_fp4_bf16_mma"), bf16_bits(bf16[:, -1, 2]), "candidate_trace_final"),
                     compare(bf16_on_ref, bf16, "bf16_trace_across_devices")]
        peak = {"reference": ref.peak, "candidate": candidate.peak}
    checks = [compare(native[:, :, i], bf16[:, :, i], name)
              for i, name in enumerate(("block_dot", "scaled_block", "running_sum"))]
    exact = [compare(dots, native[:, :, 0], "float64_oracle_vs_native_block_dot"),
             compare(dots, bf16[:, :, 0], "float64_oracle_vs_bf16_block_dot")]
    final = compare(bf16_bits(native[:, -1, 2]), bf16_bits(bf16[:, -1, 2]), "BF16_projection_output")
    divergent = np.argwhere(native[:, :, 0].view(np.uint32) != bf16[:, :, 0].view(np.uint32))
    examples = []
    for row, group in sorted(divergent.tolist(), key=lambda index: (index[1], index[0]))[:8]:
        examples.append({"row": row, "block": group, "native_FP8_dot": float(native[row, group, 0]),
                         "candidate_BF16_dot": float(bf16[row, group, 0]),
                         "native_hex": f"0x{int(native[row, group, 0].view(np.uint32)):08x}",
                         "candidate_hex": f"0x{int(bf16[row, group, 0].view(np.uint32)):08x}"})
    return {"probe": "reduction_diagnostic", "status": "pass" if all(c["status"] == "pass" for c in integrity + checks) else "fail",
            "model": MODEL, "revision": REVISION, "seed": seed, "device": info, "reference_device": ref_info,
            "integrity_checks": integrity, "stage_comparisons": checks, "independent_dot_checks": exact,
            "projection_comparison": final, "first_dot_differences": examples,
            "peak_explicit_device_allocation_bytes": peak, "numpy_version": np.__version__,
            "compiled_cubin_sha256": {"reference": hashlib.sha256(ref_cubin).hexdigest(),
                                      "candidate": hashlib.sha256(candidate_cubin).hexdigest()},
            "limitations": ["The native MMA trace is a diagnostic path, not the original TileLang kernel.",
                            "Float64 dot agreement does not establish official-kernel parity.",
                            "FP8 and BF16 MMA reduction order/rounding are not assumed interchangeable."],
            "official_kernel_parity": "separate_gate", "full_model_quality": "not_run"}
