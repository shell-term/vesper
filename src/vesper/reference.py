"""Optional official-kernel comparison. Missing dependencies are a blocked gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from .checkpoint import fetch_reference, load_fixture
from .conformance import _run, compare
from .cuda import Device, Driver, compile_probe
from .formats import bf16_bits, bf16_values
from .reference_runtime import configure_reference_runtime


def compare_official(fixture: Path | None, reference_directory: Path, ordinal: int,
                     reference_ordinal: int | None = None, seed: int = 20261018) -> dict:
    reference_ordinal = ordinal if reference_ordinal is None else reference_ordinal
    missing = [name for name in ("torch", "tilelang") if importlib.util.find_spec(name) is None]
    if missing:
        return {"probe": "official_kernel_parity", "status": "blocked", "missing_dependencies": missing,
                "required": "torch>=2.10.0 with CUDA and tilelang==0.1.12, matching the pinned reference",
                "full_model_quality": "not_run"}
    import torch
    import tilelang
    if not torch.cuda.is_available():
        return {"probe": "official_kernel_parity", "status": "blocked", "reason": "PyTorch CUDA is unavailable"}
    if not hasattr(torch, "float8_e8m0fnu") or not hasattr(torch, "float4_e2m1fn_x2"):
        return {"probe": "official_kernel_parity", "status": "blocked", "reason": "PyTorch lacks native reference dtypes"}
    if fixture is None:
        raise ValueError("Supply the revision-pinned --fixture before comparing official kernels")
    manifest = fetch_reference(reference_directory)
    spec = importlib.util.spec_from_file_location("vesper_pinned_official_kernel", reference_directory / "kernel.py")
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    workarounds = configure_reference_runtime(official, tilelang.__version__)
    rng = np.random.default_rng(seed)
    fixture_data = load_fixture(fixture)
    weight, scales = fixture_data["weight"], fixture_data["scales"]
    x = bf16_values(bf16_bits(rng.standard_normal((1, weight.shape[1] * 2)).astype(np.float32)))
    driver = Driver()
    info = next(d for d in driver.inventory()["devices"] if d["ordinal"] == ordinal)
    # Refuse a heavyweight reference probe if the same conservative device reserve is unavailable.
    with Device(driver, reference_ordinal) as device:
        if device.memory()["free"] < 1536 * 1024**2:
            return {"probe": "official_kernel_parity", "status": "blocked", "reason": "Insufficient reference-probe VRAM headroom"}
    with torch.cuda.device(reference_ordinal), torch.inference_mode():
        original_default = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            tx = torch.from_numpy(x).to(device=f"cuda:{reference_ordinal}", dtype=torch.bfloat16)
            tw = torch.from_numpy(weight.copy()).to(device=f"cuda:{reference_ordinal}").view(torch.float4_e2m1fn_x2)
            ts = torch.from_numpy(scales.copy()).to(device=f"cuda:{reference_ordinal}").view(torch.float8_e8m0fnu)
            qa, sa = official.act_quant(tx, 32, "ue8m0", torch.float8_e8m0fnu)
            expected = official.fp4_gemm(qa, sa, tw, ts, torch.float8_e8m0fnu, act_block_size=32)
            q = qa.view(torch.uint8).cpu().numpy()
            s = sa.view(torch.uint8).cpu().numpy()
            expected_bits = expected.view(torch.uint16).cpu().numpy().reshape(-1)
        finally:
            torch.set_default_dtype(original_default)
    with Device(driver, ordinal) as device:
        module = device.load(compile_probe(info["compute_capability"].replace(".", "")))
        actual_q, actual_s = _run(device, module, "activation32", [x], [(q.shape, np.uint8), (s.shape, np.uint8)],
                                     [x.size // 32], x.size // 32)
        use_mma = tuple(map(int, info["compute_capability"].split("."))) >= (8, 9)
        candidate_kernel = "gemv_fp4_mma" if use_mma else "gemv_fp4_bf16_mma"
        actual, = _run(device, module, candidate_kernel, [q, s, weight, scales],
                       [(expected_bits.shape, np.uint16)], [weight.shape[0], weight.shape[1] * 2], weight.shape[0] * 4)
    checks = [compare(q, actual_q, "official_act_quant_E4M3"), compare(s, actual_s, "official_act_quant_E8M0"),
              compare(expected_bits, actual, "official_fp4_gemm_BF16_output")]
    return {"probe": "official_kernel_parity", "status": "pass" if all(c["status"] == "pass" for c in checks) else "fail",
            "device": info, "checks": checks, "reference": manifest, "torch_version": torch.__version__,
            "reference_device_ordinal": reference_ordinal, "candidate_kernel": candidate_kernel,
            "seed": seed,
            "reference_runtime_workarounds": workarounds,
            "tilelang_version": tilelang.__version__, "full_model_quality": "not_run"}
