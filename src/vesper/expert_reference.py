"""Full-expert checks using unchanged functions extracted from pinned model.py."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from .checkpoint import fetch_reference, load_expert_fixture
from .conformance import compare
from .cuda import Device, Driver
from .expert import ResidentExpert
from .formats import bf16_bits, bf16_values
from .reference_runtime import configure_reference_runtime


class OfficialExpert:
    """Runs the original linear() and Expert.forward() bodies without a full model.

    The AST selection removes unrelated imports/classes, never rewrites the
    selected function bodies. Projection wrappers only record inputs/outputs.
    """

    def __init__(self, arrays, manifest, directory: Path, ordinal=0):
        import torch
        from torch.nn import functional as F

        self.torch, self.ordinal, self.limit = torch, ordinal, manifest["swiglu_limit"]
        self.manifest = fetch_reference(directory, include_model=True)
        spec = importlib.util.spec_from_file_location("vesper_expert_official_kernel", directory / "kernel.py")
        self.kernel = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.kernel)
        import tilelang
        self.runtime_workarounds = configure_reference_runtime(self.kernel, tilelang.__version__)
        tree = ast.parse((directory / "model.py").read_text(encoding="utf-8"))
        linear = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "linear")
        expert = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Expert")
        forward = next(node for node in expert.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
        scope = {"torch": torch, "F": F, "act_quant": self._quantize,
                 "fp4_gemm": self.kernel.fp4_gemm, "fp8_gemm": self.kernel.fp8_gemm,
                 "fp8_block_size": 32, "scale_fmt": "ue8m0", "scale_dtype": torch.float8_e8m0fnu}
        exec(compile(ast.Module(body=[linear, forward], type_ignores=[]), "pinned_deepseek_model.py", "exec"), scope)
        self.linear, self.forward_function = scope["linear"], scope["forward"]
        self.weights = {}
        with torch.cuda.device(ordinal):
            for name in ("w1", "w3", "w2"):
                weight = torch.from_numpy(arrays[f"{name}_weight"].copy()).to(f"cuda:{ordinal}").view(torch.float4_e2m1fn_x2)
                weight.scale = torch.from_numpy(arrays[f"{name}_scale"].copy()).to(f"cuda:{ordinal}").view(torch.float8_e8m0fnu)
                self.weights[name] = weight
        self.expert = SimpleNamespace(swiglu_limit=self.limit,
                                      **{name: self._projection(name) for name in self.weights})

    def _quantize(self, x, *args, **kwargs):
        q, scales = self.kernel.act_quant(x, *args, **kwargs)
        self.quantized[self.active] = (q, scales)
        return q, scales

    def _projection(self, name):
        def project(x):
            self.active = name
            if name == "w2":
                self.down_input = x
            result = self.linear(x, self.weights[name])
            self.outputs[name] = result
            return result
        return project

    def forward(self, x, routing_weight=None):
        torch = self.torch
        F = torch.nn.functional
        self.outputs, self.quantized = {}, {}
        with torch.cuda.device(self.ordinal), torch.inference_mode():
            default = torch.get_default_dtype()
            try:
                torch.set_default_dtype(torch.bfloat16)
                tx = torch.from_numpy(np.ascontiguousarray(x.reshape(1, -1))).to(f"cuda:{self.ordinal}", dtype=torch.bfloat16)
                route = None if routing_weight is None else torch.tensor([[routing_weight]], device=tx.device, dtype=torch.float32)
                output = self.forward_function(self.expert, tx, route)
                gate, up = self.outputs["w1"].float(), self.outputs["w3"].float()
                if self.limit > 0:
                    gate = torch.clamp(gate, max=self.limit)
                    up = torch.clamp(up, min=-self.limit, max=self.limit)
                silu = F.silu(gate)
                product = silu * up
                routed = product if route is None else route * product
                assert torch.equal(routed.to(tx.dtype).view(torch.uint16), self.down_input.view(torch.uint16))
                trace = torch.stack((gate, up, silu, product, routed), dim=-1).reshape(-1, 5).cpu().numpy()
                def bits(tensor, dtype):
                    return tensor.view(dtype).cpu().numpy().reshape(-1).copy()
                q, scales = self.quantized["w1"]
                q3, s3 = self.quantized["w3"]
                assert torch.equal(q.view(torch.uint8), q3.view(torch.uint8))
                assert torch.equal(scales.view(torch.uint8), s3.view(torch.uint8))
                dq, ds = self.quantized["w2"]
                return {"q": bits(q, torch.uint8), "s": bits(scales, torch.uint8),
                        "w1": bits(self.outputs["w1"], torch.uint16), "w3": bits(self.outputs["w3"], torch.uint16),
                        "trace": trace, "down_input": bits(self.down_input, torch.uint16),
                        "down_q": bits(dq, torch.uint8), "down_s": bits(ds, torch.uint8),
                        "w2": bits(output, torch.uint16)}
            finally:
                torch.set_default_dtype(default)


def compare_expert(fixture: Path, reference_directory: Path, ordinal=0, reference_ordinal=0,
                   seeds=(20261018,), routing_weights=(None, 0.25, 1.5, 0.0), input_scales=(1.0,)) -> dict:
    missing = [name for name in ("torch", "tilelang") if importlib.util.find_spec(name) is None]
    if missing:
        return {"probe": "full_expert_parity", "status": "blocked", "missing_dependencies": missing}
    import torch
    import tilelang

    if not torch.cuda.is_available() or not all(hasattr(torch, name) for name in ("float8_e8m0fnu", "float4_e2m1fn_x2")):
        return {"probe": "full_expert_parity", "status": "blocked", "reason": "CUDA PyTorch with native FP4/E8M0 dtypes is required"}
    arrays, manifest = load_expert_fixture(fixture)
    driver = Driver()
    with Device(driver, reference_ordinal) as device:
        if device.memory()["free"] < 1536 * 1024**2:
            return {"probe": "full_expert_parity", "status": "blocked", "reason": "Insufficient oracle VRAM headroom"}
    reference = OfficialExpert(arrays, manifest, reference_directory, reference_ordinal)
    records = []
    with ResidentExpert(arrays, ordinal, manifest["swiglu_limit"]) as candidate:
        for seed in seeds:
            base = np.random.default_rng(seed).standard_normal(5120).astype(np.float32)
            for scale in input_scales:
                x = bf16_values(bf16_bits(base * np.float32(scale)))
                for route in routing_weights:
                    expected, actual = reference.forward(x, route), candidate.forward(x, route)
                    actual["down_input"] = bf16_bits(actual["down_input"])
                    checks = [compare(expected[name], actual[name], name) for name in
                              ("q", "s", "w1", "w3", "down_input", "down_q", "down_s", "w2")]
                    checks += [compare(expected["trace"][:, i], actual["trace"][:, i], name) for i, name in
                               enumerate(("clamped_gate", "clamped_up", "silu_fp32", "product_fp32", "routed_fp32"))]
                    records.append({"seed": seed, "input_scale": scale, "routing_weight": route, "checks": checks,
                                    "status": "pass" if all(c["status"] == "pass" for c in checks) else "fail"})
        info, peak, cubin = candidate.info, candidate.device.peak, candidate.cubin_sha256
    return {"probe": "full_expert_parity", "status": "pass" if records and all(r["status"] == "pass" for r in records) else "fail",
            "device": info, "reference_device_ordinal": reference_ordinal, "reference": reference.manifest,
            "fixture": manifest, "cases": records, "peak_explicit_device_allocation_bytes": peak,
            "compiled_cubin_sha256": cubin, "torch_version": torch.__version__, "tilelang_version": tilelang.__version__,
            "numpy_version": np.__version__, "torch_cuda_version": torch.version.cuda,
            "compiler_options": ["fmad=false", "ftz=false"],
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "scope": f"One original layer-{manifest['layer']} expert-{manifest['expert']}; single-token calls; FP32 diagnostic trace and BF16 outputs",
            "reference_method": "Unchanged AST function bodies: linear and Expert.forward from hash-checked model.py",
            "reference_runtime_workarounds": reference.runtime_workarounds,
            "full_layer_quality": "not_run", "full_model_quality": "not_run"}
