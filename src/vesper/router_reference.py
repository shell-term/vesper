"""Unchanged Gate.forward, with its projection explicitly replaced by supplied logits."""

import ast
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .checkpoint import fetch_reference
from .conformance import compare
from .cuda import Device, Driver
from .router import ROUTER_CONFIG, RouterProbe, UnsupportedRoutingTie, load_router_fixture
from .resources import MiB


class OfficialRouter:
    def __init__(self, arrays, reference_directory, ordinal):
        import torch
        from torch.nn import functional as F
        self.torch, self.ordinal = torch, ordinal
        self.source = fetch_reference(reference_directory, include_model=True)
        tree = ast.parse((reference_directory / "model.py").read_text(encoding="utf-8"))
        gate = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Gate")
        forward = next(n for n in gate.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
        scope = {"torch": torch, "F": F, "linear": lambda x, unused_weight: x}
        exec(compile(ast.Module(body=[forward], type_ignores=[]), "pinned_gate_forward.py", "exec"), scope)
        self.forward_function = scope["forward"]
        device = f"cuda:{ordinal}"
        self.bias = torch.from_numpy(arrays["bias"].copy()).to(device)
        self.bias_vl = torch.from_numpy(arrays["bias_vl"].copy()).to(device)
        self.gate = SimpleNamespace(weight=torch.empty(0, dtype=torch.float32, device=device),
                                    bias=self.bias, bias_vl=self.bias_vl, topk=6, gate_temp=1.0,
                                    score_func="sqrtsoftplus", norm_topk_prob=True, route_scale=1.5)

    def inspect(self, logits, image_span=False, synthetic_zero_bias=False):
        torch = self.torch
        with torch.cuda.device(self.ordinal), torch.inference_mode():
            x = torch.from_numpy(logits.reshape(1, 384)).to(f"cuda:{self.ordinal}")
            mask = torch.tensor([image_span], dtype=torch.bool, device=x.device)
            self.gate.bias = torch.zeros_like(self.bias) if synthetic_zero_bias else self.bias
            self.gate.bias_vl = torch.zeros_like(self.bias_vl) if synthetic_zero_bias else self.bias_vl
            weights, indices = self.forward_function(self.gate, x, mask)
            scores = torch.nn.functional.softplus(x).sqrt()
            bias = self.gate.bias_vl if image_span else self.gate.bias
            selection = scores + bias
            raw = scores.gather(1, indices)
            denominator = raw.sum(dim=-1, keepdim=True) + 1e-20
            assert torch.equal(weights.view(torch.int32), ((raw / denominator) * 1.5).view(torch.int32))
            values = {"scores": scores, "selection": selection, "raw": raw,
                      "denominator": denominator, "weights": weights, "indices": indices.to(torch.int32)}
            return {name: value.cpu().numpy().reshape(-1).copy() for name, value in values.items()}


def compare_router(fixture: Path, reference_directory: Path, ordinal=0, reference_ordinal=0):
    if importlib.util.find_spec("torch") is None:
        return {"probe": "router_postprocessing_parity", "status": "blocked", "missing_dependencies": ["torch"]}
    import torch
    if not torch.cuda.is_available():
        return {"probe": "router_postprocessing_parity", "status": "blocked", "reason": "CUDA PyTorch is required"}
    arrays, manifest = load_router_fixture(fixture)
    with Device(Driver(), reference_ordinal) as device:
        if device.memory()["free"] < 1536 * MiB:
            return {"probe": "router_postprocessing_parity", "status": "blocked", "reason": "Insufficient oracle VRAM reserve"}
    oracle = OfficialRouter(arrays, Path(reference_directory), reference_ordinal)
    inputs = []
    for seed in (20261018, 0, 1, 42, 127, 512, 4096, 65535):
        base = np.random.default_rng(seed).standard_normal(384).astype(np.float32)
        for scale in (0.0625, 1.0, 16.0, 128.0):
            inputs.append(({"kind": "random", "seed": seed, "scale": scale}, base * np.float32(scale), False))
    for name, x in (
        ("zeros", np.zeros(384, np.float32)),
        ("large_positive", np.full(384, 128, np.float32)),
        ("large_negative", np.full(384, -128, np.float32)),
        ("ramp", np.linspace(-80, 80, 384, dtype=np.float32)),
        ("softplus_threshold", np.resize(np.array([np.nextafter(np.float32(20), np.float32(0)), 20,
                                                  np.nextafter(np.float32(20), np.float32(np.inf))], np.float32), 384)),
        ("tiny_values", np.resize(np.array([0, -0.0, 1e-40, -1e-40], np.float32), 384)),
    ):
        inputs.append(({"kind": name}, x, False))
    inputs += [({"kind": "synthetic_exact_tie"}, np.zeros(384, np.float32), True),
               ({"kind": "synthetic_near_tie"}, np.linspace(-0.0002, 0.0002, 384, dtype=np.float32), True)]
    # Equal corrected scores can still have unequal unbiased scores. Construct
    # logits that approximately cancel the real trained bias, then test the
    # resulting FP32 ties and normalization, without changing either bias.
    for bias_name in ("bias", "bias_vl"):
        bias = arrays[bias_name].astype(np.float64)
        for margin in (1.0, 2.0, 4.0, 0.13, 0.3, 0.71, 1.37, 2.77, 3.14159265):
            level = float(np.max(np.abs(bias))) + margin
            square = (level - bias) ** 2
            logits = (square + np.log(-np.expm1(-square))).astype(np.float32)
            inputs.append(({"kind": "bias_cancelled", "construction_bias": bias_name, "margin": margin}, logits, False))
    cases = []
    with RouterProbe(arrays, ordinal) as candidate:
        for metadata, x, zero_bias in inputs:
            for image_span in (False, True):
                expected = oracle.inspect(x, image_span, zero_bias)
                actual = candidate.inspect(x, image_span, zero_bias)
                checks = [compare(expected[name], actual[name], name) for name in expected]
                tied = bool(actual["tied"][0])
                admission = "diagnostic_synthetic_bias"
                if not zero_bias:
                    try:
                        admitted_weights, admitted_indices = candidate.decision(x, image_span)
                    except UnsupportedRoutingTie:
                        admission = "refused_tie"
                        assert tied
                    else:
                        admission = "allowed_non_tied"
                        assert not tied
                        assert np.array_equal(admitted_weights.view(np.uint32), actual["weights"].view(np.uint32))
                        assert np.array_equal(admitted_indices, actual["indices"])
                mismatch_details = {}
                for name in ("scores", "selection", "weights", "denominator"):
                    ref, got = expected[name].view(np.uint32), actual[name].view(np.uint32)
                    positions = np.flatnonzero(ref != got)
                    mismatch_details[name] = [{"index": int(i), "reference_bits": int(ref[i]), "candidate_bits": int(got[i])}
                                              for i in positions]
                same_set = set(expected["indices"].tolist()) == set(actual["indices"].tolist())
                map_mismatches = None
                if same_set:
                    ref_map = dict(zip(expected["indices"].tolist(), expected["weights"].view(np.uint32).tolist()))
                    got_map = dict(zip(actual["indices"].tolist(), actual["weights"].view(np.uint32).tolist()))
                    map_mismatches = sum(ref_map[key] != got_map[key] for key in ref_map)
                cases.append({**metadata, "image_span": image_span, "synthetic_zero_bias": zero_bias,
                              "input_sha256": hashlib.sha256(memoryview(x)).hexdigest(), "checks": checks,
                              "candidate_tie_flag": tied, "candidate_indices_diagnostic_only": actual["indices"].tolist(),
                              "decision_admission": admission,
                              "reference_indices": expected["indices"].tolist(), "all_mismatch_details": mismatch_details,
                              "reference_weight_bits": expected["weights"].view(np.uint32).tolist(),
                              "candidate_weight_bits_diagnostic_only": actual["weights"].view(np.uint32).tolist(),
                              "expert_set_matches": same_set, "expert_weight_map_mismatches": map_mismatches,
                              "status": "unsupported_tie" if tied else "pass" if all(c["status"] == "pass" for c in checks) else "fail"})
        info, peak, cubin = candidate.info, candidate.device.peak, candidate.cubin_sha256
    passed = sum(c["status"] == "pass" for c in cases)
    ties = sum(c["status"] == "unsupported_tie" for c in cases)
    failed = sum(c["status"] == "fail" for c in cases)
    return {"probe": "router_postprocessing_parity", "status": "pass" if passed == len(cases) else "partial",
            "native_candidate_qualified": passed == len(cases), "fixture": manifest, "reference": oracle.source,
            "device": info, "reference_device_ordinal": reference_ordinal, "config": ROUTER_CONFIG,
            "cases": cases, "cases_passed": passed, "cases_failed": failed, "unsupported_tie_cases": ties,
            "reference_forward_calls": len(cases), "native_inspections_completed": candidate.inspections,
            "peak_explicit_device_bytes": peak, "cleanup_explicit_device_bytes": sum(candidate.device_object.allocations.values()),
            "compiled_cubin_sha256": cubin, "torch_version": torch.__version__, "torch_git_version": torch.version.git_version,
            "torch_cuda_version": torch.version.cuda, "numpy_version": np.__version__,
            "compiler_options": ["fmad=false", "ftz=false"],
            "memory_scope": "Explicit native probe buffers only; excludes CUDA contexts, compiler, PyTorch oracle and returned diagnostic arrays",
            "reference_method": "Unchanged Gate.forward AST; linear projection is an identity stub for supplied FP32 logits",
            "tie_policy": "Diagnostic lowest-index candidate order only; tied routing is unsupported and is not admitted to inference",
            "gate_projection": "not_run", "full_moe_sublayer": "not_run", "full_model_quality": "not_run",
            "tracked_issue": "https://github.com/shell-term/vesper/issues/3"}
