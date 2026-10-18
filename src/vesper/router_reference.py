"""Unchanged Gate.forward, with its projection explicitly replaced by supplied logits."""

import ast
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import urllib.request

import numpy as np

from .checkpoint import fetch_reference
from .conformance import compare
from .cuda import Device, Driver
from .router import (ROUTER_CONFIG, ROUTER_TOPK_POLICY, ROUTER_TORCH_REVISION, RouterProbe,
                     load_router_fixture, top6_indices)
from .resources import MiB

TOPK_SOURCE_HASHES = {
    "TensorTopK.cpp": "59660e9db8eb469518adc5fdd58de4035c949091568ccfc2898402502751b8c0",
    "TensorTopK.cu": "4a52bc2171023e8916b93c6aca080e7879e928c3f24a14c7810c3175e4381cd4",
    "Sort.cu": "c86fc6cb4eebd24c78cdeb4bdc352b852c273d9577a0d83c1ea9396863c03984",
    "SortUtils.cuh": "ee7be444befcb0d2fdcdfcd2408125b21db12e5ca54d163535aa8e1f299e37a3",
    "SortingCommon.cuh": "70e01e301e7c9066c5b247f5dfccacbe1b78078b8c0995c122a8fd10deac20d1",
    "SortingRadixSelect.cuh": "2c2cfe4c299d776dc3e9b35a75958ced6aa7b9c2bc99ba7241c7231530e376d9",
    "LICENSE": "bd018feef8825e88181c84eb7e3aa4eafb8f08a20d9fd6ef948569610c4a3e43",
}


def fetch_topk_reference(directory):
    """Bounded, hash-checked source evidence; no C++ reference code is executed."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, expected in TOPK_SOURCE_HASHES.items():
        source = name if name == "LICENSE" else "aten/src/ATen/native/cuda/" + name
        url = f"https://raw.githubusercontent.com/pytorch/pytorch/{ROUTER_TORCH_REVISION}/{source}"
        path = directory / name
        if path.exists():
            with path.open("rb") as stream:
                raw = stream.read(262145)
        else:
            with urllib.request.urlopen(url, timeout=30) as response:
                raw = response.read(262145)
        if len(raw) > 262144 or hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("PyTorch top-k source exceeds its bound or differs from the pinned hash")
        if not path.exists():
            path.write_bytes(raw)
        records[name] = {"url": url, "bytes": len(raw), "sha256": expected}
    return {"revision": ROUTER_TORCH_REVISION, "files": records}


class OfficialRouter:
    def __init__(self, arrays, reference_directory, ordinal):
        import torch
        from torch.nn import functional as F
        self.torch, self.ordinal = torch, ordinal
        self.source = fetch_reference(reference_directory, include_model=True)
        self.topk_source = fetch_topk_reference(reference_directory / "pytorch-topk")
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

    def inspect_selection(self, scores, selection):
        """Direct unchanged PyTorch operators; synthetic selection-only inputs."""
        torch = self.torch
        with torch.cuda.device(self.ordinal), torch.inference_mode():
            scores, selection = (torch.from_numpy(value.reshape(1, 384)).to(f"cuda:{self.ordinal}")
                                 for value in (scores, selection))
            indices = selection.topk(6, dim=-1)[1]
            raw = scores.gather(1, indices)
            denominator = raw.sum(dim=-1, keepdim=True) + 1e-20
            values = {"scores": scores, "selection": selection, "raw": raw, "denominator": denominator,
                      "weights": (raw / denominator) * 1.5, "indices": indices.to(torch.int32)}
            return {name: value.cpu().numpy().reshape(-1).copy() for name, value in values.items()}


def selection_inputs():
    """Independent synthetic scores stress boundary ties and normalization order."""
    for seed in range(32):
        rng = np.random.default_rng(seed + 10000)
        for bins in (1, 2, 4, 8, 32, 384):
            selection = rng.integers(-bins, bins + 1, 384).astype(np.float32) / np.float32(16)
            scores = np.ldexp(rng.uniform(0.5, 1, 384).astype(np.float32), rng.integers(-60, 61, 384))
            yield {"kind": "discrete_selection", "seed": seed + 10000, "bins": bins}, scores, selection
    rng = np.random.default_rng(38906)
    for higher in range(6):
        for equal in sorted({7 - higher, 16, 64, 384 - higher}):
            ids = rng.permutation(384)
            selection = np.full(384, -10, np.float32)
            selection[ids[:higher]] = np.arange(30, 30 - higher, -1, dtype=np.float32)
            selection[ids[higher:higher + equal]] = 10
            scores = rng.uniform(0.001, 20, 384).astype(np.float32)
            yield {"kind": "boundary_tie", "strictly_higher": higher, "equal_to_boundary": equal}, scores, selection
    for name, selection in (
        ("all_equal", np.zeros(384, np.float32)),
        ("signed_zero", np.resize(np.array([-0.0, 0.0], np.float32), 384)),
        ("subnormal_selection", np.arange(-192, 192, dtype=np.float32) * np.nextafter(np.float32(0), np.float32(1))),
        ("finite_extreme_selection", np.linspace(-3e38, 3e38, 384, dtype=np.float64).astype(np.float32)),
    ):
        for repeat in range(4):
            scores = np.linspace(0.0001, 30, 384, dtype=np.float32)
            yield {"kind": name, "repeat": repeat}, scores, selection.copy()


def compare_router(fixture: Path, reference_directory: Path, ordinal=0, reference_ordinal=0):
    if importlib.util.find_spec("torch") is None:
        return {"probe": "router_postprocessing_parity", "status": "blocked", "missing_dependencies": ["torch"]}
    import torch
    if not torch.cuda.is_available():
        return {"probe": "router_postprocessing_parity", "status": "blocked", "reason": "CUDA PyTorch is required"}
    if torch.version.git_version != ROUTER_TORCH_REVISION:
        return {"probe": "router_postprocessing_parity", "status": "blocked",
                "reason": "The CUDA top-k policy requires the pinned reference build",
                "expected_torch_revision": ROUTER_TORCH_REVISION, "actual_torch_revision": torch.version.git_version}
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
    for seed in range(16):
        base = np.random.default_rng(seed + 20000).standard_normal(384).astype(np.float32)
        for scale in (0.001, 4.0, 64.0, 1024.0):
            inputs.append(({"kind": "extended_random", "seed": seed + 20000, "scale": scale}, base * np.float32(scale), False))
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
    cases, selection_cases = [], []
    with RouterProbe(arrays, ordinal) as candidate:
        for metadata, x, zero_bias in inputs:
            for image_span in (False, True):
                expected = oracle.inspect(x, image_span, zero_bias)
                actual = candidate.inspect(x, image_span, zero_bias)
                checks = [compare(expected[name], actual[name], name) for name in expected]
                tied = bool(actual["tied"][0])
                admission = "diagnostic_synthetic_bias"
                if not zero_bias:
                    admitted_weights, admitted_indices = candidate.decision(x, image_span)
                    admission = "allowed_tied" if tied else "allowed_non_tied"
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
                              "candidate_tie_flag": tied, "candidate_indices": actual["indices"].tolist(),
                              "decision_admission": admission,
                              "reference_indices": expected["indices"].tolist(), "all_mismatch_details": mismatch_details,
                              "reference_weight_bits": expected["weights"].view(np.uint32).tolist(),
                              "candidate_weight_bits": actual["weights"].view(np.uint32).tolist(),
                              "expert_set_matches": same_set, "expert_weight_map_mismatches": map_mismatches,
                              "status": "pass" if all(c["status"] == "pass" for c in checks) else "fail"})
        for metadata, scores, selection in selection_inputs():
            expected = oracle.inspect_selection(scores, selection)
            actual = candidate.inspect_selection(scores, selection)
            cpu_indices = top6_indices(selection)
            checks = [compare(expected[name], actual[name], name) for name in expected]
            checks.append(compare(expected["indices"], cpu_indices, "CPU_top6_specification"))
            details = {}
            for name in expected:
                ref, got = expected[name], actual[name]
                if ref.dtype == np.float32:
                    ref, got = ref.view(np.uint32), got.view(np.uint32)
                positions = np.flatnonzero(ref != got)
                details[name] = [{"index": int(i), "reference": int(ref[i]), "candidate": int(got[i])} for i in positions]
            selection_cases.append({**metadata, "input_scores_sha256": hashlib.sha256(memoryview(scores)).hexdigest(),
                                    "input_selection_sha256": hashlib.sha256(memoryview(selection)).hexdigest(),
                                    "checks": checks, "all_mismatch_details": details,
                                    "reference_indices": expected["indices"].tolist(), "candidate_indices": actual["indices"].tolist(),
                                    "reference_weight_bits": expected["weights"].view(np.uint32).tolist(),
                                    "candidate_weight_bits": actual["weights"].view(np.uint32).tolist(),
                                    "candidate_tie_flag": bool(actual["tied"][0]),
                                    "status": "pass" if all(c["status"] == "pass" for c in checks) else "fail"})
        info, peak, cubin = candidate.info, candidate.device.peak, candidate.cubin_sha256
    passed = sum(c["status"] == "pass" for c in cases)
    ties = sum(c["candidate_tie_flag"] for c in cases)
    failed = sum(c["status"] == "fail" for c in cases)
    selection_passed = sum(c["status"] == "pass" for c in selection_cases)
    qualified = passed == len(cases) and selection_passed == len(selection_cases)
    return {"probe": "router_postprocessing_parity", "status": "pass" if qualified else "partial",
            "native_candidate_qualified": qualified, "fixture": manifest, "reference": oracle.source,
            "pytorch_topk_source": oracle.topk_source,
            "device": info, "reference_device_ordinal": reference_ordinal, "config": ROUTER_CONFIG,
            "cases": cases, "cases_passed": passed, "cases_failed": failed, "tied_cases": ties, "unsupported_tie_cases": 0,
            "selection_cases": selection_cases, "selection_cases_passed": selection_passed,
            "reference_selection_calls": len(selection_cases),
            "reference_forward_calls": len(cases), "native_inspections_completed": candidate.inspections,
            "peak_explicit_device_bytes": peak, "cleanup_explicit_device_bytes": sum(candidate.device_object.allocations.values()),
            "compiled_cubin_sha256": cubin, "torch_version": torch.__version__, "torch_git_version": torch.version.git_version,
            "torch_cuda_version": torch.version.cuda, "numpy_version": np.__version__,
            "compiler_options": ["fmad=false", "ftz=false"],
            "memory_scope": "Explicit native device allocations only; excludes kernel local/shared storage, CUDA contexts/modules, compiler, PyTorch oracle and returned diagnostic arrays",
            "reference_method": "Unchanged Gate.forward AST; linear projection is an identity stub for supplied FP32 logits",
            "tie_policy": ROUTER_TOPK_POLICY,
            "gate_projection": "not_run", "full_moe_sublayer": "not_run", "full_model_quality": "not_run",
            "tracked_issue": "https://github.com/shell-term/vesper/issues/3"}
