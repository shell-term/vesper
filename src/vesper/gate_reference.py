"""Compare original gate projection and connected routing with unchanged methods."""

import ast
import hashlib
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import urllib.request

import numpy as np

from .blas import Cublas, GATE_BLAS_WORKSPACE
from .checkpoint import fetch_reference
from .conformance import compare
from .cuda import Device, Driver
from .formats import bf16_bits, bf16_values
from .gate import GateProjection, load_gate_fixture
from .resources import MemoryBudget, MiB
from .router import ROUTER_CONFIG, ROUTER_TORCH_REVISION, RouterProbe, load_router_fixture
from .router_reference import fetch_topk_reference

GATE_SOURCE_HASHES = {
    "aten/src/ATen/cuda/CUDABlas.cpp": "1d81d9e5e68ba45636e8635cb8e7905f24044e1f4eb1449ed26be505be6bbcee",
    "aten/src/ATen/native/cuda/Blas.cpp": "237271bb965e49ac3f12f950609181f8c2d95a9afd646b4da362505e3044203a",
    "aten/src/ATen/cuda/CublasHandlePool.cpp": "2dfaea7f42cb654bf4db6bddedbfd5f2661d92c6d96f2c90652f3d9c162baf2c",
}


def fetch_gate_reference(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for source, expected in GATE_SOURCE_HASHES.items():
        path = directory / Path(source).name
        url = f"https://raw.githubusercontent.com/pytorch/pytorch/{ROUTER_TORCH_REVISION}/{source}"
        if path.exists():
            with path.open("rb") as stream:
                raw = stream.read(262145)
        else:
            with urllib.request.urlopen(url, timeout=30) as stream:
                raw = stream.read(262145)
        if len(raw) > 262144 or hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("Gate reference source exceeds its bound or differs from the pinned hash")
        if not path.exists():
            path.write_bytes(raw)
        files[source] = {"url": url, "bytes": len(raw), "sha256": expected}
    return {"revision": ROUTER_TORCH_REVISION, "files": files}


def library_record(path):
    path = Path(path)
    result = {"file": path.name}
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while raw := stream.read(MiB):
                digest.update(raw)
        result.update(bytes=path.stat().st_size, sha256=digest.hexdigest())
    return result


class OfficialGate:
    def __init__(self, weight, biases, directory, ordinal):
        import torch
        from torch.nn import functional as F
        self.torch, self.ordinal = torch, ordinal
        self.source = fetch_reference(directory, include_model=True)
        self.topk_source = fetch_topk_reference(directory / "pytorch-topk")
        self.gate_source = fetch_gate_reference(directory / "pytorch-gate")
        tree = ast.parse((directory / "model.py").read_text(encoding="utf-8"))
        linear = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "linear")
        gate = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Gate")
        forward = next(node for node in gate.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
        scope = {"torch": torch, "F": F}
        exec(compile(ast.Module(body=[linear, forward], type_ignores=[]), "pinned_gate.py", "exec"), scope)
        self.linear, self.forward = scope["linear"], scope["forward"]
        device = f"cuda:{ordinal}"
        self.weight = torch.from_numpy(weight.copy()).view(torch.bfloat16).to(device)
        bias, bias_vl = (torch.from_numpy(biases[name].copy()).to(device) for name in ("bias", "bias_vl"))
        self.gate = SimpleNamespace(weight=self.weight, bias=bias, bias_vl=bias_vl, topk=6,
                                    gate_temp=1.0, score_func="sqrtsoftplus", norm_topk_prob=True, route_scale=1.5)
        self.promotion = compare(bf16_values(weight), self.weight.float().cpu().numpy(), "original_BF16_weight_FP32_promotion")
        # The installed Windows oracle build supplies this library. Other
        # platforms still report torch/CUDA/source versions without guessing it.
        path = Path(torch.__file__).parent / "lib/cublas64_13.dll"
        self.library = None
        if path.is_file():
            profile = Cublas(path)
            try:
                self.library = {**library_record(path), "version": profile.version}
            finally:
                profile.close()

    def inspect(self, x, input_kind, image_span):
        torch = self.torch
        with torch.cuda.device(self.ordinal), torch.inference_mode():
            if input_kind == "bf16":
                tx = torch.from_numpy(bf16_bits(x).reshape(1, 5120)).view(torch.bfloat16)
            else:
                tx = torch.from_numpy(x.reshape(1, 5120))
            tx = tx.to(f"cuda:{self.ordinal}")
            mask = torch.tensor([image_span], dtype=torch.bool, device=tx.device)
            weights, indices = self.forward(self.gate, tx, mask)
            promoted = tx.float()
            logits = self.linear(promoted, self.weight.float())
            scores = torch.nn.functional.softplus(logits).sqrt()
            selection = scores + (self.gate.bias_vl if image_span else self.gate.bias)
            raw = scores.gather(1, indices)
            denominator = raw.sum(dim=-1, keepdim=True) + 1e-20
            assert torch.equal(weights.view(torch.int32), ((raw / denominator) * 1.5).view(torch.int32))
            values = {"input_promoted": promoted, "logits": logits, "scores": scores, "selection": selection,
                      "raw": raw, "denominator": denominator, "weights": weights, "indices": indices.to(torch.int32)}
            return {name: value.cpu().numpy().reshape(-1).copy() for name, value in values.items()}


def gate_inputs(weight):
    for seed in range(32):
        base = np.random.default_rng(30000 + seed).standard_normal(5120).astype(np.float32)
        for scale in (0.00390625, 1.0, 16.0, 256.0):
            x = base * np.float32(scale)
            for kind in ("bf16", "fp32"):
                yield {"kind": "random", "seed": 30000 + seed, "scale": scale, "input_kind": kind}, bf16_values(bf16_bits(x)) if kind == "bf16" else x
    for name, x in (
        ("zero", np.zeros(5120, np.float32)), ("ones", np.ones(5120, np.float32)),
        ("negative_ones", np.full(5120, -1, np.float32)),
        ("ramp", np.linspace(-8, 8, 5120, dtype=np.float32)),
        ("alternating", np.resize(np.array([-1, 1], np.float32), 5120)),
        ("signed_zero", np.resize(np.array([-0.0, 0.0], np.float32), 5120)),
        ("tiny", np.full(5120, 1e-38, np.float32)),
        ("subnormal", np.full(5120, 1e-40, np.float32)),
        ("mixed_magnitudes", np.resize(np.array([65536, -65536, 2**-16, -(2**-16)], np.float32), 5120)),
    ):
        for kind in ("bf16", "fp32"):
            yield {"kind": name, "input_kind": kind}, bf16_values(bf16_bits(x)) if kind == "bf16" else x
    for column in (0, 1, 31, 32, 255, 256, 511, 512, 1023, 1024, 4095, 5119):
        x = np.zeros(5120, np.float32)
        x[column] = 1
        yield {"kind": "one_hot", "column": column, "input_kind": "bf16"}, x
    promoted = bf16_values(weight)
    for row in (0, 1, 63, 127, 255, 383):
        for sign in (-1, 1):
            yield {"kind": "weight_aligned", "row": row, "sign": sign, "input_kind": "bf16"}, promoted[row] * np.float32(sign)
    for a, b in ((0, 1), (5, 128), (63, 64), (382, 383)):
        x = promoted[a] - promoted[b]
        for kind in ("bf16", "fp32"):
            yield {"kind": "row_difference", "rows": [a, b], "input_kind": kind}, bf16_values(bf16_bits(x)) if kind == "bf16" else x
    yield {"kind": "final_zero_reuse", "input_kind": "bf16"}, np.zeros(5120, np.float32)


def compare_gate(fixture, router_fixture, directory, ordinal=0, reference_ordinal=0, library=None):
    if importlib.util.find_spec("torch") is None:
        return {"probe": "gate_projection_parity", "status": "blocked", "missing_dependencies": ["torch"]}
    import torch
    if not torch.cuda.is_available():
        return {"probe": "gate_projection_parity", "status": "blocked", "reason": "CUDA PyTorch is required"}
    if torch.version.git_version != ROUTER_TORCH_REVISION:
        return {"probe": "gate_projection_parity", "status": "blocked", "reason": "The pinned PyTorch build is required",
                "expected_torch_revision": ROUTER_TORCH_REVISION, "actual_torch_revision": torch.version.git_version}
    profile = {"torch_revision": torch.version.git_version, "float32_precision": torch.get_float32_matmul_precision(),
               "blas_backend": str(torch.backends.cuda.preferred_blas_library()),
               "tunable_enabled": torch.cuda.tunable.is_enabled(),
               "environment": {name: os.environ.get(name) for name in ("CUBLAS_WORKSPACE_CONFIG", "NVIDIA_TF32_OVERRIDE",
                                                                      "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "TORCH_CUBLASLT_UNIFIED_WORKSPACE")}}
    if (profile["torch_revision"] != ROUTER_TORCH_REVISION or profile["float32_precision"] != "highest"
            or profile["blas_backend"] != "_BlasBackend.Cublas" or profile["tunable_enabled"]
            or profile["environment"]["CUBLAS_WORKSPACE_CONFIG"] is not None
            or any(profile["environment"][name] not in (None, "0") for name in (
                "NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "TORCH_CUBLASLT_UNIFIED_WORKSPACE"))):
        return {"probe": "gate_projection_parity", "status": "blocked", "reference_profile": profile,
                "reason": "Pinned PyTorch, highest FP32, default cuBLAS/workspace and disabled tuning/TF32 overrides are required"}
    host = MemoryBudget(cap=64 * MiB)
    host.acquire(32 * MiB)
    try:
        weight, provenance = load_gate_fixture(fixture)
        biases, router_provenance = load_router_fixture(router_fixture)
        if provenance["layer"] != router_provenance["layer"]:
            raise ValueError("Gate matrix and router biases must come from the same original layer")
        with Device(Driver(), reference_ordinal) as device:
            if device.memory()["free"] < 1536 * MiB:
                return {"probe": "gate_projection_parity", "status": "blocked", "reason": "Insufficient reference VRAM reserve"}
        oracle = OfficialGate(weight, biases, Path(directory), reference_ordinal)
        cases = []
        with GateProjection(weight, ordinal, library, host_budget=host) as native, RouterProbe(biases, ordinal) as router:
            for metadata, x in gate_inputs(weight):
                logits = native.project(x)
                for image_span in (False, True):
                    expected = oracle.inspect(x, metadata["input_kind"], image_span)
                    routing = router.inspect(logits, image_span)
                    actual = {"input_promoted": x, "logits": logits, **routing}
                    admitted_weights, admitted_indices = router.decision(logits, image_span)
                    checks = [compare(expected[name], actual[name], name) for name in expected]
                    checks += [compare(expected["weights"], admitted_weights, "admitted_weights"),
                               compare(expected["indices"], admitted_indices, "admitted_indices")]
                    details = {}
                    for name in expected:
                        ref, got = expected[name], actual[name]
                        if ref.dtype == np.float32:
                            ref, got = ref.view(np.uint32), got.view(np.uint32)
                        positions = np.flatnonzero(ref != got)
                        details[name] = [{"index": int(i), "reference": int(ref[i]), "candidate": int(got[i])} for i in positions]
                    cases.append({**metadata, "image_span": image_span, "input_sha256": hashlib.sha256(memoryview(x)).hexdigest(),
                                  "reference_logits_sha256": hashlib.sha256(memoryview(expected["logits"])).hexdigest(),
                                  "candidate_logits_sha256": hashlib.sha256(memoryview(logits)).hexdigest(),
                                  "checks": checks, "all_mismatch_details": details,
                                  "reference_indices": expected["indices"].tolist(), "candidate_indices": routing["indices"].tolist(),
                                  "reference_weight_bits": expected["weights"].view(np.uint32).tolist(),
                                  "candidate_weight_bits": routing["weights"].view(np.uint32).tolist(),
                                  "selected_tie": bool(routing["tied"][0]),
                                  "status": "pass" if all(c["status"] == "pass" for c in checks) else "fail"})
            calls, peak = native.calls, native.device.peak
            library_profile = {**library_record(native.blas.library), "version": native.blas.version, "math_mode": native.blas.math_mode,
                               "workspace_bytes": GATE_BLAS_WORKSPACE, "api": "cublasSgemm_v2", "m": 384, "n": 1, "k": 5120,
                               "transa": "T", "transb": "N", "alpha": 1, "beta": 0, "pointer_mode": "host", "stream": "default"}
            router_peak, router_calls, cubin = router.device.peak, router.inspections, router.cubin_sha256
        passed = sum(case["status"] == "pass" for case in cases)
        qualified = passed == len(cases) and oracle.promotion["status"] == "pass"
        report = {"probe": "gate_projection_parity", "status": "pass" if qualified else "partial", "native_candidate_qualified": qualified,
                "fixture": provenance, "router_fixture": router_provenance, "config": ROUTER_CONFIG,
                "reference": oracle.source, "pytorch_gate_source": oracle.gate_source, "pytorch_topk_source": oracle.topk_source,
                "weight_promotion": oracle.promotion, "cases": cases, "cases_passed": passed, "cases_failed": len(cases) - passed,
                "device": native.info, "reference_device_ordinal": reference_ordinal, "candidate_blas": library_profile,
                "reference_blas": oracle.library, "torch_version": torch.__version__, "torch_git_version": torch.version.git_version,
                "reference_profile": profile,
                "torch_cuda_version": torch.version.cuda, "torch_float32_matmul_precision": torch.get_float32_matmul_precision(),
                "numpy_version": np.__version__, "native_projection_calls": calls, "native_router_inspections": router_calls,
                "reference_gate_forwards": len(cases), "reference_linear_calls_including_diagnostics": 2 * len(cases),
                "native_projection_peak_explicit_device_bytes": peak, "native_router_peak_explicit_device_bytes": router_peak,
                "cleanup_projection_explicit_device_bytes": sum(native.device_object.allocations.values()),
                "cleanup_router_explicit_device_bytes": sum(router.device_object.allocations.values()),
                "host_reservation_peak_bytes": host.peak, "router_cubin_sha256": cubin,
                "memory_scope": "Explicit native allocations/reservations only; excludes cuBLAS internal state, CUDA contexts/modules, compiler, PyTorch oracle, Python/NumPy metadata and accumulated diagnostic records",
                "reference_method": "Unchanged linear and Gate.forward AST, original BF16 weights and real projection; no identity stub",
                "full_moe_sublayer": "not_run", "full_model_quality": "not_run", "tracked_issue": "https://github.com/shell-term/vesper/issues/4"}
    finally:
        host.release(32 * MiB)
    report["cleanup_host_reservation_bytes"] = host.reserved
    return report
