"""Aggregation checks through the unchanged single-rank MoE.forward body."""

from __future__ import annotations

import ast
from contextlib import ExitStack
import importlib.util
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from .checkpoint import fetch_reference, load_expert_fixture
from .conformance import compare
from .expert import ResidentExpert
from .expert_reference import OfficialExpert
from .formats import bf16_bits, bf16_values
from .moe import EXPERTS, HIDDEN, MoEAccumulator, participation
from .resources import MemoryBudget, MiB, physical_memory
from .shared import ResidentSharedExpert, load_shared_fixture
from .shared_reference import digest, expert_stages


def aggregation_inputs():
    """Synthetic BF16 contributions test control flow, order and rounding."""
    pool = np.array([0, 1, 3, 31, 32, 95, 96, 127, 128, 255, 256, 383], np.int64)
    for seed in (20261018, 0, 1, 42, 127, 512, 4096, 65535):
        rng = np.random.default_rng(seed)
        for tokens in (1, 2, 8, 32):
            indices = np.stack([rng.choice(pool, 6, replace=False) for _ in range(tokens)])
            selected = participation(indices)
            contributions = {}
            for expert, rows in selected.items():
                values = np.ldexp(rng.standard_normal((len(rows), HIDDEN)).astype(np.float32),
                                  rng.integers(-24, 25, (len(rows), HIDDEN)))
                contributions[expert] = bf16_bits(values)
            shared = bf16_bits(rng.standard_normal((tokens, HIDDEN)).astype(np.float32))
            yield {"input": f"random_seed{seed}_tokens{tokens}", "indices": indices,
                   "contributions": contributions, "shared": shared,
                   "shape": (2, 4, HIDDEN) if tokens == 8 else (tokens, HIDDEN)}
    indices = np.array([[0, 1, 2, 3, 4, 383]], np.int64)
    values = {expert: np.zeros((1, HIDDEN), np.float32) for expert in indices[0]}
    values[0][:, 0::3], values[1][:, 0::3], values[2][:, 0::3] = 2.0**26, 1.0, -2.0**26
    values[0][:, 1::3], values[1][:, 1::3], values[2][:, 1::3] = 1.0, 2.0**-8, 2.0**-8
    values[0][:, 2::3], values[1][:, 2::3] = 2.0**26, -2.0**26
    contributions = {int(e): bf16_bits(a) for e, a in values.items()}
    shared = np.zeros((1, HIDDEN), np.float32)
    shared[:, 2::3] = 1.0
    for order in ((0, 1, 2, 3, 4, 383), (0, 2, 1, 383, 4, 3), (383, 4, 3, 2, 1, 0)):
        yield {"input": "cancellation_rank_" + "_".join(map(str, order)),
               "indices": np.array([order], np.int64), "contributions": contributions,
               "shared": bf16_bits(shared), "invariance_group": "same_contributions_permuted_ranks"}
    signed = np.resize(np.array([0, 0x8000, 1, 0x8001, 0x007f, 0x807f, 0x0080, 0x8080], np.uint16), (2, HIDDEN))
    indices = np.tile(indices, (2, 1))
    yield {"input": "signed_zero_and_BF16_subnormals", "indices": indices,
           "contributions": {int(e): signed.copy() for e in indices[0]}, "shared": signed.copy()}
    yield {"input": "zero_contributions", "indices": indices,
           "contributions": {int(e): np.zeros((2, HIDDEN), np.uint16) for e in indices[0]},
           "shared": np.zeros((2, HIDDEN), np.uint16)}
    yield {"input": "empty_token_batch", "indices": np.empty((0, 6), np.int64),
           "contributions": {}, "shared": np.empty((0, HIDDEN), np.uint16)}


class OfficialAggregation:
    """Record actual indexed writes/adds; never rewrite the selected AST body.

    Supplied gate outputs and BF16 expert callbacks isolate aggregation. The
    optional original-expert test below separately checks their arithmetic.
    """

    def __init__(self, directory: Path, ordinal=0):
        import torch
        self.torch, self.ordinal = torch, ordinal
        self.manifest = fetch_reference(directory, include_model=True)
        tree = ast.parse((directory / "model.py").read_text(encoding="utf-8"))
        moe = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MoE")
        forward = next(n for n in moe.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
        scope = {"torch": torch, "world_size": 1}
        exec(compile(ast.Module(body=[forward], type_ignores=[]), "pinned_deepseek_model.py", "exec"), scope)
        self.forward_function = scope["forward"]

    def forward(self, indices, contributions, shared, *, shape=None, routing_weights=None,
                input_values=None, expert_functions=None, shared_function=None, gate_function=None, image_span=None):
        torch = self.torch
        from torch.utils._python_dispatch import TorchDispatchMode
        selected = participation(indices)
        tokens = indices.shape[0]
        device = f"cuda:{self.ordinal}"

        class Trace(TorchDispatchMode):
            def __init__(self):
                super().__init__()
                self.accumulator, self.active = None, None
                self.snapshots = {}

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                result = func(*args, **(kwargs or {}))
                if (self.accumulator is None and func == torch.ops.aten.zeros_like.default
                        and result.dtype == torch.float32):
                    self.accumulator = result
                if (self.accumulator is not None and args and isinstance(args[0], torch.Tensor)
                        and args[0].data_ptr() == self.accumulator.data_ptr()):
                    if func == torch.ops.aten.index_put_.default:
                        self.snapshots[f"after_expert_{self.active}_fp32"] = self.accumulator.clone()
                    elif func == torch.ops.aten.add_.Tensor:
                        self.snapshots["after_shared_fp32"] = self.accumulator.clone()
                return result

        with torch.cuda.device(self.ordinal), torch.inference_mode():
            ids = torch.from_numpy(np.ascontiguousarray(indices)).to(device=device, dtype=torch.int64)
            weights = (np.arange(1, tokens * 6 + 1, dtype=np.float32).reshape(tokens, 6) / np.float32(256)
                       if routing_weights is None else np.asarray(routing_weights, dtype=np.float32))
            tw = torch.from_numpy(np.ascontiguousarray(weights)).to(device)
            x = (torch.arange(tokens, device=device, dtype=torch.bfloat16).view(tokens, 1).expand(tokens, HIDDEN).contiguous()
                 if input_values is None else torch.from_numpy(np.ascontiguousarray(input_values)).to(device, dtype=torch.bfloat16))
            def upload(bits):
                return torch.from_numpy(np.ascontiguousarray(bits)).to(device).view(torch.bfloat16)
            outputs = {e: upload(bits) for e, bits in contributions.items()}
            shared_tensor = upload(shared)
            tracer, calls = Trace(), []
            def expert_callback(expert):
                def call(tx, route):
                    tracer.active = expert
                    rows, top = np.where(indices == expert)
                    assert rows.tolist() == selected[expert].tolist()
                    assert torch.equal(tx.view(torch.uint16), x[rows].view(torch.uint16))
                    expected_route = tw[rows, top].view(-1, 1)
                    assert torch.equal(route, expected_route)
                    calls.append(expert)
                    if expert_functions is not None:
                        original_output = expert_functions[expert](tx, route)
                        assert torch.equal(original_output.view(torch.uint16), outputs[expert].view(torch.uint16))
                        return original_output
                    return outputs[expert]
                return call
            def shared_callback(tx):
                assert torch.equal(tx.view(torch.uint16), x.view(torch.uint16))
                tracer.snapshots["before_shared_fp32"] = tracer.accumulator.clone()
                tracer.active = "shared"
                if shared_function is not None:
                    original_output = shared_function(tx)
                    assert torch.equal(original_output.view(torch.uint16), shared_tensor.view(torch.uint16))
                    return original_output
                return shared_tensor
            def gate_callback(tx, mask):
                if gate_function is None:
                    return tw, ids
                original_weights, original_indices = gate_function(tx, mask)
                assert torch.equal(original_weights.view(torch.int32), tw.view(torch.int32))
                assert torch.equal(original_indices, ids)
                return original_weights, original_indices
            model = SimpleNamespace(dim=HIDDEN, n_routed_experts=EXPERTS,
                                    experts_start_idx=0, experts_end_idx=EXPERTS,
                                    gate=gate_callback,
                                    experts={e: expert_callback(e) for e in selected}, shared_experts=shared_callback)
            shape = shape or (tokens, HIDDEN)
            default = torch.get_default_dtype()
            try:
                torch.set_default_dtype(torch.bfloat16)
                with tracer:
                    mask = None if image_span is None else torch.tensor([image_span], dtype=torch.bool, device=device)
                    output = self.forward_function(model, x.view(shape), mask)
            finally:
                torch.set_default_dtype(default)
            assert tuple(output.shape) == tuple(shape)
            assert calls == list(selected)
            stages = {name: tensor.cpu().numpy().copy() for name, tensor in tracer.snapshots.items()}
            stages["output_bf16"] = output.view(torch.uint16).reshape(tokens, HIDDEN).cpu().numpy().copy()
            required = {f"after_expert_{e}_fp32" for e in selected} | {
                "before_shared_fp32", "after_shared_fp32", "output_bf16"}
            if set(stages) != required:
                raise RuntimeError("The dispatch trace did not capture every original aggregation boundary")
            return stages


def _negative_controls(case, expected):
    """Deliberately wrong arithmetic must be detected on a crafted input."""
    indices, contributions, shared = case["indices"], case["contributions"], case["shared"]
    if not case["input"].startswith("cancellation_rank_"):
        return []
    values = {e: bf16_values(bits) for e, bits in contributions.items()}
    wrong = {}
    total = np.zeros_like(bf16_values(shared))
    for expert in indices[0]:
        total = np.add(total, values[int(expert)], dtype=np.float32)
    wrong["route_rank_order"] = bf16_bits(np.add(total, bf16_values(shared), dtype=np.float32))
    total = np.zeros_like(total)
    for expert in sorted(values):
        total = bf16_values(bf16_bits(np.add(total, values[expert], dtype=np.float32)))
    wrong["BF16_after_each_expert"] = bf16_bits(np.add(total, bf16_values(shared), dtype=np.float32))
    total = bf16_values(shared).copy()
    for expert in sorted(values):
        total = np.add(total, values[expert], dtype=np.float32)
    wrong["shared_first"] = bf16_bits(total)
    return [{"mutation": name, "final_mismatches": int(np.count_nonzero(bits != expected["output_bf16"]))}
            for name, bits in wrong.items()]


def compare_aggregation(reference_directory: Path, ordinal=0, reference_ordinal=0):
    if importlib.util.find_spec("torch") is None:
        return {"probe": "moe_aggregation_parity", "status": "blocked", "missing_dependencies": ["torch"]}
    import torch
    budget = MemoryBudget(cap=192 * MiB)
    budget.acquire(192 * MiB)
    try:
        available_samples = [physical_memory()["available"]]
        reference = OfficialAggregation(reference_directory, reference_ordinal)
        records, invariance = [], {}
        with MoEAccumulator(ordinal) as candidate:
            for case in aggregation_inputs():
                available_samples.append(physical_memory()["available"])
                budget.acquire(0, available=available_samples[-1])
                expected = reference.forward(case["indices"], case["contributions"], case["shared"], shape=case.get("shape"))
                actual = candidate.run(case["indices"], case["contributions"], case["shared"])
                checks = [compare(expected[name], actual[name], name) for name in expected]
                group = case.get("invariance_group")
                if group:
                    if group not in invariance:
                        invariance[group] = expected["output_bf16"].copy()
                    checks.append(compare(invariance[group], actual["output_bf16"], "rank_permutation_invariance"))
                records.append({"input": case["input"], "tokens": case["indices"].shape[0],
                                "input_shape": list(case.get("shape", (case["indices"].shape[0], HIDDEN))),
                                "indices": case["indices"].tolist(), "expert_order": list(participation(case["indices"])),
                                "zero_participation_experts": EXPERTS - len(case["contributions"]),
                                "contribution_sha256": {str(e): digest(a) for e, a in case["contributions"].items()},
                                "shared_sha256": digest(case["shared"]),
                                "checks": checks, "expected_sha256": {n: digest(a) for n, a in expected.items()},
                                "negative_controls": _negative_controls(case, expected),
                                "status": "pass" if all(c["status"] == "pass" for c in checks) else "fail"})
            peak, info, cubin = candidate.device.peak, candidate.info, candidate.cubin_sha256
            device_object = candidate.device
        detected = {name: any(c["final_mismatches"] > 0 for r in records for c in r["negative_controls"]
                              if c["mutation"] == name)
                    for name in ("route_rank_order", "BF16_after_each_expert", "shared_first")}
        return {"probe": "moe_aggregation_parity", "schema_version": 1,
                "status": "pass" if records and all(r["status"] == "pass" for r in records) and all(detected.values()) else "fail",
                "device": info, "reference_device_ordinal": reference_ordinal, "reference": reference.manifest,
                "reference_method": "Unchanged MoE.forward AST body; CUDA dispatch records indexed FP32 writes/adds",
                "scope": "Single rank; supplied unique top-6 IDs and BF16 contributions; 0/1/2/8/32 tokens",
                "duplicate_scope": "A token contributes through multiple experts; repeated expert IDs within a token are rejected",
                "cases": records, "negative_controls_detected": detected,
                "compiled_cubin_sha256": cubin, "peak_explicit_device_allocation_bytes": peak,
                "cleanup_explicit_device_allocation_bytes": sum(device_object.allocations.values()),
                "accumulator_cleanup_host_reservation_bytes": candidate.budget.reserved,
                "minimum_observed_host_available_bytes": min(available_samples),
                "host_probe_reservation_bytes": budget.peak + 4 * MiB,
                "host_available_floor_bytes": budget.available_floor,
                "resource_exclusions": ["Python/NumPy runtime and metadata", "CUDA modules/contexts/compiler",
                                        "PyTorch oracle allocations", "caller-retained diagnostic snapshots"],
                "torch_version": torch.__version__, "torch_git_version": torch.version.git_version,
                "torch_cuda_version": torch.version.cuda, "numpy_version": np.__version__,
                "compiler_options": ["fmad=false", "ftz=false"],
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "original_expert_arithmetic": "separate_gate", "connected_original_router": "not_run",
                "full_model_quality": "not_run"}
    finally:
        budget.release(192 * MiB)


def original_expert_inputs():
    """Supplied routes isolate this gate from original gate projection/routing."""
    for seed in (20261018, 0, 42, 65535):
        base = np.random.default_rng(seed).standard_normal(HIDDEN).astype(np.float32)
        for scale_index, scale in enumerate((0.0625, 1.0, 16.0, 128.0)):
            weights = np.roll(np.array([0.0, 0.1, 0.25, 0.3, 0.85, 1.5], np.float32), (seed + scale_index) % 6)
            order = np.random.default_rng(seed + scale_index).permutation(6).astype(np.int64)
            yield {"input": f"original_seed{seed}_scale{scale:g}", "seed": seed, "input_scale": scale,
                   "x": bf16_values(bf16_bits(base * np.float32(scale))),
                   "indices": order.reshape(1, 6), "weights": weights[order].reshape(1, 6)}


def compare_original_accumulation(fixtures, shared_fixture: Path, reference_directory: Path,
                                  ordinal=0, reference_ordinal=0):
    """Six actual original experts + shared expert, with deliberately supplied routes.

    This is not connected original Gate.forward or bounded offload integration.
    All seven reference Expert.forward calls also run inside original MoE.forward.
    """
    missing = [name for name in ("torch", "tilelang") if importlib.util.find_spec(name) is None]
    if missing:
        return {"probe": "original_expert_accumulation", "status": "blocked", "missing_dependencies": missing}
    import torch
    import tilelang

    if len(fixtures) != 6:
        raise ValueError("This original-expert probe requires exactly six fixtures")
    budget = MemoryBudget(cap=512 * MiB)
    budget.acquire(512 * MiB)
    try:
        available_samples = [physical_memory()["available"]]
        loaded = [load_expert_fixture(path) for path in fixtures]
        identities = [(manifest["layer"], manifest["expert"]) for _, manifest in loaded]
        if sorted(identities) != [(0, e) for e in range(6)]:
            raise ValueError("This probe requires distinct original layer-0 experts 0 through 5")
        loaded.sort(key=lambda item: item[1]["expert"])
        shared_arrays, shared_manifest = load_shared_fixture(shared_fixture)
        if shared_manifest["layer"] != 0:
            raise ValueError("Shared expert must belong to the same layer")
        official = [OfficialExpert(a, m, reference_directory, reference_ordinal) for a, m in loaded]
        official_shared = OfficialExpert(shared_arrays, shared_manifest, reference_directory, reference_ordinal,
                                         weight_dtype=torch.float8_e4m3fn)
        aggregation = OfficialAggregation(reference_directory, reference_ordinal)
        def original_function(reference):
            def call(tx, route=None):
                reference.outputs, reference.quantized = {}, {}
                return reference.forward_function(reference.expert, tx, route)
            return call
        functions = {e: original_function(ref) for e, ref in enumerate(official)}
        shared_function = original_function(official_shared)
        records = []
        with ExitStack() as stack:
            candidates = [stack.enter_context(ResidentExpert(a, ordinal)) for a, _ in loaded]
            candidate_shared = stack.enter_context(ResidentSharedExpert(shared_arrays, ordinal))
            accumulator = stack.enter_context(MoEAccumulator(ordinal, max_tokens=1))
            for case in original_expert_inputs():
                available_samples.append(physical_memory()["available"])
                budget.acquire(0, available=available_samples[-1])
                x, indices, weights = case["x"], case["indices"], case["weights"]
                expert_records, expected_contributions, actual_contributions = [], {}, {}
                for expert in range(6):
                    route = np.float32(weights[0, np.where(indices[0] == expert)[0][0]])
                    expected = expert_stages(official[expert].forward(x, float(route)))
                    actual = expert_stages(candidates[expert].forward(x, float(route)), native=True)
                    checks = [compare(expected[name], actual[name], name) for name in expected]
                    expected_contributions[expert] = expected["w2"].reshape(1, HIDDEN)
                    actual_contributions[expert] = actual["w2"].reshape(1, HIDDEN)
                    # A deliberately wrong placement, using the original unweighted
                    # reference output, demonstrates that multiplying w2 is invalid.
                    unweighted = official[expert].forward(x)["w2"]
                    post_weighted = bf16_bits(bf16_values(unweighted) * route)
                    expert_records.append({"expert": expert, "routing_weight": float(route), "checks": checks,
                                           "expected_sha256": {n: digest(a) for n, a in expected.items()},
                                           "post_w2_weighting_mismatches": int(np.count_nonzero(post_weighted != expected["w2"])),
                                           "status": "pass" if all(c["status"] == "pass" for c in checks) else "fail"})
                expected_shared = expert_stages(official_shared.forward(x))
                actual_shared = expert_stages(candidate_shared.forward(x), native=True)
                shared_checks = [compare(expected_shared[n], actual_shared[n], n) for n in expected_shared]
                expected = aggregation.forward(indices, expected_contributions, expected_shared["w2"].reshape(1, HIDDEN),
                                               routing_weights=weights, input_values=x.reshape(1, HIDDEN),
                                               expert_functions=functions, shared_function=shared_function)
                actual = accumulator.run(indices, actual_contributions, actual_shared["w2"].reshape(1, HIDDEN))
                aggregation_checks = [compare(expected[n], actual[n], n) for n in expected]
                records.append({"input": case["input"], "seed": case["seed"], "input_scale": case["input_scale"],
                                "input_bf16_sha256": digest(bf16_bits(x)), "indices": indices.tolist(),
                                "routing_weights": weights.tolist(), "expert_order": list(range(6)),
                                "experts": expert_records, "shared_checks": shared_checks,
                                "shared_expected_sha256": {n: digest(a) for n, a in expected_shared.items()},
                                "aggregation_checks": aggregation_checks,
                                "aggregation_expected_sha256": {n: digest(a) for n, a in expected.items()},
                                "status": "pass" if all(e["status"] == "pass" for e in expert_records)
                                and all(c["status"] == "pass" for c in shared_checks + aggregation_checks) else "fail"})
            peaks = {"routed_experts": [c.device.peak for c in candidates],
                     "shared_expert": candidate_shared.device.peak, "accumulator": accumulator.device.peak}
            explicit_total = sum(peaks["routed_experts"]) + peaks["shared_expert"] + peaks["accumulator"]
            info = accumulator.info
            cubins = {"routed": candidates[0].cubin_sha256, "shared": candidate_shared.fp8_cubin_sha256,
                      "aggregation": accumulator.cubin_sha256}
            device_objects = [c.device for c in candidates] + [candidate_shared.device, accumulator.device]
        placement_detected = any(e["post_w2_weighting_mismatches"] > 0 for r in records for e in r["experts"])
        return {"probe": "original_expert_accumulation", "schema_version": 1,
                "status": "pass" if records and all(r["status"] == "pass" for r in records) and placement_detected else "fail",
                "device": info, "reference_device_ordinal": reference_ordinal, "reference": aggregation.manifest,
                "reference_method": "Unchanged MoE.forward calls unchanged Expert.forward/linear with original weights; supplied gate IDs/weights",
                "fixtures": [m for _, m in loaded], "shared_fixture": shared_manifest, "cases": records,
                "post_w2_weighting_negative_control_detected": placement_detected,
                "compiled_cubin_sha256": cubins, "peak_explicit_device_allocation_bytes": explicit_total,
                "peak_explicit_device_allocation_by_component": peaks,
                "cleanup_explicit_device_allocation_bytes": sum(sum(d.allocations.values()) for d in device_objects),
                "accumulator_cleanup_host_reservation_bytes": accumulator.budget.reserved,
                "minimum_observed_host_available_bytes": min(available_samples),
                "host_probe_reservation_bytes": budget.peak + 4 * MiB,
                "host_available_floor_bytes": budget.available_floor,
                "resource_exclusions": ["Python/NumPy runtime and metadata", "CUDA modules/contexts/compiler",
                                        "PyTorch/TileLang oracle allocations", "caller-retained diagnostic snapshots"],
                "torch_version": torch.__version__, "torch_git_version": torch.version.git_version,
                "torch_cuda_version": torch.version.cuda, "tilelang_version": tilelang.__version__, "numpy_version": np.__version__,
                "reference_runtime_workarounds": official_shared.runtime_workarounds,
                "compiler_options": ["fmad=false", "ftz=false"],
                "scope": "Layer 0; six original FP4 experts and one FP8 shared expert; finite BF16 single-token inputs; supplied routes",
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "connected_original_router": "not_run", "bounded_offload": "not_run", "full_model_quality": "not_run"}
    finally:
        budget.release(512 * MiB)
