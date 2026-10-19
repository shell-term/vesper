"""Scoped connected MoE comparisons; generated inputs do not clear the real-input gate."""

from __future__ import annotations

from collections import OrderedDict
import copy
from dataclasses import replace
import importlib.util
import os
from pathlib import Path

import numpy as np

from .bandwidth import EXPERT_BYTES
from .checkpoint import MODEL, REVISION, load_expert_fixture
from .conformance import compare
from .connected import ConnectedConfig, ConnectedMoE
from .connected_fixture import connected_inputs, load_connected_plan, validate_plan_inputs
from .expert_reference import OfficialExpert
from .formats import bf16_bits, bf16_values
from .gate import load_gate_fixture
from .gate_reference import OfficialGate
from .moe_reference import OfficialAggregation
from .resources import MemoryBudget, MiB, physical_memory
from .router import ROUTER_TORCH_REVISION, load_router_fixture
from .shared import load_shared_fixture
from .shared_reference import digest, expert_stages
from .storage import expert_key


def connected_stages(result):
    stages = {"gate.input_promoted": result["input_promoted"], "gate.logits": result["logits"]}
    stages.update({f"gate.{name}": result["routing"][name] for name in
                   ("scores", "selection", "raw", "denominator", "weights", "indices")})
    for expert, output in result["experts"].items():
        stages.update({f"expert_{expert}.{name}": array for name, array in output.items()})
    stages.update({f"shared.{name}": array for name, array in result["shared"].items()})
    stages.update({f"aggregation.{name}": array for name, array in result["aggregation"].items()})
    return stages


def native_connected_stages(result):
    return connected_stages({"input_promoted": bf16_values(result["input_bf16"]),
                             "logits": result["logits"], "routing": result["routing"],
                             "experts": {e: expert_stages(out, native=True) for e, out in result["experts"].items()},
                             "shared": expert_stages(result["shared"], native=True), "aggregation": result["aggregation"]})


class OfficialConnected:
    """Original gate, experts, shared expert and MoE.forward; six-entry oracle LRU."""

    def __init__(self, gate_fixture, router_fixture, shared_fixture, expert_directory, reference_directory, ordinal=0, *, layer=0):
        if type(layer) is not int or layer not in (0, 2, 3, 20, 21, 22, 23, 24, 25):
            raise ValueError("Unregistered connected MoE oracle layer")
        self.layer = layer
        import torch
        self.torch = torch
        if (torch.version.git_version != ROUTER_TORCH_REVISION or torch.get_float32_matmul_precision() != "highest"
                or str(torch.backends.cuda.preferred_blas_library()) != "_BlasBackend.Cublas"
                or torch.cuda.tunable.is_enabled() or os.environ.get("CUBLAS_WORKSPACE_CONFIG") is not None
                or any(os.environ.get(n) not in (None, "0") for n in
                       ("NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "TORCH_CUBLASLT_UNIFIED_WORKSPACE"))):
            raise ValueError("Connected oracle requires the qualified PyTorch/FP32/cuBLAS profile")
        self.ordinal, self.directory, self.expert_directory = ordinal, Path(reference_directory), Path(expert_directory)
        weight, self.gate_manifest = load_gate_fixture(gate_fixture)
        biases, self.router_manifest = load_router_fixture(router_fixture)
        shared, self.shared_manifest = load_shared_fixture(shared_fixture)
        if any(m["layer"] != layer for m in (self.gate_manifest, self.router_manifest, self.shared_manifest)):
            raise ValueError("Original MoE oracle artifact layers differ")
        self.gate = OfficialGate(weight, biases, self.directory, ordinal)
        self.shared = OfficialExpert(shared, self.shared_manifest, self.directory, ordinal,
                                     weight_dtype=torch.float8_e4m3fn)
        self.aggregation = OfficialAggregation(self.directory, ordinal)
        self.experts, self.provenance = OrderedDict(), {}

    def _expert(self, expert):
        if expert in self.experts:
            self.experts.move_to_end(expert)
            return self.experts[expert]
        if len(self.experts) == 6:
            self.experts.popitem(last=False)
        arrays, manifest = load_expert_fixture(self.expert_directory / f"expert-{expert:03d}/native-expert.npz")
        if manifest["layer"] != self.layer or manifest["expert"] != expert:
            raise ValueError("Oracle expert identity differs from the actual gate selection")
        self.provenance[expert_key(self.layer, expert)] = manifest
        self.experts[expert] = OfficialExpert(arrays, manifest, self.directory, self.ordinal)
        return self.experts[expert]

    @staticmethod
    def _function(reference):
        def call(tx, route=None):
            reference.outputs, reference.quantized = {}, {}
            return reference.forward_function(reference.expert, tx, route)
        return call

    def forward(self, x, image_span):
        gate = self.gate.inspect(x, "bf16", image_span)
        indices, weights = gate["indices"], gate["weights"]
        outputs, functions = {}, {}
        for expert in sorted(map(int, indices)):
            reference = self._expert(expert)
            route = float(weights[int(np.flatnonzero(indices == expert)[0])])
            outputs[expert] = expert_stages(reference.forward(x, route))
            functions[expert] = self._function(reference)
        shared = expert_stages(self.shared.forward(x))
        aggregation = self.aggregation.forward(indices.reshape(1, 6), {e: out["w2"].reshape(1, 5120) for e, out in outputs.items()},
                                                shared["w2"].reshape(1, 5120), routing_weights=weights.reshape(1, 6),
                                                input_values=x.reshape(1, 5120), expert_functions=functions,
                                                shared_function=self._function(self.shared), image_span=image_span,
                                                gate_function=lambda tx, mask: self.gate.forward(self.gate.gate, tx, mask))
        return connected_stages({"input_promoted": gate["input_promoted"], "logits": gate["logits"], "routing": gate,
                                 "experts": outputs, "shared": shared, "aggregation": aggregation})


def exercise_connected(factory, rows, checker):
    """Shared manual matrix: real transfers and failures, no model-route changes."""
    records, phases, faults = [], [], []
    def check(runtime, phase, index):
        metadata, x = rows[index]
        result = runtime.forward(x, metadata["image_span"])
        checks = checker(index, result)
        records.append({"phase": phase, "case_index": index, "input": metadata["input"],
                        "image_span": metadata["image_span"], "indices": result["routing"]["indices"].tolist(),
                        "weight_bits": result["routing"]["weights"].view(np.uint32).tolist(),
                        "expert_events": result["expert_events"], "attempt": copy.deepcopy(runtime.last_attempt),
                        "checks": checks, "status": "pass" if all(c["status"] == "pass" for c in checks) else "fail"})
        return result
    def fail(runtime, name, index, exception_type):
        metadata, x = rows[index]
        try:
            runtime.forward(x, metadata["image_span"])
        except exception_type:
            attempt = copy.deepcopy(runtime.last_attempt)
            if attempt.get("indices") is not None:
                assert attempt["indices"] == metadata["indices"] and attempt["weight_bits"] == metadata["weight_bits"]
            faults.append({"name": name, "case_index": index, "no_output_returned": True, "attempt": attempt})
        else:
            raise AssertionError(f"{name} did not refuse the incomplete call")
    with factory("six_slots") as runtime:
        check(runtime, "cold", 0)
        # Cache-state preparation changes residency only. No router/expert math.
        for expert in sorted(rows[0][0]["indices"], reverse=True):
            with runtime.experts.cache.acquire(expert_key(0, expert)):
                pass
        before = runtime.snapshot()["cache"]["store_direct_read_calls"]
        result = check(runtime, "primed_gpu_helper_ram", 0)
        assert {e["source_tier"] for e in result["expert_events"]} == {"primary_gpu", "helper_gpu", "ram"}
        assert runtime.snapshot()["cache"]["store_direct_read_calls"] == before
        for index in range(len(rows)):
            check(runtime, "changing_actual_gate_decisions", index)
            check(runtime, "repeat_actual_gate_decisions", index)
        saved = runtime.experts.cache.available
        before = runtime.snapshot()["cache"]["store_direct_read_calls"]
        try:
            runtime.experts.cache.available = lambda: runtime.config.cache.available_floor - 1
            fail(runtime, "pressure_before_gate", 0, MemoryError)
        finally:
            runtime.experts.cache.available = saved
        assert not runtime.experts.cache.ram and runtime.experts.store.read_calls == before
        check(runtime, "pressure_recovery", 0)
        original_forward, completed = runtime.experts.forward, 0
        def pressure_after_three(*args, **kwargs):
            nonlocal completed
            result = original_forward(*args, **kwargs)
            completed += 1
            if completed == 3:
                runtime.experts.cache.available = lambda: runtime.config.cache.available_floor - 1
            return result
        try:
            runtime.experts.forward = pressure_after_three
            fail(runtime, "pressure_after_three_experts", 0, MemoryError)
        finally:
            runtime.experts.forward, runtime.experts.cache.available = original_forward, saved
        assert len(runtime.last_attempt["completed_expert_ids"]) == 3
        check(runtime, "retry_discards_partial_sum", 0)
        phases.append({"name": "six_slots", "snapshot": runtime.snapshot()})
    phases[-1]["cleanup"] = runtime.cleanup
    with factory("two_slots_no_ram") as runtime:
        for index in range(len(rows)):
            check(runtime, "zero_ram_forced_reload", index)
        write, uploads = runtime.experts.backend.write, 0
        def corrupt_fourth(tier, slot, data):
            nonlocal uploads
            if tier == "primary":
                uploads += 1
                if uploads == 4:
                    changed = data.copy()
                    changed[0] ^= 1
                    return write(tier, slot, changed)
            return write(tier, slot, data)
        try:
            runtime.experts.backend.write = corrupt_fourth
            fail(runtime, "corrupt_fourth_primary_upload", 0, ValueError)
        finally:
            runtime.experts.backend.write = write
        assert len(runtime.last_attempt["completed_expert_ids"]) == 3
        check(runtime, "retry_after_rejected_upload", 0)
        read, reads = runtime.experts.store.reader.read, 0
        def fail_fourth_read(*args, **kwargs):
            nonlocal reads
            reads += 1
            if reads == 4:
                raise OSError("Injected pre-read I/O failure")
            return read(*args, **kwargs)
        try:
            runtime.experts.store.reader.read = fail_fourth_read
            fail(runtime, "fourth_store_read_refused", 0, OSError)
        finally:
            runtime.experts.store.reader.read = read
        assert len(runtime.last_attempt["completed_expert_ids"]) == 3
        check(runtime, "retry_after_io_failure", 0)
        phases.append({"name": "two_slots_no_ram", "snapshot": runtime.snapshot()})
    phases[-1]["cleanup"] = runtime.cleanup
    return records, phases, faults


def compare_connected(plan_file, store, gate_fixture, router_fixture, shared_fixture, expert_directory,
                      reference_directory, ordinal=0, helper_ordinal=1, captured_inputs=None):
    missing = [n for n in ("torch", "tilelang") if importlib.util.find_spec(n) is None]
    if missing:
        return {"probe": "connected_moe_parity", "status": "blocked", "missing_dependencies": missing}
    import torch
    import tilelang
    plan = load_connected_plan(plan_file)
    validate_plan_inputs(plan, captured_inputs)
    budget = MemoryBudget(cap=512 * MiB)
    budget.acquire(512 * MiB)
    try:
        oracle = OfficialConnected(gate_fixture, router_fixture, shared_fixture, expert_directory, reference_directory, ordinal)
        rows, expected, minimum = [], [], physical_memory()["available"]
        for metadata, x in connected_inputs(captured_inputs):
            for image_span in (False, True):
                planned = plan["cases"][len(rows)]
                if planned["input"] != metadata["input"] or planned["image_span"] != image_span or planned["input_bf16_sha256"] != digest(bf16_bits(x)):
                    raise ValueError("Connected inputs differ from the immutable import plan")
                rows.append((planned, x))
                reference = oracle.forward(x, image_span)
                if (reference["gate.indices"].tolist() != planned["indices"]
                        or reference["gate.weights"].view(np.uint32).tolist() != planned["weight_bits"]
                        or digest(reference["gate.logits"]) != planned["logits_sha256"]):
                    raise ValueError("Original gate differs from the planned native decisions")
                expected.append(reference)
                minimum = min(minimum, physical_memory()["available"])
                budget.acquire(0, minimum)
        if len(rows) != len(plan["cases"]):
            raise ValueError("Connected plan case count differs")
        def factory(mode):
            config = ConnectedConfig()
            cache = replace(config.cache, primary_device=ordinal, helper_device=helper_ordinal)
            if mode == "two_slots_no_ram":
                cache = replace(cache, primary_slots=1, helper_slots=1, ram_cache_bytes=0)
            return ConnectedMoE(store, gate_fixture, router_fixture, shared_fixture, replace(config, cache=cache))
        def checker(index, actual):
            actual = native_connected_stages(actual)
            if set(actual) != set(expected[index]):
                raise ValueError("Connected boundary sets differ")
            return [compare(expected[index][name], actual[name], name) for name in expected[index]]
        records, phases, faults = exercise_connected(factory, rows, checker)
        passed = bool(records) and all(r["status"] == "pass" for r in records) and all(not any(p["cleanup"].values()) for p in phases)
        return {"probe": "connected_moe_parity", "status": "pass" if passed else "fail", "model": MODEL, "revision": REVISION,
                "plan": plan, "cases": records, "phases": phases, "faults": faults,
                "reference_goldens": [{"case_index": i, "stage_sha256": {n: digest(a) for n, a in data.items()}} for i, data in enumerate(expected)],
                "gate_fixture": oracle.gate_manifest, "router_fixture": oracle.router_manifest, "shared_fixture": oracle.shared_manifest,
                "expert_fixtures": list(oracle.provenance.values()), "reference": oracle.gate.source,
                "reference_method": "Unchanged linear/Gate.forward/Expert.forward/MoE.forward; actual original gate called inside MoE",
                "reference_runtime_workarounds": oracle.shared.runtime_workarounds,
                "torch_version": torch.__version__, "torch_git_version": torch.version.git_version, "torch_cuda_version": torch.version.cuda,
                "tilelang_version": tilelang.__version__, "numpy_version": np.__version__, "reference_blas": oracle.gate.library,
                "minimum_observed_host_available_bytes": minimum, "oracle_host_reservation_bytes": budget.peak,
                "scope": "Layer-0 single-token connected MoE; actual gate decisions; " + plan["input_source"] + "; audited two-GPU offload",
                "input_source": plan["input_source"], "actual_model_activation_gate": "not_run" if captured_inputs is None else "pass" if passed else "fail",
                "issue6_acceptance_complete": captured_inputs is not None and passed, "full_model_quality": "not_run", "generation_speed": "not_measured"}
    finally:
        budget.release(512 * MiB)
