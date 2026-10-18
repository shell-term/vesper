"""Explicit, bounded import of only the original experts selected by real gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .bandwidth import EXPERT_BYTES
from .checkpoint import MODEL, REVISION, fetch_expert_fixture, load_expert_fixture
from .formats import bf16_bits, bf16_values
from .gate import GateProjection, load_gate_fixture
from .prefix import load_moe_inputs, moe_input_provenance
from .resources import MemoryBudget, MiB
from .router import RouterProbe, load_router_fixture
from .storage import build_store, validate_store_limit


def connected_inputs(captured_inputs=None):
    if captured_inputs is not None:
        rows, _ = load_moe_inputs(captured_inputs)
        for metadata, x in rows:
            yield {"input": metadata["input"], "origin": "captured_original_prefix"}, x
        return
    for seed, scale in ((20261018, 1.0), (42, 0.0625), (0, 16.0), (65535, 1.0)):
        x = np.random.default_rng(seed).standard_normal(5120).astype(np.float32) * np.float32(scale)
        yield {"input": f"normal_seed{seed}_scale{scale:g}", "origin": "generated_not_model_activation",
               "seed": seed, "scale": scale}, bf16_values(bf16_bits(x))


def plan_connected(gate_fixture, router_fixture, captured_inputs=None, ordinal=0, *, max_experts=48):
    validate_store_limit(max_experts)
    weight, gate_manifest = load_gate_fixture(gate_fixture)
    biases, router_manifest = load_router_fixture(router_fixture)
    if gate_manifest["layer"] != 0 or router_manifest["layer"] != 0:
        raise ValueError("Connected plan requires layer-0 gate artifacts")
    cases, selected = [], set()
    with GateProjection(weight, ordinal) as gate, RouterProbe(biases, ordinal) as router:
        for metadata, x in connected_inputs(captured_inputs):
            logits = gate.project(x)
            for image_span in (False, True):
                routing = router.inspect(logits, image_span)
                indices = routing["indices"].tolist()
                selected.update(indices)
                if len(selected) > max_experts:
                    raise ValueError("Selected expert union exceeds the explicit import limit; no model bytes fetched")
                cases.append({**metadata, "image_span": image_span,
                              "input_bf16_sha256": hashlib.sha256(bf16_bits(x).tobytes()).hexdigest(),
                              "logits_sha256": hashlib.sha256(logits.tobytes()).hexdigest(),
                              "indices": indices, "weight_bits": routing["weights"].view(np.uint32).tolist()})
        profile = {"cublas_version": gate.blas.version, "router_cubin_sha256": router.cubin_sha256}
    return {"probe": "connected_moe_plan", "status": "pass", "model": MODEL, "revision": REVISION, "layer": 0,
            "gate_fixture": gate_manifest, "router_fixture": router_manifest, "cases": cases,
            "expert_ids": sorted(selected), "max_experts": max_experts,
            "selected_weight_bytes": len(selected) * EXPERT_BYTES,
            "maximum_weight_bytes": max_experts * EXPERT_BYTES, "native_profile": profile,
            "input_source": "generated" if captured_inputs is None else "captured_original_prefix",
            "captured_input_provenance": None if captured_inputs is None else moe_input_provenance(captured_inputs),
            "scope": "Actual original gate decisions; no forced IDs, cache bias or expert downloads"}


def load_connected_plan(path):
    path = Path(path)
    if path.stat().st_size > 2 * MiB:
        raise ValueError("Connected plan exceeds the bounded metadata size")
    plan = json.loads(path.read_bytes())
    if (plan.get("probe"), plan.get("status"), plan.get("model"), plan.get("revision"), plan.get("layer")) != (
            "connected_moe_plan", "pass", MODEL, REVISION, 0):
        raise ValueError("Connected plan differs from the pinned source")
    validate_store_limit(plan["max_experts"])
    ids = plan.get("expert_ids", [])
    if (not 1 <= len(ids) <= plan["max_experts"] or ids != sorted(set(ids))
            or any(type(e) is not int or not 0 <= e < 384 for e in ids)):
        raise ValueError("Connected plan has invalid expert identities")
    union = set()
    if not 1 <= len(plan.get("cases", [])) <= 32:
        raise ValueError("Connected plan case count is outside the probe")
    for case in plan["cases"]:
        if (type(case["image_span"]) is not bool or len(case["indices"]) != 6 or len(set(case["indices"])) != 6
                or any(type(e) is not int or not 0 <= e < 384 for e in case["indices"])):
            raise ValueError("Connected plan requires six distinct original expert IDs per case")
        union.update(case["indices"])
    if sorted(union) != ids or plan.get("selected_weight_bytes") != len(ids) * EXPERT_BYTES:
        raise ValueError("Connected import scope differs from the gate decision union")
    return plan


def validate_plan_inputs(plan, captured_inputs):
    expected = "generated" if captured_inputs is None else "captured_original_prefix"
    if plan.get("input_source") != expected:
        raise ValueError("Input source differs from the immutable connected plan")
    if captured_inputs is not None and plan.get("captured_input_provenance") != moe_input_provenance(captured_inputs):
        raise ValueError("Qualified prefix provenance differs from the connected plan")


def prepare_connected_store(plan_file, expert_directory, store_directory):
    """Network access occurs only here, bounded by the explicit gate-derived plan."""
    plan = load_connected_plan(plan_file)
    expert_directory, store_directory = Path(expert_directory), Path(store_directory)
    if store_directory.exists():
        raise FileExistsError("Refusing to replace an existing connected native store")
    budget = MemoryBudget(cap=96 * MiB)
    budget.acquire(96 * MiB)
    try:
        fixtures, fetched, reused = [], 0, 0
        for expert in plan["expert_ids"]:
            directory = expert_directory / f"expert-{expert:03d}"
            path = directory / "native-expert.npz"
            if path.exists():
                arrays, manifest = load_expert_fixture(path)
                del arrays
                if manifest["layer"] != 0 or manifest["expert"] != expert:
                    raise ValueError("Cached fixture identity differs from the import plan")
                reused += EXPERT_BYTES
            else:
                fetch_expert_fixture(directory, expert=expert)
                fetched += EXPERT_BYTES
            fixtures.append(path)
        result = build_store(fixtures, store_directory, max_experts=plan["max_experts"])
        return {"probe": "prepare_connected_store", "status": "pass", "experts": len(fixtures),
                "expert_ids": plan["expert_ids"], "new_weight_bytes_fetched": fetched,
                "verified_existing_weight_bytes_reused": reused, "store": result,
                "max_experts": plan["max_experts"], "maximum_weight_bytes": plan["maximum_weight_bytes"],
                "scope": "Explicit selected experts only; bounded reads; no full checkpoint shard"}
    finally:
        budget.release(96 * MiB)
