"""Independent complete-boundary gate for an actual short text continuation."""

from contextlib import ExitStack
import importlib.util
from pathlib import Path
import json
import sys
from unittest.mock import patch
import numpy as np

from .backbone import LAYERS
from .backbone_assets import load as load_bindings
from .backbone_prepare import PreparedMoE
from .backbone_runtime import open_backbone
from .backbone_validation import capture, compare, native_state, LAYER_KEYS, SHARING_KEYS
from .checkpoint import MODEL, REVISION, validate_expert_provenance
from .engram_rows import EngramRows
from .generation import GenerationRequest, TextGeneration
from .generation_assets import load_token_rows
from .model_artifacts import canonical, digest
from .output_assets import load as load_head
from .resources import GiB, MemoryBudget
from .tokenizer import PinnedTokenizer
from .topology_validation import state_digest

QUALIFIED_IDENTITY = "c1343c08fcaf3722b18e072373ce0cfc8564ca7cb03adf43490fd1603ad2d007"
QUALIFIED_GOLDENS = "916a704b2dfaea4f62439b693d6c6e9f253129a316aa4e468d31bdd760a22708"
ORIGINAL_GENERATE_METHOD_SHA256 = "1b8b80d2139ac8c391f2d837b3479ce2d41b898f84b6b4d39adbd00356dd8297"
REQUEST = GenerationRequest("Once upon a", max_new_tokens=3)


def scope():
    return {"model": MODEL, "revision": REVISION, "layers": list(LAYERS), "batch": 1, "rank": 1, "queries_per_call": 1,
        "maximum_model_calls": 8, "original_hash_capacity": 8,
        "request": {"prompt": REQUEST.prompt, "max_new_tokens": REQUEST.max_new_tokens, "eos_id": REQUEST.eos_id},
        "temperature": .7, "seed": 2026, "raw_text_completion": True, "serial_prompt_ingestion": True,
        "prompt_intermediate_sampling": False, "generated_ids_fed_back": True, "head_attached_after_layer": 39,
        "reference_uses_upstream_generate_loop": True, "original_prompt_chunk_serialized_by_adapter": True,
        "chunk_prefill_comparison_run": False, "chat_template_run": False, "speculative_MTP_run": False, "vision_run": False,
        "full_model_quality": "not_run", "reasoning_coding_tool_evaluation": "not_run", "end_to_end_speed": "not_measured"}


def load_plan(root, run):
    root, run = Path(root), Path(run)
    if not run.resolve().is_relative_to((root / "generation").resolve()): raise ValueError("Expected a separate generation artifact directory")
    if (run / "plan.json").stat().st_size > 4 * 1024**2: raise ValueError("Excessive generation plan metadata")
    plan = json.loads((run / "plan.json").read_bytes())
    checked = {k: v for k, v in plan.items() if k != "plan_sha256"}
    if (plan.get("status"), plan.get("probe"), plan.get("model"), plan.get("revision"), plan.get("request"),
            plan.get("layers"), plan.get("temperature"), plan.get("seed"), plan.get("plan_sha256")) != (
            "pass", "text_generation_artifact_plan", MODEL, REVISION, scope()["request"], list(LAYERS), .7, 2026, canonical(checked)):
        raise ValueError("Generation artifact plan identity differs")
    if set(plan["experts"]) != set(map(str, LAYERS)): raise ValueError("Incomplete generation expert plan")
    for layer, p in plan["experts"].items():
        if (not 1 <= len(p["ids"]) <= 48 or p["ids"] != sorted(set(p["ids"])) or any(type(i) is not int or not 0 <= i < 384 for i in p["ids"])
                or p["store"] != (run.resolve().relative_to(root.resolve()) / f"stores/layer{layer}/{canonical(p['ids'])[:20]}").as_posix()):
            raise ValueError("Unbounded or incorrectly bound original expert store")
    return plan


def expert_plan(plan):
    return {str(l): {"ids": plan["experts"][str(l)]["ids"],
        "store": f"stores/layer{l}/{canonical(plan['experts'][str(l)]['ids'])[:20]}"} for l in LAYERS}


def plan_identity(plan):
    # The raw preparation receipt remains self-hashed by load_plan. Reusing
    # already downloaded bytes must not change the execution fixture identity.
    stable = {k: plan[k] for k in ("model", "revision", "request", "layers", "temperature", "seed", "bindings_sha256",
        "output_fixture", "tokenizer", "calls", "completion", "assets")}
    stable["experts"] = expert_plan(plan)
    return canonical(stable)


def record(meta, result, state):
    return {"meta": meta, "boundaries": capture(result, state), "state_sha256": state_digest(state),
        "rng_offset": state["rng_offset"], "ordered_routes": {str(l): {
            "indices": result[f"layer{l}"]["moe_gate.indices"].tolist(),
            "weight_bits": result[f"layer{l}"]["moe_gate.weights"].view("uint32").tolist()} for l in LAYERS}}


def identity(report):
    names = ("scope", "plan_sha256", "bindings_sha256", "output_fixture", "expert_plan", "expert_provenance", "assets", "tokenizer",
        "reference_methods", "original_source", "original_ngram_source_sha256", "Engram_fixtures", "Engram_controls",
        "original_embedding_transport", "original_grouped_conversion", "original_generate_method_sha256")
    stable = {n: report[n] for n in names}
    a = report["original_attention_adapter"]
    stable["original_attention_adapter"] = {k: a[k] for k in ("model", "revision", "reference", "profiles", "torch_version",
        "torch_git_version", "torch_cuda_version", "tilelang_version", "official_arithmetic_memory_adapter_gate")}
    return canonical(stable)


def validate_trajectory(report):
    tokenizer_prompt = [16600, 4465, 260]
    cases, completion = report["cases"], report["completion"]
    if completion["prompt_ids"] != tokenizer_prompt or completion["prompt"] != REQUEST.prompt or completion["max_new_tokens"] != 3:
        raise ValueError("Changed natural-language request")
    if (len(cases) != 5 or completion["calls"] != 5 or completion["stop_reason"] != "max_new_tokens"
            or completion["eos_id"] != 1 or completion["completion_ids"] != completion["selected_ids_including_eos"]
            or len(completion["completion_ids"]) != 3 or 1 in completion["completion_ids"]):
        raise ValueError("Incomplete scoped generated continuation")
    offset = 0
    for pos, case in enumerate(cases):
        token = tokenizer_prompt[pos] if pos < 3 else cases[pos - 1]["meta"]["selected_id"]
        sample = pos >= 2
        selected = None if not sample else completion["completion_ids"][pos - 2]
        if case["meta"] != {"position": pos, "token_id": token, "sampling_requested": sample, "selected_id": selected}:
            raise ValueError("Generated token was not fed back into the next call")
        offset += 4 if sample else 0
        if case["rng_offset"] != offset: raise ValueError("Prompt ingestion changed original RNG progression")
        if set(case["ordered_routes"]) != set(map(str, LAYERS)): raise ValueError("Missing actual ordered original routes")
        checks = case["boundaries"]
        if len(checks) != (5252 if sample else 5247) or len({c["check"] for c in checks}) != len(checks):
            raise ValueError("Incomplete generated-token boundary census")
        names = {c["check"] for c in checks}
        required = {f"layers/{l}/{n}/" for l in LAYERS for n in (*LAYER_KEYS, "attention_output_bf16")}
        required |= {f"attention/{l}/{n}/" for l in range(2, 40) for n in SHARING_KEYS}
        if not required.issubset(names) or any(c["elements"] != int(np.prod(c["shape"])) or not isinstance(c["sha256"], str)
                or len(c["sha256"]) != 64 or any(x not in "0123456789abcdef" for x in c["sha256"]) for c in checks):
            raise ValueError("Missing actual per-layer original math or invalid typed boundary")
        if not any(c["check"] == "result/head/logits_fp32/" and c["dtype"] == "float32" and c["shape"] == [1, 129280] for c in checks):
            raise ValueError("Missing complete original vocabulary logits")
    return True


def validate_original(report, *, require_qualified=True):
    if (report.get("probe"), report.get("status"), report.get("scope")) != ("original_text_generation_goldens", "pass", scope()):
        raise ValueError("Different original text-generation scope")
    validate_trajectory(report)
    if (report["original_generate_method_sha256"] != ORIGINAL_GENERATE_METHOD_SHA256
            or report["original_source"]["files"]["generate.py"] != "8668d67f7d108e32b90d50cb0d8606889ceb2219bfe95741d84e22f70768e9f0"):
        raise ValueError("Different original generation function or source")
    if (report["original_attention_adapter"]["status"] != "pass" or report["minimum_sampled_reference_device_free_bytes"] < 2 * GiB
            or set(report["Engram_fixtures"]) != {"1", "14"} or [c["layer"] for c in report["Engram_controls"]] != [1, 14] * 5
            or any(c["status"] != "pass" for c in report["Engram_controls"])):
        raise ValueError("Missing actual original Attention or Engram execution")
    expected = {f"{l}:{e}" for l in LAYERS for e in report["expert_plan"][str(l)]["ids"]}
    if set(report["expert_provenance"]) != expected: raise ValueError("Missing or additional original expert provenance")
    for key, manifest in report["expert_provenance"].items():
        validate_expert_provenance(manifest)
        if key != f"{manifest['layer']}:{manifest['expert']}": raise ValueError("Cross-layer original generation expert provenance")
    if report["identity_sha256"] != identity(report) or report["complete_goldens_sha256"] != canonical(report["cases"]):
        raise ValueError("Changed original generation identity or complete goldens")
    if require_qualified and (not QUALIFIED_IDENTITY or not QUALIFIED_GOLDENS or
            (report["identity_sha256"], report["complete_goldens_sha256"]) != (QUALIFIED_IDENTITY, QUALIFIED_GOLDENS)):
        raise ValueError("Original generation capture has not been independently qualified")
    if report["weight_payloads_fetched_in_reference"] != 0 or report["python_http_disabled_during_reference_execution"] is not True:
        raise ValueError("Artifact preparation was counted as an independent reference")
    if report["reference_torch_device_allocated_after_cleanup_bytes"] != report["reference_torch_device_allocated_baseline_bytes"]:
        raise ValueError("Original generation retained framework GPU allocations")
    validate_memory(report, 32 * GiB)
    return True


def validate_memory(report, cap):
    m = report["memory_after_policy_close"]
    if (m["host_cap_bytes"] != cap or m["available_floor_bytes"] != 32 * GiB or m["charged_host_peak_bytes"] > cap
            or m["minimum_observed_available_bytes"] < 32 * GiB or m["charged_host_bytes"] or m["pinned_bytes"]
            or any(m["device_used_bytes"].values()) or m["live_leases"] or m["live_reclaimers"] or m["inflight_operations"]
            or m["monitor_thread_stopped"] is not True or m["reservation_is_RSS_or_OS_limit"] is not False):
        raise ValueError("Generation memory admission, available floor or cleanup differs")


def run_native(root, run, original, progress=None, *, provider_factory=None):
    validate_original(original)
    root, run = Path(root), Path(run); plan = load_plan(root, run)
    if plan_identity(plan) != original["plan_sha256"]: raise ValueError("Native generation plan differs from original")
    bindings = json.loads((root / "backbone-controls.json").read_bytes())
    with ExitStack() as stack:
        budget = MemoryBudget(cap=12 * GiB, owner="generation_native_fixtures"); budget.acquire(budget.cap)
        stack.callback(budget.release, budget.cap)
        stack.enter_context(patch("urllib.request.urlopen", side_effect=RuntimeError("Native generation must remain offline")))
        arrays = load_bindings(root, bindings); head_arrays, head_manifest = load_head(root / "output-fixture")
        tokens = load_token_rows(root, run, plan)
        providers = {l: (PreparedMoE(root, bindings, l, plan["experts"][str(l)], original["expert_provenance"]) if provider_factory is None
            else provider_factory(root, bindings, l, plan["experts"][str(l)], original["expert_provenance"])) for l in LAYERS}
        rows = {l: stack.enter_context(EngramRows(run / f"rows/layer{l}", l)) for l in (1, 14)}
        engine = open_backbone(stack, root, arrays, tokens, providers, head_arrays, engram_rows=rows)
        if (engine.tokenizer_identity != original["tokenizer"] or engine.engram_manifests != original["Engram_fixtures"]
                or head_manifest != original["output_fixture"] or bindings["bindings_sha256"] != original["bindings_sha256"]
                or {str(l): p.converted_sha256 for l, p in engine.probe_operators["projections"].items()} != original["original_grouped_conversion"]):
            raise ValueError("Native generation original control bindings differ")
        session = TextGeneration(PinnedTokenizer(root / "import-source"), engine.step, REQUEST); cases = []
        while not session.done:
            meta, result = session.advance(); state = native_state(engine)
            case = record(meta, result, state); expected = original["cases"][len(cases)]
            if (meta, case["state_sha256"], case["ordered_routes"]) != (expected["meta"], expected["state_sha256"], expected["ordered_routes"]):
                raise RuntimeError("Generated trajectory, original routes or committed state differs")
            case["boundaries"] = compare(expected["boundaries"], result, state); case["status"] = "pass"
            cases.append(case)
            if progress: progress({"native_token": meta, "boundaries": len(case["boundaries"]), "status": "pass"})
        if session.summary() != original["completion"]: raise RuntimeError("Decoded original continuation differs")
        report = {"schema_version": 1, "probe": "native_text_generation_complete_recheck", "status": "pass", "scope": scope(),
            "plan_sha256": plan_identity(plan), "original_identity_sha256": original["identity_sha256"],
            "complete_goldens_sha256": original["complete_goldens_sha256"], "cases": cases, "completion": session.summary(),
            "transaction_active": engine.transaction.stats(), "python_http_disabled_during_native_execution": True,
            "weight_payloads_fetched_in_recheck": 0, "native_frameworks": {n: {"installed": importlib.util.find_spec(n) is not None,
                "imported": n in sys.modules} for n in ("torch", "tilelang", "transformers", "sympy")}}
    report["transaction_cleanup"] = engine.transaction.stats()
    del result, state, arrays, head_arrays, engine
    import gc
    gc.collect()
    return report


def validate_native(report, original, original_raw):
    validate_original(original)
    if (report.get("probe"), report.get("status"), report.get("scope"), report.get("reference_report_sha256"), report.get("plan_sha256"),
            report.get("original_identity_sha256"), report.get("complete_goldens_sha256"), report.get("completion")) != (
            "native_text_generation_complete_recheck", "pass", scope(), digest(original_raw), original["plan_sha256"],
            QUALIFIED_IDENTITY, QUALIFIED_GOLDENS, original["completion"]): raise ValueError("Native generation is not bound to the complete sealed original")
    validate_trajectory(report)
    for actual, expected in zip(report["cases"], original["cases"]):
        if (actual["meta"], actual["state_sha256"], actual["ordered_routes"], actual["status"]) != (
                expected["meta"], expected["state_sha256"], expected["ordered_routes"], "pass"):
            raise ValueError("Different actual native generated trajectory")
        for a, e in zip(actual["boundaries"], expected["boundaries"]):
            if (any(a.get(k) != e[k] for k in ("check", "dtype", "shape", "elements", "sha256")) or a.get("status") != "pass"
                    or a.get("actual_sha256") != e["sha256"] or a.get("expected_sha256") != e["sha256"]):
                raise ValueError("Changed generated-token typed/shaped boundary")
    if (report["native_frameworks"] != {n: {"installed": False, "imported": False} for n in ("torch", "tilelang", "transformers", "sympy")}
            or report["python_http_disabled_during_native_execution"] is not True or report["weight_payloads_fetched_in_recheck"] != 0):
        raise ValueError("Native generation used reference frameworks or fetched original payloads")
    for name in ("transaction_active", "transaction_cleanup"):
        t = report[name]
        if ((t["position"], t["committed_tokens"], t["failed_attempts"]) != (5, 5, 0)
                or t["pending_cleanup_owners"] or t["pending_recovery_owners"]): raise ValueError("Generation transaction retained unfinished owners")
    if report["transaction_cleanup"]["host_reserved_bytes"]: raise ValueError("Generation retained transaction admission")
    validate_memory(report, 64 * GiB)
    return True
