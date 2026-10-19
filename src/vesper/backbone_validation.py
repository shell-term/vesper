"""Fixed all-layer boundaries and whole-model rollback/retry qualification."""

from contextlib import ExitStack
import importlib.util
from pathlib import Path
from unittest.mock import patch

import numpy as np

from .attention_validation import array_digest
from .backbone import LAYERS
from .backbone_assets import load as load_bindings
from .backbone_prepare import TOKENS, PreparedMoE
from .backbone_runtime import embeddings, open_backbone
from .checkpoint import MODEL, REVISION, validate_expert_provenance
from .model_artifacts import canonical, digest
from .memory import RuntimeCancelled
from .output_assets import load as load_head
from .resources import GiB, MemoryBudget
from .topology_validation import flatten, state_digest

QUALIFIED_IDENTITY = "1746fa338af812fad3d981c3e6ccdc00db94094acc6f28e07644004c3b3afcb4"
QUALIFIED_GOLDENS = "b56cdc2062210bd0388806a3057882c84c78b3ef47bc9eaab274c95ae3f1febc"
LAYER_KEYS = ("input_hidden_bf16", "incoming_pre_fp32", "hc_attn_pre_fp32", "attn_pre_output_bf16", "attn_norm_output_bf16",
    "attn_post_output_bf16", "hc_ffn_pre_fp32", "ffn_pre_output_bf16", "ffn_norm_output_bf16", "ffn_post_output_bf16",
    *("moe_gate." + n for n in ("input_promoted", "logits", "scores", "selection", "raw", "denominator", "weights", "indices")),
    "moe_aggregation.output_bf16", "output_bf16", "next_pre_mix_fp32")
SHARING_KEYS = ("query_rope_output_bf16", "combined_KV_bf16", "combined_indices_i32", "attention_output_bf16",
    "selected_indices_i32", "window_cache_bf16", "compressed_visible_bf16")
FAULTS = ("after_Engram14", "after_layer39_experts", "after_sampling", "cancel_after_sampling")


def scope():
    return {"model": MODEL, "revision": REVISION, "layers": list(LAYERS), "tokens": list(TOKENS), "batch": 1,
        "queries_per_call": 1, "rank": 1, "text_only": True, "temperature": .7, "seed": 2026,
        "head_attached_after_layer": 39, "head_is_diagnostic_only": False, "full_backbone_execution_run": True,
        "serial_autoregressive_path": True, "speculative_MTP_run": False, "vision_run": False,
        "original_auxiliary_target_layers": [37, 38, 39], "auxiliary_main_hidden_comparison_run": False,
        "text_generation_available": False, "full_model_quality": "not_run", "end_to_end_speed": "not_measured",
        "full_checkpoint_downloaded": False, "corpus": "two_fixed_token_ids_not_a_generated_continuation"}


def boundary_values(result, state):
    layers = {}
    for l in LAYERS:
        row = result[f"layer{l}"]
        layers[str(l)] = {name: row[name] for name in LAYER_KEYS}
        layers[str(l)].update((name, a) for name, a in row.items() if name.startswith("moe_"))
        layers[str(l)]["attention_output_bf16"] = (row["attention_output_bf16"] if "attention_output_bf16" in row else
            row["attention_wo_b_output_bf16"] if l < 2 else result["sharing_attention"][f"layer{l}_wo_b_output_bf16"])
    sharing = {str(l): {name: result["sharing_attention"][f"layer{l}_{name}"] for name in SHARING_KEYS} for l in range(2, 40)}
    selected = {n: result[n] for n in ("embedding_bf16", "initial_pre_fp32", "hash_ids_i64", "compressed_i64", "lookbacks_i64",
        "engram1", "engram14", "head", "sampling", "output_bf16", "next_pre_mix_fp32")}
    return flatten({"result": selected, "layers": layers, "attention": sharing, "state": state})


def capture(result, state):
    return [{"check": n, "dtype": str(a.dtype), "shape": list(a.shape), "elements": int(a.size), "sha256": array_digest(a)}
        for n, a in sorted(boundary_values(result, state).items())]


def compare(expected, result, state):
    values = boundary_values(result, state)
    if len(expected) != len(values) or set(values) != {c["check"] for c in expected}:
        raise ValueError("Missing, additional or duplicate backbone boundary")
    checks = []
    for c in expected:
        a = values[c["check"]]; sha = array_digest(a)
        passed = (str(a.dtype), list(a.shape), a.size, sha) == (c["dtype"], c["shape"], c["elements"], c["sha256"])
        checks.append({**c, "expected_sha256": c["sha256"], "actual_sha256": sha, "status": "pass" if passed else "fail"})
    failed = [c["check"] for c in checks if c["status"] != "pass"]
    if failed: raise RuntimeError("Original all-layer boundary differs: " + ", ".join(failed[:12]))
    return checks


def native_state(engine):
    return {"window0": engine.windows[0].kv.inspect()["cache_bf16"], "window1": engine.windows[1].kv.inspect()["cache_bf16"],
        "sharing": engine.sharing.inspect()["state"], "history": engine.transaction.history.inspect()["history"],
        "position": engine.position, "rng_offset": engine.transaction.sampler.offset}


def identity(report):
    stable = {k: report[k] for k in ("scope", "bindings_sha256", "output_fixture", "expert_plan", "tokenizer",
        "expert_provenance",
        "original_source", "original_ngram_source_sha256",
        "negative_controls", "reference_methods", "Engram_controls", "Engram_fixtures",
        "original_embedding_transport", "original_grouped_conversion")}
    adapter = report["original_attention_adapter"]
    stable["original_attention_adapter"] = {k: adapter[k] for k in ("model", "revision", "reference", "profiles",
        "torch_version", "torch_git_version", "torch_cuda_version", "tilelang_version", "official_arithmetic_memory_adapter_gate")}
    return canonical(stable)


def validate_original(report, *, require_qualified=True):
    if (report.get("probe"), report.get("status"), report.get("scope")) != ("original_all40_backbone_goldens", "pass", scope()):
        raise ValueError("Original forty-layer evidence scope differs")
    if [c["meta"] for c in report["cases"]] != [{"position": i, "token_id": t} for i, t in enumerate(TOKENS)]:
        raise ValueError("Original backbone token corpus differs")
    if set(report["expert_plan"]) != set(map(str, LAYERS)): raise ValueError("Missing original expert owners")
    if (report["original_attention_adapter"]["status"] != "pass" or report["weight_payloads_fetched_in_reference"]
            or [c["layer"] for c in report["Engram_controls"]] != [1, 14, 1, 14]
            or any(c["status"] != "pass" for c in report["Engram_controls"])):
        raise ValueError("Missing actual original Attention/Engram execution")
    for l in LAYERS:
        p = report["expert_plan"][str(l)]; ids = p["ids"]
        expected = canonical({"layer": l, "ids": ids})[:20]
        if (not 6 <= len(ids) <= 12 or ids != sorted(set(ids)) or any(type(i) is not int or not 0 <= i < 384 for i in ids)
                or p["store"] != f"backbone-stores/layer{l}/{expected}"):
            raise ValueError("Invalid bounded original backbone expert plan")
    if set(report["expert_provenance"]) != {f"{l}:{e}" for l in LAYERS for e in report["expert_plan"][str(l)]["ids"]}:
        raise ValueError("Missing or additional original expert payload provenance")
    for key, manifest in report["expert_provenance"].items():
        validate_expert_provenance(manifest)
        if key != f"{manifest['layer']}:{manifest['expert']}": raise ValueError("Cross-layer original expert provenance")
    for case in report["cases"]:
        rows = case["boundaries"]
        names = {c["check"] for c in rows}
        if len(rows) != len(names) or not rows: raise ValueError("Missing or duplicate original boundary")
        if any(c["elements"] != int(np.prod(c["shape"])) or len(c["sha256"]) != 64 for c in rows):
            raise ValueError("Invalid typed original boundary")
        required = {f"layers/{l}/{n}/" for l in LAYERS for n in (*LAYER_KEYS, "attention_output_bf16")}
        required |= {f"attention/{l}/{n}/" for l in range(2, 40) for n in SHARING_KEYS}
        if not required.issubset(names): raise ValueError("Incomplete fixed per-layer model comparison")
        if set(case["ordered_routes"]) != set(map(str, LAYERS)): raise ValueError("Missing original ordered routes")
    if report["identity_sha256"] != identity(report) or report["complete_goldens_sha256"] != canonical(report["cases"]):
        raise ValueError("Original evidence identity does not bind its complete goldens")
    if require_qualified and (identity(report), canonical(report["cases"])) != (QUALIFIED_IDENTITY, QUALIFIED_GOLDENS):
        raise ValueError("Unqualified original full-backbone evidence")
    return report


def run_native(root, bindings, original, policy, progress=None, *, require_qualified=True):
    validate_original(original, require_qualified=require_qualified)
    if bindings["bindings_sha256"] != original["bindings_sha256"]: raise ValueError("Native control binding differs")
    budget = MemoryBudget(cap=12 * GiB, owner="backbone_fixture_and_diagnostics"); budget.acquire(budget.cap)
    try:
        root = Path(root); arrays = load_bindings(root, bindings)
        head_arrays, head_manifest = load_head(root / "output-fixture")
        if head_manifest != original["output_fixture"]: raise ValueError("Native original output fixture differs")
        providers = {l: PreparedMoE(root, bindings, l, original["expert_plan"][str(l)], original["expert_provenance"]) for l in LAYERS}
        with ExitStack() as stack:
            stack.enter_context(patch("urllib.request.urlopen", side_effect=RuntimeError("Native all-layer replay must remain offline")))
            engine = open_backbone(stack, root, arrays, embeddings(root), providers, head_arrays)
            if engine.tokenizer_identity != original["tokenizer"]: raise ValueError("Native original tokenizer differs")
            if engine.engram_manifests != original["Engram_fixtures"]: raise ValueError("Native original Engram fixtures differ")
            if {str(l): p.converted_sha256 for l, p in engine.probe_operators["projections"].items()} != original["original_grouped_conversion"]:
                raise ValueError("Original per-layer projection conversion differs")
            profiles = {str(l): {"projection_cubin_sha256": p.cubin_sha256, "projection_compiler": p.compiler,
                "HC_cubin_sha256": engine.hc[l].cubin_sha256} for l, p in engine.probe_operators["projections"].items()}
            rows, faults = [], []
            result = engine.step(np.array([[TOKENS[0]]], np.int64))
            state = native_state(engine)
            rows.append({"meta": original["cases"][0]["meta"], "status": "pass", "state_sha256": state_digest(state),
                "sampled_ids": result["sampling"]["ids_i64"].tolist(),
                "boundaries": compare(original["cases"][0]["boundaries"], result, state)})
            if rows[-1]["state_sha256"] != original["cases"][0]["state_sha256"]: raise RuntimeError("Original committed state differs")
            before = state_digest(state)
            if progress: progress({"native_position": 0, "status": "pass", "boundaries": len(rows[-1]["boundaries"])})
            class InjectedFailure(RuntimeError): pass
            for name in FAULTS:
                error = RuntimeCancelled if name == "cancel_after_sampling" else InjectedFailure
                triggered = False
                def fail(*args, **kwargs):
                    nonlocal triggered
                    triggered = True
                    raise InjectedFailure("Injected all-layer token failure")
                def observe(result):
                    nonlocal triggered
                    if name == "cancel_after_sampling":
                        triggered = True; policy.cancel(); return result
                    fail()
                with ExitStack() as inject:
                    if name == "after_Engram14":
                        op = engine.engrams[14]; saved = op.forward
                        def wrapped(*a, **k): saved(*a, **k); fail()
                        inject.enter_context(patch.object(op, "forward", wrapped))
                    elif name == "after_layer39_experts":
                        saved = providers[39].forward
                        def wrapped(*a, **k): saved(*a, **k); fail()
                        inject.enter_context(patch.object(providers[39], "forward", wrapped))
                    try: engine.step(np.array([[TOKENS[1]]], np.int64), observe=observe if "sampling" in name else None)
                    except error as caught: caught_name = type(caught).__name__
                    else: raise RuntimeError("Injected whole-model failure was not raised")
                if not triggered: raise RuntimeError("The intended failure point was not reached")
                cancelled = policy.cancelled
                if name == "cancel_after_sampling":
                    if not cancelled: raise RuntimeError("The runtime was never cancelled")
                    policy.resume()
                restored = native_state(engine)
                if state_digest(restored) != before or engine.transaction.pending:
                    raise RuntimeError("Whole-model retry did not restore every state owner")
                faults.append({"name": name, "status": "pass", "caught": caught_name, "injection_reached": triggered,
                    "runtime_cancelled_before_resume": cancelled, "no_output_returned": True, "all_states_restored": True,
                    "state_before_sha256": before, "state_after_sha256": state_digest(restored), "pending_recovery": False})
                if progress: progress({"fault": name, "status": "pass"})
            result = engine.step(np.array([[TOKENS[1]]], np.int64)); state = native_state(engine)
            rows.append({"meta": original["cases"][1]["meta"], "status": "pass", "state_sha256": state_digest(state),
                "sampled_ids": result["sampling"]["ids_i64"].tolist(),
                "boundaries": compare(original["cases"][1]["boundaries"], result, state)})
            if rows[-1]["state_sha256"] != original["cases"][1]["state_sha256"]: raise RuntimeError("Original retried state differs")
            transaction = engine.transaction.stats()
            if progress: progress({"native_position": 1, "status": "pass", "boundaries": len(rows[-1]["boundaries"])})
        report = {"schema_version": 1, "probe": "native_all40_backbone_complete_recheck", "status": "pass", "scope": scope(),
            "original_identity_sha256": original["identity_sha256"], "complete_goldens_sha256": original["complete_goldens_sha256"],
            "cases": rows, "faults": faults, "transaction_active": transaction, "transaction_cleanup": engine.transaction.stats(),
            "native_profiles": profiles, "native_frameworks": {n: {"installed": importlib.util.find_spec(n) is not None,
                "imported": n in __import__('sys').modules} for n in ("torch", "tilelang", "transformers", "sympy")},
            "weight_fetch_inside_forward": False, "weight_payloads_fetched_in_recheck": 0,
            "python_http_disabled_during_native_execution": True, "minimum_required_Engram_layers": [1, 14]}
        del result, state, arrays, head_arrays, engine
        return report
    finally: budget.release(budget.cap)


def validate_native(report, original, original_raw):
    validate_original(original)
    if (report.get("probe"), report.get("status"), report.get("scope"), report.get("reference_report_sha256"),
            report.get("original_identity_sha256"), report.get("complete_goldens_sha256")) != (
            "native_all40_backbone_complete_recheck", "pass", scope(), digest(original_raw), QUALIFIED_IDENTITY, QUALIFIED_GOLDENS):
        raise ValueError("Native backbone evidence is not bound to the complete sealed original")
    if report["native_frameworks"] != {n: {"installed": False, "imported": False} for n in ("torch", "tilelang", "transformers", "sympy")}:
        raise ValueError("Native full-backbone replay depends on reference frameworks")
    if len(report["cases"]) != len(original["cases"]): raise ValueError("Incomplete native backbone corpus")
    for a, e in zip(report["cases"], original["cases"]):
        if (a["meta"], a["status"], a["state_sha256"], a["sampled_ids"]) != (e["meta"], "pass", e["state_sha256"], e["sampled_ids"]):
            raise ValueError("Native full-model input, state or sampled IDs differ")
        if len(a["boundaries"]) != len(e["boundaries"]): raise ValueError("Incomplete native model boundaries")
        for actual, expected in zip(a["boundaries"], e["boundaries"]):
            if (any(actual.get(k) != expected[k] for k in ("check", "dtype", "shape", "elements", "sha256"))
                    or actual.get("status") != "pass" or actual.get("expected_sha256") != expected["sha256"]
                    or actual.get("actual_sha256") != expected["sha256"]):
                raise ValueError("Changed native typed/shaped model boundary")
    if [f["name"] for f in report["faults"]] != list(FAULTS): raise ValueError("Incomplete whole-model failure matrix")
    for f in report["faults"]:
        cancel = f["name"] == "cancel_after_sampling"
        if (f["status"] != "pass" or f["no_output_returned"] is not True or f["all_states_restored"] is not True
                or f["injection_reached"] is not True or f["runtime_cancelled_before_resume"] != cancel
                or f["caught"] != ("RuntimeCancelled" if cancel else "InjectedFailure") or f["pending_recovery"] is not False
                or f["state_before_sha256"] != original["cases"][0]["state_sha256"]
                or f["state_after_sha256"] != f["state_before_sha256"]):
            raise ValueError("Whole-model failure was not injected, caught or restored correctly")
    for name in ("transaction_active", "transaction_cleanup"):
        t = report[name]
        if ((t["position"], t["committed_tokens"], t["failed_attempts"]) != (2, 2, len(FAULTS))
                or t["pending_recovery_owners"] or t["pending_cleanup_owners"]):
            raise ValueError("Native whole-model transaction lost state ownership")
    if report["transaction_cleanup"]["host_reserved_bytes"]: raise ValueError("Closed model retained transaction admission")
    if set(report["native_profiles"]) != set(map(str, LAYERS)): raise ValueError("Missing actual native per-layer profiles")
    if (report["weight_fetch_inside_forward"] is not False or original["weight_payloads_fetched_in_reference"]
            or report["weight_payloads_fetched_in_recheck"] or report["python_http_disabled_during_native_execution"] is not True):
        raise ValueError("Weight preparation was counted as independent model execution")
    for r, cap in ((report, 64 * GiB), (original, 32 * GiB)):
        m = r["memory_after_policy_close"]
        if (m["host_cap_bytes"] != cap or m["available_floor_bytes"] != 32 * GiB or m["charged_host_peak_bytes"] > cap
                or m["minimum_observed_available_bytes"] < 32 * GiB or m["charged_host_bytes"] or m["pinned_bytes"]
                or any(m["device_used_bytes"].values()) or m["live_leases"] or m["live_reclaimers"] or m["inflight_operations"]
                or m["monitor_thread_stopped"] is not True or m["reservation_is_RSS_or_OS_limit"] is not False):
            raise ValueError("Full-backbone memory floor, admission or cleanup evidence differs")
    if (original["minimum_sampled_reference_device_free_bytes"] < 2 * GiB
            or original["reference_torch_device_allocated_baseline_bytes"] != original["reference_torch_device_allocated_after_cleanup_bytes"]):
        raise ValueError("Original framework GPU reserve or cleanup differs")
    return True
