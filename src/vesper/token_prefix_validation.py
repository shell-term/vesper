"""Complete typed prefix goldens and native whole-token refusal/retry evidence."""

from contextlib import ExitStack
import importlib.util
from pathlib import Path
from unittest.mock import patch

import numpy as np

from .attention_validation import array_digest
from .checkpoint import MODEL, REVISION
from .connected import ConnectedMoE
from .model_artifacts import canonical, digest
from .output import NativeOutput
from .output_assets import load as load_head
from .resources import MemoryBudget, GiB
from .token_prefix import LAYERS, TOKENS
from .token_prefix_runtime import load_arrays, moe_files, open_prefix
from .topology_validation import flatten, state_digest

QUALIFIED_IDENTITY = "9c982d7b7f04ab3023bfaefb8ddca56b6300a852256ed7cf7590cf465faa5317"
QUALIFIED_GOLDENS = "78dc3767fc0438bb8565a811e5921402624dab0cacbbed1772abfecac3b897e5"
FAULTS = ("after_window0", "after_Engram1", "after_window1", "after_block3", "head_readback",
          "after_sampling", "cancel_after_sampling", "window1_recovery_transfer", "publication_floor")


def boundary_values(result, state):
    return flatten({"result": result, "state": state})


def capture(result, state):
    return [{"check": name, "dtype": str(a.dtype), "shape": list(a.shape), "elements": int(a.size),
        "sha256": array_digest(a)} for name, a in sorted(boundary_values(result, state).items())]


def compare(expected, result, state):
    values = boundary_values(result, state)
    if set(values) != {c["check"] for c in expected} or len(expected) != len(values):
        raise ValueError("Incomplete or additional token-prefix boundary")
    checks = []
    for c in expected:
        a = values[c["check"]]; sha = array_digest(a)
        passed = (str(a.dtype), list(a.shape), a.size, sha) == (c["dtype"], c["shape"], c["elements"], c["sha256"])
        checks.append({**c, "expected_sha256": c["sha256"], "actual_sha256": sha, "status": "pass" if passed else "fail"})
    if any(c["status"] != "pass" for c in checks):
        names = [c["check"] for c in checks if c["status"] != "pass"]
        raise RuntimeError("Original token-prefix boundary differs: " + ", ".join(names[:8]))
    return checks


def native_state(prefix):
    return {"window0": prefix.windows[0].kv.inspect()["cache_bf16"],
        "window1": prefix.windows[1].kv.inspect()["cache_bf16"],
        "sharing": prefix.sharing.inspect()["state"], "history": prefix.transaction.history.inspect()["history"],
        "position": prefix.position, "rng_offset": prefix.transaction.sampler.offset}


def finalize(prefix, head, norm_weight, result):
    hc = prefix.hc[3].pre(result["output_bf16"], result["next_pre_mix_fp32"])
    norm = prefix.norm.norm(hc["output_bf16"], norm_weight)
    result["probe_head"] = {**{"pre_" + n: a for n, a in hc.items()},
        **{"norm_" + n: a for n, a in norm.items()}, **head.project(norm["output_bf16"][:, -1].copy())}
    result["probe_sampling"] = prefix.transaction.sampler.sample(result["probe_head"]["logits_fp32"], .7)
    return result


def scope():
    return {"model": MODEL, "revision": REVISION, "layers": list(LAYERS), "tokens": list(TOKENS),
        "batch": 1, "queries_per_call": 1, "rank": 1, "text_only": True, "head_attached_after_layer": 3,
        "head_is_diagnostic_only": True, "full_model_execution_run": False, "text_generation_available": False,
        "full_model_quality": "not_run", "end_to_end_speed": "not_measured", "full_checkpoint_downloaded": False}


def identity(report):
    return canonical({k: report[k] for k in ("scope", "fixtures", "additional_identities", "output_fixture", "reference_methods", "expert_plan", "original_controls", "negative_controls", "original_grouped_conversion")})


def goldens(report):
    return canonical(report["cases"])


def validate_original(report, *, require_qualified=True):
    if (report.get("probe"), report.get("status"), report.get("scope")) != ("original_connected_token_prefix_goldens", "pass", scope()):
        raise ValueError("Original token-prefix evidence scope differs")
    if [c["meta"] for c in report["cases"]] != [{"position": i, "token_id": t} for i, t in enumerate(TOKENS)]:
        raise ValueError("Original token-prefix corpus differs")
    if set(report["expert_plan"]) != {str(l) for l in LAYERS}:
        raise ValueError("Original prefix expert owners differ")
    for layer in LAYERS:
        plan = report["expert_plan"][str(layer)]; ids = plan["ids"]
        if (not 6 <= len(ids) <= 12 or ids != sorted(set(ids)) or any(type(e) is not int or not 0 <= e < 384 for e in ids)
                or plan["store"] != f"token-prefix-stores/layer{layer}/experts-{len(ids)}"):
            raise ValueError("Original prefix store is not the bounded registered relative path")
    for case in report["cases"]:
        checks = case["boundaries"]
        if len(checks) != len({c["check"] for c in checks}) or not checks:
            raise ValueError("Missing or duplicate original prefix boundaries")
        if any(c["elements"] != int(np.prod(c["shape"])) or len(c["sha256"]) != 64 for c in checks):
            raise ValueError("Original prefix typed boundary differs")
    if require_qualified and (QUALIFIED_IDENTITY is None or identity(report) != QUALIFIED_IDENTITY or goldens(report) != QUALIFIED_GOLDENS):
        raise ValueError("Unqualified original token-prefix identity or complete goldens")
    if report.get("identity_sha256") != identity(report) or report.get("complete_goldens_sha256") != goldens(report):
        raise ValueError("Recorded token-prefix identity does not bind its complete evidence")
    return report


def validate_native(report, original, original_raw):
    validate_original(original)
    if (report.get("probe"), report.get("status"), report.get("scope"), report.get("original_identity_sha256"),
            report.get("complete_goldens_sha256"), report.get("reference_report_sha256")) != (
            "native_connected_token_prefix_complete_recheck", "pass", scope(), QUALIFIED_IDENTITY, QUALIFIED_GOLDENS, digest(original_raw)):
        raise ValueError("Native token-prefix receipt does not bind the sealed original report")
    if report.get("native_frameworks") != {n: {"installed": False, "imported": False} for n in ("torch", "tilelang", "transformers", "sympy")}:
        raise ValueError("Native token-prefix replay is not framework-independent")
    def checks(actual, expected):
        if len(actual) != len(expected): raise ValueError("Incomplete native token-prefix comparisons")
        for a, e in zip(actual, expected):
            if (any(a.get(k) != e[k] for k in ("check", "dtype", "shape", "elements", "sha256"))
                    or a.get("status") != "pass" or a.get("expected_sha256") != e["sha256"] or a.get("actual_sha256") != e["sha256"]):
                raise ValueError("Native typed/shaped token-prefix comparison differs")
    if len(report["cases"]) != len(original["cases"]): raise ValueError("Native token-prefix corpus is incomplete")
    for a, e in zip(report["cases"], original["cases"]):
        if a["meta"] != e["meta"] or a["state_sha256"] != e["state_sha256"] or a["status"] != "pass":
            raise ValueError("Native token-prefix input/state differs")
        checks(a["boundaries"], e["boundaries"])
    if [f["condition"] for f in report["faults"]] != list(FAULTS): raise ValueError("Incomplete whole-token fault matrix")
    for f in report["faults"]:
        if (f["status"] != "pass" or not f["caught"] or f["partial_token_returned"] is not False
                or f["state_before_sha256"] != original["cases"][0]["state_sha256"]
                or f["restored_state_sha256"] != f["state_before_sha256"] or f["retry_state_sha256"] != original["cases"][1]["state_sha256"]
                or f["transaction"]["position"] != 2 or f["transaction"]["committed_tokens"] != 2
                or f["transaction"]["pending_recovery_owners"] or f["transaction"]["pending_cleanup_owners"]):
            raise ValueError("Whole-token failure/recovery evidence differs")
        pending = f["recovery_pending_before_disarm"]
        expected = (["window1"] if f["condition"] == "window1_recovery_transfer" else
                    ["window0", "window1", "sharing2_3", "__history__", "__sampler__"] if f["condition"] == "publication_floor" else [])
        if pending != expected: raise ValueError("Blocked recovery lost snapshot ownership")
        checks(f["retry_boundaries"], original["cases"][1]["boundaries"])
    if report["head_cleanup"]["device_owned_bytes"] or report["head_cleanup"]["host_reserved_bytes"]:
        raise ValueError("Native prefix head retained owned buffers")
    checkpoint = report["outer_KV_checkpoint"]
    if checkpoint["status"] != "pass" or [c["batch"] for c in checkpoint["cases"]] != [1, 2]:
        raise ValueError("Missing actual outer KV ring checkpoint probe")
    for c in checkpoint["cases"]:
        if (c["status"] != "pass" or (c["checkpoint_position"], c["advanced_position"], c["restored_position"]) != (128, 130, 128)
                or c["original_ring_sha256"] != c["restored_ring_sha256"] or c["readback_failure_kept_active_state"] is not True
                or c["host_checkpoint_bytes"] != c["batch"] * 128 * 512 * 2
                or c["cleanup"]["device_owned_bytes"] or c["cleanup"]["host_reserved_bytes"] or c["cleanup"]["outer_host_checkpoint_bytes"]):
            raise ValueError("Outer KV checkpoint bits/cleanup differ")
    for r in (report, original):
        m = r["memory_after_policy_close"]
        if (m["host_cap_bytes"] != 24 * GiB or m["available_floor_bytes"] != 32 * GiB or m["charged_host_peak_bytes"] > 24 * GiB
                or m["minimum_observed_available_bytes"] < 32 * GiB or m["charged_host_bytes"] or m["pinned_bytes"]
                or any(m["device_used_bytes"].values()) or m["live_leases"] or m["live_reclaimers"] or m["inflight_operations"]
                or m["monitor_thread_stopped"] is not True or m["reservation_is_RSS_or_OS_limit"] is not False):
            raise ValueError("Token prefix memory/cleanup evidence differs")
    if report["weight_payloads_fetched_in_recheck"] or original["weight_payloads_fetched_in_reference"]:
        raise ValueError("Artifact preparation leaked into the forward/replay path")
    return True


def run_native(root, original, policy, progress=None, *, require_qualified=True):
    validate_original(original, require_qualified=require_qualified)
    root = Path(root)
    budget = MemoryBudget(cap=4 * GiB, owner="token_prefix_fixture_and_diagnostics"); budget.acquire(budget.cap)
    try:
        arrays, embeddings, fixtures = load_arrays(root)
        head_arrays, output_fixture = load_head(root / "output-fixture")
        if fixtures != original["fixtures"] or output_fixture != original["output_fixture"]:
            raise ValueError("Native prefix original fixture binding differs")
        with ExitStack() as outer:
            head = outer.enter_context(NativeOutput(head_arrays["head.weight"]))
            moe = {l: outer.enter_context(ConnectedMoE(root / original["expert_plan"][str(l)]["store"],
                *moe_files(root, l), layer=l, profile="token_prefix")) for l in LAYERS}
            rows, faults = [], []
            with ExitStack() as stack:
                prefix = open_prefix(stack, root, arrays, embeddings, moe)
                if prefix.probe_identities != original["additional_identities"]:
                    raise ValueError("Native prefix tokenizer/Engram bindings differ")
                native_profiles = {str(l): {"projection_cubin_sha256": prefix.probe_operators["projections"][l].cubin_sha256,
                    "projection_compiler": prefix.probe_operators["projections"][l].compiler,
                    "grouped_converted_sha256": prefix.probe_operators["projections"][l].converted_sha256,
                    "HC_cubin_sha256": prefix.hc[l].cubin_sha256} for l in LAYERS}
                if {l: p["grouped_converted_sha256"] for l, p in native_profiles.items()} != original["original_grouped_conversion"]:
                    raise RuntimeError("Native prefix original grouped conversion differs")
                for i, token in enumerate(TOKENS):
                    result = prefix.step(np.array([[token]], np.int64), finalize=lambda r: finalize(prefix, head, head_arrays["norm.weight"], r))
                    state = native_state(prefix)
                    if state_digest(state) != original["cases"][i]["state_sha256"]:
                        raise RuntimeError("Native prefix committed state differs")
                    checks = compare(original["cases"][i]["boundaries"], result, state)
                    rows.append({"meta": original["cases"][i]["meta"], "status": "pass", "boundaries": checks,
                        "state_sha256": state_digest(state)})
                    if progress: progress({"native_prefix_position": i, "boundaries": len(checks), "status": "pass"})
            for mode in FAULTS:
                with ExitStack() as stack:
                    prefix = open_prefix(stack, root, arrays, embeddings, moe)
                    finish = lambda r: finalize(prefix, head, head_arrays["norm.weight"], r)
                    prefix.step(np.array([[TOKENS[0]]], np.int64), finalize=finish)
                    before = state_digest(native_state(prefix))
                    def after_call(original_call):
                        def failed(*args, **kwargs):
                            original_call(*args, **kwargs)
                            raise OSError("Injected late prefix component failure")
                        return failed
                    caught = None
                    with ExitStack() as injected:
                        if mode in ("after_window0", "after_window1"):
                            target = prefix.windows[0 if mode.endswith("0") else 1]
                            injected.enter_context(patch.object(target, "step", after_call(target.step)))
                        elif mode == "after_Engram1":
                            injected.enter_context(patch.object(prefix.engram, "forward", after_call(prefix.engram.forward)))
                        elif mode == "after_block3":
                            injected.enter_context(patch.object(moe[3], "forward", after_call(moe[3].forward)))
                        elif mode == "head_readback":
                            original_download = head.device.download
                            def download(pointer, shape, dtype):
                                if tuple(shape) == (1, 129280) and dtype == np.float32:
                                    raise OSError("Injected head logits download failure")
                                return original_download(pointer, shape, dtype)
                            injected.enter_context(patch.object(head.device, "download", download))
                        elif mode == "window1_recovery_transfer":
                            injected.enter_context(patch.object(prefix.windows[1].kv, "restore", side_effect=OSError("Injected outer KV restore transfer failure")))
                        def fail_finalizer(result):
                            result = finish(result)
                            if mode in ("after_sampling", "window1_recovery_transfer"):
                                raise OSError("Injected failure after diagnostic sampling")
                            if mode == "cancel_after_sampling":
                                policy.cancel("Injected cancellation after diagnostic sampling")
                            if mode == "publication_floor":
                                injected.enter_context(patch("vesper.resources.physical_memory", return_value={"available": 16 * GiB}))
                            return result
                        try:
                            prefix.step(np.array([[TOKENS[1]]], np.int64), finalize=fail_finalizer)
                        except BaseException as error:
                            caught = type(error).__name__
                        else:
                            raise RuntimeError("Injected prefix failure returned a partial token")
                        pending = list(prefix.transaction.pending)
                        if pending:
                            try:
                                prefix.step(np.array([[TOKENS[1]]], np.int64), finalize=finish)
                            except RuntimeError:
                                pass
                            else:
                                raise RuntimeError("Pending whole-token recovery allowed another call")
                    if mode == "cancel_after_sampling":
                        policy.resume()
                    if prefix.transaction.pending:
                        prefix.recover()
                    restored = state_digest(native_state(prefix))
                    if restored != before:
                        raise RuntimeError("Whole-token failure changed committed state")
                    result = prefix.step(np.array([[TOKENS[1]]], np.int64), finalize=finish)
                    state = native_state(prefix)
                    checks = compare(original["cases"][1]["boundaries"], result, state)
                    if state_digest(state) != original["cases"][1]["state_sha256"]:
                        raise RuntimeError("Whole-token retry changed original state")
                    faults.append({"condition": mode, "status": "pass", "caught": caught, "partial_token_returned": False,
                        "recovery_pending_before_disarm": pending, "state_before_sha256": before, "restored_state_sha256": restored,
                        "retry_state_sha256": state_digest(state), "retry_boundaries": checks, "transaction": prefix.transaction.stats()})
                    if progress: progress({"whole_token_fault": mode, "retry_boundaries": len(checks), "status": "pass"})
            head_stats = head.stats()
        return {"schema_version": 1, "probe": "native_connected_token_prefix_complete_recheck", "status": "pass", "scope": scope(),
            "original_identity_sha256": identity(original), "complete_goldens_sha256": goldens(original), "cases": rows, "faults": faults,
            "native_profiles": native_profiles,
            "head_active": head_stats, "head_cleanup": head.stats(),
            "native_frameworks": {n: {"installed": importlib.util.find_spec(n) is not None, "imported": n in __import__('sys').modules}
                for n in ("torch", "tilelang", "transformers", "sympy")}, "weight_payloads_fetched_in_recheck": 0}
    finally:
        budget.release(budget.cap)
