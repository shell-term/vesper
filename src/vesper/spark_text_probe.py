"""Explicit local-only Spark text preparation and short functional generation.

This is an S1 functional request, not a lowered benchmark or a quality gate.
Chat serialization, serial prompt ingestion, token feedback and EOS handling
reuse the original-format adapters. No checkpoint acquisition, route replay,
automatic retry, fallback token or giant cross-step diagnostic archive exists.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import time
import traceback
import weakref

import numpy as np

from .checkpoint import MODEL, REVISION
from .generation import TextGeneration
from .memory import RuntimeMemory, process_memory
from .messages import EOS, encode_messages, parse_completion
from .resources import MemoryBudget, MiB, GiB
from .sequence import MAX_CALLS
from .spark_text_request import (SparkTextLimits, SparkTextRequest, MAX_COMPLETION_CHARS,
                                 MAX_USER_PROMPT_CHARS)
from .spark_attention import load_candidate
from .spark_backbone_runtime import (open_spark_backbone, spark_memory_policy, TARGET,
    SPARK_SHARED_CAP, SPARK_ASSET_CAP, SPARK_DEVICE_CAP, SPARK_COMPONENT_ALLOWANCE)
from .spark_probe import Driver, Unsupported, host_memory, require_capacity
from .spark_tensor_views import SparkModelAssets
from .spark_attention_validation import check_receipt
from .tokenizer import PinnedTokenizer, TOKENIZERS_VERSION

SCHEMA = "vesper.spark-text-probe.v1"
DEFAULT_PROMPT = "What is 2 + 2? Answer with the number only."
HOST_ADMISSION = 128 * MiB
STEP_DIAGNOSTIC_ADMISSION = 512 * MiB
REPORT_LIMIT = 32 * MiB


class ProgressFile:
    """Small atomic live snapshots, separate from immutable final evidence."""
    def __init__(self, workspace, run_id):
        self.workspace, self.run_id = workspace, run_id
        self.relative = f"s1/text/{run_id}/progress.json"
        self.temporary = f"s1/text/{run_id}/progress.tmp"
        self.context = workspace.context(run_id)
        self.started, self.written = time.perf_counter(), 0.
        self.path = workspace.artifact_path(self.relative, create_parent=True)
        if self.path.exists():
            raise FileExistsError("Refusing to replace an earlier text run's progress")

    def __call__(self, value):
        now = time.perf_counter()
        force = value.get("phase") in ("model_call_complete", "layer_complete", "finished")
        if not force and now - self.written < .5:
            return
        payload = {"schema":"vesper.spark-text-progress.v1", "run_id":self.run_id,
            "observed_unix":time.time(), "elapsed_seconds":now-self.started,
            "workspace_context":self.context, **value}
        raw = (json.dumps(payload, ensure_ascii=False)+"\n").encode()
        if len(raw) > 64 * 1024:
            raise ValueError("Text progress exceeded its explicit bound")
        temporary = self.workspace.artifact_path(self.temporary, create_parent=True)
        self.path = self.workspace.artifact_path(self.relative)
        owned = None
        try:
            with temporary.open("xb") as stream:
                info = os.fstat(stream.fileno())
                owned = info.st_dev, info.st_ino
                stream.write(raw)
            os.replace(temporary, self.path)
        finally:
            if owned is not None:
                try:
                    info = temporary.lstat()
                    if not temporary.is_symlink() and (info.st_dev,info.st_ino) == owned:
                        temporary.unlink()
                except FileNotFoundError:
                    pass
        self.written = now


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _guard(workspace, run_id):
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", run_id):
        raise ValueError("Expected a simple lowercase run identity")
    workspace.check()
    workspace.check_runtime_module(__file__, "src/vesper/spark_text_probe.py")
    from . import spark_text_request as request_module
    workspace.check_runtime_module(request_module.__file__, "src/vesper/spark_text_request.py")
    from . import workspace as workspace_module
    workspace.check_runtime_module(workspace_module.__file__, "src/vesper/workspace.py")
    context = workspace.context(run_id)
    if tuple(context[k] for k in ("platform", "profile", "model", "revision")) != (
            "dgx-spark", "native", MODEL, REVISION):
        raise ValueError("Text execution requires the pinned dgx-spark/native workspace")
    return context


def _validate(mode, prompt, max_new_tokens, temperature, seed, limits):
    if mode not in ("prepare", "generate"):
        raise ValueError("Unknown local Spark text mode")
    if (type(prompt) is not str or not prompt or len(prompt) > MAX_USER_PROMPT_CHARS
            or type(temperature) not in (float, int) or not math.isfinite(temperature) or not 0 <= temperature <= 1000
            or type(seed) is not int or not 0 <= seed < 2**64):
        raise ValueError("Expected bounded text, temperature in [0,1000] and uint64 seed")
    limits.admission(max_new_tokens)


def _array_hash(value):
    if not isinstance(value, np.ndarray) or not value.flags.c_contiguous:
        raise ValueError("Expected a contiguous native diagnostic array")
    return {"shape": list(value.shape), "dtype": value.dtype.str,
            "bytes": value.nbytes, "sha256": _sha(memoryview(value).cast("B"))}


def _diagnostic_bytes(result):
    """Bound the returned arrays while avoiding double-counting shared owners."""
    visited, owners, total = set(), set(), 0
    pending = [result]
    while pending:
        item = pending.pop()
        if id(item) in visited:
            continue
        visited.add(id(item))
        if len(visited) > 100000:
            raise ValueError("Native step diagnostic structure exceeds its bound")
        if isinstance(item, np.ndarray):
            owner = item
            while isinstance(owner.base, np.ndarray):
                owner = owner.base
            if id(owner) not in owners:
                owners.add(id(owner)); total += owner.nbytes
                if total > STEP_DIAGNOSTIC_ADMISSION:
                    raise MemoryError("Returned native step diagnostics exceed their explicit allowance")
        elif isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, (tuple, list)):
            pending.extend(item)
    return total


class RouteObserver:
    """Capture scalar original-route I/O metadata before each MoE owner closes."""
    def __init__(self, engine):
        self.pool = engine.operator_pool
        self.original = self.pool.moe_factory
        self.current = []

    def __enter__(self):
        original, observer = self.original, self
        def factory(layer):
            owner = original(layer)
            owner_ref, forward_ref = weakref.ref(owner), weakref.WeakMethod(owner.forward)
            def observed(*args, **kwargs):
                call = forward_ref()
                if call is None:
                    raise RuntimeError("Original MoE diagnostic owner expired")
                try:
                    return call(*args, **kwargs)
                finally:
                    actual = owner_ref()
                    if actual is not None:
                        observer.capture(layer, actual)
            owner.forward = observed
            return owner
        self.pool.moe_factory = factory
        return self

    def capture(self, layer, owner):
        if len(self.current) >= 40:
            raise ValueError("A serial native call returned excess original MoE routes")
        attempt = getattr(owner, "last_attempt", None) or {}
        row = {"layer": layer, "status": attempt.get("status", "unavailable")}
        for key in ("indices", "completed_expert_ids"):
            values = attempt.get(key)
            if values is not None:
                if (not isinstance(values, list) or len(values) > 6
                        or any(type(v) is not int or not 0 <= v < 384 for v in values)):
                    raise ValueError("Invalid original route metadata")
                row[key] = values.copy()
        for key in ("store_direct_read_calls", "store_direct_read_bytes", "wall_ms"):
            value = attempt.get(key)
            if type(value) in (int, float) and math.isfinite(value) and value >= 0:
                row[key] = value
        store = getattr(getattr(owner, "experts", None), "store", None)
        if store is not None:
            stats = store.stats()
            row["source_io"] = {k: stats[k] for k in ("logical_expert_reads", "logical_expert_bytes",
                "physical_data_bytes_read", "integrity_index_bytes_read", "network_payload_bytes_in_forward")
                if k in stats and type(stats[k]) is int and stats[k] >= 0}
            row["source_setup_timings"] = stats.get("source_setup_timings", {})
        row["opening_phase_seconds"] = dict(getattr(owner, "opening_phase_seconds", {}))
        self.current.append(row)

    def __exit__(self, *_):
        self.pool.moe_factory = self.original


def _step_summary(meta, result, routes, seconds):
    diagnostics = _diagnostic_bytes(result)
    layers = {}
    if [row.get("layer") for row in routes] != list(range(40)):
        raise ValueError("Successful native token must execute all forty original MoE layers")
    for layer in range(40):
        part = result[f"layer{layer}"]
        indices, weights = part["moe_gate.indices"], part["moe_gate.weights"]
        if (not isinstance(indices, np.ndarray) or indices.shape != (6,) or indices.dtype not in (np.int32, np.int64)
                or len(set(map(int, indices))) != 6 or np.any(indices < 0) or np.any(indices >= 384)
                or not isinstance(weights, np.ndarray) or weights.dtype != np.float32 or weights.shape != (6,)
                or not np.isfinite(weights).all()):
            raise ValueError("Native token returned an invalid original six-expert route")
        if routes[layer].get("indices") != indices.tolist() or routes[layer].get("status") != "pass":
            raise ValueError("Returned native route differs from the actual original gate attempt")
        io = routes[layer].get("source_io", {})
        if (set(io) != {"logical_expert_reads", "logical_expert_bytes", "physical_data_bytes_read",
                        "integrity_index_bytes_read", "network_payload_bytes_in_forward"}
                or io["network_payload_bytes_in_forward"] != 0):
            raise ValueError("Successful native route requires complete offline source-I/O evidence")
        layers[str(layer)] = {"indices": indices.tolist(), "weight_bits": weights.view(np.uint32).tolist(),
                              "output_bf16": _array_hash(part["output_bf16"])}
    logits = result["head"]["logits_fp32"]
    if logits.shape != (1, 129280) or logits.dtype != np.float32 or not np.isfinite(logits).all():
        raise ValueError("Native text requires finite full-vocabulary logits")
    return {**meta, "model_call_wall_seconds_including_native_diagnostics": seconds, "returned_array_owner_bytes": diagnostics,
            "layers": layers, "route_io": routes.copy(), "logits_fp32": _array_hash(logits),
            "engrams": {str(layer):_array_hash(result[f"engram{layer}"]["output_bf16"]) for layer in (1,14)},
            "output_bf16": _array_hash(result["output_bf16"])}


def _warning(workspace, run_id, error, stage, initial):
    """Private bounded diagnostics; public reports never include exception text."""
    relative = f"s1/text/{run_id}/failure.json"
    try:
        path = workspace.artifact_path(relative, create_parent=True)
        raw = (json.dumps({"failure_type": type(error).__name__, "stage": stage,
            "message": str(error)[:8192], "traceback": traceback.format_exc()[-32768:],
            "workspace_context": initial}, ensure_ascii=False) + "\n").encode()
        if len(raw) > 65536:
            raise ValueError("Private warning exceeds its bound")
        with path.open("xb") as stream:
            stream.write(raw)
        return {"saved": True, "artifact": relative, "sha256": _sha(raw), "bytes": len(raw)}
    except Exception:
        return {"saved": False, "reason": "Private warning could not be saved exclusively"}


def _failed(report, workspace, run_id, error, stage, initial):
    if report.get("status") in ("failed", "cancelled") and "failure_type" in report:
        report.setdefault("secondary_failures", []).append({"failure_type": type(error).__name__, "stage": stage})
        return
    report.update(status="cancelled" if isinstance(error, KeyboardInterrupt) else "failed", functional_acceptance=False, failure_type=type(error).__name__,
                  failure_stage=stage, reason="Local Spark functional execution could not complete")
    if "private_failure" not in report:
        report["private_failure"] = _warning(workspace, run_id, error, stage, initial)


def _finish_completion(report, session, tokenizer):
    summary = session.summary()
    report["completion"] = summary
    serialized = tokenizer.decode(list(session.generated))
    report["completion_serialized_including_eos"] = serialized
    report["eos_response_complete"] = summary["stop_reason"] == "eos"
    report["parsed_completion"] = None
    if report["eos_response_complete"]:
        parsed = parse_completion(serialized, thinking_mode=report["request"]["thinking_mode"])
        report["parsed_completion"] = parsed
        if not parsed["content"].strip() or parsed["tool_calls"]:
            raise ValueError("This text-only functional request requires a nonempty plain response")
    if report["request"]["user_prompt"] == DEFAULT_PROMPT:
        report["default_prompt_answer_matches_four"] = bool(
            report["parsed_completion"] and report["parsed_completion"]["content"].strip() == "4")


def _check_inventory_receipt(inventory, proof):
    """Compare the public inventory view with the qualified Driver view."""
    device = {"ordinal": inventory["ordinal"], "name": inventory["name"],
        "compute_capability": ".".join(map(str, inventory["compute_capability"])),
        "total_bytes": inventory["reported_cuda_total_bytes"]}
    if (inventory["device_count"] != 1 or device != proof["observed_device"]
            or inventory["driver_api_version"] != proof["driver_api_version"]):
        raise ValueError("Observed generation hardware or driver differs from the Attention receipt")


def _check_engine_receipt(engine, proof):
    """Bind every sparse Attention consumer before any model forward call."""
    attention = engine.sharing.attention
    inventory = attention.driver.inventory()
    if (engine.windows[0].attention is not attention or engine.windows[1].attention is not attention
            or attention.manifest_sha256 != proof["bundle_manifest_sha256"]
            or attention.device_info != proof["observed_device"]
            or inventory["devices"] != [proof["observed_device"]]
            or inventory["driver_api_version"] != proof["driver_api_version"]):
        raise ValueError("Opened generation Attention differs from the qualified bundle or runtime")


def collect(mode, workspace, *, run_id, metadata="s1/import-source", originals="s1/originals",
            integrity="s1/original-integrity", bundle="s1/attention-candidate", fixtures="s1/moe-original",
            attention_reference_result="s1/attention/reference-r02.json",
            attention_receipt_result="s1/attention/native-validation-r01.json",
            prompt=DEFAULT_PROMPT, max_new_tokens=64, temperature=0., seed=2026, progress=None, max_model_calls=MAX_CALLS,
            reuse_profile="baseline", thinking_mode="chat", reasoning_effort=None,
            max_input_tokens=MAX_CALLS, context_tokens=MAX_CALLS, max_duration_seconds=None):
    """Return an unstamped report after resource cleanup; the CLI writes it."""
    initial = _guard(workspace, run_id)
    limits = SparkTextLimits(context_tokens, max_input_tokens, max_model_calls, max_duration_seconds)
    _validate(mode, prompt, max_new_tokens, temperature, seed, limits)
    if reuse_profile not in ("baseline", "s2"):
        raise ValueError("Expected baseline or s2 native preparation reuse")
    if (thinking_mode not in ("chat", "thinking") or
            (reasoning_effort is not None and (type(reasoning_effort) is not int or not 1 <= reasoning_effort <= 100))):
        raise ValueError("Expected chat/thinking and an optional reasoning effort in [1,100]")
    if thinking_mode == "chat" and reasoning_effort is not None:
        raise ValueError("Reasoning effort applies only to thinking mode")
    if mode == "generate" and any(not isinstance(value, str) or not value for value in
                                   (attention_reference_result, attention_receipt_result)):
        raise ValueError("Generation requires both current Attention reference and native validation results")
    paths = {name: workspace.artifact_path(value) for name, value in
             (("metadata", metadata), ("originals", originals), ("integrity", integrity), ("bundle", bundle), ("fixtures", fixtures))}
    messages = [{"role": "user", "content": prompt}]
    serialized = encode_messages(messages, thinking_mode=thinking_mode, reasoning_effort=reasoning_effort)
    host_admission = HOST_ADMISSION * limits.metadata_multiplier
    report_limit = REPORT_LIMIT * limits.metadata_multiplier
    report = {"schema": SCHEMA, "mode": mode, "status": "unsupported", "model": MODEL, "revision": REVISION,
        "execution_target": TARGET,
        "scope": ("short native all40 text functional request" if context_tokens == MAX_CALLS else
                  "bounded native all40 text functional request"), "sequence_scope": limits.scope(),
        "functional_acceptance": False, "eos_response_complete": False, "full_model_quality_qualified": False,
        "original_kernel_parity_qualified": False, "general_performance_claim": False,
        "network_in_forward": False, "checkpoint_download_started": False, "automatic_retry": False,
        "request": {"messages": messages, "serialized_chat": serialized, "user_prompt": prompt,
            "thinking_mode": thinking_mode, "thinking_enabled": thinking_mode == "thinking",
            "reasoning_effort": (75 if reasoning_effort is None else reasoning_effort) if thinking_mode == "thinking" else None,
            "temperature": float(temperature), "seed": seed,
            "max_new_tokens": max_new_tokens, "maximum_model_calls": max_model_calls,
            "requested_limits": limits.admission(max_new_tokens), "admitted_limits": None,
            "purpose": "explicit bounded functional request; no benchmark condition changed"},
        "artifact_inputs": {"metadata": metadata, "originals": originals, "integrity": integrity,
                            "bundle": bundle, "fixtures": fixtures},
        "limits": {"shared_host_cap_bytes": SPARK_SHARED_CAP, "physical_available_floor_bytes": 32 * GiB,
            "asset_host_cap_bytes": SPARK_ASSET_CAP, "device_cap_bytes": SPARK_DEVICE_CAP,
            "conservative_component_allowance_bytes": SPARK_COMPONENT_ALLOWANCE,
            "probe_host_admission_bytes": host_admission, "step_diagnostic_admission_bytes": STEP_DIAGNOSTIC_ADMISSION,
            "report_limit_bytes": report_limit, "reservation_is_RSS_or_OS_limit": False},
        "memory_observations": [], "steps": [], "owned_resources_closed": False, "memory_runtime_closed": False,
        "attention_qualification": None}
    report["preparation_reuse_profile"] = reuse_profile
    report["timing_marks_seconds_from_start"] = {}
    memory, admission = None, None
    engine = assets = tokenizer = session = result = None
    stage, started = "platform", time.perf_counter()
    class DurationBudgetExceeded(Exception):
        pass
    def duration_exceeded():
        return max_duration_seconds is not None and time.perf_counter() - started >= max_duration_seconds
    def check_duration():
        if duration_exceeded():
            raise DurationBudgetExceeded()
    def record_error(error):
        if isinstance(error, DurationBudgetExceeded):
            report.update(status="partial_duration_budget", duration_budget_exceeded=True,
                          duration_budget_stop_stage=stage)
            if memory is None:
                report.update(owned_resources_closed=True, memory_runtime_closed=True)
        else:
            _failed(report, workspace, run_id, error, stage, initial)
    def mark(name):
        report["timing_marks_seconds_from_start"][name] = time.perf_counter() - started
    def update(event=None):
        if progress is not None:
            value = {"status":report["status"], "stage":stage,
                "completed_model_calls":0 if session is None else session.position,
                "prompt_tokens":len(report["request"].get("prompt_ids", [])),
                "generated_tokens":0 if session is None else len(session.generated),
                "generated_ids":[] if session is None else list(session.generated), **(event or {})}
            progress(value)
    def check(pending=0):
        workspace.check()
        observation = host_memory()
        require_capacity(observation, pending)
        report["memory_observations"].append(observation)
        if memory is not None and memory.opened:
            memory.checkpoint()
        return observation
    try:
        if platform.system() != "Linux" or platform.machine().lower() not in ("aarch64", "arm64"):
            raise Unsupported("Spark text execution requires Linux AArch64")
        report["observed_platform"] = {"system": "Linux", "architecture": platform.machine().lower(),
                                      "single_physical_pool": True}
        check_duration()
        check(host_admission)
        if mode == "generate":
            # Authenticate both current source-bound reports and the immutable
            # bundle before assets, CUDA inventory, or any model operator opens.
            stage = "attention_qualification"
            proof = check_receipt(workspace, paths["bundle"], attention_reference_result, attention_receipt_result)
            report["artifact_inputs"].update(attention_reference_result=attention_reference_result,
                                            attention_receipt_result=attention_receipt_result)
            report["attention_qualification"] = {"status": proof["status"], "scope": proof["scope"],
                "bundle_manifest_sha256": proof["bundle_manifest_sha256"],
                "reference_result_sha256": proof["reference_result_sha256"],
                "native_receipt_result_sha256": proof["receipt_result_sha256"],
                "case_count": len(proof["cases"]), "fault_count": len(proof["faults"]),
                "full_model_quality_qualified": False}
        stage = "memory_runtime"
        check_duration()
        memory = RuntimeMemory(spark_memory_policy())
        with memory:
            admission = MemoryBudget(cap=host_admission + STEP_DIAGNOSTIC_ADMISSION, owner="spark_text_probe")
            admission.acquire(host_admission, category="diagnostics")
            try:
                with ExitStack() as stack:
                    stage = "assets"
                    mark("assets_open_start")
                    catalog = None
                    if mode == "generate" and reuse_profile == "s2":
                        from .spark_artifacts import AuthenticatedModelCatalog, CATALOG_REUSE_BYTES
                        catalog = stack.enter_context(AuthenticatedModelCatalog(paths["metadata"], paths["originals"],
                            paths["integrity"], MemoryBudget(cap=CATALOG_REUSE_BYTES, owner="spark_text_catalog"),
                            workspace=workspace))
                    reuse = {} if catalog is None else {"catalog": catalog}
                    assets = stack.enter_context(SparkModelAssets(paths["metadata"], paths["originals"], paths["integrity"],
                        workspace=workspace, fixtures=paths["fixtures"],
                        budget=MemoryBudget(cap=SPARK_ASSET_CAP, owner="spark_model_assets"), **reuse))
                    mark("assets_open_complete")
                    check_duration()
                    if mode == "prepare":
                        stage = "moe_preparation"
                        report["preparation"] = assets.prepare_moe_fixtures()
                        report["status"] = "local_moe_prepared"
                        report["cuda_inventory_requested"] = False
                    else:
                        check_duration()
                        stage = "prepared_local_assets"
                        for layer in range(40):
                            assets.moe_paths(layer)
                        manifest, binaries, manifest_sha = load_candidate(paths["bundle"])
                        if manifest_sha != proof["bundle_manifest_sha256"]:
                            raise ValueError("Attention bundle changed after receipt validation")
                        report["attention_bundle"] = {"manifest_sha256": manifest_sha,
                            "qualification": manifest["qualification"], "compute_capability": manifest["compute_capability"],
                            "nvrtc_version": manifest["nvrtc_version"],
                            "profiles": [{k: p[k] for k in ("heads", "dimension", "diagnostics", "cubin_sha256")}
                                         for p in manifest["profiles"]]}
                        del manifest, binaries
                        stage = "hardware_inventory"
                        report["observed_hardware"] = Driver().inventory()
                        _check_inventory_receipt(report["observed_hardware"], proof)
                        check()
                        stage = "tokenizer"
                        tokenizer = PinnedTokenizer(paths["metadata"])
                        eos = tokenizer.encode(EOS)
                        if len(eos) != 1 or tokenizer.decode(eos) != EOS:
                            raise ValueError("Original tokenizer EOS must be one preserved special token")
                        request = SparkTextRequest(serialized, max_new_tokens=max_new_tokens, eos_id=eos[0], limits=limits)
                        report["request"]["prompt_ids"] = list(request.encode(tokenizer))
                        report["request"]["eos_id"] = eos[0]
                        report["request"]["admitted_limits"] = {**limits.admission(max_new_tokens),
                            "actual_input_tokens": len(report["request"]["prompt_ids"]),
                            "maximum_required_model_calls": len(report["request"]["prompt_ids"]) + max_new_tokens - 1}
                        report["tokenizer"] = {"version": TOKENIZERS_VERSION, "vocabulary": len(tokenizer)}
                        stage = "model_open"
                        check_duration()
                        mark("model_open_start")
                        update({"phase":"model_open"})
                        reuse = {} if reuse_profile == "baseline" else {"reuse_profile": reuse_profile}
                        engine = open_spark_backbone(stack, assets, paths["bundle"], temperature=float(temperature), seed=seed,
                            progress=update if progress is not None else None, **reuse)
                        mark("model_open_complete")
                        stage = "model_attention_qualification"
                        _check_engine_receipt(engine, proof)
                        session = TextGeneration(tokenizer, engine.step, request)
                        report["tokenizer"]["compressed_identity"] = engine.tokenizer_identity
                        stage = "generation"
                        mark("generation_start")
                        report["status"] = "running"
                        with RouteObserver(engine) as observer:
                            while not session.done:
                                check_duration()
                                if session.position >= max_model_calls:
                                    break
                                if session.position >= limits.context_tokens:
                                    raise ValueError("Serial model calls exceeded the explicit functional scope")
                                check(STEP_DIAGNOSTIC_ADMISSION)
                                admission.acquire(STEP_DIAGNOSTIC_ADMISSION, category="diagnostics")
                                observer.current = []
                                step_start = time.perf_counter()
                                update({"phase":"model_call_start"})
                                try:
                                    meta, result = session.advance()
                                    if meta.get("selected_id") is not None and "first_selected_output" not in report["timing_marks_seconds_from_start"]:
                                        mark("first_selected_output")
                                    report["steps"].append(_step_summary(meta, result, observer.current,
                                        time.perf_counter() - step_start))
                                    update({"phase":"model_call_complete", "last_model_call_seconds":time.perf_counter()-step_start})
                                    if meta.get("selected_id") is not None and len(tokenizer.decode(list(session.generated))) > MAX_COMPLETION_CHARS:
                                        report["completion_character_budget_exceeded"] = True
                                        break
                                except Exception:
                                    report["failed_step"] = {"driver_position_after_failure": session.position,
                                        "route_io": observer.current.copy(),
                                        "model_call_wall_seconds": time.perf_counter() - step_start}
                                    raise
                                finally:
                                    result = None
                                    admission.release(STEP_DIAGNOSTIC_ADMISSION, category="diagnostics")
                                check()
                        stage = "completion_parse"
                        mark("generation_complete")
                        if report.get("completion_character_budget_exceeded"):
                            report["completion"] = session.summary()
                            report["completion_serialized_including_eos"] = tokenizer.decode(list(session.generated))
                            report["parsed_completion"] = None
                        else:
                            _finish_completion(report, session, tokenizer)
                        report["engine_before_close"] = engine.stats()
                        report["status"] = ("partial_duration_budget" if duration_exceeded() else
                            "partial_completion_character_budget" if report.get("completion_character_budget_exceeded") else
                            "functional_response_complete" if report["eos_response_complete"] else
                            "partial_model_call_budget" if not session.done else "partial_max_new_tokens")
                    report["assets_before_close"] = assets.stats()
                    if catalog is not None:
                        report["catalog_before_close"] = catalog.stats()
                    mark("owned_cleanup_start")
            except (Exception, KeyboardInterrupt) as error:
                if session is not None and tokenizer is not None:
                    try:
                        _finish_completion(report, session, tokenizer)
                    except Exception:
                        pass
                record_error(error)
            finally:
                # Drop all per-call arrays and bound-method owners before the
                # shared governor checks its remaining resource leases.
                result = session = tokenizer = engine = assets = None
                gc.collect()
                mark("owned_cleanup_complete")
                if admission.reserved:
                    admission.release(admission.reserved, category="diagnostics")
                report["memory_after_owned_close"] = memory.snapshot()
                report["owned_resources_closed"] = bool(
                    report["memory_after_owned_close"]["charged_host_bytes"] == memory.CONTROL_METADATA_BYTES
                    and not any(report["memory_after_owned_close"]["device_used_bytes"].values())
                    and report["memory_after_owned_close"]["live_reclaimers"] == 0)
                stage = "memory_runtime_cleanup"
        report["memory_runtime_closed"] = True
    except (Exception, KeyboardInterrupt) as error:
        record_error(error)
    finally:
        if memory is not None:
            report["memory_after_runtime_close"] = memory.snapshot()
        report["wall_seconds_total"] = time.perf_counter() - started
        report["duration_budget_exceeded"] = duration_exceeded()
        mark("runtime_cleanup_complete")
        report["process_after_cleanup"] = process_memory()
        try:
            check()
        except Exception as error:
            _failed(report, workspace, run_id, error, "post_cleanup_memory", initial)
    clean = report["owned_resources_closed"] and report["memory_runtime_closed"]
    if report["duration_budget_exceeded"] and report["status"] in ("functional_response_complete", "local_moe_prepared"):
        report["status"] = "partial_duration_budget"
    if not clean and report["status"] in ("functional_response_complete", "local_moe_prepared", "partial_max_new_tokens", "partial_model_call_budget", "partial_duration_budget", "partial_completion_character_budget"):
        report.update(status="failed", reason="Owned Spark runtime cleanup could not be established")
    report["functional_acceptance"] = bool(clean and report["status"] == "functional_response_complete")
    try:
        update({"phase":"finished", "completed_model_calls":report.get("completion",{}).get("calls",len(report["steps"])),
            "validated_model_calls":len(report["steps"]),
            "generated_tokens":len(report.get("completion",{}).get("selected_ids_including_eos", [])),
            "generated_ids":report.get("completion",{}).get("selected_ids_including_eos", [])})
    except (Exception, KeyboardInterrupt) as error:
        report["final_progress_failure_type"] = type(error).__name__
    if len(json.dumps(report, ensure_ascii=False).encode()) > report_limit:
        raise ValueError("Functional report exceeded its explicit metadata bound")
    workspace.stamp_result({}, run_id=run_id, initial_context=initial)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "generate"))
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--metadata", default="s1/import-source")
    parser.add_argument("--originals", default="s1/originals")
    parser.add_argument("--integrity", default="s1/original-integrity")
    parser.add_argument("--bundle", default="s1/attention-candidate")
    parser.add_argument("--fixtures", default="s1/moe-original")
    parser.add_argument("--attention-reference-result", default="s1/attention/reference-r02.json")
    parser.add_argument("--attention-receipt-result", default="s1/attention/native-validation-r01.json")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-model-calls", type=int, default=MAX_CALLS)
    parser.add_argument("--max-input-tokens", type=int, default=MAX_CALLS)
    parser.add_argument("--context-tokens", type=int, default=MAX_CALLS)
    parser.add_argument("--max-duration-seconds", type=float)
    parser.add_argument("--thinking-mode", choices=("chat", "thinking"), default="chat")
    parser.add_argument("--reasoning-effort", type=int)
    parser.add_argument("--reuse-profile", choices=("baseline", "s2"), default="baseline")
    args = parser.parse_args(argv)
    from .workspace import Workspace
    workspace = Workspace.open(args.workspace)
    initial = _guard(workspace, args.run_id)
    relative = f"s1/text/{args.run_id}.json"
    if workspace.result_path(relative).exists():
        raise FileExistsError("Refusing to overwrite Spark text evidence")
    report = collect(args.mode, workspace, run_id=args.run_id, metadata=args.metadata, originals=args.originals,
        integrity=args.integrity, bundle=args.bundle, fixtures=args.fixtures,
        attention_reference_result=args.attention_reference_result, attention_receipt_result=args.attention_receipt_result, prompt=args.prompt,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature, seed=args.seed,
        progress=ProgressFile(workspace,args.run_id), max_model_calls=args.max_model_calls,
        reuse_profile=args.reuse_profile, thinking_mode=args.thinking_mode, reasoning_effort=args.reasoning_effort,
        max_input_tokens=args.max_input_tokens, context_tokens=args.context_tokens,
        max_duration_seconds=args.max_duration_seconds)
    path = workspace.write_result(relative, report, run_id=args.run_id, initial_context=initial)
    print(json.dumps({"mode": args.mode, "status": report["status"], "functional_acceptance": report["functional_acceptance"],
                      "result": path.relative_to(workspace.root).as_posix()}))
    return 0 if report["status"] in ("local_moe_prepared", "functional_response_complete") else 2


if __name__ == "__main__":
    raise SystemExit(main())
