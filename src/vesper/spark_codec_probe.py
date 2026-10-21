"""CPU-only bounded lossless samples over authenticated original Spark experts.

Independent frames model random-access storage; no compressed model is written,
no numerical profile is adopted, and no GPU or throughput qualification occurs.
"""

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import time
import traceback
import zlib

import numpy as np

from .bandwidth import EXPERT_BYTES
from .checkpoint import MODEL, REVISION, EXPERT_LAYOUT
from .memory import RuntimeMemory, MemoryPolicy, RuntimeCancelled
from .resources import MemoryBudget, MiB, GiB
from .spark_artifacts import OriginalTensorReader
from .storage import tensor_layout

SCHEMA = "vesper.spark-lossless-codec-sample.v1"
SAMPLES = tuple((layer, expert) for layer in (0, 20, 39) for expert in (0, 127, 255, 383))
BLOCK = 65536
LEVELS = (1, 6)
SAMPLE_CAP = 128 * MiB
CATALOG_ADMISSION = 512 * MiB
RUNTIME_CAP = 768 * MiB
CODEC_ALLOWANCE = 8 * MiB
FRAME_LIMIT = BLOCK + 1024
INDEX_RECORD_BYTES = 64
REPORT_LIMIT = 2 * MiB


def _guard(workspace, run_id):
    if not isinstance(run_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", run_id):
        raise ValueError("Expected a lowercase bounded run identity")
    workspace.check()
    for name in ("spark_codec_probe.py", "spark_artifacts.py", "artifact_repository.py", "model_artifacts.py", "workspace.py"):
        workspace.check_runtime_module(Path(__file__).parent / name, "src/vesper/" + name)
    initial = workspace.context(run_id)
    if tuple(initial[k] for k in ("platform", "profile", "model", "revision")) != (
            "dgx-spark", "native", MODEL, REVISION):
        raise ValueError("Codec sampling requires the pinned original Spark native workspace")
    return initial


def _name(layer, expert, short):
    projection, suffix = short.split("_")
    return f"layers.{layer}.ffn.experts.{expert}.{projection}.{suffix}"


def _codec_identity():
    common = {"python_version": platform.python_version(), "python_build": list(platform.python_build()),
        "linked_runtime_version": zlib.ZLIB_RUNTIME_VERSION}
    module_file = getattr(zlib, "__file__", None)
    if module_file is None:
        return {**common, "module_kind": "built-in", "extension_sha256": None,
            "extension_basename": None, "binary_identity_limitation": "Built into Python; no independent extension file hash"}
    path = Path(module_file)
    if not 0 < path.stat().st_size <= 8 * MiB:
        raise ValueError("The codec extension exceeds its identity-read bound")
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        while raw := stream.read(BLOCK): sha.update(raw)
    return {**common, "module_kind": "extension", "extension_sha256": sha.hexdigest(),
        "extension_basename": path.name}


def make_plan(workspace, run_id, *, metadata="s1/import-source", originals="s1/originals", integrity="s1/original-integrity"):
    initial = _guard(workspace, run_id)
    for path in (metadata, originals, integrity): workspace.artifact_path(path)
    return {"schema": SCHEMA, "purpose": "bounded lossless codec inventory; no adoption",
        "run_id": run_id, "source_context": initial,
        "artifact_inputs": {"metadata": metadata, "originals": originals, "integrity": integrity},
        "samples": [{"layer": layer, "expert": expert} for layer, expert in SAMPLES],
        "codec": {"name": "zlib", "levels": list(LEVELS), "version": zlib.ZLIB_VERSION,
            "runtime_version": zlib.ZLIB_RUNTIME_VERSION, "independent_frame_bytes": BLOCK,
            "implementation": _codec_identity(),
            "deflate_window_bits": zlib.MAX_WBITS, "wrapper": "zlib",
            "maximum_compressed_frame_bytes": FRAME_LIMIT, "alignment_bytes": 4096,
            "index_record_bytes": INDEX_RECORD_BYTES, "container_header_bytes": 4096},
        "limits": {"samples": 12, "maximum_live_sample_experts": 1,
            "sample_owned_memory_cap_bytes": SAMPLE_CAP, "codec_allowance_bytes": CODEC_ALLOWANCE,
            "catalog_transient_admission_bytes_separate": CATALOG_ADMISSION,
            "runtime_control_admission_bytes_separate": RuntimeMemory.CONTROL_METADATA_BYTES,
            "total_shared_admission_cap_bytes": RUNTIME_CAP, "physical_available_floor_bytes": 32 * GiB,
            "external_timeout_seconds": 300, "timeout_enforced_by": "external worker",
            "reservation_is_RSS_or_OS_limit": False},
        "gpu_requested": False, "network_payload_bytes": 0, "compressed_model_written": False,
        "automatic_retry": False, "lossy_profile_adopted": False, "general_speed_claim": False,
        "imagefunctionalquality_not_qualified": True, "vision_or_image_tensors_modified": False}


def _aligned(size): return ((size + 4095) // 4096) * 4096


def _entropy(histogram):
    total = sum(histogram)
    return -sum((n / total) * math.log2(n / total) for n in histogram if n) if total else 0.


def _histograms(target, layout):
    codes, scales = np.zeros(16, np.int64), np.zeros(256, np.int64)
    for short, region in layout.items():
        data = target[region["offset"]:region["offset"] + region["bytes"]]
        for start in range(0, data.size, BLOCK):
            part = data[start:start + BLOCK]
            if short.endswith("_weight"):
                codes += np.bincount(part & 15, minlength=16)
                codes += np.bincount(part >> 4, minlength=16)
            else:
                scales += np.bincount(part, minlength=256)
    code_list, scale_list = codes.tolist(), scales.tolist()
    return {"fp4_code_histogram": code_list, "e8m0_scale_histogram": scale_list,
        "fp4_code_entropy_bits": _entropy(code_list), "scale_entropy_bits": _entropy(scale_list),
        "signed_zero_code_fraction": (code_list[0] + code_list[8]) / sum(code_list),
        "interpretation": "Byte/code statistics only; signs and original bytes are preserved"}


def _codec(data, level, budget):
    # Covers codec-native state and simultaneous encoded/decoded block copies;
    # this conservative admission is not an allocator-level zlib RSS limit.
    transient = CODEC_ALLOWANCE + 3 * FRAME_LIMIT
    budget.acquire(transient, category="scratch")
    try:
        begin = time.perf_counter()
        original_sha = hashlib.sha256(memoryview(data)).hexdigest()
        sha_seconds = time.perf_counter() - begin
        decoded_sha = hashlib.sha256()
        compressed = padded = frames = 0
        compress_seconds = decompress_seconds = 0.
        for start in range(0, data.size, BLOCK):
            budget.acquire(0)
            part = memoryview(data)[start:start + BLOCK]
            begin = time.perf_counter(); encoded = zlib.compress(part, level, wbits=zlib.MAX_WBITS)
            compress_seconds += time.perf_counter() - begin
            if not encoded or len(encoded) > FRAME_LIMIT:
                raise ValueError("Compressed frame exceeded its fixed bound")
            begin = time.perf_counter()
            decoder = zlib.decompressobj(wbits=zlib.MAX_WBITS)
            decoded = decoder.decompress(encoded, len(part) + 1)
            if (len(decoded) != len(part) or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail):
                raise ValueError("Lossless frame extent or termination differs")
            decompress_seconds += time.perf_counter() - begin
            begin = time.perf_counter(); decoded_sha.update(decoded)
            sha_seconds += time.perf_counter() - begin
            compressed += len(encoded); padded += _aligned(len(encoded)); frames += 1
            del encoded, decoded, decoder, part
        actual = decoded_sha.hexdigest()
        if actual != original_sha:
            raise ValueError("Lossless reconstruction SHA-256 differs from authenticated original bytes")
        index = _aligned(frames * INDEX_RECORD_BYTES)
        total = 4096 + padded + index
        return {"level": level, "frames": frames, "original_bytes": data.nbytes,
            "compressed_payload_bytes": compressed, "aligned_payload_bytes": padded,
            "aligned_index_bytes": index, "container_header_bytes": 4096, "modeled_storage_bytes": total,
            "modeled_storage_ratio": total / data.nbytes, "compress_seconds": compress_seconds,
            "decompress_seconds": decompress_seconds, "sha_seconds": sha_seconds,
            "reconstructed_sha256": actual, "original_sha256": original_sha, "exact_reconstruction": True,
            "random_access_scope": "One independent <=64KiB frame, with 4KiB-aligned payload extents",
            "storage_is_accounting_only": True}
    finally:
        budget.release(transient, category="scratch")


def _failure(workspace, run_id, error):
    raw = json.dumps({"failure_type": type(error).__name__,
        "traceback": "".join(traceback.format_exception(error))}).encode()
    if len(raw) > 65536: return {"saved": False}
    try:
        relative = f"s2/codec/{run_id}/failure.json"
        with workspace.artifact_path(relative, create_parent=True).open("xb") as stream: stream.write(raw)
        return {"saved": True, "artifact": relative, "sha256": hashlib.sha256(raw).hexdigest()}
    except Exception: return {"saved": False}


def collect(workspace, *, run_id, metadata="s1/import-source", originals="s1/originals", integrity="s1/original-integrity"):
    plan = make_plan(workspace, run_id, metadata=metadata, originals=originals, integrity=integrity)
    initial = plan["source_context"]
    # Freeze the plan exclusively before catalog or payload reads begin.
    plan_relative = f"s2/codec/{run_id}/plan.json"
    plan_raw = (json.dumps(plan, sort_keys=True) + "\n").encode()
    with workspace.artifact_path(plan_relative, create_parent=True).open("xb") as stream: stream.write(plan_raw)
    report = {"schema": SCHEMA, "status": "failed", "plan": plan,
        "plan_artifact": plan_relative, "plan_sha256": hashlib.sha256(plan_raw).hexdigest(),
        "samples": [], "full_model_quality_qualified": False, "general_speed_claim": False,
        "gpu_requested": False, "compressed_model_written": False, "owned_resources_closed": False,
        "memory_runtime_closed": False, "limitations": ["Fixed 12-expert samples do not establish full-checkpoint compression.",
            "CPU codec timings exclude model execution, GPU transfer and a real compressed storage implementation.",
            "Catalog admission is separate from the 128MiB sample cap; native codec overhead is conservatively admitted, not hard capped.",
            "Image functionality/quality is not qualified; vision, aligner and image-marker tensors are not modified or classified as unused."]}
    budget = MemoryBudget(cap=SAMPLE_CAP, owner="spark_codec_sample")
    memory = RuntimeMemory(MemoryPolicy(host_cap_bytes=RUNTIME_CAP, pinned_cap_bytes=MiB,
        device_caps={0:MiB}, unified_device_memory=True))
    reader = None
    started = time.perf_counter(); stage = "memory_runtime"

    def remember_failure(error):
        # Record the original interruption before context cleanup can raise a
        # second error. Public evidence contains no traceback or local paths.
        if "failure_type" not in report:
            report.update(status="cancelled" if isinstance(error, (KeyboardInterrupt, RuntimeCancelled)) else "failed",
                failure_type=type(error).__name__, failure_stage=stage,
                private_failure=_failure(workspace, run_id, error))

    @contextmanager
    def preserve_failure():
        try: yield
        except (Exception, KeyboardInterrupt) as error:
            remember_failure(error)
            raise

    try:
        with memory, preserve_failure():
            names = tuple(_name(layer, expert, short) for layer, expert in SAMPLES for short in EXPERT_LAYOUT)
            stage = "authenticated_metadata"
            with OriginalTensorReader(workspace.artifact_path(metadata), workspace.artifact_path(originals),
                    workspace.artifact_path(integrity), budget, tensor_names=names, workspace=workspace) as reader, preserve_failure():
                layout = tensor_layout()
                for layer, expert in SAMPLES:
                    stage = f"expert-{layer}-{expert}"
                    # Check every released tensor contract before reading its payload.
                    for short, shape in EXPERT_LAYOUT.items():
                        t = reader.tensors[_name(layer, expert, short)]
                        if (t["dtype"], t["shape"], t["bytes"]) != (
                                "I8" if short.endswith("_weight") else "F8_E8M0", list(shape), layout[short]["bytes"]):
                            raise ValueError("Original native expert layout differs")
                    before = reader.stats(); read_started = time.perf_counter()
                    budget.acquire(EXPERT_BYTES, category="weights")
                    target = None
                    try:
                        target = np.empty(EXPERT_BYTES, np.uint8)
                        for short, region in layout.items():
                            reader.read_into(_name(layer, expert, short), 0,
                                target[region["offset"]:region["offset"] + region["bytes"]])
                        read_seconds = time.perf_counter() - read_started
                        histogram_started = time.perf_counter()
                        budget.acquire(MiB, category="scratch")
                        try: histograms = _histograms(target, layout)
                        finally: budget.release(MiB, category="scratch")
                        row = {"layer": layer, "expert": expert, "original_bytes": EXPERT_BYTES,
                            "fp4_packed_code_bytes": sum(r["bytes"] for n, r in layout.items() if n.endswith("_weight")),
                            "e8m0_scale_bytes": sum(r["bytes"] for n, r in layout.items() if n.endswith("_scale")),
                            "read_seconds_including_authentication": read_seconds,
                            "histogram_seconds": time.perf_counter() - histogram_started,
                            "reader_before": before, "reader_after": reader.stats(),
                            "histograms": histograms, "codecs": []}
                        report["samples"].append(row)
                        for level in LEVELS: row["codecs"].append(_codec(target, level, budget))
                    except (Exception, KeyboardInterrupt) as error:
                        remember_failure(error)
                        raise
                    finally:
                        target = None
                        budget.release(EXPERT_BYTES, category="weights")
                    memory.sample(stage)
                    memory.checkpoint()
                report["reader_before_close"] = reader.stats()
            stage = "terminal_memory_checkpoint"
            memory.checkpoint()
        report["status"] = "sampled_exact_lossless_frames"
    except (Exception, KeyboardInterrupt) as error:
        remember_failure(error)
    finally:
        gc.collect()
        report["sample_budget_peak_bytes"] = budget.peak
        report["sample_budget_reserved_after"] = budget.reserved
        after = report["memory_after"] = memory.snapshot()
        report["memory_runtime_closed"] = not memory.opened
        minimum = after["minimum_observed_available_bytes"]
        checks = report["terminal_runtime_checks"] = {
            "sample_budget_released": budget.reserved == 0,
            "runtime_charges_released": after["charged_host_bytes"] == 0 and after["pinned_bytes"] == 0
                and not any(after["device_used_bytes"].values()),
            "leases_and_reclaimers_released": after["live_leases"] == 0 and after["live_reclaimers"] == 0,
            "no_inflight_operations": after["inflight_operations"] == 0,
            "monitor_thread_stopped": after["monitor_thread_stopped"] is True,
            "monitor_observation_healthy": after["monitor_error"] is None,
            "observed_physical_floor_preserved": type(minimum) is int
                and minimum >= after["available_floor_bytes"] >= 32 * GiB,
            "memory_runtime_closed": report["memory_runtime_closed"]}
        report["owned_resources_closed"] = all(checks.values())
        report["wall_seconds"] = time.perf_counter() - started
        if report["status"] == "sampled_exact_lossless_frames" and not (
                report["owned_resources_closed"] and report["memory_runtime_closed"]):
            stage = "cleanup"
            remember_failure(RuntimeError("Terminal runtime cleanup or memory observation checks failed"))
    if len(json.dumps(report).encode()) > REPORT_LIMIT: raise ValueError("Codec report exceeds its bound")
    workspace.stamp_result({}, run_id=run_id, initial_context=initial)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--metadata", default="s1/import-source")
    parser.add_argument("--originals", default="s1/originals")
    parser.add_argument("--integrity", default="s1/original-integrity")
    args = parser.parse_args(argv)
    from .workspace import Workspace
    workspace = Workspace.open(args.workspace)
    initial = _guard(workspace, args.run_id)
    relative = f"s2/codec/{args.run_id}.json"
    if workspace.result_path(relative).exists(): raise FileExistsError("Refusing to replace earlier codec evidence")
    report = collect(workspace, run_id=args.run_id, metadata=args.metadata, originals=args.originals, integrity=args.integrity)
    path = workspace.write_result(relative, report, run_id=args.run_id, initial_context=initial)
    print(json.dumps({"status": report["status"], "samples": len(report["samples"]),
        "result": path.relative_to(workspace.root).as_posix()}))
    return 0 if report["status"] == "sampled_exact_lossless_frames" else 2


if __name__ == "__main__": raise SystemExit(main())
