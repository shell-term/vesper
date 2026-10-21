"""Guarded metadata-only native Spark tensor inventory for candidate planning.

No GPU library, model payload, transformation or downloader is opened. Structural
groups describe authenticated catalog names; they prove neither runtime use nor
safe removal. Run under the caller's external 120-second CPU-probe deadline.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import re
import stat
import time

from . import checkpoint, memory, model_artifacts, resources, workspace as workspace_module
from .checkpoint import MODEL, REVISION
from .memory import MemoryPolicy, RuntimeMemory
from .resources import GiB, MiB, MemoryBudget

SCHEMA = "vesper.spark-metadata-inventory.v1"
MAX_REPORT_BYTES = MiB
MAX_TENSORS = 200000
CATALOG_ADMISSION_BYTES = 512 * MiB
REPORT_ADMISSION_BYTES = 16 * MiB
HOST_CAP_BYTES = 2 * GiB
AVAILABLE_FLOOR_BYTES = 32 * GiB
GROUPS = ("routed_experts", "engram_tables", "engram_other", "shared_experts", "core_layers",
          "embedding", "head", "final_norm", "mtp", "vision", "aligner", "image_markers", "other")
ROLES = ("quantization_scale_storage", "weight_named_storage", "other_storage")


def _guard(workspace, run_id):
    if not isinstance(run_id, str) or re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", run_id) is None:
        raise ValueError("Expected a simple lowercase inventory run identity")
    workspace.check()
    for module, filename in ((None, "spark_inventory_probe.py"), (model_artifacts, "model_artifacts.py"),
            (checkpoint, "checkpoint.py"), (memory, "memory.py"), (resources, "resources.py"),
            (workspace_module, "workspace.py")):
        workspace.check_runtime_module(__file__ if module is None else module.__file__, "src/vesper/" + filename)
    initial = workspace.context(run_id)
    if tuple(initial.get(k) for k in ("platform", "profile", "model", "revision")) != (
            "dgx-spark", "native", MODEL, REVISION):
        raise ValueError("Inventory requires the pinned dgx-spark/native workspace")
    return initial


def _metadata_path(workspace, relative):
    if not isinstance(relative, str) or not 0 < len(relative) <= 256:
        raise ValueError("Expected a bounded metadata namespace")
    root = workspace.artifact_path(relative)
    if not root.is_dir():
        raise ValueError("Original metadata directory is missing")
    names = [*(value[0] for value in model_artifacts.FILES.values()), "tree.json"]
    names += [f"headers/model-{i:05d}-of-00048.safetensors.json" for i in range(1, 49)]
    for name in names:
        path = workspace.artifact_path(relative + "/" + name)
        info = path.lstat()
        if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
                or getattr(info, "st_file_attributes", 0) & 0x400):
            raise ValueError("Original metadata must be regular files in its bound namespace")
    return root, len(names)


def _group(name):
    if name.startswith("mtp."): return "mtp"
    if name.startswith("vision."): return "vision"
    if name.startswith("aligner."): return "aligner"
    if name.split(".", 1)[0] in ("image_start", "image_end", "image_newline"): return "image_markers"
    if re.match(r"^layers\.\d+\.ffn\.experts\.\d+\.", name): return "routed_experts"
    if re.match(r"^layers\.\d+\.engram\.embed\.", name): return "engram_tables"
    if re.match(r"^layers\.\d+\.engram\.", name): return "engram_other"
    if re.match(r"^layers\.\d+\.ffn\.shared_experts\.", name): return "shared_experts"
    if re.match(r"^layers\.\d+\.", name): return "core_layers"
    if name.startswith("embed."): return "embedding"
    if name.startswith("head."): return "head"
    if name.startswith("norm."): return "final_norm"
    return "other"


def _count(row, size):
    row["tensor_count"] += 1; row["bytes"] += size


def _empty():
    return {"tensor_count": 0, "bytes": 0}


def _census(catalog, identity):
    if (not isinstance(catalog, dict) or not 0 < len(catalog) <= MAX_TENSORS
            or identity.get("model") != MODEL or identity.get("revision") != REVISION
            or identity.get("catalog_sha256") != model_artifacts.CATALOG_SHA256):
        raise ValueError("Unqualified original catalog identity")
    shards = identity["shards"]
    expected = {f"model-{i:05d}-of-00048.safetensors" for i in range(1, 49)}
    if len(shards) != 48 or {s["shard"] for s in shards} != expected:
        raise ValueError("Incomplete original shard census")
    extents = {s["shard"]: s["file_bytes"] for s in shards}
    if any(type(size) is not int or size <= 0 for size in extents.values()):
        raise ValueError("Invalid original shard extent")
    groups = {name: {**_empty(), "dtypes": {}, "roles": {role: _empty() for role in ROLES}} for name in GROUPS}
    total = _empty(); dtypes = {}
    policy = memory.current_policy()
    for ordinal, (name, tensor) in enumerate(catalog.items()):
        if policy is not None and ordinal % 1024 == 0:
            policy.checkpoint()
        if not isinstance(name, str) or not 0 < len(name) <= 512 or tensor.get("name") != name:
            raise ValueError("Invalid original tensor identity")
        size, dtype, shape = tensor["bytes"], tensor["dtype"], tensor["shape"]
        offset, shard = tensor["shard_byte_offset"], tensor["shard"]
        if (dtype not in model_artifacts.DTYPE_BYTES or not isinstance(shape, list) or not 1 <= len(shape) <= 16
                or any(type(d) is not int or d <= 0 for d in shape) or type(size) is not int or size <= 0
                or math.prod(shape) * model_artifacts.DTYPE_BYTES[dtype] != size
                or type(offset) is not int or offset < 0 or shard not in extents or offset + size > extents[shard]):
            raise ValueError("Original tensor storage census differs")
        group = groups[_group(name)]
        role = ("quantization_scale_storage" if dtype == "F8_E8M0" else
                "weight_named_storage" if name.endswith(".weight") else "other_storage")
        _count(total, size); _count(group, size); _count(group["roles"][role], size)
        _count(group["dtypes"].setdefault(dtype, _empty()), size)
        _count(dtypes.setdefault(dtype, _empty()), size)
    if total["tensor_count"] != identity["tensor_count"] or total["bytes"] != identity["original_tensor_bytes"]:
        raise ValueError("Original tensor totals differ from the authenticated catalog")
    if (sum(g["bytes"] for g in groups.values()) != total["bytes"]
            or sum(g["tensor_count"] for g in groups.values()) != total["tensor_count"]):
        raise ValueError("Tensor classification is not an exclusive partition")
    container_bytes = sum(extents.values())
    if container_bytes < total["bytes"]:
        raise ValueError("Original shard containers are smaller than their tensor payloads")
    return {"tensor_count": total["tensor_count"], "tensor_bytes": total["bytes"],
            "original_shard_file_bytes": container_bytes,
            "shard_header_and_container_overhead_bytes": container_bytes - total["bytes"],
            "shard_count": len(shards), "groups": groups, "dtypes": dtypes}


def collect(workspace, *, run_id, metadata="s1/import-source"):
    """Return a new source-stamped report; failures publish nothing."""
    initial = _guard(workspace, run_id)
    relative = f"s2/inventory/{run_id}.json"
    if workspace.result_path(relative).exists():
        raise FileExistsError("Inventory run identity already has a result")
    path, metadata_files = _metadata_path(workspace, metadata)
    started = time.perf_counter()
    retained = MemoryBudget(cap=CATALOG_ADMISSION_BYTES, owner="spark_inventory_catalog")
    report_budget = MemoryBudget(cap=REPORT_ADMISSION_BYTES, owner="spark_inventory_report")
    # MemoryPolicy requires positive device entries. A one-byte unused sentinel
    # satisfies that structural contract; no device discovery/allocation occurs.
    policy = MemoryPolicy(host_cap_bytes=HOST_CAP_BYTES, device_caps={0: 1}, pinned_cap_bytes=1,
                         available_floor_bytes=AVAILABLE_FLOOR_BYTES, unified_device_memory=True)
    catalog = identity = None
    with RuntimeMemory(policy) as runtime:
        try:
            report_budget.acquire(REPORT_ADMISSION_BYTES, category="diagnostics")
            retained.acquire(CATALOG_ADMISSION_BYTES, category="metadata")
            loaded = time.perf_counter()
            catalog, identity = model_artifacts.load_catalog(path)
            load_seconds = time.perf_counter() - loaded
            runtime.checkpoint()
            counted = time.perf_counter()
            census = _census(catalog, identity)
            census_seconds = time.perf_counter() - counted
            runtime.checkpoint()
        finally:
            catalog = identity = None
            gc.collect()
            if retained.reserved: retained.release(retained.reserved, category="metadata")
            if report_budget.reserved: report_budget.release(report_budget.reserved, category="diagnostics")
        runtime.checkpoint()
        owned = runtime.snapshot()
        owned_closed = bool(owned["charged_host_bytes"] == runtime.CONTROL_METADATA_BYTES
            and not any(owned["device_used_bytes"].values()) and owned["live_reclaimers"] == 0
            and owned["inflight_operations"] == 0)
    final = runtime.snapshot()
    closed = bool(final["charged_host_bytes"] == final["live_leases"] == final["live_reclaimers"] == final["inflight_operations"] == 0
                  and not any(final["device_used_bytes"].values()) and final["monitor_thread_stopped"]
                  and final["monitor_error"] is None and final["minimum_observed_available_bytes"] >= AVAILABLE_FLOOR_BYTES)
    if not owned_closed or not closed:
        raise RuntimeError("Inventory metadata ownership did not close")
    report = {"schema": SCHEMA, "status": "authenticated_metadata_inventory", "model": MODEL, "revision": REVISION,
        "catalog_sha256": model_artifacts.CATALOG_SHA256,
        "shard_inventory_sha256": model_artifacts.SHARD_INVENTORY_SHA256, **census,
        "scope": "Authenticated metadata-only structural census; neither runtime dependency nor safe-removal evidence",
        "classification_is_runtime_dependency_proof": False, "model_data_removal_authorized": False,
        "desired_capabilities": ["text", "image_input"], "image_input_qualified_by_this_probe": False,
        "image_artifact_policy": "Preserve vision, aligner and image-marker artifacts for the required desired image-input capability; this census does not qualify image execution or prove use/absence.",
        "classification_rules": "MTP and vision prefixes precede text-layer groups; each tensor appears in exactly one group",
        "storage_dtype_notes": "I8 routed-expert storage contains original packed FP4 codes. F8_E8M0 identifies scale storage; storage dtype is not a new precision profile.",
        "gpu_work": False, "model_generations": 0, "checkpoint_downloads": 0, "tensor_payload_bytes_read": 0,
        "metadata_file_count": metadata_files,
        "full_model_quality_qualified": False, "original_kernel_parity_qualified": False,
        "general_performance_claim": False, "new_profile_adopted": False,
        "limits": {"host_cap_bytes": HOST_CAP_BYTES, "retained_catalog_admission_bytes": CATALOG_ADMISSION_BYTES,
            "loader_transient_admission_bytes": CATALOG_ADMISSION_BYTES, "report_admission_bytes": REPORT_ADMISSION_BYTES,
            "available_floor_bytes": AVAILABLE_FLOOR_BYTES, "maximum_report_bytes": MAX_REPORT_BYTES,
            "device_and_pinned_budget_bytes": 1, "device_budget_is_unused_policy_sentinel": True,
            "reservation_is_RSS_or_OS_limit": False},
        "timings_seconds": {"catalog_load_including_authentication": load_seconds, "classification": census_seconds,
                            "collect_including_cleanup": time.perf_counter() - started},
        "memory_after_owned_close": owned, "memory_after_runtime_close": final,
        "owned_resources_closed": owned_closed, "memory_runtime_closed": closed}
    report = workspace.stamp_result(report, run_id=run_id, initial_context=initial)
    # Match Workspace's pretty-printed on-disk representation, not only compact JSON.
    if len((json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")) > MAX_REPORT_BYTES:
        raise ValueError("Inventory report exceeds its one MiB bound")
    return report


def run(workspace, *, run_id, metadata="s1/import-source"):
    report = collect(workspace, run_id=run_id, metadata=metadata)
    context = report.pop("workspace_context")
    return workspace.write_result(f"s2/inventory/{run_id}.json", report,
                                  run_id=run_id, initial_context=context)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--metadata", default="s1/import-source")
    args = parser.parse_args(argv)
    try:
        workspace = workspace_module.Workspace.open(args.workspace)
        run(workspace, run_id=args.run_id, metadata=args.metadata)
        print(json.dumps({"status": "authenticated_metadata_inventory", "result": f"s2/inventory/{args.run_id}.json"}))
        return 0
    except (Exception, KeyboardInterrupt) as error:
        print(json.dumps({"status": "cancelled" if isinstance(error, KeyboardInterrupt) else "failed",
                          "failure_type": type(error).__name__, "result_written": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
