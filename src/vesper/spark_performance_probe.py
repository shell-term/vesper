"""Bounded layer-zero upload equivalence and timing over local original weights.

Both modes use the current verified bulk-storage path. The first mode retains
uncached validation/upload reads; the second reuses arrays only during upload.
An empty scoped compiler cache precedes warm cache reuse, so wall times are
observations, not an isolated upload speedup or a full-model qualification.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
from pathlib import Path
import platform
import re
import time
import traceback

import numpy as np

from .checkpoint import MODEL, REVISION
from .execution_target import spark_kernel_cache
from .formats import bf16_bits
from .memory import RuntimeMemory
from .resources import MemoryBudget, GiB, MiB
from .spark_backbone_runtime import LayerOperatorPool, spark_memory_policy, TARGET, SPARK_ASSET_CAP
from .spark_probe import Driver, Unsupported, host_memory, require_capacity
from .spark_tensor_views import SparkModelAssets

SCHEMA = "vesper.spark-layer-performance.v1"
SEED = 20261021
DIAGNOSTIC_ALLOWANCE = 128 * MiB
MAX_REPORT_BYTES = 4 * MiB
MAX_ARRAYS = 4096


def _guard(workspace, run_id):
    if not isinstance(run_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", run_id):
        raise ValueError("Expected a simple lowercase run identity")
    workspace.check()
    workspace.check_runtime_module(__file__, "src/vesper/spark_performance_probe.py")
    from . import workspace as workspace_module
    workspace.check_runtime_module(workspace_module.__file__, "src/vesper/workspace.py")
    initial = workspace.context(run_id)
    if tuple(initial[k] for k in ("platform", "profile", "model", "revision")) != (
            "dgx-spark", "native", MODEL, REVISION):
        raise ValueError("Layer performance execution requires the pinned dgx-spark/native workspace")
    return initial


def _hash_arrays(value, prefix="result"):
    """Record every returned array's exact storage bits, including FP32 stages."""
    result, total, visited = {}, 0, 0
    def visit(item, path):
        nonlocal total, visited
        visited += 1
        if visited > 100000:
            raise ValueError("Native diagnostic structure exceeds its bound")
        if isinstance(item, np.ndarray):
            if not item.flags.c_contiguous or item.dtype.hasobject:
                raise ValueError("Expected contiguous numeric native diagnostic arrays")
            total += item.nbytes
            if total > DIAGNOSTIC_ALLOWANCE or len(result) >= MAX_ARRAYS:
                raise MemoryError("Layer diagnostic arrays exceed their bounded allowance")
            result[path] = {"dtype": item.dtype.str, "shape": list(item.shape), "bytes": item.nbytes,
                "sha256": hashlib.sha256(memoryview(item).cast("B")).hexdigest()}
        elif isinstance(item, dict):
            for key, child in item.items():
                if type(key) not in (str, int) or len(str(key)) > 160:
                    raise ValueError("Native diagnostic keys exceed their bound")
                visit(child, f"{path}/{key}")
        elif isinstance(item, (tuple, list)):
            for number, child in enumerate(item): visit(child, f"{path}/{number}")
    visit(value, prefix)
    return result


class _UncachedUploadAssets:
    def __init__(self, assets): self.assets = assets
    def __getattr__(self, name): return getattr(self.assets, name)
    @contextmanager
    def upload_arrays(self, layer):
        yield self.assets.arrays_for_layer(layer)


class _ObservedMoE:
    """Observe existing native ownership without changing its entry/exit rules."""
    def __init__(self, owner, rows): self.owner, self.rows = owner, rows
    def __getattr__(self, name): return getattr(self.owner, name)
    def __enter__(self):
        self.owner.__enter__()
        return self
    def forward(self, *args, **kwargs):
        try:
            return self.owner.forward(*args, **kwargs)
        finally:
            store = getattr(getattr(self.owner, "experts", None), "store", None)
            self.rows.append({"layer": self.owner.layer,
                "attempt": dict(self.owner.last_attempt) if self.owner.last_attempt is not None else None,
                "source": store.stats() if store is not None else None})
    def __exit__(self, *error): return self.owner.__exit__(*error)


def _mode(mode, assets, values, cache, memory, reports):
    current_assets = _UncachedUploadAssets(assets) if mode == "uncached_upload" else assets
    report = {"mode": mode, "timings_seconds": {}, "array_hashes": {}, "moe": [],
              "reader_before": assets.reader.stats(), "kernel_cache_before": cache.stats()}
    reports.append(report)
    started = time.perf_counter()
    pool = LayerOperatorPool(current_assets)
    factory = pool.moe_factory
    pool.moe_factory = lambda layer: _ObservedMoE(factory(layer), report["moe"])
    def measured(name, operation):
        begin = time.perf_counter()
        try: return operation()
        finally: report["timings_seconds"][name] = time.perf_counter() - begin
    def record_arrays(value, name):
        report["array_hashes"].update(_hash_arrays(value, name))
        if sum(row["bytes"] for row in report["array_hashes"].values()) > DIAGNOSTIC_ALLOWANCE:
            raise MemoryError("Combined layer diagnostic arrays exceed their bounded allowance")
    try:
        with pool:
            hc = pool.proxy("hc", 0)
            mixes = measured("hc_mixes_including_first_open", lambda: hc.mixes(values, "ffn"))
            record_arrays(mixes, "hc_mixes")
            collapsed = measured("hc_pre", lambda: hc.pre(values, mixes["pre_fp32"]))
            record_arrays(collapsed, "hc_pre")
            activation = collapsed["output_bf16"]
            projection = measured("projection_wq_a_including_open", lambda:
                pool.proxy("projection", 0).fp8("wq_a", activation))
            record_arrays(projection, "projection_wq_a")
            moe = measured("moe_forward_including_open", lambda:
                pool.proxy("moe", 0).forward(activation.reshape(5120), False))
            record_arrays(moe, "moe")
            route = moe["routing"]["indices"]
            if (not isinstance(route, np.ndarray) or route.shape != (6,)
                    or not np.issubdtype(route.dtype, np.integer) or len(set(route.tolist())) != 6
                    or np.any(route < 0) or np.any(route >= 384)):
                raise ValueError("The native layer probe requires six distinct original expert routes")
            if not report["array_hashes"]:
                raise ValueError("Layer execution returned no diagnostic arrays")
            report["native_operators"] = {kind: operator.stats()
                for kind, (_, operator, _) in pool.active.items()}
            # Hash records are sufficient; do not retain two modes' diagnostics.
            del mixes, collapsed, activation, projection, moe
    finally:
        report["pool_after"] = pool.stats()
        report["reader_after"] = assets.reader.stats()
        report["kernel_cache_after"] = cache.stats()
        report["wall_seconds"] = time.perf_counter() - started
        memory.sample(f"layer_performance_{mode}")
    return report


def _failure(workspace, run_id, error, stage):
    """Keep machine paths in a private, exclusive artifact only."""
    relative = f"s1/performance/{run_id}/failure.json"
    value = {"failure_type": type(error).__name__, "stage": stage,
        "traceback": "".join(traceback.format_exception(error))}
    raw = (json.dumps(value, ensure_ascii=False) + "\n").encode()
    if len(raw) > 65536: return {"saved": False, "reason": "Private diagnostic exceeded its bound"}
    try:
        path = workspace.artifact_path(relative, create_parent=True)
        with path.open("xb") as stream: stream.write(raw)
        return {"saved": True, "artifact": relative, "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest()}
    except Exception:
        return {"saved": False, "reason": "Private diagnostic could not be saved exclusively"}


def collect(workspace, *, run_id, metadata="s1/import-source", originals="s1/originals",
            integrity="s1/original-integrity", fixtures="s1/moe-original"):
    initial = _guard(workspace, run_id)
    paths = {name: workspace.artifact_path(value) for name, value in
        (("metadata", metadata), ("originals", originals), ("integrity", integrity), ("fixtures", fixtures))}
    report = {"schema": SCHEMA, "status": "failed", "scope": "layer0 HC + wq_a + native six-route MoE",
        "model": MODEL, "revision": REVISION, "execution_target": TARGET, "seed": SEED,
        "layers": [0], "input_shape": [1, 1, 4, 5120], "input_storage": "BF16",
        "mode_order": ["uncached_upload", "deduplicated_upload"], "modes": [],
        "both_modes_use_current_verified_bulk_storage": True,
        "timing_order_limitation": "Uncached mode starts with an empty scoped kernel cache; deduplicated mode can reuse its entries. Other OS, driver and storage caches are not controlled. Wall times do not isolate upload deduplication or the former 4 KiB path.",
        "full_model_functional_acceptance": False, "full_model_quality_qualified": False,
        "original_kernel_parity_qualified": False, "qualification_receipt_issued": False,
        "general_performance_claim": False, "network_in_forward": False,
        "automatic_retry": False, "checkpoint_download_started": False,
        "limits": {"asset_cap_bytes": SPARK_ASSET_CAP, "physical_available_floor_bytes": 32 * GiB,
                   "diagnostic_admission_bytes": DIAGNOSTIC_ALLOWANCE,
                   "worker_timeout_seconds": 300, "timeout_enforced_by": "external worker"},
        "array_bits_exact": False, "owned_resources_closed": False, "memory_runtime_closed": False}
    memory = assets = cache = admission = None
    stage, started = "platform", time.perf_counter()
    try:
        if platform.system() != "Linux" or platform.machine().lower() not in ("aarch64", "arm64"):
            raise Unsupported("Spark layer execution requires Linux AArch64")
        report["physical_memory_before"] = host_memory()
        require_capacity(report["physical_memory_before"], DIAGNOSTIC_ALLOWANCE)
        report["hardware"] = Driver().inventory()
        stage = "memory_runtime"
        memory = RuntimeMemory(spark_memory_policy())
        with memory:
            with spark_kernel_cache() as cache:
                admission = MemoryBudget(cap=DIAGNOSTIC_ALLOWANCE, owner="spark_layer_performance")
                admission.acquire(DIAGNOSTIC_ALLOWANCE, category="diagnostics")
                try:
                    stage = "assets"
                    assets = SparkModelAssets(paths["metadata"], paths["originals"], paths["integrity"],
                        workspace=workspace, fixtures=paths["fixtures"],
                        budget=MemoryBudget(cap=SPARK_ASSET_CAP, owner="spark_model_assets"))
                    with assets:
                        assets.moe_paths(0)
                        rng = np.random.default_rng(SEED)
                        values = bf16_bits(rng.normal(0., .1, (1, 1, 4, 5120)).astype(np.float32))
                        values.setflags(write=False)
                        report["input"] = _hash_arrays(values, "input")["input"]
                        for mode in report["mode_order"]:
                            stage = mode
                            _mode(mode, assets, values, cache, memory, report["modes"])
                        del values
                        first, second = (mode["array_hashes"] for mode in report["modes"])
                        report["array_bits_exact"] = first == second
                        report["compared_array_count"] = len(first)
                        report["mismatched_arrays"] = sorted(name for name in set(first) | set(second)
                            if first.get(name) != second.get(name))
                        if not report["array_bits_exact"]:
                            raise ValueError("Cached and uncached upload modes returned different exact array bits")
                        report["assets_before_close"] = assets.stats()
                finally:
                    admission.release(DIAGNOSTIC_ALLOWANCE, category="diagnostics")
            report["kernel_cache_after_close"] = cache.stats()
        report["status"] = "equivalent_layer_components"
    except Exception as error:
        report.update(failure_type=type(error).__name__, failure_stage=stage,
                      private_failure=_failure(workspace, run_id, error, stage))
    finally:
        gc.collect()
        if assets is not None:
            report["assets_after_close"] = assets.stats()
            report["owned_resources_closed"] = (assets.reader is None and assets.leases["bytes"] == 0)
        if memory is not None:
            report["memory_after"] = memory.snapshot()
            report["memory_runtime_closed"] = not memory.opened
            report["owned_resources_closed"] = (report["owned_resources_closed"]
                and not any(report["memory_after"]["device_used_bytes"].values())
                and report["memory_after"]["live_leases"] == 0
                and report["memory_after"]["live_reclaimers"] == 0)
        if cache is not None: report["kernel_cache_after_close"] = cache.stats()
        if admission is not None: report["diagnostic_reserved_bytes_after"] = admission.reserved
        report["wall_seconds"] = time.perf_counter() - started
        if report["status"] == "equivalent_layer_components" and not (
                report["owned_resources_closed"] and report["memory_runtime_closed"]):
            report.update(status="failed", failure_stage="cleanup", failure_type="RuntimeError")
    if len(json.dumps(report).encode()) > MAX_REPORT_BYTES:
        raise ValueError("Layer performance report exceeded its explicit bound")
    workspace.stamp_result({}, run_id=run_id, initial_context=initial)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--metadata", default="s1/import-source")
    parser.add_argument("--originals", default="s1/originals")
    parser.add_argument("--integrity", default="s1/original-integrity")
    parser.add_argument("--fixtures", default="s1/moe-original")
    args = parser.parse_args(argv)
    from .workspace import Workspace
    workspace = Workspace.open(args.workspace)
    initial = _guard(workspace, args.run_id)
    relative = f"s1/performance/{args.run_id}.json"
    if workspace.result_path(relative).exists():
        raise FileExistsError("Refusing to overwrite earlier Spark layer evidence")
    report = collect(workspace, run_id=args.run_id, metadata=args.metadata, originals=args.originals,
                     integrity=args.integrity, fixtures=args.fixtures)
    path = workspace.write_result(relative, report, run_id=args.run_id, initial_context=initial)
    print(json.dumps({"status": report["status"], "array_bits_exact": report["array_bits_exact"],
                      "result": path.relative_to(workspace.root).as_posix()}))
    return 0 if report["status"] == "equivalent_layer_components" else 2


if __name__ == "__main__":
    raise SystemExit(main())
