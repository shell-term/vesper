"""Bounded, predeclared native component comparison for Spark S2 reuse.

Both profiles use the current bulk verified reader and identical diagnostics.
The baseline has no runtime catalog/backend reuse; S2 opts into both. Separate
asset and compiler-cache lifetimes prevent a warm compiler cache from crossing
profiles. This is a component regression/calibration, never a full-model or
practical-quality qualification. The owning worker enforces the 300 s limit.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
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
from .resources import MemoryBudget, MiB, GiB
from .spark_backbone_runtime import LayerOperatorPool, spark_memory_policy, TARGET, SPARK_ASSET_CAP
from .spark_probe import Driver, Unsupported, host_memory, require_capacity
from .spark_tensor_views import SparkModelAssets

SCHEMA = "vesper.spark-s2-components.v1"
SEED = 20261021
LAYERS = (0, 1, 0)
MODES = ("baseline", "s2")
DIAGNOSTIC_ALLOWANCE = 128 * MiB
MAX_REPORT_BYTES = 4 * MiB
MAX_ARRAYS = 4096
WORKER_TIMEOUT_SECONDS = 300


def _catalog(metadata, originals, integrity, budget, workspace):
    from .spark_artifacts import AuthenticatedModelCatalog
    return AuthenticatedModelCatalog(metadata, originals, integrity, budget, workspace=workspace)


def _backends(workspace):
    from .spark_operator_backends import SparkOperatorBackends
    return SparkOperatorBackends(workspace, ordinal=0)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _guard(workspace, run_id):
    if not isinstance(run_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", run_id):
        raise ValueError("Expected a simple lowercase run identity")
    workspace.check()
    workspace.check_runtime_module(__file__, "src/vesper/spark_s2_probe.py")
    from . import workspace as workspace_module
    workspace.check_runtime_module(workspace_module.__file__, "src/vesper/workspace.py")
    initial = workspace.context(run_id)
    if tuple(initial[k] for k in ("platform", "profile", "model", "revision")) != (
            "dgx-spark", "native", MODEL, REVISION):
        raise ValueError("S2 component execution requires the pinned dgx-spark/native workspace")
    return initial


def _hash_arrays(value, prefix="result"):
    records, total, visits = {}, 0, 0
    def visit(item, path):
        nonlocal total, visits
        visits += 1
        if visits > 100000:
            raise ValueError("Native diagnostic structure exceeds its bound")
        if isinstance(item, np.ndarray):
            if not item.flags.c_contiguous or item.dtype.hasobject:
                raise ValueError("Expected contiguous numeric native diagnostic arrays")
            total += item.nbytes
            if total > DIAGNOSTIC_ALLOWANCE or len(records) >= MAX_ARRAYS:
                raise MemoryError("Component diagnostics exceed their bounded allowance")
            records[path] = {"dtype": item.dtype.str, "shape": list(item.shape), "bytes": item.nbytes,
                             "sha256": _sha(memoryview(item).cast("B"))}
        elif isinstance(item, dict):
            for key, child in item.items():
                if type(key) not in (str, int) or len(str(key)) > 160:
                    raise ValueError("Native diagnostic keys exceed their bound")
                visit(child, f"{path}/{key}")
        elif isinstance(item, (tuple, list)):
            for index, child in enumerate(item):
                visit(child, f"{path}/{index}")
    visit(value, prefix)
    return records


def _inputs():
    rng = np.random.default_rng(SEED)
    result = {"hidden": bf16_bits(rng.normal(0., .1, (1, 1, 4, 5120)).astype(np.float32)),
              "grouped": bf16_bits(rng.normal(0., .1, (1, 1, 8, 4096)).astype(np.float32))}
    for value in result.values():
        value.setflags(write=False)
    return result


def _plan(workspace, run_id, initial, inputs):
    """Publish an exclusive immutable plan before any hardware/model entry."""
    plan = {"schema": "vesper.spark-s2-components-plan.v1", "model": MODEL, "revision": REVISION,
        "workspace_context": initial, "execution_target": TARGET, "seed": SEED,
        "mode_order": list(MODES), "layer_sequence_per_mode": list(LAYERS),
        "operations_per_visit": ["hc.ffn.mixes", "hc.ffn.pre", "projection.wq_a",
                                  "projection.grouped_wo_a", "moe.native_six_routes"],
        "input_arrays": _hash_arrays(inputs, "input"),
        "mode_lifetimes": "fresh assets and empty scoped compiler cache per mode; S2 runtime-owned catalog and backends",
        "comparison": "exact dtype/shape/bytes/SHA256 of every returned diagnostic array; exact layer0 revisit",
        "fault_or_quality_qualification": False, "automatic_retry": False,
        "worker_timeout_seconds": WORKER_TIMEOUT_SECONDS, "timeout_enforced_by": "external worker",
        "diagnostic_admission_bytes": DIAGNOSTIC_ALLOWANCE, "report_limit_bytes": MAX_REPORT_BYTES,
        "physical_available_floor_bytes": 32 * GiB,
        "timing_limitation": "fixed baseline then S2 order; OS/storage/driver state is not reset; no isolated cache speedup claim"}
    relative = f"s2/components/{run_id}/plan.json"
    raw = (json.dumps(plan, sort_keys=True, ensure_ascii=False) + "\n").encode()
    path = workspace.artifact_path(relative, create_parent=True)
    with path.open("xb") as stream:
        stream.write(raw)
    return {"artifact": relative, "sha256": _sha(raw), "bytes": len(raw), "content": plan}


class _ObservedMoE:
    def __init__(self, owner, rows):
        self.owner, self.rows = owner, rows
    def __getattr__(self, name):
        return getattr(self.owner, name)
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
    def __exit__(self, *error):
        return self.owner.__exit__(*error)


def _route(result):
    route = result["routing"]["indices"]
    if (not isinstance(route, np.ndarray) or route.shape != (6,)
            or not np.issubdtype(route.dtype, np.integer) or len(set(route.tolist())) != 6
            or np.any(route < 0) or np.any(route >= 384)):
        raise ValueError("S2 components require six distinct original expert routes")


def _exercise(pool, inputs, report):
    def measured(name, operation):
        started = time.perf_counter()
        try:
            value = operation()
            records = _hash_arrays(value, name)
            if not records:
                raise ValueError("Native component returned no diagnostic arrays")
            report["array_hashes"].update(records)
            if (len(report["array_hashes"]) > MAX_ARRAYS or sum(
                    row["bytes"] for row in report["array_hashes"].values()) > DIAGNOSTIC_ALLOWANCE):
                raise MemoryError("Combined component diagnostics exceed their bounded allowance")
            return value
        finally:
            report["timings_seconds"][name] = time.perf_counter() - started
    for visit, layer in enumerate(LAYERS):
        prefix = f"visit{visit}-layer{layer}"
        hc = pool.proxy("hc", layer)
        mixes = measured(f"{prefix}/hc_mixes", lambda: hc.mixes(inputs["hidden"], "ffn"))
        collapsed = measured(f"{prefix}/hc_pre", lambda: hc.pre(inputs["hidden"], mixes["pre_fp32"]))
        activation = collapsed["output_bf16"]
        projection = pool.proxy("projection", layer)
        projected = measured(f"{prefix}/projection_wq_a", lambda: projection.fp8("wq_a", activation))
        del projected
        grouped = measured(f"{prefix}/projection_grouped", lambda: projection.grouped(inputs["grouped"]))
        del grouped
        moe = measured(f"{prefix}/moe", lambda: pool.proxy("moe", layer).forward(activation.reshape(5120), False))
        _route(moe)
        del mixes, collapsed, activation, moe
    records = report["array_hashes"]
    first = {k.split("/", 1)[1]: v for k, v in records.items() if k.startswith("visit0-layer0/")}
    last = {k.split("/", 1)[1]: v for k, v in records.items() if k.startswith("visit2-layer0/")}
    report["revisited_layer_array_bits_exact"] = bool(first and first == last)
    if not report["revisited_layer_array_bits_exact"]:
        raise ValueError("Revisited original layer changed exact diagnostic bits")


def _mode(mode, workspace, paths, inputs, reports):
    report = {"mode": mode, "array_hashes": {}, "timings_seconds": {}, "moe": [],
              "revisited_layer_array_bits_exact": False, "owned_resources_closed": False}
    reports.append(report)
    memory = assets = pool = catalog = backends = cache = admission = None
    started = time.perf_counter()
    try:
        memory = RuntimeMemory(spark_memory_policy())
        with memory:
            with ExitStack() as stack:
                admission = MemoryBudget(cap=DIAGNOSTIC_ALLOWANCE, owner="spark_s2_component_diagnostics")
                admission.acquire(DIAGNOSTIC_ALLOWANCE, category="diagnostics")
                stack.callback(admission.release, DIAGNOSTIC_ALLOWANCE, category="diagnostics")
                cache = stack.enter_context(spark_kernel_cache())
                report["kernel_cache_before"] = cache.stats()
                if mode == "s2":
                    catalog = stack.enter_context(_catalog(paths["metadata"], paths["originals"], paths["integrity"],
                        MemoryBudget(cap=512 * MiB, owner="spark_s2_authenticated_catalog"), workspace))
                kwargs = {"workspace": workspace, "fixtures": paths["fixtures"],
                          "budget": MemoryBudget(cap=SPARK_ASSET_CAP, owner="spark_s2_model_assets")}
                if catalog is not None:
                    kwargs["catalog"] = catalog
                assets = stack.enter_context(SparkModelAssets(paths["metadata"], paths["originals"], paths["integrity"], **kwargs))
                for layer in sorted(set(LAYERS)):
                    assets.moe_paths(layer)
                report["reader_before"] = assets.reader.stats()
                if mode == "s2":
                    backends = stack.enter_context(_backends(workspace))
                kwargs = {"backend_pool": backends} if backends is not None else {}
                pool = stack.enter_context(LayerOperatorPool(assets, **kwargs))
                factory = pool.moe_factory
                pool.moe_factory = lambda layer: _ObservedMoE(factory(layer), report["moe"])
                _exercise(pool, inputs, report)
                report["native_operators"] = {kind: operator.stats()
                    for kind, (_, operator, _) in pool.active.items()}
                report["reader_before_close"] = assets.reader.stats()
                report["assets_before_close"] = assets.stats()
                report["kernel_cache_before_close"] = cache.stats()
                if catalog is not None:
                    report["catalog_before_close"] = catalog.stats()
                if backends is not None:
                    report["backends_before_close"] = backends.stats()
    finally:
        gc.collect()
        if pool is not None:
            report["pool_after_close"] = pool.stats()
        if assets is not None:
            report["assets_after_close"] = assets.stats()
        if catalog is not None:
            report["catalog_after_close"] = catalog.stats()
        if backends is not None:
            report["backends_after_close"] = backends.stats()
        if cache is not None:
            report["kernel_cache_after_close"] = cache.stats()
        if admission is not None:
            report["diagnostic_reserved_bytes_after"] = admission.reserved
        if memory is not None:
            after = memory.snapshot()
            report["memory_after_close"] = after
            report["memory_runtime_closed"] = not memory.opened
            report["owned_resources_closed"] = bool(not memory.opened
                and after["charged_host_bytes"] == 0 and not any(after["device_used_bytes"].values())
                and after["live_leases"] == 0 and after["live_reclaimers"] == 0
                and after["inflight_operations"] == 0
                and after["monitor_thread_stopped"]
                and (assets is None or assets.reader is None and assets.leases["bytes"] == 0)
                and (admission is None or admission.reserved == 0))
        report["wall_seconds_including_initialization_and_cleanup"] = time.perf_counter() - started
    if not report["owned_resources_closed"]:
        raise RuntimeError("S2 component ownership cleanup could not be established")


def _failure(workspace, run_id, error, stage):
    relative = f"s2/components/{run_id}/failure.json"
    raw = (json.dumps({"failure_type": type(error).__name__, "stage": stage,
        "traceback": "".join(traceback.format_exception(error))}, ensure_ascii=False) + "\n").encode()
    if len(raw) > 65536:
        return {"saved": False, "reason": "Private diagnostic exceeded its bound"}
    try:
        path = workspace.artifact_path(relative, create_parent=True)
        with path.open("xb") as stream:
            stream.write(raw)
        return {"saved": True, "artifact": relative, "bytes": len(raw), "sha256": _sha(raw)}
    except Exception:
        return {"saved": False, "reason": "Private diagnostic could not be saved exclusively"}


def collect(workspace, *, run_id, metadata="s1/import-source", originals="s1/originals",
            integrity="s1/original-integrity", fixtures="s1/moe-original"):
    initial = _guard(workspace, run_id)
    paths = {name: workspace.artifact_path(value) for name, value in
        (("metadata", metadata), ("originals", originals), ("integrity", integrity), ("fixtures", fixtures))}
    inputs = _inputs()
    plan = _plan(workspace, run_id, initial, inputs)
    report = {"schema": SCHEMA, "status": "failed", "scope": "native HC/wq_a/grouped wo_a/MoE layer-switch regression",
        "model": MODEL, "revision": REVISION, "execution_target": TARGET, "plan": plan,
        "mode_order": list(MODES), "layer_sequence_per_mode": list(LAYERS), "modes": [],
        "array_bits_exact": False, "owned_resources_closed": False,
        "full_model_functional_acceptance": False, "full_model_quality_qualified": False,
        "original_kernel_parity_qualified": False, "qualification_receipt_issued": False,
        "general_performance_claim": False, "network_in_forward": False,
        "checkpoint_download_started": False, "automatic_retry": False}
    stage, started = "platform", time.perf_counter()
    try:
        if platform.system() != "Linux" or platform.machine().lower() not in ("aarch64", "arm64"):
            raise Unsupported("Spark S2 execution requires Linux AArch64")
        report["physical_memory_before"] = host_memory()
        require_capacity(report["physical_memory_before"], DIAGNOSTIC_ALLOWANCE)
        report["hardware"] = Driver().inventory()
        for mode in MODES:
            stage = mode
            _mode(mode, workspace, paths, inputs, report["modes"])
        first, second = (mode["array_hashes"] for mode in report["modes"])
        report["array_bits_exact"] = first == second
        report["compared_array_count"] = len(first)
        report["mismatched_arrays"] = sorted(name for name in set(first) | set(second)
            if first.get(name) != second.get(name))
        if not report["array_bits_exact"]:
            raise ValueError("Baseline and S2 returned different exact component diagnostic bits")
        report["status"] = "equivalent_s2_components"
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="cancelled" if isinstance(error, KeyboardInterrupt) else "failed",
            failure_type=type(error).__name__, failure_stage=stage,
            private_failure=_failure(workspace, run_id, error, stage))
    finally:
        del inputs
        gc.collect()
        report["owned_resources_closed"] = bool(report["modes"] and all(
            mode["owned_resources_closed"] for mode in report["modes"]))
        report["wall_seconds"] = time.perf_counter() - started
    if len(json.dumps(report, ensure_ascii=False).encode()) > MAX_REPORT_BYTES:
        raise ValueError("S2 component report exceeded its explicit bound")
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
    relative = f"s2/components/{args.run_id}.json"
    if workspace.result_path(relative).exists():
        raise FileExistsError("Refusing to overwrite earlier S2 component evidence")
    report = collect(workspace, run_id=args.run_id, metadata=args.metadata, originals=args.originals,
        integrity=args.integrity, fixtures=args.fixtures)
    path = workspace.write_result(relative, report, run_id=args.run_id, initial_context=initial)
    print(json.dumps({"status": report["status"], "array_bits_exact": report["array_bits_exact"],
        "result": path.relative_to(workspace.root).as_posix()}))
    return 0 if report["status"] == "equivalent_s2_components" else 2


if __name__ == "__main__":
    raise SystemExit(main())
