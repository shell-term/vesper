"""Separate-process GB10 full-vocabulary head checks over one original shard.

This component repository intentionally authenticates only head.weight and its
complete released shard. It neither relaxes all-48 runtime registration nor
downloads checkpoint/reference data. Native execution imports no Torch. The
reference lifts bounded BF16 rows into one device FP32 matrix and executes the
unchanged pinned ParallelHead.forward with world_size=1.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import ExitStack
import ctypes as C
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import time
from types import SimpleNamespace
import uuid
import weakref
import zipfile

import numpy as np

from .artifact_repository import VerifiedBlob, register_blob, BLOCK
from .blas import Cublas
from .checkpoint import MODEL, REVISION, REFERENCE_HASHES
from .conformance import compare
from .execution_target import resolve_execution_target
from .formats import bf16_bits, bf16_values
from .memory import RuntimeMemory, MemoryPolicy, process_memory
from .model_artifacts import load_catalog, shard_inventory, CATALOG_SHA256, SHARD_INVENTORY_SHA256
from .output_assets import VOCAB, WIDTH, TENSOR_SHA256
from .resources import MemoryBudget, MiB, GiB
from .spark_artifacts import _scope, _regular_below, _metadata_paths
from .spark_output import SparkOutputCandidate, validate_rows, HEAD_DEVICE_CAP
from .spark_probe import Driver, Unsupported, host_memory, require_capacity, validate_device

SCHEMA = "vesper.spark-head-probe.v1"
HEAD_SHA256 = TENSOR_SHA256["head.weight"]
TORCH_REVISION = "5c4886908584029761b579af026dcfb627c84070"
NATIVE_ADMISSION, REFERENCE_ADMISSION = 256 * MiB, 6 * GiB
DIAGNOSTIC_CAP, ROWS, REFERENCE_LIMIT = 64 * MiB, 512, MiB
PROCESS_INSTANCE = uuid.uuid4().hex
_EXECUTION_MODE = None
SCOPE = "original BF16 full-vocabulary head lifted exactly to FP32; generated BF16 hidden inputs"


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _read_json(path, limit=2 * MiB):
    with Path(path).open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("Excessive head component metadata")
    return json.loads(raw)


def _guard(workspace, run_id):
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", run_id):
        raise ValueError("Expected a simple head component run identity")
    _scope(workspace)
    workspace.check_runtime_module(__file__, "src/vesper/spark_head_probe.py")
    return workspace.context(run_id)


def head_spec(metadata):
    """Small pinned catalog only; no shard payload or network access."""
    _metadata_paths(metadata)
    catalog, _ = load_catalog(metadata)
    tensor = catalog["head.weight"]
    if (tensor["dtype"], tensor["shape"], tensor["bytes"]) != ("BF16", [VOCAB, WIDTH], VOCAB * WIDTH * 2):
        raise ValueError("Original full-vocabulary head layout differs")
    inventory = {row["file"]: row for row in shard_inventory(metadata)}
    expected = inventory[tensor["shard"]]
    if tensor["shard_byte_offset"] < 8 or tensor["shard_byte_offset"] + tensor["bytes"] > expected["bytes"]:
        raise ValueError("Original head leaves its released complete shard")
    return dict(tensor), dict(expected)


def component_plan(metadata, originals, integrity, *, workspace):
    _scope(workspace, metadata, originals, integrity)
    tensor, expected = head_spec(metadata)
    path = Path(originals) / expected["file"]
    present = path.is_file()
    if present:
        _regular_below(originals, path)
    return {"kind": "spark_head_component_plan", "model": MODEL, "revision": REVISION,
        "head_tensor": tensor, "head_tensor_sha256": HEAD_SHA256, "selected_original_shard": expected,
        "selected_shard_present": present, "selected_shard_size_matches": present and path.stat().st_size == expected["bytes"],
        "required_complete_original_shards": 1, "full_checkpoint_available_or_registered": False,
        "source_payload_opened": False, "network_payload_bytes": 0, "second_head_copy_created": False}


def _head_hash(blob, tensor, budget):
    sha = hashlib.sha256()
    for offset in range(0, tensor["bytes"], 4096):
        budget.acquire(0)
        sha.update(blob.read(tensor["shard_byte_offset"] + offset, min(4096, tensor["bytes"] - offset)))
    return sha.hexdigest()


def register_component(metadata, originals, integrity, *, workspace):
    """Explicit released whole-shard SHA plus pinned head SHA registration."""
    _scope(workspace, metadata, originals, integrity)
    tensor, expected = head_spec(metadata)
    source, target = Path(originals) / expected["file"], Path(integrity)
    _regular_below(originals, source)
    if source.stat().st_size != expected["bytes"]:
        raise ValueError("The component requires its complete original shard, never a partial")
    if target.exists() and any(target.iterdir()):
        raise FileExistsError("Refusing to replace an existing head component integrity repository")
    index_bytes = ((((expected["bytes"] + BLOCK - 1) // BLOCK) * 32 + 4095) // 4096) * 4096
    parent = target
    while not parent.exists():
        parent = parent.parent
    if shutil.disk_usage(parent).free < 64 * GiB + index_bytes:
        raise OSError("Head component registration would consume the SSD reserve")
    target.mkdir(parents=True, exist_ok=True)
    budget = MemoryBudget(cap=32 * MiB, owner="spark_head_registration")
    budget.acquire(8 * MiB, category="metadata")
    try:
        receipt = register_blob(source, expected, target / (expected["file"] + ".sha256"), budget)
        blob = VerifiedBlob(source, target, receipt, expected, budget)
        try:
            if _head_hash(blob, tensor, budget) != HEAD_SHA256:
                raise ValueError("Original head tensor SHA-256 differs")
        finally:
            blob.close()
        manifest = {"kind": "spark_registered_head_component", "model": MODEL, "revision": REVISION,
            "catalog_sha256": CATALOG_SHA256, "shard_inventory_sha256": SHARD_INVENTORY_SHA256,
            "tensor": tensor, "tensor_sha256": HEAD_SHA256, "receipt": receipt,
            "whole_selected_shard_sha256_verified": True, "head_tensor_sha256_verified": True,
            "required_complete_original_shards": 1, "full_checkpoint_available_or_registered": False,
            "source_open_mode": "read_only_direct", "source_hardlinks_allowed": True,
            "source_link_count_observed": source.stat().st_nlink, "network_payload_bytes": 0,
            "original_payload_duplicated": False}
        with (target / "manifest.json").open("xb") as stream:
            stream.write(_json(manifest)); stream.flush(); os.fsync(stream.fileno())
        return manifest
    finally:
        budget.release(8 * MiB, category="metadata")


class HeadRows:
    """Only the one selected verified shard; each returned owner is leased."""
    def __init__(self, metadata, originals, integrity, *, workspace):
        _scope(workspace, metadata, originals, integrity)
        workspace.check_runtime_module(__file__, "src/vesper/spark_head_probe.py")
        self.metadata, self.originals, self.integrity = map(Path, (metadata, originals, integrity))
        self.workspace, self.blob = workspace, None
        self.metadata_owned = False
        self.budget = MemoryBudget(cap=32 * MiB, owner="spark_head_component_rows")
        self.calls = self.logical_bytes = 0

    def __enter__(self):
        if self.blob is not None or self.metadata_owned:
            raise RuntimeError("Head component is already open")
        self.workspace.check()
        tensor, expected = head_spec(self.metadata)
        path = self.integrity / "manifest.json"
        _regular_below(self.integrity, path)
        manifest = _read_json(path, 65536)
        if any(manifest.get(k) != v for k, v in {
            "kind": "spark_registered_head_component", "model": MODEL, "revision": REVISION,
            "catalog_sha256": CATALOG_SHA256, "shard_inventory_sha256": SHARD_INVENTORY_SHA256,
            "tensor": tensor, "tensor_sha256": HEAD_SHA256, "whole_selected_shard_sha256_verified": True,
            "head_tensor_sha256_verified": True, "required_complete_original_shards": 1,
            "full_checkpoint_available_or_registered": False}.items()):
            raise ValueError("Head component registration differs from the pinned original")
        source = self.originals / expected["file"]
        _regular_below(self.originals, source)
        _regular_below(self.integrity, self.integrity / (expected["file"] + ".sha256"))
        self.budget.acquire(8 * MiB, category="metadata")
        self.metadata_owned = True
        try:
            self.blob = VerifiedBlob(source, self.integrity, manifest["receipt"], expected, self.budget)
            if _head_hash(self.blob, tensor, self.budget) != HEAD_SHA256:
                raise ValueError("Registered head payload differs from its pinned tensor SHA-256")
            self.tensor, self.expected, self.manifest = tensor, expected, manifest
            self.manifest_sha256 = _sha(path.read_bytes())
            return self
        except BaseException:
            self.__exit__()
            raise

    def __call__(self, start, count):
        if self.blob is None:
            raise RuntimeError("Use an open verified head component")
        if (type(start) is not int or type(count) is not int or start < 0 or not 1 <= count <= ROWS
                or start + count > VOCAB):
            raise ValueError("Expected one to 512 original head rows")
        size = count * WIDTH * 2
        self.budget.acquire(size, category="weights")
        try:
            owner = np.empty((count, WIDTH), np.uint16)
        except BaseException:
            self.budget.release(size, category="weights")
            raise
        weakref.finalize(owner, self.budget.release, size, category="weights")
        view, base = owner.view(np.uint8).reshape(-1), self.tensor["shard_byte_offset"] + start * WIDTH * 2
        for offset in range(0, size, 4096):
            self.budget.acquire(0)
            raw = self.blob.read(base + offset, min(4096, size - offset))
            view[offset:offset + len(raw)] = np.frombuffer(raw, np.uint8)
        validate_rows(owner, count)
        owner.setflags(write=False)
        self.calls += 1; self.logical_bytes += size
        return owner

    def stats(self):
        return {"head_only_component": True, "required_complete_original_shards": 1,
            "full_checkpoint_available_or_registered": False, "row_calls": self.calls,
            "logical_requested_row_bytes": self.logical_bytes, "maximum_rows_per_call": ROWS,
            "maximum_verified_source_span_bytes": 4096, "whole_host_head_materialized": False,
            "host_reserved_bytes": self.budget.reserved, "network_payload_bytes": 0,
            "manifest_sha256": getattr(self, "manifest_sha256", None)}

    def __exit__(self, *_):
        try:
            if self.blob is not None:
                self.blob.close()
        finally:
            self.blob = None
            if self.metadata_owned:
                self.budget.release(8 * MiB, category="metadata")
                self.metadata_owned = False


def cases():
    for batch in (1, 2):
        for pattern, seed in (("zero", None), ("pattern", None), ("random", 17), ("random", 2026)):
            meta = {"batch": batch, "pattern": pattern, "seed": seed}
            if pattern == "zero":
                values = np.zeros((batch, 1, WIDTH), np.float32)
            elif pattern == "pattern":
                values = np.resize(np.array([-8., -1., -.125, -0., 0., .125, 1., 8.], np.float32), (batch, 1, WIDTH))
            else:
                values = np.random.default_rng(seed).standard_normal((batch, 1, WIDTH)).astype(np.float32)
            yield meta, bf16_bits(values)


def case_identity():
    return [{**meta, "input_sha256": _sha(x.tobytes()), "shape": list(x.shape), "dtype": x.dtype.str} for meta, x in cases()]


def load_head_method(source):
    source = Path(source)
    _regular_below(source.parent, source)
    if not 0 < source.stat().st_size <= REFERENCE_LIMIT:
        raise ValueError("Original model source exceeds its bounded scope")
    raw = source.read_bytes()
    if _sha(raw) != REFERENCE_HASHES["model.py"]:
        raise ValueError("Original ParallelHead source differs from the pinned release")
    license = source.parent / "LICENSE"
    _regular_below(source.parent, license)
    if not 0 < license.stat().st_size <= 65536 or _sha(license.read_bytes()) != REFERENCE_HASHES["LICENSE"]:
        raise ValueError("The pinned original head source license is required")
    tree = ast.parse(raw)
    parent = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ParallelHead")
    node = next(n for n in parent.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    return node, {"model_source_sha256": _sha(raw), "license_sha256": REFERENCE_HASHES["LICENSE"],
                  "method": "ParallelHead.forward", "method_ast_sha256": _sha(ast.dump(node).encode()),
                  "method_body_modified": False, "world_size": 1}


class ReferenceHead:
    def __init__(self, rows, source, check):
        self.rows, self.source, self.check = rows, source, check
        self.stack, self.weight = ExitStack(), None

    def __enter__(self):
        node, self.source_identity = load_head_method(self.source)
        import torch
        self.torch = torch
        if (torch.__version__ != "2.14.1+cu130" or torch.version.git_version != TORCH_REVISION
                or torch.version.cuda != "13.0"):
            raise ValueError("Unregistered original head reference framework")
        self.previous_precision = torch.get_float32_matmul_precision()
        self.previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            self.stack.callback(torch.set_float32_matmul_precision, self.previous_precision)
            self.stack.callback(setattr, torch.backends.cuda.matmul, "allow_tf32", self.previous_tf32)
            if str(torch.backends.cuda.preferred_blas_library()) != "_BlasBackend.Cublas":
                raise ValueError("The reference requires the original cuBLAS backend")
            self.stack.enter_context(torch.inference_mode())
            self.stack.enter_context(torch.cuda.device(0))
            scope = {"torch": torch, "F": torch.nn.functional, "world_size": 1}
            exec(compile(ast.Module(body=[node], type_ignores=[]), "<pinned-original-ParallelHead.forward>", "exec"), scope)
            self.forward = scope["forward"]
            self.check()
            self.weight = torch.empty((VOCAB, WIDTH), dtype=torch.float32, device="cuda:0")
            for start in range(0, VOCAB, ROWS):
                self.check()
                original = self.rows(start, min(ROWS, VOCAB - start))
                host_block = original.copy()
                chunk = torch.from_numpy(host_block).view(torch.bfloat16).to("cuda:0", dtype=torch.float32)
                self.weight[start:start + len(original)].copy_(chunk)
                del chunk, original, host_block
            torch.cuda.synchronize(0)
            self.library = Cublas(); self.stack.callback(self.library.close)
            self.blas = self._blas_record()
            return self
        except BaseException:
            self.close()
            raise

    def _blas_record(self):
        handle = C.c_void_p(self.torch.cuda.current_blas_handle())
        version, mode = C.c_int(), C.c_int()
        self.library.check(self.library.cublasGetVersion_v2(handle, C.byref(version)))
        self.library.check(self.library.cublasGetMathMode(handle, C.byref(mode)))
        record = resolve_execution_target("spark-sm121-candidate").validate_blas(version.value, mode.value)
        return {**record, "handle_source": "torch.cuda.current_blas_handle", "preferred_backend": "Cublas"}

    def run(self, x):
        self.check()
        torch = self.torch
        tx = torch.from_numpy(x).view(torch.bfloat16).to("cuda:0")
        logits = self.forward(SimpleNamespace(weight=self.weight), tx)
        torch.cuda.synchronize(0)
        self.blas = self._blas_record()
        return {"promoted_fp32": tx[:, -1].float().cpu().numpy().copy(),
                "logits_fp32": logits.cpu().numpy().copy(),
                "greedy_ids_i64": logits.argmax(dim=-1).to(torch.int64).cpu().numpy().copy()}

    def close(self):
        torch = getattr(self, "torch", None)
        self.weight = None
        gc.collect()
        try:
            if torch is not None:
                torch.cuda.synchronize(0)
                torch.cuda.empty_cache()
        finally:
            self.stack.close()


def _layout():
    return {f"c{i:02d}_{name}": (shape, np.dtype(dtype)) for i, (meta, _) in enumerate(cases())
            for name, shape, dtype in (("promoted_fp32", (meta["batch"], WIDTH), np.float32),
                ("logits_fp32", (meta["batch"], VOCAB), np.float32), ("greedy_ids_i64", (meta["batch"],), np.int64))}


def _save(workspace, run_id, arrays):
    _validate_arrays(arrays)
    relative = f"s1/head/{run_id}/diagnostics.npz"
    path = workspace.artifact_path(relative, create_parent=True)
    with path.open("xb") as stream:
        np.savez(stream, **arrays)
    if path.stat().st_size > DIAGNOSTIC_CAP:
        raise ValueError("Head archive exceeds its explicit limit")
    return {"artifact": relative, "bytes": path.stat().st_size, "sha256": _sha(path.read_bytes())}


def _validate_arrays(arrays):
    expected = _layout()
    if set(arrays) != set(expected) or sum(a.nbytes for a in arrays.values()) > DIAGNOSTIC_CAP:
        raise ValueError("Incomplete or excessive full-vocabulary head diagnostics")
    for name, (shape, dtype) in expected.items():
        a = arrays[name]
        if not isinstance(a, np.ndarray) or a.shape != shape or a.dtype != dtype or not a.flags.c_contiguous:
            raise ValueError("Head diagnostic layout differs from the fixed corpus")
        if dtype == np.float32 and not np.isfinite(a).all():
            raise ValueError("Head diagnostics must contain finite complete logits")
    for i, (_, x) in enumerate(cases()):
        promoted = bf16_values(x[:, -1].copy())
        if not np.array_equal(arrays[f"c{i:02d}_promoted_fp32"].view(np.uint32), promoted.view(np.uint32)):
            raise ValueError("Head inputs were not promoted exactly from the fixed original BF16 corpus")
        if not np.array_equal(arrays[f"c{i:02d}_greedy_ids_i64"], np.argmax(arrays[f"c{i:02d}_logits_fp32"], axis=-1)):
            raise ValueError("Head greedy IDs do not select the first complete-vocabulary maximum")


def _load(workspace, report):
    receipt = report["diagnostics"]
    path = workspace.artifact_path(receipt["artifact"])
    if not 0 < path.stat().st_size <= DIAGNOSTIC_CAP or path.stat().st_size != receipt["bytes"]:
        raise ValueError("Head diagnostic archive extent differs")
    if _sha(path.read_bytes()) != receipt["sha256"]:
        raise ValueError("Head diagnostic archive SHA differs")
    expected = _layout()
    with zipfile.ZipFile(path) as archive:
        if sorted(archive.namelist()) != sorted(name + ".npy" for name in expected):
            raise ValueError("Head archive members differ")
        for name, (shape, dtype) in expected.items():
            info = archive.getinfo(name + ".npy")
            if info.file_size > int(np.prod(shape)) * dtype.itemsize + 1024:
                raise ValueError("Head NPY exceeds its bounded layout")
            with archive.open(info) as stream:
                version = np.lib.format.read_magic(stream)
                if version not in ((1, 0), (2, 0)):
                    raise ValueError("Unsupported head NPY header")
                reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                if reader(stream, max_header_size=1024) != (shape, False, dtype):
                    raise ValueError("Head NPY allocation differs from its fixed layout")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in expected}
    _validate_arrays(arrays)
    return arrays


def _claim(mode):
    global _EXECUTION_MODE
    if _EXECUTION_MODE is not None:
        raise Unsupported("Run native and reference head execution in separate fresh processes")
    if mode == "native" and "torch" in sys.modules:
        raise Unsupported("Native head execution must use a process without Torch")
    _EXECUTION_MODE = mode


def collect(mode, workspace, *, run_id, metadata="s1/head-component/metadata", originals="s1/head-component/originals",
            integrity="s1/head-component/integrity", reference_source="s1/head-component/reference/model.py"):
    initial = _guard(workspace, run_id)
    if mode not in ("plan", "register", "native", "reference"):
        raise ValueError("Unknown head component mode")
    paths = {k: workspace.artifact_path(v) for k, v in (("metadata", metadata), ("originals", originals),
                ("integrity", integrity), ("reference_source", reference_source))}
    report = {"schema": SCHEMA, "mode": mode, "status": "unsupported", "model": MODEL, "revision": REVISION,
        "scope": SCOPE, "case_identity": case_identity(), "cases": [], "original_kernel_parity_qualified": False,
        "full_model_quality_qualified": False, "general_performance_claim": False,
        "execution_process_instance": PROCESS_INSTANCE, "separate_process_mode": True,
        "full_checkpoint_available_or_registered": False, "network_payload_bytes": 0,
        "head_storage_dtype": "BF16", "head_compute_dtype": "float32", "full_vocabulary": VOCAB,
        "native_block_rows": ROWS, "native_device_cap_bytes": HEAD_DEVICE_CAP,
        "native_head_retained_on_device": True,
        "reference_admission_bytes": REFERENCE_ADMISSION, "diagnostic_cap_bytes": DIAGNOSTIC_CAP,
        "physical_available_floor_bytes": 32 * GiB, "framework_and_context_memory_not_hard_capped": True,
        "memory_observations": [], "owned_resources_closed": False, "memory_runtime_closed": False}
    policy = candidate = rows = None
    outputs = {}
    def check(pending=0):
        workspace.check()
        data = host_memory(); require_capacity(data, pending)
        report["memory_observations"].append(data)
        if policy is not None and policy.opened:
            policy.checkpoint()
    try:
        if platform.system() != "Linux" or platform.machine().lower() not in ("aarch64", "arm64"):
            raise Unsupported("Head component execution requires Linux AArch64")
        admission = REFERENCE_ADMISSION if mode == "reference" else NATIVE_ADMISSION
        check(admission + (HEAD_DEVICE_CAP if mode == "native" else 0))
        with RuntimeMemory(MemoryPolicy(host_cap_bytes=8 * GiB, pinned_cap_bytes=128 * MiB,
                device_caps={0:HEAD_DEVICE_CAP}, unified_device_memory=True)) as policy:
            budget = MemoryBudget(cap=admission + DIAGNOSTIC_CAP, owner="spark_head_phase")
            budget.acquire(admission)
            budget.acquire(DIAGNOSTIC_CAP, category="diagnostics")
            try:
                if mode == "plan":
                    report["plan"] = component_plan(paths["metadata"], paths["originals"], paths["integrity"], workspace=workspace)
                    report["status"] = "head_component_planned"
                elif mode == "register":
                    report["component"] = register_component(paths["metadata"], paths["originals"], paths["integrity"], workspace=workspace)
                    report["status"] = "head_component_registered"
                else:
                    _claim(mode)
                    report["observed_hardware"] = Driver().inventory()
                    with HeadRows(paths["metadata"], paths["originals"], paths["integrity"], workspace=workspace) as rows:
                        report["component"] = {k: rows.manifest[k] for k in ("model", "revision", "catalog_sha256", "tensor", "tensor_sha256", "receipt")}
                        started = time.perf_counter()
                        if mode == "native":
                            candidate = SparkOutputCandidate(rows, block_rows=ROWS)
                            candidate.__enter__()
                        else:
                            candidate = ReferenceHead(rows, paths["reference_source"], check).__enter__()
                            report["reference"] = candidate.source_identity
                            report["reference"].update(torch_version=candidate.torch.__version__, torch_revision=TORCH_REVISION,
                                torch_cuda_version=candidate.torch.version.cuda, full_device_fp32_head_bytes=VOCAB * WIDTH * 4,
                                maximum_host_weight_rows=ROWS, whole_host_head_materialized=False)
                        report["initialization_wall_seconds"] = time.perf_counter() - started
                        for i, (meta, x) in enumerate(cases()):
                            check()
                            started = time.perf_counter()
                            actual = candidate.project(x[:, -1].copy()) if mode == "native" else candidate.run(x)
                            outputs.update((f"c{i:02d}_{name}", value) for name, value in actual.items())
                            report["cases"].append({"id": i, "wall_seconds": time.perf_counter() - started})
                        report["blas"] = candidate.blas_provenance if mode == "native" else candidate.blas
                        if mode == "native":
                            report["native_before_close"] = candidate.stats()
                        else:
                            report["reference_framework_memory"] = {"peak_allocated_bytes": candidate.torch.cuda.max_memory_allocated(0),
                                "peak_reserved_bytes": candidate.torch.cuda.max_memory_reserved(0), "not_added_to_physical_capacity": True}
                        candidate.__exit__() if mode == "native" else candidate.close()
                        if mode == "native":
                            report["native_after_close"] = candidate.stats()
                            report["native_torch_imported"] = "torch" in sys.modules
                        if mode == "reference":
                            report["reference_framework_memory"].update(
                                allocated_bytes_after_weight_close=candidate.torch.cuda.memory_allocated(0),
                                reserved_bytes_after_weight_close=candidate.torch.cuda.memory_reserved(0),
                                CUDA_context_released_on_process_exit=True)
                        candidate = None
                        report["rows"] = rows.stats()
                    rows = None
                    report["diagnostics"] = _save(workspace, run_id, outputs)
                    report["status"] = f"{mode}_execution_pass"
            except Exception as error:
                report.update(status="unsupported", failure_type=type(error).__name__,
                    reason="Head component could not complete; no parity or model-quality claim")
            finally:
                try:
                    if candidate is not None:
                        candidate.__exit__() if mode == "native" else candidate.close()
                        candidate = None
                except Exception as error:
                    report.update(status="unsupported", cleanup_failure_type=type(error).__name__)
                outputs.clear(); actual = x = rows = None
                gc.collect()
                budget.release(DIAGNOSTIC_CAP, category="diagnostics")
                budget.release(admission)
                report["memory_after_owned_close"] = policy.snapshot()
                report["owned_resources_closed"] = bool(policy.host == policy.CONTROL_METADATA_BYTES
                    and not any(policy.device_used.values()) and not policy.reclaimers)
        report["memory_runtime_closed"] = True
        check()
    except Exception as error:
        report.update(status="unsupported", failure_type=type(error).__name__,
                      reason="Head component could not complete; no parity or model-quality claim")
    finally:
        if policy is not None:
            report["memory_after_runtime_close"] = policy.snapshot()
        report["process_after_cleanup"] = process_memory()
    if report["status"].endswith("_pass") and not (report["owned_resources_closed"] and report["memory_runtime_closed"]):
        report["status"] = "unsupported"
    workspace.stamp_result({}, run_id=run_id, initial_context=initial)
    return report


def compare_runs(workspace, native_result, reference_result, *, metadata="s1/head-component/metadata",
                 reference_source="s1/head-component/reference/model.py"):
    _guard(workspace, "head-comparison")
    metadata_path, source_path = workspace.artifact_path(metadata), workspace.artifact_path(reference_source)
    _scope(workspace, metadata_path, source_path)
    tensor, original = head_spec(metadata_path)
    _, source_identity = load_head_method(source_path)
    reports = []
    for relative, mode in ((native_result, "native"), (reference_result, "reference")):
        workspace.check_result(relative)
        report = _read_json(workspace.result_path(relative))
        if (report.get("schema"), report.get("mode"), report.get("status")) != (SCHEMA, mode, f"{mode}_execution_pass"):
            raise ValueError("Two successful separate-process head executions are required")
        if report.get("case_identity") != case_identity() or len(report.get("cases", [])) != len(case_identity()):
            raise ValueError("Head execution corpus differs")
        if (tuple(report.get(k) for k in ("model", "revision", "scope", "head_storage_dtype", "head_compute_dtype", "full_vocabulary"))
                != (MODEL, REVISION, SCOPE, "BF16", "float32", VOCAB)
                or report.get("owned_resources_closed") is not True or report.get("memory_runtime_closed") is not True):
            raise ValueError("Head execution source, scope or cleanup is unqualified")
        component = report.get("component", {})
        if (component.get("model"), component.get("revision")) != (MODEL, REVISION):
            raise ValueError("Head component belongs to another checkpoint")
        if (component.get("tensor") != tensor or component.get("tensor_sha256") != HEAD_SHA256
                or component.get("catalog_sha256") != CATALOG_SHA256
                or any(component.get("receipt", {}).get(k) != v for k, v in original.items())):
            raise ValueError("Head component is not bound to the qualified original tensor and shard")
        blas = report.get("blas", {})
        resolve_execution_target("spark-sm121-candidate").validate_blas(blas.get("version"), blas.get("math_mode"))
        hardware = report.get("observed_hardware")
        try:
            if (not isinstance(hardware, dict) or hardware.get("ordinal") != 0
                    or type(hardware.get("driver_api_version")) is not int or hardware["driver_api_version"] < 13000):
                raise ValueError("Missing GB10 driver profile")
            validate_device(hardware)
        except (KeyError, TypeError, Unsupported) as error:
            raise ValueError("Missing or foreign single-GB10 unified-memory profile") from error
        reports.append(report)
    native, reference = reports
    if (not all(isinstance(r.get("execution_process_instance"), str) and r["execution_process_instance"] for r in reports)
            or native["execution_process_instance"] == reference["execution_process_instance"]):
        raise ValueError("Head native/reference executions must come from separate processes")
    for key in ("model", "revision", "scope", "component", "observed_hardware", "head_storage_dtype", "head_compute_dtype", "full_vocabulary"):
        if native.get(key) != reference.get(key):
            raise ValueError("Head source, observed hardware or execution profile differs")
    if any(reference.get("reference", {}).get(k) != v for k, v in source_identity.items()):
        raise ValueError("Reference did not preserve the unchanged original ParallelHead method")
    ref = reference["reference"]
    if (tuple(ref.get(k) for k in ("torch_version", "torch_revision", "torch_cuda_version"))
            != ("2.14.1+cu130", TORCH_REVISION, "13.0")
            or reference["blas"].get("handle_source") != "torch.cuda.current_blas_handle"
            or reference["blas"].get("preferred_backend") != "Cublas"):
        raise ValueError("Reference framework or actual Torch BLAS handle profile differs")
    before, after = native.get("native_before_close", {}), native.get("native_after_close", {})
    if (any(before.get(k) != v for k, v in {"calls": len(case_identity()), "rows_read": VOCAB,
            "block_rows": ROWS, "full_vocabulary_rows": VOCAB, "head_compute_dtype": "float32", "TF32_enabled": False,
            "retained_head": True, "head_device_bytes": VOCAB * WIDTH * 4, "device_cap_bytes": HEAD_DEVICE_CAP}.items())
            or after.get("device_owned_bytes") != 0 or after.get("host_reserved_bytes") != 0
            or type(before.get("device_peak_owned_bytes")) is not int
            or not VOCAB * WIDTH * 4 <= before["device_peak_owned_bytes"] <= HEAD_DEVICE_CAP
            or native.get("native_torch_imported") is not False):
        raise ValueError("Native head full-vocabulary coverage or resource cleanup receipt differs")
    a, expected = _load(workspace, native), _load(workspace, reference)
    checks = [compare(expected[name].view(np.uint32) if expected[name].dtype == np.float32 else expected[name],
                      a[name].view(np.uint32) if a[name].dtype == np.float32 else a[name], name) for name in sorted(expected)]
    passed = all(c["status"] == "pass" for c in checks)
    return {"schema": SCHEMA, "mode": "compare", "status": "pass" if passed else "fail", "model": MODEL, "revision": REVISION,
        "scope": SCOPE, "checks": checks, "full_vocabulary": VOCAB, "component": native["component"],
        "original_kernel_parity_qualified": passed, "qualification_scope": "only the eight fixed generated full-vocabulary head inputs",
        "full_model_quality_qualified": False, "general_performance_claim": False,
        "native_blas": native["blas"], "reference_blas": reference["blas"], "reference": reference.get("reference"),
        "same_device_evidence": "same bound Spark workspace and observed GB10 profile; no GPU individual-identity attestation",
        "execution_result_sha256": {mode: _sha(workspace.result_path(relative).read_bytes()) for mode, relative in
                                    (("native", native_result), ("reference", reference_result))}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "register", "native", "reference", "compare"))
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--metadata", default="s1/head-component/metadata")
    parser.add_argument("--originals", default="s1/head-component/originals")
    parser.add_argument("--integrity", default="s1/head-component/integrity")
    parser.add_argument("--reference-source", default="s1/head-component/reference/model.py")
    parser.add_argument("--native-result")
    parser.add_argument("--reference-result")
    args = parser.parse_args(argv)
    if args.mode == "compare" and not (args.native_result and args.reference_result):
        parser.error("compare requires both independently executed result paths")
    from .workspace import Workspace
    workspace = Workspace.open(args.workspace)
    initial = _guard(workspace, args.run_id)
    relative = f"s1/head/{args.run_id}.json"
    if workspace.result_path(relative).exists():
        raise FileExistsError("Refusing to overwrite original head evidence")
    report = (compare_runs(workspace, args.native_result, args.reference_result, metadata=args.metadata,
        reference_source=args.reference_source) if args.mode == "compare" else
        collect(args.mode, workspace, run_id=args.run_id, metadata=args.metadata, originals=args.originals,
            integrity=args.integrity, reference_source=args.reference_source))
    path = workspace.write_result(relative, report, run_id=args.run_id, initial_context=initial)
    print(json.dumps({"mode": args.mode, "status": report["status"], "result": path.relative_to(workspace.root).as_posix()}))
    return 0 if report["status"] in ("head_component_planned", "head_component_registered", "native_execution_pass", "reference_execution_pass", "pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
