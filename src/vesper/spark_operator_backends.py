"""Explicit Spark-only reuse of two bounded native operator backends.

Layer weights remain owned by the borrowing operator. This owner retains only
Driver/context, exact kernel module, cuBLAS handle and its original workspace.
Nothing here qualifies arithmetic or changes the default discrete-GPU path.
"""

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import threading
import time

from .blas import Cublas, GATE_BLAS_WORKSPACE
from .checkpoint import MODEL, REVISION
from .cuda import Device, Driver, compile_probe, load_toolkit_library, _compiler_library_identity
from .execution_target import (SPARK_TARGET, compile_execution_kernel,
    kernel_source_sha256, resolve_execution_target)
from .hc import HC_CUBIN_SHA256
from .memory import current_policy
from .projections import GroupedBF16, PROJECTION_CUBIN_SHA256
from .resources import MemoryBudget, MiB, GiB

SPECS = {"hc": (128 * MiB, 64 * MiB, HC_CUBIN_SHA256),
         "projection": (512 * MiB, 256 * MiB, PROJECTION_CUBIN_SHA256)}


def _sources():
    root = Path(__file__).parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("spark_operator_backends.py", "hc.py", "projections.py")}


class _Backend:
    def __init__(self, pool, kind):
        self.pool, self.kind = pool, kind
        allowance, cap, _ = SPECS[kind]
        self.budget = MemoryBudget(cap=allowance, owner=f"spark_{kind}_backend")
        self.allowance, self.cap = allowance, cap
        self.driver = self.device = self.blas = self.module = self.nvrtc = None
        self.admitted = self.borrowed = self.poisoned = False
        self.compiler, self.provenance = {}, {}

    def open(self):
        target = resolve_execution_target(SPARK_TARGET)
        self.budget.acquire(self.allowance)
        self.admitted = True
        try:
            self.driver = Driver()
            self.inventory = self.driver.inventory()
            self.info = next(d for d in self.inventory["devices"] if d["ordinal"] == self.pool.ordinal)
            target.validate_device(self.info, self.inventory)
            self.device = Device(self.driver, self.pool.ordinal, cap=self.cap, free_floor=2 * GiB)
            flags = {"hc" if self.kind == "hc" else "projections": True}
            binary, self.provenance = compile_execution_kernel(compile_probe, target,
                compiler_info=self.compiler, pc_cubin_sha256=SPECS[self.kind][2], **flags)
            self.provenance["device"] = target.validate_device(self.info, self.inventory)
            self.source_sha256 = kernel_source_sha256(execution_target=target, **flags)
            self.nvrtc = load_toolkit_library("nvrtc")
            self.compiler_identity = _compiler_library_identity(self.nvrtc)
            if self.compiler_identity is None:
                raise ValueError("A reusable Spark backend requires an identified compiler library")
            self.module = self.device.load(binary)
            self.blas = Cublas() if self.kind == "hc" else GroupedBF16()
            self.provenance["cublas"] = target.validate_blas(self.blas.version, self.blas.math_mode)
            pointer = self.device.allocate(GATE_BLAS_WORKSPACE)
            self.blas.workspace(pointer, GATE_BLAS_WORKSPACE)
            self.base_allocations = dict(self.device.allocations)
            return self
        except BaseException:
            self.close()
            raise

    def validate(self):
        if (self.poisoned or not self.admitted or self.device is None or not self.device.context
                or self.blas is None or not self.blas.handle or self.module is None
                or self.module not in self.device.modules
                or self.device.allocations != self.base_allocations):
            raise RuntimeError("Reusable Spark backend is incomplete or requires cleanup")
        flags = {"hc" if self.kind == "hc" else "projections": True}
        target = resolve_execution_target(SPARK_TARGET)
        if (kernel_source_sha256(execution_target=target, **flags) != self.source_sha256
                or _compiler_library_identity(self.nvrtc) != self.compiler_identity
                or self.driver.inventory() != self.inventory):
            raise ValueError("Reusable Spark source/compiler/device identity changed")
        target.validate_blas(self.blas.version, self.blas.math_mode)
        self.device.current()

    def close(self):
        """Retain failed cleanup ownership so an explicit close can retry."""
        if self.borrowed:
            raise RuntimeError("A Spark backend still has a borrowing layer")
        try:
            if self.device is not None and self.device.context:
                self.device.current()
                # Free before Device.close: a failed free must not release its
                # primary context and make a live allocation impossible to retry.
                for pointer in list(self.device.allocations): self.device.free(pointer)
            if self.blas is not None:
                original_handle = self.blas.handle
                try: self.blas.close()
                except BaseException:
                    self.blas.handle = original_handle
                    raise
                self.blas = None
            if self.device is not None:
                while self.device.modules:
                    self.driver.check(self.driver.cuModuleUnload(self.device.modules[-1]))
                    self.device.modules.pop()
                self.device.close()
                self.device = None
            if self.admitted:
                self.budget.release(self.allowance)
                self.admitted = False
            self.module = self.nvrtc = self.driver = None
        except BaseException:
            self.poisoned = True
            raise


class SparkOperatorBackends:
    """One runtime owns at most one HC and one projection backend.

    Enter after RuntimeMemory, before layer operators. An idle backend can be
    reclaimed under pressure. Active layer owners cannot be evicted. Layers
    must free every non-workspace allocation before returning their borrow.
    """
    def __init__(self, workspace, ordinal=0):
        if type(ordinal) is not int or ordinal != 0:
            raise ValueError("Spark backend reuse requires the sole device ordinal zero")
        self.workspace, self.ordinal = workspace, ordinal
        self.lock = threading.RLock()
        self.entries, self.failed = {}, []
        self.opened = self.closed = self.poisoned = False
        self.opens = self.hits = self.evictions = self.borrows = 0
        self.open_seconds = 0.
        self.reclaimer = self.policy = None

    def __enter__(self):
        with self.lock:
            if self.opened or self.closed: raise RuntimeError("Spark backend owners open once")
            self.workspace.check()
            for filename in ("spark_operator_backends.py", "hc.py", "projections.py"):
                self.workspace.check_runtime_module(Path(__file__).parent / filename, "src/vesper/" + filename)
            self.context = self.workspace.context()
            if tuple(self.context[k] for k in ("platform", "profile", "model", "revision")) != (
                    "dgx-spark", "native", MODEL, REVISION):
                raise ValueError("Backend reuse requires the pinned Spark native workspace")
            self.policy = current_policy()
            if self.policy is None or not self.policy.config.unified_device_memory:
                raise ValueError("Backend reuse requires an active unified runtime")
            self.sources = _sources()
            self.reclaimer = self.policy.register_reclaimer(self.reclaim)
            self.opened = True
            return self

    def _ready(self):
        if not self.opened or self.closed or self.poisoned:
            raise RuntimeError("Spark backend owner is closed or requires cleanup")
        if current_policy() is not self.policy:
            raise RuntimeError("Spark backend owner belongs to another runtime")
        self.policy.checkpoint()
        if _sources() != self.sources:
            self.poisoned = True
            raise ValueError("Reusable Spark operator source changed")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") is not None or os.environ.get("NVIDIA_TF32_OVERRIDE") == "1":
            raise ValueError("Spark backend reuse requires the original cuBLAS math policy")

    @contextmanager
    def acquire(self, kind, ordinal, execution_target, library=None):
        target = resolve_execution_target(execution_target)
        if (type(kind) is not str or kind not in SPECS or type(ordinal) is not int
                or ordinal != self.ordinal or target.name != SPARK_TARGET or library is not None):
            raise ValueError("Expected an original Spark HC/projection backend and default library")
        if self.policy is None:
            raise RuntimeError("Open the Spark backend owner before borrowing")
        # Runtime policy callbacks take the same order to avoid pressure races.
        with self.policy.lock, self.lock:
            self._ready()
            backend = self.entries.get(kind)
            if backend is not None and backend.borrowed:
                raise RuntimeError("A backend cannot be shared by simultaneous layer owners")
            if backend is None:
                candidate = _Backend(self, kind)
                started = time.perf_counter()
                try: backend = candidate.open()
                except BaseException:
                    if candidate.admitted or candidate.device is not None:
                        self.failed.append(candidate); self.poisoned = True
                    raise
                finally: self.open_seconds += time.perf_counter() - started
                self.entries[kind] = backend
                self.opens += 1
            else:
                try: backend.validate()
                except BaseException:
                    backend.poisoned = self.poisoned = True
                    raise
                self.hits += 1
            backend.borrowed = True
            self.borrows += 1
        try:
            yield backend
        finally:
            with self.policy.lock, self.lock:
                backend.borrowed = False
                if backend.device is not None and backend.device.allocations != backend.base_allocations:
                    backend.poisoned = self.poisoned = True
                    raise RuntimeError("A layer left live allocations in its reusable backend")

    def reclaim(self, requested):
        with self.policy.lock, self.lock:
            freed = 0
            for kind, backend in list(self.entries.items()):
                if backend.borrowed or backend.poisoned: continue
                before = self.policy.host
                try:
                    backend.close()
                except BaseException:
                    # Partial cleanup can already have freed the workspace.
                    # Keep its owner and reject all new work until close retries.
                    backend.poisoned = self.poisoned = True
                    raise
                del self.entries[kind]
                self.evictions += 1
                freed += before - self.policy.host
                if freed >= requested: break
            return freed

    def stats(self):
        with self.lock:
            return {"opened": self.opened, "closed": self.closed, "poisoned": self.poisoned,
                "maximum_entries": 2, "entries": len(self.entries), "failed_cleanup_owners": len(self.failed),
                "opens": self.opens, "reuse_hits": self.hits, "borrows": self.borrows,
                "evictions": self.evictions, "open_seconds": self.open_seconds,
                "borrowers": sum(b.borrowed for b in self.entries.values()),
                "host_admission_bytes": sum(b.budget.reserved for b in [*self.entries.values(), *self.failed]),
                "device_owned_bytes": sum(sum(b.device.allocations.values()) for b in [*self.entries.values(), *self.failed]
                    if b.device is not None),
                "maximum_device_workspace_bytes": 2 * GATE_BLAS_WORKSPACE,
                "retained_layer_weights": False, "qualification_receipt_issued": False,
                "source_sha256": dict(getattr(self, "sources", {})),
                "backends": {kind: {"compiler": deepcopy(b.compiler), "execution": deepcopy(b.provenance),
                    "compiler_library_identity_sha256": hashlib.sha256(repr(b.compiler_identity).encode()).hexdigest()}
                    for kind, b in self.entries.items()}}

    def close(self):
        if self.policy is None: return
        with self.policy.lock, self.lock:
            errors = []
            for kind, backend in list(self.entries.items()):
                try: backend.close()
                except BaseException as error: errors.append(error)
                else: del self.entries[kind]
            for backend in list(self.failed):
                try: backend.close()
                except BaseException as error: errors.append(error)
                else: self.failed.remove(backend)
            if errors:
                self.poisoned = True
                raise RuntimeError("Spark backend cleanup requires an explicit retry") from errors[0]
            if self.reclaimer is not None:
                self.policy.unregister_reclaimer(self.reclaimer)
                self.reclaimer = None
            self.opened, self.closed = False, True

    def __exit__(self, *_): self.close()
