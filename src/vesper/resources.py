"""Explicit allocation reservations and read-only physical-memory checks."""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass,field
from contextlib import contextmanager,nullcontext
import threading
from pathlib import Path

GiB = 1024**3
MiB = 1024**2


def physical_memory() -> dict[str, int]:
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong)
                for name in ("total", "available", "page_total", "page_available", "virtual_total", "virtual_available", "extended")
            ]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("Cannot read physical memory status")
        return {"total": status.total, "available": status.available}
    # Observe the kernel afresh for every admission/checkpoint. Parsing only
    # the two required fields avoids work without reusing a stale observation.
    fields = {}
    with Path("/proc/meminfo").open("r", encoding="ascii") as stream:
        for line in stream:
            name, separator, value = line.partition(":")
            if name not in ("MemTotal", "MemAvailable"):
                continue
            parts = value.split()
            if (not separator or name in fields or len(parts) != 2 or parts[1] != "kB"
                    or not parts[0].isascii() or not parts[0].isdecimal()):
                raise ValueError("Invalid required physical-memory field")
            fields[name] = int(parts[0]) * 1024
            if len(fields) == 2:
                break
    if (len(fields) != 2 or fields["MemTotal"] <= 0
            or not 0 <= fields["MemAvailable"] <= fields["MemTotal"]):
        raise ValueError("Incomplete or inconsistent physical-memory observation")
    return {"total": fields["MemTotal"], "available": fields["MemAvailable"]}


@dataclass
class MemoryBudget:
    """Tracks owned allocations, not total process RSS or external allocations."""

    cap: int = 96 * GiB
    available_floor: int = 32 * GiB
    reserved: int = 0
    peak: int = 0
    owner: str = 'legacy_component'
    category: str = 'admission'
    _policy: object = field(default=None,init=False,repr=False,compare=False)
    _lock: object = field(default_factory=threading.RLock,init=False,repr=False,compare=False)
    _token: object = field(default_factory=object,init=False,repr=False,compare=False)

    def __post_init__(self) -> None:
        if self.cap <= 0 or self.available_floor < 0:
            raise ValueError("Invalid memory budget")

    def acquire(self, size: int, available: int | None = None, *, category: str | None = None) -> None:
        from .memory import current_policy
        policy=self._policy or current_policy()
        with (policy.lock if policy is not None else nullcontext()),self._lock:
            if policy is not None and self._policy is None and self.reserved:
                raise RuntimeError('Start the runtime memory policy before opening component allocations')
            self._acquire(size,available,category,policy)

    def _acquire(self,size,available,category,policy):
        explicit_available=available is not None
        if size < 0:
            raise ValueError("Negative allocation")
        if available is None:
            available = physical_memory()["available"]
        if size > self.cap - self.reserved:
            raise MemoryError("Allocation exceeds the owned-memory cap")
        if size > available - self.available_floor:
            if policy is None: raise MemoryError("Allocation would consume the system memory reserve")
            policy.checkpoint()
            if explicit_available: raise MemoryError("Component observation remains below its memory reserve; retry with a fresh observation")
            # A component's stricter local reserve also has to be restored.
            available=physical_memory()['available']
            if size>available-self.available_floor: raise MemoryError("Allocation would consume the component memory reserve")
        if policy is not None:
            policy.acquire_budget(self,size,category or self.category)
            if size: self._policy=policy
        self.reserved += size
        self.peak = max(self.peak, self.reserved)

    def release(self, size: int, *, category: str | None = None) -> None:
        policy=self._policy
        with (policy.lock if policy is not None else nullcontext()),self._lock:
            if size < 0 or size > self.reserved: raise ValueError("Invalid release")
            if policy is not None: policy.release_budget(self,size,category or self.category)
            self.reserved -= size
            if not self.reserved: self._policy=None

    @contextmanager
    def publication(self):
        from .memory import current_policy
        policy=self._policy or current_policy()
        with (policy.publication() if policy is not None else nullcontext()):
            self.acquire(0); yield

    @contextmanager
    def restoration(self):
        from .memory import current_policy
        policy=self._policy or current_policy()
        with (policy.restoration() if policy is not None else nullcontext()): yield

    def register_reclaimer(self,callback):
        from .memory import current_policy
        policy=self._policy or current_policy()
        return None if policy is None else (policy,policy.register_reclaimer(callback))

    @staticmethod
    def unregister_reclaimer(handle):
        if handle is not None: handle[0].unregister_reclaimer(handle[1])
