"""Explicit allocation reservations and read-only physical-memory checks."""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
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
    fields = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, value = line.split(":", 1)
        fields[name] = int(value.split()[0]) * 1024
    return {"total": fields["MemTotal"], "available": fields["MemAvailable"]}


@dataclass
class MemoryBudget:
    """Tracks owned allocations, not total process RSS or external allocations."""

    cap: int = 96 * GiB
    available_floor: int = 32 * GiB
    reserved: int = 0
    peak: int = 0

    def __post_init__(self) -> None:
        if self.cap <= 0 or self.available_floor < 0:
            raise ValueError("Invalid memory budget")

    def acquire(self, size: int, available: int | None = None) -> None:
        if size < 0:
            raise ValueError("Negative allocation")
        if available is None:
            available = physical_memory()["available"]
        if size > self.cap - self.reserved:
            raise MemoryError("Allocation exceeds the owned-memory cap")
        if size > available - self.available_floor:
            raise MemoryError("Allocation would consume the system memory reserve")
        self.reserved += size
        self.peak = max(self.peak, self.reserved)

    def release(self, size: int) -> None:
        if size < 0 or size > self.reserved:
            raise ValueError("Invalid release")
        self.reserved -= size
