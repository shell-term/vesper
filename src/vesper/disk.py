"""Direct-I/O probes with one aligned buffer and bounded reads.

An existing model file is opened read-only. Synthetic scratch uses a unique
new file and deletes only that file, never a directory or a user-provided file.
"""

from __future__ import annotations

import ctypes as C
import mmap
import os
from pathlib import Path
import statistics
import time
import uuid

import numpy as np

from .bandwidth import EXPERT_BYTES
from .resources import GiB, MiB, MemoryBudget


class DirectReader:
    def __init__(self, path: Path, capacity: int, budget: MemoryBudget):
        self.capacity, self.budget = capacity, budget
        self.handle, self.buffer = None, None
        budget.acquire(capacity)
        try:
            if os.name == "nt":
                self.lib = C.WinDLL("kernel32", use_last_error=True)
                self.lib.CreateFileW.argtypes = [C.c_wchar_p, C.c_uint32, C.c_uint32, C.c_void_p,
                                               C.c_uint32, C.c_uint32, C.c_void_p]
                self.lib.CreateFileW.restype = C.c_void_p
                self.lib.VirtualAlloc.argtypes = [C.c_void_p, C.c_size_t, C.c_uint32, C.c_uint32]
                self.lib.VirtualAlloc.restype = C.c_void_p
                self.lib.VirtualFree.argtypes = [C.c_void_p, C.c_size_t, C.c_uint32]
                self.lib.CloseHandle.argtypes = [C.c_void_p]
                self.lib.SetFilePointerEx.argtypes = [C.c_void_p, C.c_int64, C.c_void_p, C.c_uint32]
                self.lib.ReadFile.argtypes = [C.c_void_p, C.c_void_p, C.c_uint32, C.POINTER(C.c_uint32), C.c_void_p]
                self.handle = self.lib.CreateFileW(str(path), 0x80000000, 3, None, 3, 0x20000000, None)
                if self.handle == C.c_void_p(-1).value:
                    self.handle = None
                    raise C.WinError(C.get_last_error())
                self.buffer = self.lib.VirtualAlloc(None, capacity, 0x3000, 4)
                if not self.buffer:
                    raise C.WinError(C.get_last_error())
            else:
                if not hasattr(os, "O_DIRECT") or not hasattr(os, "preadv"):
                    raise RuntimeError("Direct I/O is unavailable; buffered fallback is intentionally disabled")
                self.handle = os.open(path, os.O_RDONLY | os.O_DIRECT)
                self.buffer = mmap.mmap(-1, capacity)
        except Exception:
            self.close()
            raise

    def read(self, offset: int, size: int) -> bytes:
        if offset < 0 or offset % 4096 or size <= 0 or size % 4096 or size > self.capacity:
            raise ValueError("Direct read is not aligned or exceeds the buffer")
        if os.name == "nt":
            count = C.c_uint32()
            if not self.lib.SetFilePointerEx(self.handle, offset, None, 0):
                raise C.WinError(C.get_last_error())
            if not self.lib.ReadFile(self.handle, self.buffer, size, C.byref(count), None):
                raise C.WinError(C.get_last_error())
            if count.value != size:
                raise OSError("Short direct read")
            return C.string_at(self.buffer, 4096)
        view = memoryview(self.buffer)[:size]
        try:
            if os.preadv(self.handle, [view], offset) != size:
                raise OSError("Short direct read")
            return bytes(view[:4096])
        finally:
            view.release()

    def close(self):
        if os.name == "nt":
            if self.buffer:
                self.lib.VirtualFree(self.buffer, 0, 0x8000)
            if self.handle is not None:
                self.lib.CloseHandle(self.handle)
        else:
            if self.buffer is not None:
                self.buffer.close()
            if self.handle is not None:
                os.close(self.handle)
        self.buffer, self.handle = None, None
        if self.budget is not None:
            self.budget.release(self.capacity)
            self.budget = None

    def array(self) -> np.ndarray:
        """Borrow the aligned read buffer; callers must not retain it after close."""
        if self.buffer is None:
            raise RuntimeError("Direct reader is closed")
        if os.name == "nt":
            return np.ctypeslib.as_array((C.c_uint8 * self.capacity).from_address(self.buffer))
        return np.frombuffer(self.buffer, dtype=np.uint8, count=self.capacity)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def benchmark(directory: Path, existing_file: Path | None = None) -> dict:
    directory = Path(directory).resolve()
    created = False
    budget = MemoryBudget(cap=64 * MiB, available_floor=32 * GiB)
    if existing_file is None:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"vesper-{uuid.uuid4().hex}.scratch"
        budget.acquire(3 * MiB)
        try:
            rng = np.random.default_rng(20261018)
            with path.open("xb", buffering=0) as stream:
                created = True
                for _ in range(288):
                    stream.write(rng.bytes(MiB))
                os.fsync(stream.fileno())
        except Exception:
            if created and path.resolve().parent == directory:
                path.unlink()
            raise
        finally:
            budget.release(3 * MiB)
    else:
        path = Path(existing_file).resolve(strict=True)
    try:
        before = path.stat()
        if before.st_size < EXPERT_BYTES:
            raise ValueError("The probe file must contain at least one expert-sized region")
        max_start_page = (before.st_size - EXPERT_BYTES) // 4096
        rng = np.random.default_rng(20261018)
        patterns = {
            "expert_sized_sequential": (EXPERT_BYTES, [i * EXPERT_BYTES for i in range(min(12, before.st_size // EXPERT_BYTES))]),
            "expert_sized_random": (EXPERT_BYTES, [int(i) * 4096 for i in rng.integers(0, max_start_page + 1, 12)]),
            "engram_page_random": (4096, [int(i) * 4096 for i in rng.integers(0, before.st_size // 4096, 64)]),
        }
        measurements = []
        with DirectReader(path, EXPERT_BYTES, budget) as reader, path.open("rb", buffering=0) as verifier:
            for label, (size, offsets) in patterns.items():
                samples = []
                for offset in offsets:
                    started = time.perf_counter()
                    head = reader.read(offset, size)
                    elapsed = time.perf_counter() - started
                    verifier.seek(offset)
                    if head != verifier.read(4096):
                        raise AssertionError("Direct read and ordinary read returned different page bytes")
                    samples.append(elapsed)
                median = statistics.median(samples)
                measurements.append({"pattern": label, "request_bytes": size, "requests": len(samples),
                                     "median_ms": median * 1000, "min_ms": min(samples) * 1000,
                                     "effective_GBps": size / median / 1e9,
                                     "first_page_at_every_offset_verified": True})
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError("Probe file changed during measurement")
        return {"probe": "disk_bandwidth", "status": "pass", "direct_io": True, "queue_depth": 1,
                "file_kind": "unique_synthetic_scratch" if created else "existing_read_only_file",
                "file_bytes": before.st_size, "peak_reserved_buffer_bytes": budget.peak,
                "measurements": measurements,
                "limitations": ["Queue-depth-one direct reads; no buffered fallback or asynchronous overlap.",
                                "Drive firmware caches and other workload contention can affect results.",
                                "4 KiB reads model physical Engram pages, not the complete Engram lookup algorithm."]}
    finally:
        if created:
            if path.resolve().parent != directory:
                raise RuntimeError("Scratch target moved outside its owned directory; refusing cleanup")
            path.unlink()
