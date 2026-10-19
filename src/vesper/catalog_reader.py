"""Reuse authenticated small metadata; fetch only pinned payload ranges."""

from pathlib import Path
import struct
import threading
import time
import urllib.error

from .checkpoint import LIMIT, read_range
from .model_artifacts import load_catalog, read_checked, digest
from .resources import MemoryBudget, MiB


class CatalogReader:
    def __init__(self, source):
        self.source = Path(source)
        self.budget = MemoryBudget(cap=512 * MiB, owner="pinned_catalog_reader")
        self.lock = threading.RLock()
        self.metadata, self.headers, self.catalog = {}, {}, None
        self.payload_bytes = self.network_ranges = self.metadata_hits = 0

    def __enter__(self):
        self.budget.acquire(self.budget.cap)
        try:
            self.catalog, self.identity = load_catalog(self.source)
            self.shards = {s["shard"]: s for s in self.identity["shards"]}
            return self
        except BaseException:
            self.budget.release(self.budget.cap)
            raise

    def small(self, path):
        with self.lock:
            if self.catalog is None: raise RuntimeError("Use an open pinned catalog reader")
            self.budget.acquire(0)
            if path not in self.metadata: self.metadata[path] = read_checked(self.source, path)
            self.metadata_hits += 1
            return self.metadata[path]

    def range(self, shard, start, size):
        with self.lock:
            if self.catalog is None or shard not in self.shards:
                raise ValueError("Range is outside the authenticated original catalog")
            self.budget.acquire(0)
            meta = self.shards[shard]
            if type(start) is not int or type(size) is not int or start < 0 or not 0 < size <= LIMIT or start + size > meta["file_bytes"]:
                raise ValueError("Range exceeds the original shard extent or bounded transfer")
            if (start, size) in ((0, 8), (8, meta["header_bytes"])):
                if shard not in self.headers:
                    raw = (self.source / "headers" / (shard + ".json")).read_bytes()
                    if len(raw) != meta["header_bytes"] or digest(raw) != meta["header_sha256"]:
                        raise ValueError("Detached original header changed after catalog validation")
                    self.headers[shard] = raw
                self.metadata_hits += 1
                return struct.pack("<Q", meta["header_bytes"]) if start == 0 else self.headers[shard]
            if start < 8 + meta["header_bytes"]:
                raise ValueError("Expected an exact detached header read or original tensor payload")
        for attempt in range(4):
            self.budget.acquire(0)
            try:
                raw = read_range(shard, start, size)
                break
            except urllib.error.HTTPError as error:
                if error.code not in (429, 500, 502, 503, 504) or attempt == 3: raise
                time.sleep(2 ** attempt)
        with self.lock:
            self.payload_bytes += len(raw); self.network_ranges += 1
        return raw

    def stats(self):
        with self.lock:
            return {"payload_bytes_fetched": self.payload_bytes, "network_payload_ranges": self.network_ranges,
                "local_metadata_hits": self.metadata_hits, "maximum_range_bytes": LIMIT,
                "host_reserved_bytes": self.budget.reserved, "host_allowance_bytes": self.budget.cap}

    def __exit__(self, *_):
        self.catalog = None; self.metadata.clear(); self.headers.clear()
        self.budget.release(self.budget.cap)
