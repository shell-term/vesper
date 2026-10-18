"""Immutable, aligned native-expert store; no model-wide mapping or page cache."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np

from .bandwidth import EXPERT_BYTES
from .checkpoint import EXPERT_LAYOUT, MODEL, REVISION, load_expert_fixture, validate_expert_identity, validate_expert_provenance
from .disk import DirectReader
from .resources import MemoryBudget, MiB


def expert_key(layer: int, expert: int) -> str:
    validate_expert_identity(layer, expert)
    return f"{layer}:{expert}"


def tensor_layout() -> dict:
    offset, result = 0, {}
    for name, shape in EXPERT_LAYOUT.items():
        size = int(np.prod(shape))
        result[name] = {"offset": offset, "bytes": size, "shape": list(shape)}
        offset += size
    if offset != EXPERT_BYTES:
        raise ValueError("Native expert layout differs from the fixed expert size")
    return result


def build_store(fixtures: list[Path], directory: Path) -> dict:
    """Create a new store from at most four individually verified expert fixtures."""
    if not 1 <= len(fixtures) <= 4:
        raise ValueError("The store-building probe accepts one to four real expert fixtures")
    directory = Path(directory).resolve()
    if directory.exists():
        raise FileExistsError("Refusing to replace an existing native store")
    budget = MemoryBudget(cap=64 * MiB)
    budget.acquire(64 * MiB)
    created = False
    try:
        directory.mkdir(parents=True, exist_ok=False)
        created = True
        entries, seen = [], set()
        with (directory / "experts.bin").open("xb", buffering=0) as stream:
            for fixture in fixtures:
                arrays, provenance = load_expert_fixture(fixture)
                key = expert_key(provenance["layer"], provenance["expert"])
                if key in seen:
                    raise ValueError("Store entries must identify different original experts")
                seen.add(key)
                digest = hashlib.sha256()
                offset = stream.tell()
                for name in EXPERT_LAYOUT:
                    raw = memoryview(arrays[name]).cast("B")
                    digest.update(raw)
                    if stream.write(raw) != len(raw):
                        raise OSError("Short native-store write")
                entries.append({"key": key, "offset": offset, "bytes": EXPERT_BYTES,
                                "sha256": digest.hexdigest(), "provenance": provenance})
                del arrays, raw
            os.fsync(stream.fileno())
        result = {"schema_version": 1, "kind": "vesper_native_expert_store", "model": MODEL, "revision": REVISION,
                  "expert_bytes": EXPERT_BYTES, "file": "experts.bin", "file_bytes": len(entries) * EXPERT_BYTES,
                  "layout": tensor_layout(), "entries": entries}
        (directory / "manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return {"probe": "build_native_store", "status": "pass", "experts": len(entries),
                "file_bytes": result["file_bytes"], "model": MODEL, "revision": REVISION,
                "note": "Original packed bytes only; a small immutable store, not a full model."}
    except Exception:
        if created:
            # Only these files in the newly created directory belong to this call.
            for name in ("manifest.json", "experts.bin"):
                (directory / name).unlink(missing_ok=True)
            directory.rmdir()
        raise
    finally:
        budget.release(64 * MiB)


class NativeStore:
    """Validated metadata plus one fixed direct-I/O buffer, opened read-only."""

    def __init__(self, directory: Path, budget: MemoryBudget):
        self.directory = Path(directory).resolve(strict=True)
        self.budget, self.reader = budget, None
        self.read_calls, self.bytes_read = 0, 0
        with (self.directory / "manifest.json").open("rb") as stream:
            raw = stream.read(MiB + 1)
        if len(raw) > MiB:
            raise ValueError("Native-store metadata exceeds the probe's bounded scope")
        self.manifest = json.loads(raw)
        m = self.manifest
        if (m.get("schema_version"), m.get("kind"), m.get("model"), m.get("revision"), m.get("expert_bytes"), m.get("file")) != (
                1, "vesper_native_expert_store", MODEL, REVISION, EXPERT_BYTES, "experts.bin"):
            raise ValueError("Native store does not match the pinned format")
        if m.get("layout") != tensor_layout() or not 1 <= len(m.get("entries", [])) <= 4:
            raise ValueError("Unexpected native-store layout or probe size")
        self.entries = {}
        for ordinal, entry in enumerate(m["entries"]):
            p = entry["provenance"]
            validate_expert_provenance(p)
            key = expert_key(p.get("layer"), p.get("expert"))
            if (entry.get("key") != key or key in self.entries or entry.get("offset") != ordinal * EXPERT_BYTES
                    or entry.get("bytes") != EXPERT_BYTES or not re.fullmatch(r"[0-9a-f]{64}", entry.get("sha256", ""))):
                raise ValueError("Invalid or overlapping native-store entry")
            self.entries[key] = entry
        self.size = EXPERT_BYTES
        self.path = self.directory / "experts.bin"
        if m.get("file_bytes") != len(self.entries) * EXPERT_BYTES or self.path.stat().st_size != m["file_bytes"]:
            raise ValueError("Native store file size differs from its complete entries")

    def __enter__(self):
        if self.reader is not None:
            raise RuntimeError("Native store is already open")
        self.reader = DirectReader(self.path, self.size, self.budget)
        return self

    def validate(self, key: str, data: np.ndarray, *, tensors: bool = False):
        entry = self.entries[key]
        if data.dtype != np.uint8 or data.shape != (self.size,) or not data.flags.c_contiguous:
            raise ValueError("Native-store payload layout differs from one expert")
        if hashlib.sha256(memoryview(data)).hexdigest() != entry["sha256"]:
            raise ValueError("Native expert bytes differ from the store hash")
        if tensors:
            layout = self.manifest["layout"]
            for tensor in entry["provenance"]["tensors"]:
                region = layout[tensor["name"]]
                view = data[region["offset"]:region["offset"] + region["bytes"]]
                if hashlib.sha256(memoryview(view)).hexdigest() != tensor["sha256"]:
                    raise ValueError("Native expert tensor differs from its original source hash")

    def read_into(self, key: str, target: np.ndarray):
        if self.reader is None:
            raise RuntimeError("Native store is closed")
        entry = self.entries[key]
        self.reader.read(entry["offset"], self.size)
        self.read_calls += 1
        self.bytes_read += self.size
        np.copyto(target, self.reader.array())
        self.validate(key, target, tensors=True)

    def __exit__(self, *_):
        if self.reader is not None:
            self.reader.close()
            self.reader = None
