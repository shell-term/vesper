"""Lazy, leased views of verified local native Spark model tensors.

Opening assets authenticates metadata, not every possible embedding value.
Each array lookup verifies its original shard blocks and validates finite values
before publication. Arrays own their memory lease until the last NumPy view is
released; closing assets does not prematurely release borrowed arrays.
"""

from collections.abc import Mapping
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import threading
import weakref

import numpy as np

from .backbone_assets import controls
from .checkpoint import MODEL, REVISION
from .engram_assets import CONTROL_LAYOUT
from .gate import fetch_gate_fixture, load_gate_fixture
from .model_artifacts import CATALOG_SHA256, FILES, digest, load_catalog, read_checked
from .output_assets import VOCAB, WIDTH
from .resources import MemoryBudget, MiB, GiB
from .router import fetch_router_fixture, load_router_fixture
from .shared import fetch_shared_fixture, load_shared_fixture
from .spark_artifacts import OriginalTensorReader, CHUNK_BYTES, _scope, _regular_below, _metadata_paths, _download_lock

CONTROL_COUNT = 1251
CONTROL_BYTES = 7199083968
MAX_ARRAY_BYTES = 256 * MiB
FIXTURE_SCRATCH_BYTES = 160 * MiB
FIXTURE_QUOTA_BYTES = 2 * GiB
FIXTURE_LAYER_ALLOWANCE = 40 * MiB
STORAGE = {"BF16": np.dtype("<u2"), "F32": np.dtype("<f4"), "F16": np.dtype("<f2"),
           "I64": np.dtype("<i8"), "I32": np.dtype("<i4"), "I8": np.dtype("i1"),
           "U8": np.dtype("u1"), "F8_E4M3": np.dtype("u1"), "F8_E8M0": np.dtype("u1")}


def _layer(layer):
    if type(layer) is not int or not 0 <= layer < 40:
        raise ValueError("Expected an original backbone layer from zero to 39")
    return layer


def _finite(array, dtype):
    # Keep validation scratch bounded even for the 150 MiB Engram projection.
    for start in range(0, array.size, MiB):
        part = array.reshape(-1)[start:start + MiB]
        bad = (np.any((part & 0x7f80) == 0x7f80) if dtype == "BF16" else
               np.any(part == 255) if dtype == "F8_E8M0" else
               np.any((part & 127) == 127) if dtype == "F8_E4M3" else
               not np.isfinite(part).all() if dtype in ("F32", "F16") else False)
        if bad:
            raise ValueError("Original tensor contains nonfinite native values")


class _TensorMapping(Mapping):
    """Metadata enumeration is cheap; each value access owns a new array."""
    def __init__(self, assets, names):
        self.assets, self.names = assets, dict(names)

    def __len__(self):
        return len(self.names)

    def __iter__(self):
        return iter(self.names)

    def __contains__(self, name):
        return name in self.names

    def __getitem__(self, name):
        return self.assets._array(self.names[name])


class OriginalEmbeddingRows:
    """Explicit original-vocabulary provider; values validate on each lookup.

    This is deliberately not a Mapping of prevalidated prepared rows. There is
    no items() enumeration claiming that unrequested vocabulary rows were read.
    Membership denotes an addressable original token, not a completed read.
    """
    vocabulary, width = VOCAB, WIDTH
    validation = "verified-local-per-lookup"

    def __init__(self, assets):
        self.assets = assets

    def __contains__(self, token):
        return type(token) is int and 0 <= token < self.vocabulary

    def __getitem__(self, token):
        return self.assets.embedding(token)


class _LocalFixtureReader:
    """Existing fixture builders receive authenticated local ranges only."""
    def __init__(self, assets):
        self.assets = assets

    def small(self, name):
        return self.assets._small(name)

    def range(self, shard, start, size):
        a = self.assets
        with a.lock:
            a._check_open()
            if type(start) is not int or type(size) is not int or start < 0 or not 0 < size <= CHUNK_BYTES:
                raise ValueError("Local fixture range exceeds its bounded original extent")
            header = a.shards[shard]
            if (start, size) in ((0, 8), (8, header["header_bytes"])):
                path = a.metadata / "headers" / (shard + ".json")
                _regular_below(a.metadata, path)
                raw = path.read_bytes()
                if len(raw) != header["header_bytes"] or digest(raw) != header["header_sha256"]:
                    raise ValueError("Qualified original header changed before local fixture preparation")
                return struct.pack("<Q", len(raw)) if start == 0 else raw
            for name in a.shard_names.get(shard, ()):
                t = a.reader.tensors[name]
                offset = start - t["shard_byte_offset"]
                if 0 <= offset and offset + size <= t["bytes"]:
                    raw = bytearray(size)
                    a.reader.read_into(name, offset, raw)
                    return bytes(raw)
            raise ValueError("Local fixture range is outside the selected original tensors")


class SparkModelAssets:
    """One qualified catalog, small live shard set and uncached leased arrays.

    prepare_moe_fixtures() is an explicit SSD-only preparation step. moe_paths()
    never fetches or writes a missing fixture during model forward. The supplied
    budget is shared with the caller; the default 8 GiB cap is an admission
    ledger, not an RSS or unified-memory hardware limit.
    """
    def __init__(self, metadata, originals, integrity, *, workspace=None, budget=None, fixtures=None):
        self.workspace = _scope(workspace, metadata, originals, integrity)
        self.workspace.check_runtime_module(__file__, "src/vesper/spark_tensor_views.py")
        self.metadata, self.originals, self.integrity = map(Path, (metadata, originals, integrity))
        self.fixtures = Path(fixtures) if fixtures is not None else self.workspace.artifact_path("s1/moe-original")
        _scope(self.workspace, self.fixtures)
        self.budget = budget if budget is not None else MemoryBudget(cap=8 * GiB, owner="spark_model_assets")
        if self.budget.available_floor < 32 * GiB:
            raise ValueError("Spark model assets require the 32 GiB physical available-memory reserve")
        self.lock = threading.RLock()
        self.reader = None
        self.arrays = _TensorMapping(self, {})
        self.embedding_rows = OriginalEmbeddingRows(self)
        self.small_cache, self.moe_ready, self.source_hashes = {}, {}, {}
        self.small_bytes = 0
        self.leases = {"bytes": 0, "arrays": 0, "peak_bytes": 0}
        self._lease_lock = threading.RLock()

    def __enter__(self):
        with self.lock:
            if self.reader is not None:
                raise RuntimeError("Spark model assets are already open")
            self.workspace.check()
            _metadata_paths(self.metadata)
            catalog, identity = load_catalog(self.metadata)
            selected = controls(catalog)
            if len(selected) != CONTROL_COUNT or sum(t["bytes"] for t in selected.values()) != CONTROL_BYTES:
                raise ValueError("The complete original forty-layer control census differs")
            for name, shape in (("embed.weight", [VOCAB, WIDTH]), ("head.weight", [VOCAB, WIDTH]), ("norm.weight", [WIDTH])):
                t = catalog[name]
                if t["dtype"] != "BF16" or t["shape"] != shape or t["bytes"] != int(np.prod(shape)) * 2:
                    raise ValueError("Original embedding/head/norm layout differs")
            names = tuple(selected) + ("norm.weight", "embed.weight", "head.weight")
            self.shards = {s["shard"]: {k: s[k] for k in ("header_bytes", "header_sha256")} for s in identity["shards"]}
            del selected, catalog
            reader = OriginalTensorReader(self.metadata, self.originals, self.integrity, self.budget,
                                          tensor_names=names, workspace=self.workspace)
            try:
                self.reader = reader.__enter__()
                self.arrays = _TensorMapping(self, {n: n for n in names if n not in ("embed.weight", "head.weight")})
                self.shard_names = {}
                for name, t in self.reader.tensors.items():
                    self.shard_names.setdefault(t["shard"], []).append(name)
                return self
            except BaseException:
                self.reader = None
                raise

    def _check_open(self):
        if self.reader is None:
            raise RuntimeError("Use open Spark model assets")
        self.budget.acquire(0)

    @contextmanager
    def _scratch(self, size):
        self.budget.acquire(size, category="scratch")
        try:
            yield
        finally:
            self.budget.release(size, category="scratch")

    def _allocate(self, shape, dtype):
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        if not 0 < size <= MAX_ARRAY_BYTES:
            raise ValueError("A Spark tensor array must be at most 256 MiB")
        self.budget.acquire(size, category="weights")
        try:
            owner = np.empty(shape, dtype)
        except BaseException:
            self.budget.release(size, category="weights")
            raise
        tracker, lock, budget = self.leases, self._lease_lock, self.budget
        with lock:
            tracker["bytes"] += size; tracker["arrays"] += 1
            tracker["peak_bytes"] = max(tracker["peak_bytes"], tracker["bytes"])
        def release():
            with lock:
                budget.release(size, category="weights")
                tracker["bytes"] -= size; tracker["arrays"] -= 1
        # np.empty(shape) is the allocation owner. Derived NumPy views keep that
        # owner alive; no cache eviction or provider close ends this lease.
        weakref.finalize(owner, release)
        return owner

    def _array(self, name, *, byte_offset=0, shape=None):
        with self.lock:
            self._check_open()
            t = self.reader.tensors[name]
            dtype = STORAGE[t["dtype"]]
            shape = tuple(t["shape"] if shape is None else shape)
            owner = self._allocate(shape, dtype)
            view = owner.view(np.uint8).reshape(-1)
            if byte_offset < 0 or byte_offset + owner.nbytes > t["bytes"]:
                raise ValueError("Original tensor array leaves its source extent")
            for offset in range(0, owner.nbytes, CHUNK_BYTES):
                self.reader.read_into(name, byte_offset + offset, view[offset:offset + CHUNK_BYTES])
            with self._scratch(8 * MiB):
                _finite(owner, t["dtype"])
            owner.setflags(write=False)
            return owner

    def layer_arrays(self, layer):
        self._check_open()
        layer = _layer(layer)
        return _TensorMapping(self, {n: n for n in self.arrays if n.startswith(f"layers.{layer}.")})

    arrays_for_layer = layer_arrays

    def compression_arrays(self):
        return self.arrays

    def indexer_arrays(self):
        return self.arrays

    def engram_controls(self, layer):
        self._check_open()
        if type(layer) is not int or layer not in (1, 14):
            raise ValueError("Expected one of the two original Engram layers")
        names = {short: f"layers.{layer}.engram.{short}" for short in CONTROL_LAYOUT}
        for short, (dtype, shape) in CONTROL_LAYOUT.items():
            t = self.reader.tensors[names[short]]
            if t["dtype"] != dtype or t["shape"] != shape:
                raise ValueError("Original Engram control alias/layout differs")
        return _TensorMapping(self, names)

    def embedding(self, token):
        if type(token) is not int or not 0 <= token < VOCAB:
            raise ValueError("Expected an original vocabulary token")
        return self._array("embed.weight", byte_offset=token * WIDTH * 2, shape=(WIDTH,))

    def norm_weight(self):
        return self._array("norm.weight")

    def head_rows(self, start, count):
        if (type(start) is not int or type(count) is not int or start < 0 or count <= 0 or start + count > VOCAB
                or count * WIDTH * 2 > MAX_ARRAY_BYTES):
            raise ValueError("Expected bounded contiguous original output rows")
        return self._array("head.weight", byte_offset=start * WIDTH * 2, shape=(count, WIDTH))

    def _small(self, name):
        with self.lock:
            self._check_open()
            if name not in self.small_cache:
                size = FILES[name][1]
                self.budget.acquire(size, category="metadata")
                try:
                    raw = read_checked(self.metadata, name)
                    self.small_cache[name] = raw; self.small_bytes += size
                except BaseException:
                    self.budget.release(size, category="metadata")
                    raise
            return self.small_cache[name]

    def _paths(self, layer, root=None):
        base = self.fixtures / f"layer{_layer(layer)}" if root is None else root
        return tuple(base / part / f"native-{part}.npz" for part in ("gate", "router", "shared"))

    def _source_hash(self, name):
        if name not in self.source_hashes:
            t = self.reader.tensors[name]
            buffer = np.empty(min(CHUNK_BYTES, t["bytes"]), np.uint8)
            sha = hashlib.sha256()
            for offset in range(0, t["bytes"], buffer.nbytes):
                part = buffer[:min(buffer.nbytes, t["bytes"] - offset)]
                self.reader.read_into(name, offset, part)
                sha.update(part)
            self.source_hashes[name] = sha.hexdigest()
        return self.source_hashes[name]

    def _validate_moe(self, layer, root=None):
        paths, identities = self._paths(layer, root), []
        for part, path, loader in zip(("gate", "router", "shared"), paths,
                                      (load_gate_fixture, load_router_fixture, load_shared_fixture)):
            _regular_below(self.fixtures, path)
            manifest_path = path.parent / "manifest.json"
            _regular_below(self.fixtures, manifest_path)
            if not 0 < manifest_path.stat().st_size <= 65536:
                raise ValueError("Original MoE fixture metadata exceeds its bound")
            values, manifest = loader(path)
            del values
            if manifest.get("layer") != layer:
                raise ValueError("Original MoE fixture belongs to another layer")
            entries = [manifest["tensor"]] if part == "gate" else manifest["tensors"]
            for entry in entries:
                name = entry.get("tensor", entry["name"])
                t = self.reader.tensors[name]
                if any(entry.get(k) != t[k] for k in ("dtype", "shape", "bytes", "shard", "shard_byte_offset")):
                    raise ValueError("Local MoE fixture differs from the qualified original catalog")
                if entry.get("sha256") != self._source_hash(name):
                    raise ValueError("Local MoE fixture bytes differ from registered original tensor bytes")
            identities.append(digest(manifest_path.read_bytes()))
        return paths, tuple(identities)

    def prepare_moe_fixtures(self, layers=tuple(range(40))):
        """Explicit bounded SSD derivative preparation; no network is available."""
        layers = tuple(layers)
        if not layers or len(set(layers)) != len(layers):
            raise ValueError("Expected unique original MoE layers")
        for layer in layers:
            _layer(layer)
        with self.lock:
            self._check_open(); self.workspace.check()
            self.fixtures.mkdir(parents=True, exist_ok=True)
            guard = self.fixtures / ".download.lock"
            if guard.exists() or guard.is_symlink():
                _regular_below(self.fixtures, guard)
                if guard.stat().st_nlink != 1:
                    raise ValueError("Local MoE fixture lock must have one owned link")
            with _download_lock(self.fixtures), self._scratch(FIXTURE_SCRATCH_BYTES):
                for layer in layers:
                    self.budget.acquire(0); self.workspace.check()
                    target = self.fixtures / f"layer{layer}"
                    if not target.exists():
                        occupied = 0
                        for path in self.fixtures.rglob("*"):
                            if path.is_symlink():
                                raise ValueError("Local MoE fixture repository cannot contain links")
                            if path.is_file():
                                _regular_below(self.fixtures, path); occupied += path.stat().st_size
                        if occupied + FIXTURE_LAYER_ALLOWANCE > FIXTURE_QUOTA_BYTES:
                            raise OSError("Local MoE derivatives exceed their two GiB SSD quota")
                        if shutil.disk_usage(self.fixtures).free < 64 * GiB + FIXTURE_LAYER_ALLOWANCE:
                            raise OSError("Local MoE preparation would consume the SSD reserve")
                        stage = self.fixtures / f".layer{layer}.part"
                        stage.mkdir(exist_ok=False)
                        reader = _LocalFixtureReader(self)
                        for part, fetch in (("gate", fetch_gate_fixture), ("router", fetch_router_fixture), ("shared", fetch_shared_fixture)):
                            fetch(stage / part, layer=layer, reader=reader)
                        self._validate_moe(layer, stage)
                        if target.exists():
                            raise FileExistsError("Refusing to replace an original MoE fixture layer")
                        os.rename(stage, target)
                    self.moe_ready[layer] = self._validate_moe(layer)
            return {"kind": "spark_original_moe_derivatives", "model": MODEL, "revision": REVISION,
                    "catalog_sha256": CATALOG_SHA256, "layers": list(layers), "network_payload_bytes": 0,
                    "source": "registered_original_shards", "maximum_source_read_bytes": CHUNK_BYTES,
                    "second_expert_pack_created": False, "runtime_or_quality_qualified": False}

    def moe_paths(self, layer):
        with self.lock:
            self._check_open(); _layer(layer)
            if layer not in self.moe_ready:
                if not all(p.is_file() for p in self._paths(layer)):
                    raise FileNotFoundError("Prepare verified local MoE fixtures before model forward")
                with self._scratch(FIXTURE_SCRATCH_BYTES):
                    self.moe_ready[layer] = self._validate_moe(layer)
            paths, identities = self.moe_ready[layer]
            for path, expected in zip(paths, identities):
                _regular_below(self.fixtures, path)
                manifest = path.parent / "manifest.json"
                _regular_below(self.fixtures, manifest)
                if manifest.stat().st_size > 65536 or digest(manifest.read_bytes()) != expected:
                    raise ValueError("Prepared original MoE fixture manifest changed")
            return paths

    def stats(self):
        with self.lock, self._lease_lock:
            return {"kind": "spark_lazy_original_tensor_assets", "live_array_bytes": self.leases["bytes"],
                    "live_arrays": self.leases["arrays"], "peak_array_bytes": self.leases["peak_bytes"],
                    "maximum_array_bytes": MAX_ARRAY_BYTES, "maximum_source_read_bytes": CHUNK_BYTES,
                    "resident_array_cache": False, "whole_embedding_or_head_materialized": False,
                    "embedding_validation": OriginalEmbeddingRows.validation, "network_payload_bytes": 0,
                    "prepared_moe_layers": sorted(self.moe_ready), "reservation_is_RSS_limit": False}

    def __exit__(self, *_):
        with self.lock:
            try:
                if self.reader is not None:
                    self.reader.__exit__()
            finally:
                self.reader = None
                self.arrays = _TensorMapping(self, {})
                self.small_cache.clear()
                self.budget.release(self.small_bytes, category="metadata"); self.small_bytes = 0
                self.moe_ready.clear()
                self.source_hashes.clear()
