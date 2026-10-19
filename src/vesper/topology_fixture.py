"""Bounded per-layer original fixtures for the registered sharing groups."""

import hashlib
import json
from pathlib import Path
import struct
import zipfile

import numpy as np

from .attention_topology import REGISTERED_INPUT_LAYERS, role, compression_layout, indexer_layout
from .checkpoint import MODEL, REVISION, LIMIT, read_small, read_range
from .compressed_attention_fixture import invalid
from .prefix import STORAGE_DTYPES
from .projections import FP8_LAYOUT
from .resources import MemoryBudget, MiB


def layout(layer, *, profile="qualified"):
    from .prefix_profile import validate_layer
    validate_layer(layer, REGISTERED_INPUT_LAYERS, profile)
    rows = {}
    for part, (r, w) in {**FP8_LAYOUT, "wo_a": (8192, 4096)}.items():
        rows[f"layers.{layer}.attn.{part}.weight"] = ("F8_E4M3", (r, w))
        rows[f"layers.{layer}.attn.{part}.scale"] = ("F8_E8M0", (r // 32, w // 32))
    for part, width in (("q_norm", 1280), ("kv_norm", 512)):
        rows[f"layers.{layer}.attn.{part}.weight"] = ("BF16", (width,))
    rows[f"layers.{layer}.attn.attn_sink"] = ("F32", (64,))
    if role(layer).owns_kv: rows.update({n: ("BF16", s) for n, s in compression_layout((layer,)).items()})
    if role(layer).owns_index: rows.update(indexer_layout((layer,)))
    return rows


def fetch_layer(directory, layer, *, reuse=None, profile="qualified"):
    directory, specs = Path(directory), layout(layer, profile=profile)
    if directory.exists(): raise FileExistsError("Refusing to replace an existing original layer fixture")
    reuse = {} if reuse is None else reuse
    budget = MemoryBudget(cap=384 * MiB)
    budget.acquire(384 * MiB)
    try:
        index = json.loads(read_small("model.safetensors.index.json"))["weight_map"]
        arrays, entries, headers, fetched = {}, [], {}, 0
        for name, (dtype, shape) in specs.items():
            budget.acquire(0)
            size = int(np.prod(shape)) * STORAGE_DTYPES[dtype].itemsize
            shard = index[name]
            if shard not in headers:
                n = struct.unpack("<Q", read_range(shard, 0, 8))[0]
                if not 0 < n <= LIMIT: raise ValueError("Excessive topology shard header")
                headers[shard] = json.loads(read_range(shard, 8, n)), n + 8
            header, start = headers[shard]
            t = header[name]
            begin, end = t["data_offsets"]
            if t["dtype"] != dtype or t["shape"] != list(shape) or begin < 0 or end - begin != size:
                raise ValueError("Original topology tensor layout differs")
            if name in reuse:
                a = reuse[name]
                if a.dtype != STORAGE_DTYPES[dtype] or a.shape != shape or not a.flags.c_contiguous:
                    raise ValueError("Reused original tensor layout differs")
            else:
                a = np.empty(shape, STORAGE_DTYPES[dtype])
                view = a.view(np.uint8).reshape(-1)
                for offset in range(0, size, LIMIT):
                    raw = read_range(shard, start + begin + offset, min(LIMIT, size - offset))
                    view[offset:offset + len(raw)] = np.frombuffer(raw, np.uint8)
                fetched += size
            if invalid(a, dtype): raise ValueError("Original topology tensor is nonfinite")
            arrays[name] = a
            entries.append({"name": name, "dtype": dtype, "shape": list(shape), "bytes": size,
                            "shard": shard, "shard_byte_offset": start + begin, "sha256": hashlib.sha256(a.tobytes()).hexdigest()})
        manifest = {"kind": "original_topology_layer_fixture", "model": MODEL, "revision": REVISION,
                    "layer": layer, "tensors": entries, "tensor_bytes": sum(e["bytes"] for e in entries),
                    "weights_fetched_bytes": fetched, "weights_reused_bytes": sum(e["bytes"] for e in entries) - fetched,
                    "maximum_range_bytes": LIMIT}
        directory.mkdir(parents=True, exist_ok=False)
        np.savez(directory / "weights.npz", **arrays)
        (directory / "manifest.json").write_bytes((json.dumps(manifest, indent=2) + "\n").encode())
        return manifest
    finally: budget.release(384 * MiB)


def load_layer(directory, layer, *, profile="qualified"):
    directory, specs = Path(directory), layout(layer, profile=profile)
    size = sum(int(np.prod(s)) * STORAGE_DTYPES[d].itemsize for d, s in specs.values())
    meta, path = directory / "manifest.json", directory / "weights.npz"
    if meta.stat().st_size > 65536 or path.stat().st_size > size + 65536: raise ValueError("Layer fixture exceeds its bound")
    manifest = json.loads(meta.read_bytes())
    entries = manifest.get("tensors", [])
    if ((manifest.get("kind"), manifest.get("model"), manifest.get("revision"), manifest.get("layer")) !=
            ("original_topology_layer_fixture", MODEL, REVISION, layer) or manifest.get("tensor_bytes") != size
            or len(entries) != len(specs) or {e.get("name") for e in entries} != set(specs)):
        raise ValueError("Original topology fixture identity/layout differs")
    if (any(type(manifest.get(k)) is not int or manifest[k] < 0 for k in ("weights_fetched_bytes", "weights_reused_bytes"))
            or manifest["weights_fetched_bytes"] + manifest["weights_reused_bytes"] != size or manifest.get("maximum_range_bytes") != LIMIT):
        raise ValueError("Original topology fixture retrieval scope differs")
    for e in entries:
        dtype, shape = specs[e["name"]]
        if (e.get("dtype"), e.get("shape"), e.get("bytes")) != (dtype, list(shape), int(np.prod(shape)) * STORAGE_DTYPES[dtype].itemsize):
            raise ValueError("Topology tensor provenance layout differs")
    with zipfile.ZipFile(path) as archive:
        if sorted(archive.namelist()) != sorted(n + ".npy" for n in specs): raise ValueError("Topology archive members differ")
        for name, (dtype, shape) in specs.items():
            if archive.getinfo(name + ".npy").file_size > int(np.prod(shape)) * STORAGE_DTYPES[dtype].itemsize + 1024:
                raise ValueError("Excessive topology array member")
            with archive.open(name + ".npy") as stream:
                version = np.lib.format.read_magic(stream)
                reader = {(1, 0): np.lib.format.read_array_header_1_0, (2, 0): np.lib.format.read_array_header_2_0}.get(version)
                if reader is None: raise ValueError("Unsupported topology array header")
                actual, order, storage = reader(stream, max_header_size=1024)
                if actual != shape or order or storage != STORAGE_DTYPES[dtype]: raise ValueError("Topology array header exceeds layout")
    with np.load(path, allow_pickle=False) as archive: arrays = {n: archive[n] for n in specs}
    for e in entries:
        a = arrays[e["name"]]
        if invalid(a, e["dtype"]) or hashlib.sha256(a.tobytes()).hexdigest() != e["sha256"]:
            raise ValueError("Topology tensor bytes differ")
    return arrays, manifest


def fetch_topology_fixture(directory, reuse_paths=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    reuse, manifests = {}, []
    budget = MemoryBudget(cap=768 * MiB)
    budget.acquire(768 * MiB)
    try:
        if reuse_paths:
            from .compressed_attention_fixture import load_compressed_attention_fixture
            from .compression_fixture import load_compression_fixture
            from .indexer_fixture import load_indexer_fixture
            if len(reuse_paths) != 3: raise ValueError("All three original reuse fixtures are required")
            for loader, path in zip((load_compressed_attention_fixture, load_compression_fixture, load_indexer_fixture), reuse_paths):
                values, _ = loader(path)
                reuse.update(values)
        for layer in QUALIFIED_INPUT_LAYERS:
            target = directory / str(layer)
            if target.exists(): values, manifest = load_layer(target, layer)
            else: manifest = fetch_layer(target, layer, reuse=reuse)
            manifests.append(manifest)
    finally: budget.release(768 * MiB)
    return {"probe": "fetch_topology_fixture", "status": "pass", "layers": list(QUALIFIED_INPUT_LAYERS),
            "tensor_bytes": sum(m["tensor_bytes"] for m in manifests),
            "weights_fetched_bytes": sum(m["weights_fetched_bytes"] for m in manifests),
            "weights_reused_bytes": sum(m["weights_reused_bytes"] for m in manifests)}
