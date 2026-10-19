"""Bounded original Attention tensors for compressed profiles 2/20/24."""

import hashlib
import json
from pathlib import Path
import struct
import zipfile

import numpy as np

from .checkpoint import LIMIT, MODEL, REVISION, read_range, read_small
from .prefix import STORAGE_DTYPES
from .projections import FP8_LAYOUT
from .resources import MemoryBudget, MiB

LAYERS = (2, 20, 24)
LAYOUT = {}
for layer in LAYERS:
    for part, (rows, width) in {**FP8_LAYOUT, "wo_a": (8192, 4096)}.items():
        LAYOUT[f"layers.{layer}.attn.{part}.weight"] = ("F8_E4M3", (rows, width))
        LAYOUT[f"layers.{layer}.attn.{part}.scale"] = ("F8_E8M0", (rows // 32, width // 32))
    for part, width in (("q_norm", 1280), ("kv_norm", 512)):
        LAYOUT[f"layers.{layer}.attn.{part}.weight"] = ("BF16", (width,))
    LAYOUT[f"layers.{layer}.attn.attn_sink"] = ("F32", (64,))
FIXTURE_BYTES = sum(int(np.prod(s)) * STORAGE_DTYPES[d].itemsize for d, s in LAYOUT.values())


def invalid(a, dtype):
    if dtype == "BF16": return np.any((a & 0x7f80) == 0x7f80)
    if dtype == "F8_E4M3": return np.any((a & 127) == 127)
    if dtype == "F8_E8M0": return np.any(a == 255)
    return not np.isfinite(a).all()


def fetch_compressed_attention_fixture(directory):
    directory = Path(directory)
    if directory.exists(): raise FileExistsError("Refusing to replace a compressed Attention fixture")
    budget = MemoryBudget(cap=768 * MiB)
    budget.acquire(768 * MiB)
    try:
        index = json.loads(read_small("model.safetensors.index.json"))["weight_map"]
        arrays, headers, entries = {}, {}, []
        for name, (dtype, shape) in LAYOUT.items():
            shard = index[name]
            if shard not in headers:
                size = struct.unpack("<Q", read_range(shard, 0, 8))[0]
                if not 0 < size <= LIMIT: raise ValueError("Excessive Attention shard header")
                headers[shard] = json.loads(read_range(shard, 8, size)), size + 8
            header, start = headers[shard]
            tensor = header[name]
            begin, end = tensor["data_offsets"]
            size = int(np.prod(shape)) * STORAGE_DTYPES[dtype].itemsize
            if tensor["dtype"] != dtype or tensor["shape"] != list(shape) or begin < 0 or end - begin != size:
                raise ValueError("Original compressed Attention layout differs")
            a = np.empty(shape, STORAGE_DTYPES[dtype])
            view = a.view(np.uint8).reshape(-1)
            for offset in range(0, size, LIMIT):
                raw = read_range(shard, start + begin + offset, min(LIMIT, size - offset))
                view[offset:offset + len(raw)] = np.frombuffer(raw, np.uint8)
            if invalid(a, dtype): raise ValueError("Original Attention tensor is nonfinite")
            arrays[name] = a
            entries.append({"name": name, "dtype": dtype, "shape": list(shape), "bytes": size,
                            "shard": shard, "shard_byte_offset": start + begin, "sha256": hashlib.sha256(a.tobytes()).hexdigest()})
        manifest = {"kind": "original_compressed_attention_fixture", "model": MODEL, "revision": REVISION,
                    "layers": list(LAYERS), "tensors": entries, "weights_fetched_bytes": FIXTURE_BYTES,
                    "maximum_range_bytes": LIMIT, "scope": "39 original Attention tensors only; compressor/indexer fixtures are reused; no full shard/model"}
        directory.mkdir(parents=True, exist_ok=False)
        np.savez(directory / "native-compressed-attention.npz", **arrays)
        (directory / "manifest.json").write_bytes((json.dumps(manifest, indent=2) + "\n").encode())
        return {"probe": "fetch_compressed_attention_fixture", "status": "pass", **manifest}
    finally: budget.release(768 * MiB)


def load_compressed_attention_fixture(path):
    path = Path(path)
    meta = path.parent / "manifest.json"
    if meta.stat().st_size > 65536 or path.stat().st_size > FIXTURE_BYTES + 65536:
        raise ValueError("Compressed Attention fixture exceeds its bound")
    manifest = json.loads(meta.read_bytes())
    if (manifest.get("kind"), manifest.get("model"), manifest.get("revision"), manifest.get("layers")) != (
            "original_compressed_attention_fixture", MODEL, REVISION, list(LAYERS)):
        raise ValueError("Original compressed Attention fixture identity differs")
    entries = manifest.get("tensors", [])
    if manifest.get("weights_fetched_bytes") != FIXTURE_BYTES or len(entries) != len(LAYOUT) or {e.get("name") for e in entries} != set(LAYOUT):
        raise ValueError("Incomplete compressed Attention fixture")
    for e in entries:
        dtype, shape = LAYOUT[e["name"]]
        if (e.get("dtype"), e.get("shape"), e.get("bytes")) != (dtype, list(shape), int(np.prod(shape)) * STORAGE_DTYPES[dtype].itemsize):
            raise ValueError("Original Attention provenance layout differs")
    with zipfile.ZipFile(path) as archive:
        if sorted(archive.namelist()) != sorted(n + ".npy" for n in LAYOUT): raise ValueError("Attention archive members differ")
        for name, (dtype, shape) in LAYOUT.items():
            if archive.getinfo(name + ".npy").file_size > int(np.prod(shape)) * STORAGE_DTYPES[dtype].itemsize + 1024:
                raise ValueError("Excessive Attention array size")
            with archive.open(name + ".npy") as stream:
                version = np.lib.format.read_magic(stream)
                reader = {(1, 0): np.lib.format.read_array_header_1_0, (2, 0): np.lib.format.read_array_header_2_0}.get(version)
                if reader is None: raise ValueError("Unsupported Attention array header")
                actual, order, storage = reader(stream, max_header_size=1024)
                if actual != shape or order or storage != STORAGE_DTYPES[dtype]: raise ValueError("Attention array header exceeds bounded layout")
    with np.load(path, allow_pickle=False) as archive: arrays = {name: archive[name] for name in LAYOUT}
    for e in entries:
        a = arrays[e["name"]]
        if invalid(a, e["dtype"]) or hashlib.sha256(a.tobytes()).hexdigest() != e["sha256"]:
            raise ValueError("Attention bytes differ from finite original fixture")
    return arrays, manifest
