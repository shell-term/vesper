"""Eight original HC/RMS tensors per registered compressed block."""

import hashlib
import json
from pathlib import Path
import struct
import zipfile

import numpy as np

from .checkpoint import MODEL, REVISION, LIMIT, read_range, read_small
from .compressed_attention_fixture import invalid
from .prefix import STORAGE_DTYPES
from .resources import MemoryBudget, MiB

def block_layout(layer):
    if type(layer) is not int or layer not in (2, 3): raise ValueError("Unregistered compressed block fixture layer")
    rows = {f"layers.{layer}.hc_{p}_{n}": ("F32", s) for p in ("attn", "ffn")
            for n, s in (("fn", (24, 20480)), ("scale", (3,)), ("base", (24,)))}
    rows.update({f"layers.{layer}.{p}_norm.weight": ("BF16", (5120,)) for p in ("attn", "ffn")})
    return rows


LAYOUT = block_layout(2)
TENSOR_BYTES = sum(int(np.prod(s)) * STORAGE_DTYPES[d].itemsize for d, s in LAYOUT.values())


def fetch_block_fixture(directory, *, layer=2):
    layout = block_layout(layer)
    directory = Path(directory)
    if directory.exists(): raise FileExistsError("Refusing to replace an original block fixture")
    budget = MemoryBudget(cap=32 * MiB)
    budget.acquire(32 * MiB)
    try:
        index = json.loads(read_small("model.safetensors.index.json"))["weight_map"]
        arrays, entries, headers = {}, [], {}
        for name, (dtype, shape) in layout.items():
            budget.acquire(0)
            shard = index[name]
            if shard not in headers:
                size = struct.unpack("<Q", read_range(shard, 0, 8))[0]
                if not 0 < size <= LIMIT: raise ValueError("Excessive block shard header")
                headers[shard] = json.loads(read_range(shard, 8, size)), size + 8
            header, start = headers[shard]
            tensor = header[name]
            begin, end = tensor["data_offsets"]
            size = int(np.prod(shape)) * STORAGE_DTYPES[dtype].itemsize
            if (tensor["dtype"], tensor["shape"], end - begin) != (dtype, list(shape), size) or begin < 0:
                raise ValueError("Original block tensor layout differs")
            raw = read_range(shard, start + begin, size)
            a = np.frombuffer(raw, STORAGE_DTYPES[dtype]).reshape(shape).copy()
            if invalid(a, dtype): raise ValueError("Original block tensor is nonfinite")
            arrays[name] = a
            entries.append({"name": name, "dtype": dtype, "shape": list(shape), "bytes": size,
                "shard": shard, "shard_byte_offset": start + begin, "sha256": hashlib.sha256(raw).hexdigest()})
        manifest = {"kind": f"original_layer{layer}_block_fixture", "model": MODEL, "revision": REVISION, "layer": layer,
            "tensors": entries, "tensor_bytes": TENSOR_BYTES, "weights_fetched_bytes": TENSOR_BYTES, "maximum_range_bytes": LIMIT}
        directory.mkdir(parents=True, exist_ok=False)
        np.savez(directory / "weights.npz", **arrays)
        (directory / "manifest.json").write_bytes((json.dumps(manifest, indent=2) + "\n").encode())
        return manifest
    finally: budget.release(32 * MiB)


def load_block_fixture(directory, *, layer=2):
    layout = block_layout(layer)
    directory = Path(directory)
    path, meta = directory / "weights.npz", directory / "manifest.json"
    if path.stat().st_size > TENSOR_BYTES + 16384 or meta.stat().st_size > 16384:
        raise ValueError("Block fixture exceeds its bound")
    manifest = json.loads(meta.read_bytes())
    entries = manifest.get("tensors", [])
    if ((manifest.get("kind"), manifest.get("model"), manifest.get("revision"), manifest.get("layer")) !=
            (f"original_layer{layer}_block_fixture", MODEL, REVISION, layer) or manifest.get("tensor_bytes") != TENSOR_BYTES
            or manifest.get("weights_fetched_bytes") != TENSOR_BYTES or manifest.get("maximum_range_bytes") != LIMIT
            or len(entries) != len(layout) or {e.get("name") for e in entries} != set(layout)):
        raise ValueError("Original block fixture identity differs")
    for e in entries:
        d, s = layout[e["name"]]
        if (e.get("dtype"), e.get("shape"), e.get("bytes")) != (d, list(s), int(np.prod(s)) * STORAGE_DTYPES[d].itemsize):
            raise ValueError("Original block tensor provenance differs")
    with zipfile.ZipFile(path) as archive:
        if sorted(archive.namelist()) != sorted(n + ".npy" for n in layout): raise ValueError("Block archive members differ")
        for name, (d, s) in layout.items():
            if archive.getinfo(name + ".npy").file_size > int(np.prod(s)) * STORAGE_DTYPES[d].itemsize + 1024:
                raise ValueError("Excessive block array member")
            with archive.open(name + ".npy") as stream:
                version = np.lib.format.read_magic(stream)
                reader = {(1, 0): np.lib.format.read_array_header_1_0, (2, 0): np.lib.format.read_array_header_2_0}.get(version)
                if reader is None: raise ValueError("Unsupported block array header")
                shape, fortran, dtype = reader(stream, max_header_size=1024)
                if shape != s or fortran or dtype != STORAGE_DTYPES[d]: raise ValueError("Block array header differs")
    with np.load(path, allow_pickle=False) as archive: arrays = {n: archive[n] for n in layout}
    for e in entries:
        a = arrays[e["name"]]
        if invalid(a, e["dtype"]) or hashlib.sha256(a.tobytes()).hexdigest() != e["sha256"]:
            raise ValueError("Original block tensor bytes differ")
    return arrays, manifest
