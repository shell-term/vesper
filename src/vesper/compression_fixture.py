"""Bounded original compressor matrices for source layers 2 (r=2) and 20 (r=1)."""

import hashlib
import json
from pathlib import Path
import struct
import zipfile

import numpy as np

from .checkpoint import LIMIT, MODEL, REVISION, read_range, read_small
from .resources import MemoryBudget, MiB

COMPRESSION_LAYOUT = {}
for layer in (2, 20):
    for part, shape in (("wkv", (512, 5120)), ("norm", (512,))):
        COMPRESSION_LAYOUT[f"layers.{layer}.attn.compressor.{part}.weight"] = shape
COMPRESSION_LAYOUT["layers.2.attn.compressor.wgate.weight"] = (512, 5120)
COMPRESSION_BYTES = sum(int(np.prod(s)) * 2 for s in COMPRESSION_LAYOUT.values())


def fetch_compression_fixture(directory):
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError("Refusing to replace an existing compression fixture")
    budget = MemoryBudget(cap=128 * MiB)
    budget.acquire(128 * MiB)
    try:
        index = json.loads(read_small("model.safetensors.index.json"))["weight_map"]
        headers, arrays, entries = {}, {}, []
        for name, shape in COMPRESSION_LAYOUT.items():
            shard = index[name]
            if shard not in headers:
                size = struct.unpack("<Q", read_range(shard, 0, 8))[0]
                if not 0 < size <= LIMIT:
                    raise ValueError("Excessive compressor shard header")
                headers[shard] = json.loads(read_range(shard, 8, size)), size + 8
            header, data_start = headers[shard]
            tensor = header[name]
            begin, end = tensor["data_offsets"]
            size = int(np.prod(shape)) * 2
            if tensor["dtype"] != "BF16" or tensor["shape"] != list(shape) or begin < 0 or end - begin != size:
                raise ValueError("Original compressor tensor layout differs")
            raw = read_range(shard, data_start + begin, size)
            arrays[name] = np.frombuffer(raw, dtype="<u2").reshape(shape).copy()
            entries.append({"name": name, "dtype": "BF16", "shape": list(shape), "bytes": size,
                            "shard": shard, "shard_byte_offset": data_start + begin,
                            "sha256": hashlib.sha256(raw).hexdigest()})
        manifest = {"kind": "original_compression_fixture", "model": MODEL, "revision": REVISION,
                    "layers_and_ratios": {"2": 2, "20": 1}, "tensors": entries,
                    "weights_fetched_bytes": COMPRESSION_BYTES, "maximum_range_bytes": LIMIT,
                    "scope": "Five original BF16 compressor tensors only; no full shard or model"}
        directory.mkdir(parents=True, exist_ok=False)
        np.savez(directory / "native-compression.npz", **arrays)
        (directory / "manifest.json").write_bytes((json.dumps(manifest, indent=2) + "\n").encode())
        return {"probe": "fetch_compression_fixture", "status": "pass", **manifest}
    finally:
        budget.release(128 * MiB)


def load_compression_fixture(path):
    path = Path(path)
    manifest_path = path.parent / "manifest.json"
    if manifest_path.stat().st_size > 65536 or path.stat().st_size > COMPRESSION_BYTES + 16384:
        raise ValueError("Compression fixture exceeds its bounded size")
    manifest = json.loads(manifest_path.read_bytes())
    if (manifest.get("kind"), manifest.get("model"), manifest.get("revision")) != (
            "original_compression_fixture", MODEL, REVISION):
        raise ValueError("Compressor fixture source identity differs")
    if manifest.get("layers_and_ratios") != {"2": 2, "20": 1} or manifest.get("weights_fetched_bytes") != COMPRESSION_BYTES:
        raise ValueError("Compressor fixture scope differs")
    entries = manifest.get("tensors", [])
    if len(entries) != len(COMPRESSION_LAYOUT) or {e["name"] for e in entries} != set(COMPRESSION_LAYOUT):
        raise ValueError("Incomplete compressor fixture")
    for e in entries:
        shape = COMPRESSION_LAYOUT[e["name"]]
        if e["dtype"] != "BF16" or e["shape"] != list(shape) or e["bytes"] != int(np.prod(shape)) * 2:
            raise ValueError("Compressor provenance layout differs")
    with zipfile.ZipFile(path) as archive:
        if sorted(archive.namelist()) != sorted(name + ".npy" for name in COMPRESSION_LAYOUT):
            raise ValueError("Compressor archive members differ")
        for name, shape in COMPRESSION_LAYOUT.items():
            if archive.getinfo(name + ".npy").file_size > int(np.prod(shape)) * 2 + 1024:
                raise ValueError("Compressor member exceeds its bound")
            with archive.open(name + ".npy") as stream:
                version = np.lib.format.read_magic(stream)
                reader = {(1, 0): np.lib.format.read_array_header_1_0, (2, 0): np.lib.format.read_array_header_2_0}.get(version)
                if reader is None:
                    raise ValueError("Unsupported compressor array header")
                actual, order, dtype = reader(stream, max_header_size=1024)
                if actual != shape or order or dtype != np.dtype("<u2"):
                    raise ValueError("Compressor header exceeds the bounded layout")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in COMPRESSION_LAYOUT}
    for e in entries:
        a = arrays[e["name"]]
        if np.any((a & 0x7f80) == 0x7f80) or hashlib.sha256(a.tobytes()).hexdigest() != e["sha256"]:
            raise ValueError("Compressor bytes differ from finite original fixture")
    return arrays, manifest
