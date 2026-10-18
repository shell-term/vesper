"""Bounded original indexer weights for owners 2/20 and candidate consumer 24."""

import hashlib
import json
from pathlib import Path
import struct
import zipfile

import numpy as np

from .checkpoint import LIMIT, MODEL, REVISION, read_range, read_small
from .resources import MemoryBudget, MiB

INDEXER_LAYOUT = {}
for layer in (2, 20, 24):
    parts = {"wq_b.weight": ("F8_E4M3", (4096, 1280)), "wq_b.scale": ("F8_E8M0", (128, 40)),
             "weights_proj.weight": ("BF16", (32, 5120))}
    if layer != 24:
        parts.update({"wk.weight": ("BF16", (128, 512)), "k_norm.weight": ("BF16", (128,))})
    INDEXER_LAYOUT.update((f"layers.{layer}.attn.indexer.{part}", spec) for part, spec in parts.items())
DTYPES = {"F8_E4M3": np.dtype("u1"), "F8_E8M0": np.dtype("u1"), "BF16": np.dtype("<u2")}
INDEXER_BYTES = sum(int(np.prod(s)) * DTYPES[d].itemsize for d, s in INDEXER_LAYOUT.values())


def fetch_indexer_fixture(directory):
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError("Refusing to replace an existing indexer fixture")
    budget = MemoryBudget(cap=128 * MiB)
    budget.acquire(128 * MiB)
    try:
        index = json.loads(read_small("model.safetensors.index.json"))["weight_map"]
        arrays, headers, entries = {}, {}, []
        for name, (dtype, shape) in INDEXER_LAYOUT.items():
            shard = index[name]
            if shard not in headers:
                n = struct.unpack("<Q", read_range(shard, 0, 8))[0]
                if not 0 < n <= LIMIT:
                    raise ValueError("Excessive indexer shard header")
                headers[shard] = json.loads(read_range(shard, 8, n)), n + 8
            header, start = headers[shard]
            tensor = header[name]
            begin, end = tensor["data_offsets"]
            size = int(np.prod(shape)) * DTYPES[dtype].itemsize
            if tensor["dtype"] != dtype or tensor["shape"] != list(shape) or begin < 0 or end - begin != size:
                raise ValueError("Original indexer layout differs")
            raw = read_range(shard, start + begin, size)
            arrays[name] = np.frombuffer(raw, DTYPES[dtype]).reshape(shape).copy()
            entries.append({"name": name, "dtype": dtype, "shape": list(shape), "bytes": size,
                            "shard": shard, "shard_byte_offset": start + begin, "sha256": hashlib.sha256(raw).hexdigest()})
        manifest = {"kind": "original_indexer_fixture", "model": MODEL, "revision": REVISION,
                    "layers": [2, 20, 24], "tensors": entries, "weights_fetched_bytes": INDEXER_BYTES,
                    "maximum_range_bytes": LIMIT, "scope": "Original indexer matrices/scales/norms only; no full shard/model"}
        directory.mkdir(parents=True, exist_ok=False)
        np.savez(directory / "native-indexer.npz", **arrays)
        (directory / "manifest.json").write_bytes((json.dumps(manifest, indent=2) + "\n").encode())
        return {"probe": "fetch_indexer_fixture", "status": "pass", **manifest}
    finally:
        budget.release(128 * MiB)


def load_indexer_fixture(path):
    path = Path(path)
    meta = path.parent / "manifest.json"
    if meta.stat().st_size > 65536 or path.stat().st_size > INDEXER_BYTES + 32768:
        raise ValueError("Indexer fixture exceeds its bound")
    manifest = json.loads(meta.read_bytes())
    entries = manifest.get("tensors", [])
    if (manifest.get("kind"), manifest.get("model"), manifest.get("revision"), manifest.get("layers")) != (
            "original_indexer_fixture", MODEL, REVISION, [2, 20, 24]):
        raise ValueError("Original indexer fixture identity differs")
    if (manifest.get("weights_fetched_bytes") != INDEXER_BYTES or len(entries) != len(INDEXER_LAYOUT)
            or {e.get("name") for e in entries} != set(INDEXER_LAYOUT)):
        raise ValueError("Incomplete indexer fixture provenance")
    for e in entries:
        dtype, shape = INDEXER_LAYOUT[e["name"]]
        if (e.get("dtype"), e.get("shape"), e.get("bytes")) != (dtype, list(shape), int(np.prod(shape)) * DTYPES[dtype].itemsize):
            raise ValueError("Original indexer provenance layout differs")
    with zipfile.ZipFile(path) as archive:
        if sorted(archive.namelist()) != sorted(n + ".npy" for n in INDEXER_LAYOUT):
            raise ValueError("Indexer archive members differ")
        for name, (dtype, shape) in INDEXER_LAYOUT.items():
            if archive.getinfo(name + ".npy").file_size > int(np.prod(shape)) * DTYPES[dtype].itemsize + 1024:
                raise ValueError("Excessive indexer array size")
            with archive.open(name + ".npy") as stream:
                version = np.lib.format.read_magic(stream)
                reader = {(1, 0): np.lib.format.read_array_header_1_0, (2, 0): np.lib.format.read_array_header_2_0}.get(version)
                if reader is None:
                    raise ValueError("Unsupported indexer array header")
                actual, order, actual_dtype = reader(stream, max_header_size=1024)
                if actual != shape or order or actual_dtype != DTYPES[dtype]:
                    raise ValueError("Indexer array header exceeds its bounded layout")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in INDEXER_LAYOUT}
    for e in entries:
        a = arrays[e["name"]]
        invalid = (a & 0x7f80) == 0x7f80 if e["dtype"] == "BF16" else ((a & 127) == 127 if e["dtype"] == "F8_E4M3" else a == 255)
        if np.any(invalid) or hashlib.sha256(a.tobytes()).hexdigest() != e["sha256"]:
            raise ValueError("Indexer bytes differ from finite original weights")
    return arrays, manifest
