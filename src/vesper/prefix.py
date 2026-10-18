"""Small original layer-0 prefix fixture; never import or download a full model."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import struct
import zipfile

import numpy as np

from .checkpoint import LIMIT, MODEL, REVISION, read_range, read_small
from .formats import bf16_values
from .resources import MemoryBudget, MiB

PREFIX_SEQUENCES = ((1, 42), (1000, 4096))
PREFIX_IDS = tuple(sorted({token for sequence in PREFIX_SEQUENCES for token in sequence}))
PREFIX_LAYOUT = {}
for part in ("attn", "ffn"):
    for name, shape in (("fn", (24, 20480)), ("base", (24,)), ("scale", (3,))):
        PREFIX_LAYOUT[f"layers.0.hc_{part}_{name}"] = ("F32", shape)
for projection, rows, width in (("wq_a", 1280, 5120), ("wq_b", 32768, 1280),
                                 ("wkv", 512, 5120), ("wo_a", 8192, 4096), ("wo_b", 5120, 8192)):
    PREFIX_LAYOUT[f"layers.0.attn.{projection}.weight"] = ("F8_E4M3", (rows, width))
    PREFIX_LAYOUT[f"layers.0.attn.{projection}.scale"] = ("F8_E8M0", (rows // 32, width // 32))
for key, width in (("attn.q_norm", 1280), ("attn.kv_norm", 512), ("attn_norm", 5120), ("ffn_norm", 5120)):
    PREFIX_LAYOUT[f"layers.0.{key}.weight"] = ("BF16", (width,))
PREFIX_LAYOUT["layers.0.attn.attn_sink"] = ("F32", (64,))
STORAGE_DTYPES = {"F32": np.dtype("<f4"), "BF16": np.dtype("<u2"),
                  "F8_E4M3": np.dtype("u1"), "F8_E8M0": np.dtype("u1")}
PREFIX_BYTES = sum(int(np.prod(s)) * STORAGE_DTYPES[d].itemsize for d, s in PREFIX_LAYOUT.values()) + len(PREFIX_IDS) * 10240


def _shape_dtype():
    return {key: (shape, STORAGE_DTYPES[dtype]) for key, (dtype, shape) in PREFIX_LAYOUT.items()} | {
        "embedding_rows": ((len(PREFIX_IDS), 5120), np.dtype("<u2"))}


def fetch_prefix_fixture(directory: Path):
    """Only fixed prefix matrices and four original embedding rows, in <=8 MiB ranges."""
    budget = MemoryBudget(cap=512 * MiB)
    budget.acquire(512 * MiB)
    try:
        directory = Path(directory)
        if directory.exists():
            raise FileExistsError("Refusing to replace an existing prefix fixture")
        index = json.loads(read_small("model.safetensors.index.json"))["weight_map"]
        headers, arrays, entries = {}, {}, []
        def location(key):
            shard = index[key]
            if shard not in headers:
                size = struct.unpack("<Q", read_range(shard, 0, 8))[0]
                if not 0 < size <= LIMIT:
                    raise ValueError("Excessive prefix tensor header")
                headers[shard] = (json.loads(read_range(shard, 8, size)), 8 + size)
            tensors, start = headers[shard]
            return shard, tensors[key], start
        for key, (dtype, shape) in PREFIX_LAYOUT.items():
            shard, tensor, data_start = location(key)
            begin, end = tensor["data_offsets"]
            storage = STORAGE_DTYPES[dtype]
            size = int(np.prod(shape)) * storage.itemsize
            if tensor["dtype"] != dtype or tensor["shape"] != list(shape) or begin < 0 or end - begin != size:
                raise ValueError("Prefix tensor differs from its pinned layout")
            array = np.empty(shape, storage)
            byte_view = array.view(np.uint8).reshape(-1)
            for offset in range(0, size, LIMIT):
                raw = read_range(shard, data_start + begin + offset, min(LIMIT, size - offset))
                byte_view[offset:offset + len(raw)] = np.frombuffer(raw, np.uint8)
            arrays[key] = array
            entries.append({"name": key, "dtype": dtype, "shape": list(shape), "bytes": size,
                            "shard": shard, "shard_byte_offset": data_start + begin,
                            "sha256": hashlib.sha256(array.tobytes()).hexdigest()})
        shard, tensor, start = location("embed.weight")
        if tensor["dtype"] != "BF16" or tensor["shape"] != [129280, 5120] or tensor["data_offsets"][1] - tensor["data_offsets"][0] != 129280 * 10240:
            raise ValueError("Original embedding layout differs")
        rows = np.empty((len(PREFIX_IDS), 5120), np.uint16)
        row_entries = []
        for i, token in enumerate(PREFIX_IDS):
            offset = start + tensor["data_offsets"][0] + token * 10240
            raw = read_range(shard, offset, 10240)
            rows[i] = np.frombuffer(raw, np.uint16)
            row_entries.append({"token_id": token, "bytes": 10240, "shard_byte_offset": offset,
                                "sha256": hashlib.sha256(raw).hexdigest()})
        arrays["embedding_rows"] = rows
        manifest = {"kind": "original_layer0_prefix_fixture", "model": MODEL, "revision": REVISION,
                    "layer": 0, "tensors": entries, "embedding_tensor": "embed.weight", "embedding_shard": shard,
                    "embedding_rows": row_entries, "weights_fetched_bytes": PREFIX_BYTES,
                    "maximum_range_bytes": LIMIT, "sequences": PREFIX_SEQUENCES,
                    "scope": "Original prefix only; no tokenizer, Engram, experts or full model"}
        directory.mkdir(parents=True, exist_ok=False)
        np.savez(directory / "native-prefix.npz", **arrays)
        (directory / "manifest.json").write_bytes((json.dumps(manifest, indent=2) + "\n").encode())
        return {"probe": "fetch_prefix_fixture", "status": "pass", **manifest}
    finally:
        budget.release(512 * MiB)


def load_prefix_fixture(path):
    path = Path(path)
    manifest = json.loads((path.parent / "manifest.json").read_bytes())
    if (manifest.get("kind"), manifest.get("model"), manifest.get("revision"), manifest.get("layer")) != (
            "original_layer0_prefix_fixture", MODEL, REVISION, 0):
        raise ValueError("Prefix fixture differs from the pinned source")
    entries = manifest.get("tensors", [])
    if len(entries) != len(PREFIX_LAYOUT) or {e["name"] for e in entries} != set(PREFIX_LAYOUT):
        raise ValueError("Incomplete prefix fixture")
    for entry in entries:
        dtype, shape = PREFIX_LAYOUT[entry["name"]]
        if entry["dtype"] != dtype or entry["shape"] != list(shape) or entry["bytes"] != int(np.prod(shape)) * STORAGE_DTYPES[dtype].itemsize:
            raise ValueError("Prefix provenance layout differs")
    if [e["token_id"] for e in manifest.get("embedding_rows", [])] != list(PREFIX_IDS):
        raise ValueError("Prefix embedding row identities differ")
    layouts = _shape_dtype()
    if path.stat().st_size > PREFIX_BYTES + 64 * 1024:
        raise ValueError("Prefix archive exceeds its bounded size")
    with zipfile.ZipFile(path) as archive:
        if sorted(archive.namelist()) != sorted(f"{n}.npy" for n in layouts):
            raise ValueError("Prefix archive members differ")
        for name, (shape, dtype) in layouts.items():
            info = archive.getinfo(f"{name}.npy")
            if info.file_size > int(np.prod(shape)) * dtype.itemsize + 1024:
                raise ValueError("Prefix member exceeds its bounded size")
            with archive.open(info) as stream:
                version = np.lib.format.read_magic(stream)
                reader = {(1, 0): np.lib.format.read_array_header_1_0, (2, 0): np.lib.format.read_array_header_2_0}.get(version)
                if reader is None:
                    raise ValueError("Unsupported prefix array header")
                actual_shape, order, actual_dtype = reader(stream, max_header_size=1024)
                if actual_shape != shape or order or actual_dtype != dtype:
                    raise ValueError("Prefix header differs from its bounded layout")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in layouts}
    for entry in entries:
        if hashlib.sha256(arrays[entry["name"]].tobytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Prefix bytes differ from their original source hash")
    for i, entry in enumerate(manifest["embedding_rows"]):
        if hashlib.sha256(arrays["embedding_rows"][i].tobytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Embedding row differs from its original source hash")
    return arrays, manifest


def load_moe_inputs(path):
    path = Path(path)
    if path.stat().st_size > 2 * MiB:
        raise ValueError("MoE input record exceeds its bounded size")
    report = json.loads(path.read_bytes())
    if (report.get("probe"), report.get("status"), report.get("model"), report.get("revision")) != (
            "original_prefix_moe_inputs", "pass", MODEL, REVISION):
        raise ValueError("Expected passing original-prefix input provenance")
    if not 2 <= len(report.get("cases", [])) <= 16:
        raise ValueError("Expected two to sixteen captured single-token inputs")
    result = []
    for case in report["cases"]:
        encoded = case["activation_bf16_base64"]
        if not isinstance(encoded, str) or len(encoded) > 13656:
            raise ValueError("Captured activation exceeds one BF16 row")
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) != 10240 or hashlib.sha256(raw).hexdigest() != case["activation_bf16_sha256"]:
            raise ValueError("Captured activation bits differ from their source hash")
        values = bf16_values(np.frombuffer(raw, np.uint16))
        if not np.isfinite(values).all():
            raise ValueError("Captured activation leaves the finite BF16 scope")
        result.append((case, values))
    return result, report
