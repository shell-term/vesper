"""Bounded, revision-pinned reads of a public checkpoint. Never fetch full shards."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import struct
import urllib.request
import zipfile

import numpy as np

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"
BASE = f"https://huggingface.co/{MODEL}"
LIMIT = 8 * 1024**2
REFERENCE_HASHES = {
    "kernel.py": "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455",
    "LICENSE": "f2c6c602815669d292889e5be8c802f2ed950653b77999b1584e8e6aed25d040",
    "model.py": "e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d",
}


def read_small(path: str, limit=LIMIT) -> bytes:
    with urllib.request.urlopen(f"{BASE}/raw/{REVISION}/{path}", timeout=45) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError("Reference metadata exceeds the bounded read limit")
    return data


def read_range(shard: str, start: int, size: int) -> bytes:
    if not re.fullmatch(r"model-\d{5}-of-\d{5}\.safetensors", shard):
        raise ValueError("Expected a checkpoint shard filename")
    if start < 0 or not 0 < size <= LIMIT:
        raise ValueError("Invalid or excessive byte range")
    request = urllib.request.Request(
        f"{BASE}/resolve/{REVISION}/{shard}?vesper_range={start}-{size}",
        headers={"Range": f"bytes={start}-{start + size - 1}", "Accept-Encoding": "identity"},
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        if response.status != 206:
            raise ValueError("Server ignored Range; refusing to read the shard")
        if not response.headers.get("Content-Range", "").startswith(f"bytes {start}-{start + size - 1}/"):
            raise ValueError("Server returned a different byte range")
        data = response.read(size + 1)
    if len(data) != size:
        raise ValueError("Server returned an unexpected byte count")
    return data


def fetch_fixture(directory: Path, rows=128) -> dict:
    if not 1 <= rows <= 128:
        raise ValueError("This small fixture is limited to 128 rows")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    index = json.loads(read_small("model.safetensors.index.json"))
    config = json.loads(read_small("config.json"))
    keys = ["layers.0.ffn.experts.0.w1.weight", "layers.0.ffn.experts.0.w1.scale"]
    headers = {}
    payloads, provenance = [], []
    for key in keys:
        shard = index["weight_map"][key]
        if shard not in headers:
            header_size = struct.unpack("<Q", read_range(shard, 0, 8))[0]
            if not 0 < header_size <= LIMIT:
                raise ValueError("Excessive safetensors header size")
            headers[shard] = (json.loads(read_range(shard, 8, header_size)), 8 + header_size)
        header, data_start = headers[shard]
        tensor = header[key]
        if tensor["dtype"] not in ("I8", "F8_E8M0") or len(tensor["shape"]) != 2:
            raise ValueError("Unexpected native expert tensor layout")
        width = tensor["shape"][1]
        size = rows * width
        begin, end = tensor["data_offsets"]
        if end - begin != tensor["shape"][0] * width or size > end - begin:
            raise ValueError("Tensor shape and byte range disagree")
        raw = read_range(shard, data_start + begin, size)
        payloads.append(np.frombuffer(raw, dtype=np.uint8).reshape(rows, width).copy())
        provenance.append({"tensor": key, "dtype": tensor["dtype"], "original_shape": tensor["shape"],
                           "selected_shape": [rows, width], "shard": shard,
                           "shard_byte_offset": data_start + begin, "bytes": len(raw),
                           "sha256": hashlib.sha256(raw).hexdigest()})
    if payloads[0].shape[1] * 2 != config["text_config"]["hidden_size"]:
        raise ValueError("Fixture width does not match the pinned model")
    if payloads[1].shape[1] * 32 != payloads[0].shape[1] * 2:
        raise ValueError("Scale width does not match the native FP4 block size")
    np.savez(directory / "native-weight-slice.npz", weight=payloads[0], scales=payloads[1])
    result = {"probe": "fetch_fixture", "status": "pass", "model": MODEL, "revision": REVISION,
              "weights_fetched_bytes": sum(p.nbytes for p in payloads), "tensors": provenance,
              "note": "Original packed bytes and scales; a small W1 slice, not a full expert or model."}
    (directory / "manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def fetch_reference(directory: Path, *, include_model: bool = False) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"model": MODEL, "revision": REVISION, "files": {}}
    sources = [("inference/kernel.py", "kernel.py"), ("LICENSE", "LICENSE")]
    if include_model:
        sources.append(("inference/model.py", "model.py"))
    for source, destination in sources:
        path = directory / destination
        raw = path.read_bytes() if path.exists() else read_small(source)
        digest = hashlib.sha256(raw).hexdigest()
        if digest != REFERENCE_HASHES[destination]:
            raise ValueError("Official reference source hash differs from the pinned artifact")
        if not path.exists():
            path.write_bytes(raw)
        manifest["files"][destination] = digest
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_fixture(path: Path) -> dict[str, np.ndarray]:
    path = Path(path)
    manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("model") != MODEL or manifest.get("revision") != REVISION:
        raise ValueError("Fixture provenance does not match the pinned checkpoint")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in ("weight", "scales")}
    entries = manifest.get("tensors", [])
    if len(entries) != 2:
        raise ValueError("Incomplete fixture provenance")
    for name, entry in zip(("weight", "scales"), entries):
        array = arrays[name]
        if array.dtype != np.uint8 or list(array.shape) != entry["selected_shape"]:
            raise ValueError("Fixture tensor layout differs from its manifest")
        if hashlib.sha256(array.tobytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Fixture bytes differ from the downloaded source hash")
    weight, scales = arrays["weight"], arrays["scales"]
    if weight.ndim != 2 or scales.shape != (weight.shape[0], weight.shape[1] // 16):
        raise ValueError("Fixture scale blocks do not match native FP4 packing")
    return arrays


EXPERT_SHAPES = {"w1": (2304, 5120), "w3": (2304, 5120), "w2": (5120, 2304)}
EXPERT_LAYOUT = {f"{p}_{s}": (r, k // d) for p, (r, k) in EXPERT_SHAPES.items()
                 for s, d in (("weight", 2), ("scale", 32))}


def validate_expert_arrays(arrays: dict[str, np.ndarray]) -> None:
    if set(arrays) != set(EXPERT_LAYOUT):
        raise ValueError("A complete expert requires exactly six native tensors")
    for name, shape in EXPERT_LAYOUT.items():
        if arrays[name].dtype != np.uint8 or arrays[name].shape != shape:
            raise ValueError("Expert tensor layout differs from the pinned shape")


def fetch_expert_fixture(directory: Path) -> dict:
    """Fetch exactly layer 0 / expert 0: six bounded tensors, 18,800,640 bytes."""
    from .resources import MemoryBudget, MiB

    budget = MemoryBudget(cap=64 * MiB)
    budget.acquire(64 * MiB)
    directory = Path(directory)
    try:
        index = json.loads(read_small("model.safetensors.index.json"))
        config = json.loads(read_small("config.json"))["text_config"]
        if (config["hidden_size"], config["moe_intermediate_size"], config["swiglu_limit"]) != (5120, 2304, 10.0):
            raise ValueError("Pinned expert configuration differs from this probe's scope")
        headers, arrays, entries = {}, {}, []
        for projection, (rows, width) in EXPERT_SHAPES.items():
            for suffix, divisor, dtype in (("weight", 2, "I8"), ("scale", 32, "F8_E8M0")):
                key = f"layers.0.ffn.experts.0.{projection}.{suffix}"
                shard = index["weight_map"][key]
                if shard not in headers:
                    header_size = struct.unpack("<Q", read_range(shard, 0, 8))[0]
                    if not 0 < header_size <= LIMIT:
                        raise ValueError("Excessive safetensors header size")
                    headers[shard] = (json.loads(read_range(shard, 8, header_size)), 8 + header_size)
                header, data_start = headers[shard]
                tensor = header[key]
                expected_shape = [rows, width // divisor]
                begin, end = tensor["data_offsets"]
                size = rows * (width // divisor)
                if tensor["dtype"] != dtype or tensor["shape"] != expected_shape or end - begin != size:
                    raise ValueError("Unexpected complete expert tensor layout")
                raw = read_range(shard, data_start + begin, size)
                name = f"{projection}_{suffix}"
                arrays[name] = np.frombuffer(raw, dtype=np.uint8).reshape(expected_shape).copy()
                entries.append({"name": name, "tensor": key, "dtype": dtype, "shape": expected_shape,
                                "shard": shard, "shard_byte_offset": data_start + begin,
                                "bytes": size, "sha256": hashlib.sha256(raw).hexdigest()})
        total = sum(a.nbytes for a in arrays.values())
        if total != 18_800_640:
            raise ValueError("Complete expert size differs from its bounded scope")
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(directory / "native-expert.npz", **arrays)
        manifest = {"probe": "fetch_expert_fixture", "status": "pass", "kind": "complete_expert",
                    "model": MODEL, "revision": REVISION, "layer": 0, "expert": 0,
                    "swiglu_limit": config["swiglu_limit"], "weights_fetched_bytes": total, "tensors": entries,
                    "note": "One original native expert; not the full model or an inference download cache."}
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return manifest
    finally:
        budget.release(64 * MiB)


def load_expert_fixture(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    path = Path(path)
    manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("model"), manifest.get("revision"), manifest.get("kind"), manifest.get("layer"),
            manifest.get("expert"), manifest.get("swiglu_limit")) != (MODEL, REVISION, "complete_expert", 0, 0, 10.0):
        raise ValueError("Expert provenance differs from the pinned checkpoint and probe scope")
    expected = EXPERT_LAYOUT
    entries = manifest.get("tensors", [])
    if len(entries) != len(expected) or {e["name"] for e in entries} != set(expected):
        raise ValueError("Incomplete expert fixture provenance")
    # Inspect every NPY header before NumPy can allocate from a declared shape.
    # Central-directory sizes alone do not bound an invalid NPY shape.
    with zipfile.ZipFile(path) as archive:
        if sorted(archive.namelist()) != sorted(f"{name}.npy" for name in expected):
            raise ValueError("Expert archive does not contain exactly six tensors")
        for name, shape in expected.items():
            info = archive.getinfo(f"{name}.npy")
            if info.file_size > int(np.prod(shape)) + 1024:
                raise ValueError("Expert archive member exceeds its bounded size")
            with archive.open(info) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    header = np.lib.format.read_array_header_1_0(stream, max_header_size=1024)
                elif version == (2, 0):
                    header = np.lib.format.read_array_header_2_0(stream, max_header_size=1024)
                else:
                    raise ValueError("Unsupported expert array header")
                stored_shape, fortran_order, dtype = header
                if stored_shape != shape or fortran_order or dtype != np.dtype(np.uint8):
                    raise ValueError("Expert archive header differs from the bounded native layout")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in expected}
    validate_expert_arrays(arrays)
    for entry in entries:
        name = entry["name"]
        array = arrays[name]
        if array.dtype != np.uint8 or array.shape != expected[name] or list(array.shape) != entry["shape"]:
            raise ValueError("Expert fixture tensor layout differs from its pinned shape")
        if hashlib.sha256(array.tobytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Expert fixture bytes differ from the downloaded source hash")
    return arrays, manifest
