"""Bounded, revision-pinned reads of a public checkpoint. Never fetch full shards."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import struct
import urllib.request

import numpy as np

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"
BASE = f"https://huggingface.co/{MODEL}"
LIMIT = 8 * 1024**2
REFERENCE_HASHES = {
    "kernel.py": "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455",
    "LICENSE": "f2c6c602815669d292889e5be8c802f2ed950653b77999b1584e8e6aed25d040",
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


def fetch_reference(directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"model": MODEL, "revision": REVISION, "files": {}}
    for source, destination in (("inference/kernel.py", "kernel.py"), ("LICENSE", "LICENSE")):
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
