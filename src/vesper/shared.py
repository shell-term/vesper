"""Bounded original FP8 shared-expert fixture and single-token arithmetic."""

from __future__ import annotations

import ctypes as C
import hashlib
import json
from pathlib import Path
import re
import struct
import zipfile

import numpy as np

from .checkpoint import EXPERT_SHAPES, LIMIT, MODEL, REVISION, read_range, read_small
from .cuda import compile_probe
from .expert import ExpertExecutor
from .resources import MemoryBudget, MiB

SHARED_LAYOUT = {
    f"{p}_{suffix}": (rows, width) if suffix == "weight" else (rows // 32, width // 32)
    for p, (rows, width) in EXPERT_SHAPES.items() for suffix in ("weight", "scale")
}
SHARED_BYTES = sum(int(np.prod(shape)) for shape in SHARED_LAYOUT.values())


def validate_shared_arrays(arrays):
    if set(arrays) != set(SHARED_LAYOUT):
        raise ValueError("A shared expert requires exactly six native tensors")
    for name, shape in SHARED_LAYOUT.items():
        if arrays[name].dtype != np.uint8 or arrays[name].shape != shape:
            raise ValueError("Shared tensor differs from the pinned FP8/block-32 layout")


def validate_shared_provenance(manifest):
    if (manifest.get("model"), manifest.get("revision"), manifest.get("kind"),
            manifest.get("swiglu_limit")) != (MODEL, REVISION, "shared_expert_fp8", 10.0):
        raise ValueError("Shared provenance differs from the pinned checkpoint")
    layer = manifest.get("layer")
    if type(layer) is not int or not 0 <= layer < 40:
        raise ValueError("Shared layer is outside the pinned backbone")
    entries = manifest.get("tensors", [])
    if len(entries) != 6 or {e["name"] for e in entries} != set(SHARED_LAYOUT):
        raise ValueError("Incomplete shared fixture provenance")
    for entry in entries:
        name = entry["name"]
        projection, suffix = name.split("_")
        if (entry.get("tensor") != f"layers.{layer}.ffn.shared_experts.{projection}.{suffix}"
                or entry.get("dtype") != ("F8_E4M3" if suffix == "weight" else "F8_E8M0")
                or entry.get("shape") != list(SHARED_LAYOUT[name])
                or entry.get("bytes") != int(np.prod(SHARED_LAYOUT[name]))
                or not re.fullmatch(r"[0-9a-f]{64}", entry.get("sha256", ""))):
            raise ValueError("Shared tensor provenance differs from its pinned identity/layout")


def fetch_shared_fixture(directory: Path, layer=0, *, reader=None):
    """Fetch 35,424,000 bytes; every HTTP range is at most 8 MiB."""
    small, ranges = (read_small, read_range) if reader is None else (reader.small, reader.range)
    if type(layer) is not int or not 0 <= layer < 40:
        raise ValueError("Shared layer is outside the pinned backbone")
    budget = MemoryBudget(cap=128 * MiB)
    budget.acquire(128 * MiB)
    try:
        index = json.loads(small("model.safetensors.index.json"))
        config = json.loads(small("config.json"))["text_config"]
        if (config["hidden_size"], config["moe_intermediate_size"], config["swiglu_limit"]) != (5120, 2304, 10.0):
            raise ValueError("Shared configuration differs from this probe's scope")
        arrays, entries, headers = {}, [], {}
        for name, shape in SHARED_LAYOUT.items():
            projection, suffix = name.split("_")
            key = f"layers.{layer}.ffn.shared_experts.{projection}.{suffix}"
            shard = index["weight_map"][key]
            if shard not in headers:
                header_size = struct.unpack("<Q", ranges(shard, 0, 8))[0]
                if not 0 < header_size <= LIMIT:
                    raise ValueError("Excessive safetensors header size")
                headers[shard] = (json.loads(ranges(shard, 8, header_size)), 8 + header_size)
            header, data_start = headers[shard]
            tensor = header[key]
            begin, end = tensor["data_offsets"]
            size = int(np.prod(shape))
            dtype = "F8_E4M3" if suffix == "weight" else "F8_E8M0"
            if tensor["dtype"] != dtype or tensor["shape"] != list(shape) or begin < 0 or end - begin != size:
                raise ValueError("Unexpected original shared-expert tensor layout")
            array = np.empty(shape, np.uint8)
            flat = array.reshape(-1)
            for offset in range(0, size, LIMIT):
                raw = ranges(shard, data_start + begin + offset, min(LIMIT, size - offset))
                flat[offset:offset + len(raw)] = np.frombuffer(raw, np.uint8)
            arrays[name] = array
            entries.append({"name": name, "tensor": key, "dtype": dtype, "shape": list(shape),
                            "shard": shard, "shard_byte_offset": data_start + begin,
                            "bytes": size, "sha256": hashlib.sha256(array.tobytes()).hexdigest()})
        validate_shared_arrays(arrays)
        manifest = {"probe": "fetch_shared_fixture", "status": "pass", "kind": "shared_expert_fp8",
                    "model": MODEL, "revision": REVISION, "layer": layer, "swiglu_limit": 10.0,
                    "weights_fetched_bytes": SHARED_BYTES, "maximum_range_bytes": LIMIT,
                    "tensors": entries, "note": "One original shared expert; no full checkpoint shard."}
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(directory / "native-shared.npz", **arrays)
        (directory / "manifest.json").write_bytes((json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
        return manifest
    finally:
        budget.release(128 * MiB)


def load_shared_fixture(path: Path):
    path = Path(path)
    manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
    validate_shared_provenance(manifest)
    if path.stat().st_size > SHARED_BYTES + 64 * 1024:
        raise ValueError("Shared archive exceeds its bounded fixture size")
    # Check NPY headers before trusting allocation sizes declared in the archive.
    with zipfile.ZipFile(path) as archive:
        if sorted(archive.namelist()) != sorted(f"{name}.npy" for name in SHARED_LAYOUT):
            raise ValueError("Shared archive must contain exactly six tensors")
        for name, shape in SHARED_LAYOUT.items():
            info = archive.getinfo(f"{name}.npy")
            if info.file_size > int(np.prod(shape)) + 1024:
                raise ValueError("Shared archive member exceeds its bounded size")
            with archive.open(info) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    stored_shape, order, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=1024)
                elif version == (2, 0):
                    stored_shape, order, dtype = np.lib.format.read_array_header_2_0(stream, max_header_size=1024)
                else:
                    raise ValueError("Unsupported shared array header")
                if stored_shape != shape or order or dtype != np.dtype(np.uint8):
                    raise ValueError("Shared archive header differs from its bounded native layout")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in SHARED_LAYOUT}
    validate_shared_arrays(arrays)
    for entry in manifest["tensors"]:
        if hashlib.sha256(arrays[entry["name"]].tobytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Shared fixture bytes differ from their source hash")
    return arrays, manifest


class ResidentSharedExpert(ExpertExecutor):
    """Always-active FP8 expert; reuses the qualified quantizer and SwiGLU."""

    def __init__(self, arrays, ordinal=0):
        validate_shared_arrays(arrays)
        super().__init__(ordinal, swiglu_limit=10.0)
        self.arrays = arrays

    def __enter__(self):
        if self.device is not None:
            raise RuntimeError("Shared executor is already open")
        super().__enter__()
        try:
            cubin = compile_probe(self.info["compute_capability"].replace(".", ""), shared=True)
            self.fp8_cubin_sha256 = hashlib.sha256(cubin).hexdigest()
            self.fp8_module = self.device.load(cubin)
            self.weights = {name: self.device.upload(array) for name, array in self.arrays.items()}
            return self
        except Exception:
            self.close()
            raise

    def _projection(self, name, q, scales, output, weights):
        rows, width = EXPERT_SHAPES[name]
        args = [C.c_uint64(p) for p in (q, scales, weights[f"{name}_weight"], weights[f"{name}_scale"], output)]
        with self.timings.section("compute.kernels"):
            self.device.launch(self.fp8_module, "gemv_fp8_shared", args + [C.c_int(rows), C.c_int(width)], rows * 4)

    def forward(self, x):
        return self.execute(x, self.weights)
