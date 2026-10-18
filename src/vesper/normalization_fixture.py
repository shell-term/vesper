"""Read only four original normalization vectors from the cached prefix archive."""

import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from .attention import read_bounded
from .checkpoint import MODEL, REVISION
from .prefix import PREFIX_BYTES

NORM_WEIGHTS = {"layers.0.attn.q_norm.weight": 1280, "layers.0.attn.kv_norm.weight": 512,
                "layers.0.attn_norm.weight": 5120, "layers.0.ffn_norm.weight": 5120}


def load_norm_weights(path):
    path = Path(path)
    manifest = json.loads(read_bounded(path.parent / "manifest.json", 65536))
    if (manifest.get("kind"), manifest.get("model"), manifest.get("revision"), manifest.get("layer")) != (
            "original_layer0_prefix_fixture", MODEL, REVISION, 0) or path.stat().st_size > PREFIX_BYTES + 65536:
        raise ValueError("Expected the bounded original layer-0 prefix archive")
    selected = [e for e in manifest.get("tensors", []) if e.get("name") in NORM_WEIGHTS]
    entries = {e["name"]: e for e in selected}
    if len(selected) != len(NORM_WEIGHTS) or set(entries) != set(NORM_WEIGHTS):
        raise ValueError("Original normalization vectors are incomplete")
    with zipfile.ZipFile(path) as archive:
        for name, width in NORM_WEIGHTS.items():
            e = entries[name]
            if (e.get("dtype"), e.get("shape"), e.get("bytes")) != ("BF16", [width], width * 2):
                raise ValueError("Original normalization vector layout differs")
            info = archive.getinfo(name + ".npy")
            if info.file_size > width * 2 + 1024:
                raise ValueError("Normalization member exceeds its bounded size")
            with archive.open(info) as stream:
                version = np.lib.format.read_magic(stream)
                reader = {(1, 0): np.lib.format.read_array_header_1_0, (2, 0): np.lib.format.read_array_header_2_0}.get(version)
                if reader is None:
                    raise ValueError("Unsupported normalization header version")
                shape, order, dtype = reader(stream, max_header_size=1024)
                if shape != (width,) or order or dtype != np.dtype("<u2"):
                    raise ValueError("Normalization header differs from the fixed layout")
    with np.load(path, allow_pickle=False) as archive:
        weights = {name: archive[name].copy() for name in NORM_WEIGHTS}
    for name, value in weights.items():
        if hashlib.sha256(value.tobytes()).hexdigest() != entries[name]["sha256"] or np.any((value & 0x7F80) == 0x7F80):
            raise ValueError("Original normalization bytes differ or are nonfinite")
    return weights, {"model": MODEL, "revision": REVISION, "kind": "selected_original_prefix_norm_vectors",
                     "tensors": list(entries.values()), "original_weight_bytes_reused": sum(w.nbytes for w in weights.values()),
                     "new_weight_bytes_fetched": 0, "other_prefix_matrices_loaded": False}
