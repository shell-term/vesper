"""Original BF16 gate weights, promoted exactly to the official FP32 projection."""

import ctypes as C
import hashlib
import json
from pathlib import Path
import struct
import zipfile

import numpy as np

from .blas import Cublas, GATE_BLAS_WORKSPACE
from .checkpoint import MODEL, REVISION, LIMIT, read_small, read_range, validate_expert_identity
from .cuda import Device, Driver
from .formats import bf16_values
from .resources import MemoryBudget, MiB

GATE_SHAPE = (384, 5120)
GATE_NATIVE_BYTES = 384 * 5120 * 2


def fetch_gate_fixture(directory, layer=0, *, reader=None):
    small, ranges = (read_small, read_range) if reader is None else (reader.small, reader.range)
    validate_expert_identity(layer, 0)
    budget = MemoryBudget(cap=64 * MiB)
    budget.acquire(64 * MiB)
    try:
        config_raw = small("config.json")
        config = json.loads(config_raw)["text_config"]
        if (config["hidden_size"], config["n_routed_experts"]) != (5120, 384):
            raise ValueError("Unexpected pinned gate shape")
        index = json.loads(small("model.safetensors.index.json"))
        key = f"layers.{layer}.ffn.gate.weight"
        shard = index["weight_map"][key]
        size = struct.unpack("<Q", ranges(shard, 0, 8))[0]
        if not 0 < size <= LIMIT:
            raise ValueError("Excessive gate shard header")
        header = json.loads(ranges(shard, 8, size))
        tensor = header[key]
        begin, end = tensor["data_offsets"]
        if (tensor["dtype"], tensor["shape"], end - begin) != ("BF16", list(GATE_SHAPE), GATE_NATIVE_BYTES):
            raise ValueError("Unexpected original gate weight layout")
        raw = ranges(shard, 8 + size + begin, end - begin)
        weight = np.frombuffer(raw, dtype="<u2").reshape(GATE_SHAPE).copy()
        validate_gate_weight(weight)
        manifest = {"probe": "fetch_gate_fixture", "status": "pass", "kind": "gate_projection_fixture",
                    "model": MODEL, "revision": REVISION, "layer": layer,
                    "config_sha256": hashlib.sha256(config_raw).hexdigest(),
                    "tensor": {"name": key, "dtype": "BF16", "shape": list(GATE_SHAPE), "bytes": GATE_NATIVE_BYTES,
                               "shard": shard, "shard_byte_offset": 8 + size + begin,
                               "sha256": hashlib.sha256(raw).hexdigest()},
                    "weights_fetched_bytes": len(raw), "scope": "One original gate matrix only; no full shard/model"}
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(directory / "native-gate.npz", weight=weight)
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return manifest
    finally:
        budget.release(64 * MiB)


def validate_gate_weight(weight):
    if weight.dtype != np.uint16 or weight.shape != GATE_SHAPE or np.any((weight & 0x7f80) == 0x7f80):
        raise ValueError("Gate weight must contain the original finite BF16 matrix")


def load_gate_fixture(path):
    path = Path(path)
    if path.stat().st_size > 4 * MiB:
        raise ValueError("Gate archive exceeds its bounded fixture size")
    with (path.parent / "manifest.json").open("rb") as stream:
        raw = stream.read(65537)
    if len(raw) > 65536:
        raise ValueError("Excessive gate fixture metadata")
    manifest = json.loads(raw)
    if (manifest.get("kind"), manifest.get("model"), manifest.get("revision")) != ("gate_projection_fixture", MODEL, REVISION):
        raise ValueError("Gate fixture differs from the pinned source")
    validate_expert_identity(manifest.get("layer"), 0)
    tensor = manifest["tensor"]
    if (tensor.get("name"), tensor.get("dtype"), tensor.get("shape"), tensor.get("bytes")) != (
            f"layers.{manifest['layer']}.ffn.gate.weight", "BF16", list(GATE_SHAPE), GATE_NATIVE_BYTES):
        raise ValueError("Gate tensor identity/layout differs from the fixture")
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != ["weight.npy"] or archive.getinfo("weight.npy").file_size > GATE_NATIVE_BYTES + 1024:
            raise ValueError("Gate archive must contain one bounded BF16 matrix")
        with archive.open("weight.npy") as stream:
            version = np.lib.format.read_magic(stream)
            if version not in ((1, 0), (2, 0)):
                raise ValueError("Unsupported gate array version")
            reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
            shape, fortran, dtype = reader(stream, max_header_size=1024)
            if shape != GATE_SHAPE or fortran or dtype != np.dtype(np.uint16):
                raise ValueError("Gate header exceeds the bounded BF16 layout")
    with np.load(path, allow_pickle=False) as archive:
        weight = archive["weight"].copy()
    validate_gate_weight(weight)
    if hashlib.sha256(memoryview(weight)).hexdigest() != tensor["sha256"]:
        raise ValueError("Gate weight differs from its original source hash")
    return weight, manifest


class GateProjection:
    """One original gate matrix, one FP32 activation row; independent of PyTorch."""

    def __init__(self, weight, ordinal=0, library=None, host_budget=None):
        validate_gate_weight(weight)
        self.weight, self.ordinal, self.library = weight, ordinal, library
        self.device, self.blas = None, None
        self.host_budget = host_budget if host_budget is not None else MemoryBudget(cap=32 * MiB)
        self.host_reserved = False
        self.calls = 0

    def __enter__(self):
        if self.device is not None:
            raise RuntimeError("Gate projection is already open")
        self.host_budget.acquire(32 * MiB)
        self.host_reserved = True
        try:
            self.driver = Driver()
            self.info = next(d for d in self.driver.inventory()["devices"] if d["ordinal"] == self.ordinal)
            self.device = Device(self.driver, self.ordinal, cap=32 * MiB)
            self.device_object = self.device
            self.blas = Cublas(self.library)
            self.work = {"weight": self.device.upload(bf16_values(self.weight)),
                         "activation": self.device.allocate(5120 * 4), "logits": self.device.allocate(384 * 4),
                         "workspace": self.device.allocate(GATE_BLAS_WORKSPACE)}
            self.blas.workspace(self.work["workspace"], GATE_BLAS_WORKSPACE)
            return self
        except BaseException:
            self.__exit__()
            raise

    def project(self, activation):
        if self.device is None:
            raise RuntimeError("Use the gate projection as a context manager")
        activation = np.ascontiguousarray(activation, dtype=np.float32)
        if activation.shape != (5120,) or not np.isfinite(activation).all():
            raise ValueError("Gate projection needs one finite 5,120-element FP32 input")
        self.device.current()
        self.driver.check(self.driver.cuMemcpyHtoD_v2(self.work["activation"], activation.ctypes.data, activation.nbytes))
        self.blas.gate(self.work["weight"], self.work["activation"], self.work["logits"])
        self.driver.check(self.driver.cuCtxSynchronize())
        self.calls += 1
        logits = self.device.download(self.work["logits"], (384,), np.float32)
        if not np.isfinite(logits).all():
            raise ValueError("Projection output leaves the finite probe scope")
        return logits

    def __exit__(self, *_):
        try:
            blas, self.blas = self.blas, None
            if blas is not None:
                if self.device is not None:
                    self.device.current()
                blas.close()
        finally:
            try:
                device, self.device = self.device, None
                if device is not None:
                    device.close()
            finally:
                if self.host_reserved:
                    self.host_budget.release(32 * MiB)
                    self.host_reserved = False
