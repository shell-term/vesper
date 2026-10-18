"""Single-row routing postprocessing for supplied FP32 logits, not a full gate."""

import ctypes as C
import hashlib
import json
from pathlib import Path
import struct
import zipfile

import numpy as np

from .checkpoint import MODEL, REVISION, LIMIT, read_small, read_range, validate_expert_identity
from .cuda import Device, Driver, compile_probe
from .resources import MemoryBudget, MiB

ROUTER_CONFIG = {"experts": 384, "topk": 6, "score_func": "sqrtsoftplus", "gate_temp": 1.0,
                 "norm_topk_prob": True, "route_scale": 1.5, "normalization_epsilon": 1e-20}
ROUTER_INFERENCE_CONFIG_HASH = "2e84f45cf1dac8c7fcbb200e96667d4b913275690668ed496f24c7747207a809"
ROUTER_TORCH_REVISION = "5c4886908584029761b579af026dcfb627c84070"
ROUTER_TOPK_POLICY = "pytorch_cuda_single_block_gather_then_bitonic32_k6"


def top6_indices(selection):
    """Finite FP32, one-row CUDA top-k specification for the pinned PyTorch build.

    Gather values above the kth threshold in ID order, then fill from equal
    values in ID order. Apply ATen's unstable 32-cell bitonic network to the six
    gathered entries. This is not a general torch.topk stability guarantee.
    See third_party/PyTorch-LICENSE and THIRD_PARTY_NOTICES.md.
    """
    selection = np.asarray(selection, dtype=np.float32)
    if selection.shape != (384,) or not np.isfinite(selection).all():
        raise ValueError("Top-six selection requires 384 finite FP32 values")
    bits = selection.view(np.uint32)
    ordered = bits ^ np.where(bits & np.uint32(0x80000000), np.uint32(0xffffffff), np.uint32(0x80000000))
    threshold = np.partition(ordered, 378)[378]
    gathered = np.concatenate((np.flatnonzero(ordered > threshold), np.flatnonzero(ordered == threshold)))[:6]
    keys = np.zeros(32, np.float32)
    indices = np.zeros(32, np.int32)
    valid = np.arange(32) < 6
    keys[:6], indices[:6] = selection[gathered], gathered

    def stage(stride, size):
        for thread in range(16):
            pos = 2 * thread - (thread & (stride - 1))
            other = pos + stride
            direction = bool(thread & (size // 2)) if size else False
            swap = bool((keys[pos] > keys[other] and valid[pos]) or not valid[other])
            if swap == direction:
                keys[pos], keys[other] = keys[other], keys[pos]
                indices[pos], indices[other] = indices[other], indices[pos]
                valid[pos], valid[other] = valid[other], valid[pos]

    for size in (2, 4, 8, 16):
        stride = size // 2
        while stride:
            stage(stride, size)
            stride //= 2
    for stride in (16, 8, 4, 2, 1):
        stage(stride, 0)
    return indices[:6].copy()


def fetch_router_fixture(directory: Path, layer=0):
    validate_expert_identity(layer, 0)
    budget = MemoryBudget(cap=64 * MiB)
    budget.acquire(64 * MiB)
    try:
        config_raw = read_small("config.json")
        config = json.loads(config_raw)["text_config"]
        if (config["n_routed_experts"], config["num_experts_per_tok"], config["scoring_func"],
                config["norm_topk_prob"], config["routed_scaling_factor"]) != (384, 6, "sqrtsoftplus", True, 1.5):
            raise ValueError("Routing configuration differs from this pinned probe")
        if any("temp" in name for name in config):
            raise ValueError("A routing temperature override requires explicit qualification")
        inference_raw = read_small("inference/config.json")
        if hashlib.sha256(inference_raw).hexdigest() != ROUTER_INFERENCE_CONFIG_HASH:
            raise ValueError("Released inference configuration hash differs from the pinned source")
        inference = json.loads(inference_raw)
        if (inference["n_routed_experts"], inference["n_activated_experts"], inference["score_func"],
                inference["route_scale"], inference.get("gate_temp", 1.0), inference.get("norm_topk_prob", True)) != (
                384, 6, "sqrtsoftplus", 1.5, 1.0, True):
            raise ValueError("Released inference routing parameters differ from the probe")
        index = json.loads(read_small("model.safetensors.index.json"))
        headers, arrays, tensors = {}, {}, []
        for name in ("bias", "bias_vl"):
            key = f"layers.{layer}.ffn.gate.{name}"
            shard = index["weight_map"][key]
            if shard not in headers:
                size = struct.unpack("<Q", read_range(shard, 0, 8))[0]
                if not 0 < size <= LIMIT:
                    raise ValueError("Excessive router shard header")
                headers[shard] = (json.loads(read_range(shard, 8, size)), 8 + size)
            header, data_start = headers[shard]
            tensor = header[key]
            begin, end = tensor["data_offsets"]
            if (tensor["dtype"], tensor["shape"], end - begin) != ("F32", [384], 1536):
                raise ValueError("Unexpected original router-bias layout")
            raw = read_range(shard, data_start + begin, end - begin)
            arrays[name] = np.frombuffer(raw, dtype="<f4").copy()
            tensors.append({"name": name, "tensor": key, "dtype": "F32", "shape": [384], "bytes": len(raw),
                            "shard": shard, "shard_byte_offset": data_start + begin,
                            "sha256": hashlib.sha256(raw).hexdigest()})
        manifest = {"probe": "fetch_router_fixture", "status": "pass", "kind": "router_bias_fixture",
                    "model": MODEL, "revision": REVISION, "layer": layer, "config": ROUTER_CONFIG,
                    "config_sha256": hashlib.sha256(config_raw).hexdigest(), "tensors": tensors,
                    "inference_config_sha256": ROUTER_INFERENCE_CONFIG_HASH,
                    "weights_fetched_bytes": 3072,
                    "temperature_source": "Released inference/config.json omits gate_temp; pinned ModelArgs default is 1.0",
                    "scope": "Two original selection biases only; gate projection and model execution excluded"}
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(directory / "native-router.npz", **arrays)
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return manifest
    finally:
        budget.release(64 * MiB)


def validate_router_arrays(arrays):
    if set(arrays) != {"bias", "bias_vl"}:
        raise ValueError("Routing needs original text and image-span biases")
    for array in arrays.values():
        if array.dtype != np.float32 or array.shape != (384,) or not np.isfinite(array).all():
            raise ValueError("Router bias must contain 384 finite FP32 values")


def load_router_fixture(path: Path):
    path = Path(path)
    if path.stat().st_size > 16384:
        raise ValueError("Router archive exceeds its bounded fixture size")
    with (path.parent / "manifest.json").open("rb") as stream:
        raw = stream.read(65537)
    if len(raw) > 65536:
        raise ValueError("Excessive router fixture metadata")
    manifest = json.loads(raw)
    if (manifest.get("kind"), manifest.get("model"), manifest.get("revision"), manifest.get("config"),
            manifest.get("inference_config_sha256")) != (
            "router_bias_fixture", MODEL, REVISION, ROUTER_CONFIG, ROUTER_INFERENCE_CONFIG_HASH):
        raise ValueError("Router fixture differs from the pinned source/configuration")
    validate_expert_identity(manifest.get("layer"), 0)
    entries = manifest.get("tensors", [])
    if len(entries) != 2 or {t["name"] for t in entries} != {"bias", "bias_vl"}:
        raise ValueError("Incomplete router provenance")
    with zipfile.ZipFile(path) as archive:
        if sorted(archive.namelist()) != ["bias.npy", "bias_vl.npy"]:
            raise ValueError("Router archive must contain exactly two bias arrays")
        for name in ("bias", "bias_vl"):
            info = archive.getinfo(name + ".npy")
            if info.file_size > 2560:
                raise ValueError("Router array exceeds its bounded layout")
            with archive.open(info) as stream:
                version = np.lib.format.read_magic(stream)
                if version not in ((1, 0), (2, 0)):
                    raise ValueError("Unsupported router array version")
                reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                shape, fortran, dtype = reader(stream, max_header_size=1024)
                if shape != (384,) or fortran or dtype != np.dtype(np.float32):
                    raise ValueError("Router header exceeds the bounded FP32 layout")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in ("bias", "bias_vl")}
    validate_router_arrays(arrays)
    for entry in entries:
        name = entry["name"]
        if (entry.get("tensor"), entry.get("dtype"), entry.get("shape"), entry.get("bytes")) != (
                f"layers.{manifest['layer']}.ffn.gate.{name}", "F32", [384], 1536):
            raise ValueError("Router tensor identity/layout differs from its manifest")
        if hashlib.sha256(memoryview(arrays[name])).hexdigest() != entry["sha256"]:
            raise ValueError("Router bias differs from its original source hash")
    return arrays, manifest


def select_scores(scores, bias):
    """CPU selection specification; transcendental score computation is excluded."""
    scores, bias = np.asarray(scores, dtype=np.float32), np.asarray(bias, dtype=np.float32)
    if scores.shape != (384,) or bias.shape != (384,) or not np.isfinite(scores).all() or not np.isfinite(bias).all() or np.any(scores < 0):
        raise ValueError("Selection requires 384 finite nonnegative scores and finite biases")
    with np.errstate(over="ignore"):
        selection = scores + bias
    if not np.isfinite(selection).all():
        raise ValueError("Selection score overflow")
    indices = top6_indices(selection)
    raw = scores[indices]
    with np.errstate(over="ignore"):
        total = np.float32(np.float32(np.float32(raw[0] + raw[4]) + raw[2])
                           + np.float32(np.float32(raw[1] + raw[5]) + raw[3]))
    denominator = np.float32(total + np.float32(1e-20))
    if not np.isfinite(denominator):
        raise ValueError("Routing normalization overflows the finite probe scope")
    weights = (raw / denominator) * np.float32(1.5)
    return weights, indices


class RouterProbe:
    """Single-token native candidate. Diagnostics do not qualify a full gate."""

    def __init__(self, arrays, ordinal=0):
        validate_router_arrays(arrays)
        self.arrays, self.ordinal, self.device = arrays, ordinal, None
        self.inspections = 0

    def __enter__(self):
        if self.device is not None:
            raise RuntimeError("Router probe is already open")
        driver = Driver()
        self.info = next(d for d in driver.inventory()["devices"] if d["ordinal"] == self.ordinal)
        self.device = Device(driver, self.ordinal, cap=MiB)
        self.device_object = self.device
        try:
            cubin = compile_probe(self.info["compute_capability"].replace(".", ""), routing=True)
            self.cubin_sha256 = hashlib.sha256(cubin).hexdigest()
            self.module = self.device.load(cubin)
            self.biases = {name: self.device.upload(array) for name, array in self.arrays.items()}
            self.layout = {"logits": ((384,), np.float32), "scores": ((384,), np.float32),
                           "selection": ((384,), np.float32), "raw": ((6,), np.float32),
                           "denominator": ((1,), np.float32), "weights": ((6,), np.float32),
                           "indices": ((6,), np.int32), "tied": ((1,), np.int32)}
            self.work = {name: self.device.allocate(int(np.prod(shape)) * np.dtype(dtype).itemsize)
                         for name, (shape, dtype) in self.layout.items()}
            return self
        except BaseException:
            self.device.close()
            self.device = None
            raise

    def inspect(self, logits, image_span=False, synthetic_zero_bias=False):
        if self.device is None:
            raise RuntimeError("Use the router probe as a context manager")
        if type(image_span) is not bool or type(synthetic_zero_bias) is not bool:
            raise ValueError("Router probe mask flags must be boolean")
        logits = np.ascontiguousarray(logits, dtype=np.float32)
        if logits.shape != (384,) or not np.isfinite(logits).all():
            raise ValueError("Router probe accepts one finite 384-element FP32 logit vector")
        d, p = self.device, self.work
        d.current()
        d.driver.check(d.driver.cuMemcpyHtoD_v2(p["logits"], logits.ctypes.data, logits.nbytes))
        args = [C.c_uint64(v) for v in (p["logits"], self.biases["bias"], self.biases["bias_vl"], p["scores"], p["selection"])]
        args += [C.c_int(image_span), C.c_int(synthetic_zero_bias)]
        d.launch(self.module, "router_scores", args, 384)
        return self._finish()

    def inspect_selection(self, scores, selection):
        """Synthetic selection-only diagnostics; never substitutes a model bias."""
        if self.device is None:
            raise RuntimeError("Use the router probe as a context manager")
        scores, selection = (np.ascontiguousarray(value, dtype=np.float32) for value in (scores, selection))
        if (scores.shape != (384,) or selection.shape != (384,) or not np.isfinite(scores).all()
                or not np.isfinite(selection).all() or np.any(scores < 0)
                or np.max(scores) > np.finfo(np.float32).max / 8):
            raise ValueError("Synthetic selection requires finite bounded raw scores and finite selection scores")
        d = self.device
        d.current()
        for name, array in (("scores", scores), ("selection", selection)):
            d.driver.check(d.driver.cuMemcpyHtoD_v2(self.work[name], array.ctypes.data, array.nbytes))
        return self._finish()

    def _finish(self):
        d, p = self.device, self.work
        args = [C.c_uint64(p[name]) for name in ("scores", "selection", "raw", "denominator", "weights", "indices", "tied")]
        d.launch(self.module, "router_top6", args, 1)
        result = {name: d.download(p[name], shape, dtype) for name, (shape, dtype) in self.layout.items() if name != "logits"}
        self.inspections += 1
        return result

    def decision(self, logits, image_span=False):
        """Single-row original-bias decision using the pinned CUDA top-k path."""
        result = self.inspect(logits, image_span=image_span)
        return result["weights"], result["indices"]

    def __exit__(self, *_):
        if self.device is not None:
            self.device.close()
            self.device = None
