"""Bounded SM89 Attention using qualified local CUBINs and the CUDA Driver.

Optional reference tools build the bundle once. Execution needs NumPy and the
driver only. This is the sparse Attention operation, not projections or KV state.
"""

from contextlib import ExitStack
import ctypes as C
import hashlib
import json
from pathlib import Path
import threading

import numpy as np

from .attention_memory import FUNCTION, validate_qualification
from .attention_storage import STAGES, trace_shapes
from .checkpoint import MODEL, REVISION, REFERENCE_HASHES
from .cuda import Driver, Device, _bind
from .resources import GiB, MiB, MemoryBudget

# Published generated CUDA profiles, with only the Query allocation in registers.
CANDIDATES = {
    (64, 256, False): ("94183e3c57f91df69c563a0d22478fc7243a1619448a936ba40c39fe4a75a6f8", 43008),
    (64, 256, True): ("356869df7d0304cfaf7a497795d3087e60f16c47115643031ab6667e906d85cd", 43008),
    (16, 512, False): ("4f929a36c5e0af46fcfca2272f629d23a0e429a3c92d6d1287162669aa221718", 69632),
    (16, 512, True): ("2fb3fc49956b96961f184e9057ce227457c15d5e441f4e6dafb6ba6fc813e297", 69632),
    (64, 512, False): ("06a4b388bfde5c5ba44155eeb29bb8fac24313deef9d6104518f8de641d1b46d", 75776),
    (64, 512, True): ("85a43cb8c9bc62116fb39189fb4968b2c3f2e88cb5833e91a8d68f8ca1701b60", 75776),
}
CANDIDATE_BINARIES = dict(zip(CANDIDATES, (
    "b042f3a97317bbc14ca60acebcbaafde17cf122573752b55b604fb4df4cfa036",
    "c21385e76f093a080cf6b3a28bedb0efc13feabae5161662de27a77db540c192",
    "dfb1802aaffcdefed72a22fe39c3f5a7cf4498de44afe433e49d7d5017d46315",
    "938e25258cb183962c0b578d5bad5964ee78c5ac18563f88d1f2b886d30ff3f1",
    "10a3f3feaec52172f578dae9608f59f6f455fb6a2acf0aa59d955a71a7d8c709",
    "79d7367ac978ef89c76565994ca0c25e1d5c47d1e8821813d1c9217a1a73bcf9",
)))


def argument_names(diagnostics):
    pointers = ["attn_sink", "kv", "o", "q", "topk_idxs"]
    return sorted([*pointers, *STAGES] if diagnostics else pointers) + ["b", "m", "n", "topk"]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_bounded(path, limit):
    with Path(path).open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("Attention artifact exceeds its byte limit")
    return data


def kernel_filename(heads, dimension, diagnostics):
    return f"h{heads}-d{dimension}-{'trace' if diagnostics else 'plain'}.cubin"


def load_bundle(directory):
    """Check every artifact before any CUDA module is loaded. No network access."""
    directory = Path(directory)
    raw = read_bounded(directory / "manifest.json", 128 * 1024)
    manifest = json.loads(raw)
    expected = {"schema_version": 1, "kind": "vesper_attention_register_query_bundle", "model": MODEL,
                "revision": REVISION, "compute_capability": "8.9", "minimum_driver_api": 13040,
                "tilelang_version": "0.1.12", "nvrtc_version": [13, 4],
                "kernel_source_sha256": REFERENCE_HASHES["kernel.py"]}
    if not isinstance(manifest, dict) or any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError("Unqualified Attention bundle or toolchain profile")
    receipt_raw = read_bounded(directory / "qualification.json", 2 * MiB)
    if digest(receipt_raw) != manifest.get("qualification_sha256"):
        raise ValueError("Attention qualification receipt hash differs")
    receipt = validate_qualification(json.loads(receipt_raw))
    profiles = manifest.get("profiles", [])
    if len(profiles) != 6 or profiles != receipt.get("native_candidate_exports"):
        raise ValueError("Attention bundle is not bound to all qualified candidate binaries")
    if digest(read_bounded(directory / "DeepSeek-LICENSE", 64 * 1024)) != REFERENCE_HASHES["LICENSE"]:
        raise ValueError("Attention bundle requires the pinned third-party license")
    binaries, seen = {}, set()
    for p in profiles:
        key = p.get("heads"), p.get("dimension"), p.get("diagnostics")
        if (type(key[0]) is not int or type(key[1]) is not int or type(key[2]) is not bool
                or key not in CANDIDATES or key in seen):
            raise ValueError("Unexpected or duplicate Attention kernel profile")
        seen.add(key)
        source_hash, shared = CANDIDATES[key]
        args = p.get("argument_names", [])
        if (p.get("generated_cuda_sha256") != source_hash or p.get("dynamic_shared_bytes") != shared
                or p.get("function") != FUNCTION or p.get("file") != kernel_filename(*key)
                or p.get("binary_reproduction") is not True
                or args != argument_names(key[2]) or p.get("cubin_sha256") != CANDIDATE_BINARIES[key]):
            raise ValueError("Attention candidate provenance or launch contract differs")
        binary = read_bounded(directory / kernel_filename(*key), 2 * MiB)
        if digest(binary) != p["cubin_sha256"]:
            raise ValueError("Attention CUBIN hash differs; refusing module load")
        binaries[key] = binary
    if sum(map(len, binaries.values())) > 8 * MiB:
        raise ValueError("Attention bundle exceeds the combined binary allowance")
    return manifest, binaries, digest(raw)


def validate_inputs(q, kv, sink, indices, diagnostics=False):
    if type(diagnostics) is not bool or not all(isinstance(a, np.ndarray) for a in (q, kv, sink, indices)):
        raise ValueError("Expected NumPy arrays and an explicit diagnostic flag")
    if q.ndim != 4 or indices.ndim != 3:
        raise ValueError("Expected rank-four Query and rank-three indices")
    batch, sequence, heads, dimension = q.shape
    shapes = trace_shapes(batch, sequence, heads, dimension, indices.shape[-1])
    if ((heads, dimension, diagnostics) not in CANDIDATES or kv.ndim != 3 or kv.shape[0] != batch
            or not 1 <= kv.shape[1] <= 257 or kv.shape[2] != dimension or sink.shape != (heads,)
            or indices.shape[:2] != (batch, sequence)
            or tuple(a.dtype for a in (q, kv, sink, indices)) != (np.uint16, np.uint16, np.float32, np.int32)
            or any(not a.flags.c_contiguous for a in (q, kv, sink, indices))):
        raise ValueError("Invalid or unbounded Attention input layout; BF16 is passed as uint16 bits")
    if (np.any((q & 0x7F80) == 0x7F80) or np.any((kv & 0x7F80) == 0x7F80)
            or not np.isfinite(sink).all() or np.any((indices < -1) | (indices >= kv.shape[1]))):
        raise ValueError("Attention requires finite inputs and valid indices or -1 padding")
    return shapes


class NativeAttention:
    """Synchronous, serialized operation; per-call device buffers are transactional."""

    def __init__(self, bundle: Path, ordinal=0):
        self.bundle, self.ordinal = Path(bundle), ordinal
        self.lock, self.opened = threading.Lock(), False
        self.budget = MemoryBudget(cap=32 * MiB)
        self.stack = ExitStack()

    def __enter__(self):
        with self.lock:
            if self.opened:
                raise RuntimeError("Native Attention is already open")
            try:
                self.budget.acquire(32 * MiB)
                self.stack.callback(self.budget.release, 32 * MiB)
                self.manifest, binaries, self.manifest_sha256 = load_bundle(self.bundle)
                self.driver = Driver()
                inventory = self.driver.inventory()
                self.device_info = next(d for d in inventory["devices"] if d["ordinal"] == self.ordinal)
                if self.device_info["compute_capability"] != "8.9" or inventory["driver_api_version"] < 13040:
                    raise ValueError("Native Attention currently requires SM89 and a CUDA 13.4-capable driver")
                self.device = self.stack.enter_context(Device(self.driver, self.ordinal, cap=16 * MiB, free_floor=2 * GiB))
                query = _bind(self.driver.lib, "cuDeviceGetAttribute", [C.POINTER(C.c_int), C.c_int, C.c_int])
                limit = C.c_int()
                self.driver.check(query(C.byref(limit), 97, self.ordinal))
                attribute = _bind(self.driver.lib, "cuFuncSetAttribute", [C.c_void_p, C.c_int, C.c_int])
                self.kernels = {}
                for p in self.manifest["profiles"]:
                    key = p["heads"], p["dimension"], p["diagnostics"]
                    if p["dynamic_shared_bytes"] > limit.value:
                        raise ValueError("Attention kernel exceeds available opt-in shared memory")
                    module = self.device.load(binaries[key])
                    function = C.c_void_p()
                    self.driver.check(self.driver.cuModuleGetFunction(C.byref(function), module, FUNCTION.encode()))
                    self.driver.check(attribute(function, 8, p["dynamic_shared_bytes"]))
                    self.kernels[key] = function, p
                self.opened = True
                return self
            except BaseException:
                self.stack.close()
                raise

    def __call__(self, q, kv, sink, indices, *, diagnostics=False):
        with self.lock:
            if not self.opened:
                raise RuntimeError("Native Attention is not open")
            shapes = validate_inputs(q, kv, sink, indices, diagnostics)
            self.budget.acquire(0)
            batch, sequence, heads, dimension = q.shape
            self.device.current()
            with ExitStack() as allocations:
                def allocate(size):
                    pointer = self.device.allocate(size)
                    allocations.callback(self.device.free, pointer)
                    return pointer
                args = {}
                for name, array in zip(("q", "kv", "attn_sink", "topk_idxs"), (q, kv, sink, indices)):
                    pointer = allocate(array.nbytes)
                    self.driver.check(self.driver.cuMemcpyHtoD_v2(pointer, array.ctypes.data, array.nbytes))
                    args[name] = C.c_uint64(pointer)
                output = allocate(q.nbytes)
                args["o"] = C.c_uint64(output)
                traces = []
                if diagnostics:
                    for name, shape in zip(STAGES, shapes):
                        pointer = allocate(int(np.prod(shape)) * 4)
                        traces.append((pointer, shape))
                        args[name] = C.c_uint64(pointer)
                args.update(b=C.c_int(batch), m=C.c_int(sequence), n=C.c_int(kv.shape[1]), topk=C.c_int(indices.shape[-1]))
                function, profile = self.kernels[heads, dimension, diagnostics]
                values = [args[name] for name in profile["argument_names"]]
                pointers = (C.c_void_p * len(values))(*[C.cast(C.byref(v), C.c_void_p) for v in values])
                self.driver.check(self.driver.cuLaunchKernel(function, sequence, batch, 1, 256, 1, 1,
                                                            profile["dynamic_shared_bytes"], None, pointers, None))
                self.driver.check(self.driver.cuCtxSynchronize())
                result = {"output_bf16": self.device.download(output, q.shape, np.uint16)}
                result.update((name, self.device.download(pointer, shape, np.float32))
                              for name, (pointer, shape) in zip(STAGES, traces))
                return result

    def __exit__(self, *_):
        with self.lock:
            self.opened = False
            self.stack.close()

    def stats(self):
        return {"host_allowance_bytes": self.budget.peak, "host_reserved_bytes": self.budget.reserved,
                "device_peak_owned_bytes": getattr(self, "device", None).peak if hasattr(self, "device") else 0,
                "device_owned_bytes": sum(self.device.allocations.values()) if hasattr(self, "device") else 0,
                "owned_context_active": bool(self.device.context) if hasattr(self, "device") else False,
                "owned_modules_active": len(self.device.modules) if hasattr(self, "device") and self.device.context else 0,
                "reservation_is_RSS_limit": False, "driver_module_memory_excluded": True}
