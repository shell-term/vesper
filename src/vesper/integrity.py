"""SHA-256 per consecutive 4 KiB chunk; source digests come from verified bytes."""

import ctypes as C
import hashlib

import numpy as np

from .cuda import compile_probe
from .timing import Timings

CHUNK_BYTES = 4096
DIGEST_BYTES = 32


def digest_shape(size):
    if type(size) is not int or not 0 < size <= 2**30:
        raise ValueError("Invalid bounded integrity payload size")
    return ((size + CHUNK_BYTES - 1) // CHUNK_BYTES, DIGEST_BYTES)


def hash_chunks(data: np.ndarray, target: np.ndarray):
    if data.dtype != np.uint8 or data.ndim != 1 or not data.flags.c_contiguous:
        raise ValueError("Integrity input must be a contiguous byte array")
    if target.dtype != np.uint8 or target.shape != digest_shape(data.size):
        raise ValueError("Integrity digest destination has the wrong size")
    if np.shares_memory(data, target):
        raise ValueError("Digest destination must not overwrite source bytes")
    view = memoryview(data)
    for i in range(target.shape[0]):
        digest = hashlib.sha256(view[i * CHUNK_BYTES:(i + 1) * CHUNK_BYTES]).digest()
        target[i] = np.frombuffer(digest, np.uint8)


class DeviceIntegrity:
    """Owns one bounded digest buffer. The caller owns the CUDA device lifecycle."""

    def __init__(self, device, capability, size, timings=None):
        self.device, self.size, self.shape = device, size, digest_shape(size)
        self.timings = timings or Timings()
        cubin = compile_probe(capability.replace(".", ""), integrity=True)
        self.cubin_sha256 = hashlib.sha256(cubin).hexdigest()
        self.module = device.load(cubin)
        self.pointer = device.allocate(int(np.prod(self.shape)))

    def digests(self, pointer, *, size=None, target=None):
        if self.pointer is None:
            raise RuntimeError("Integrity checker is closed")
        size = self.size if size is None else size
        shape = digest_shape(size)
        if size > self.size:
            raise ValueError("Integrity payload exceeds the fixed buffer scope")
        if self.device.allocations.get(pointer, 0) < size:
            raise ValueError("Integrity read exceeds an owned source allocation")
        if target is not None and (target.dtype != np.uint8 or target.shape != shape
                                   or not target.flags.c_contiguous or not target.flags.writeable):
            raise ValueError("Integrity readback must fit the complete digest table")
        with self.timings.section(f"integrity.gpu{self.device.ordinal}_sha256"):
            args = [C.c_uint64(pointer), C.c_uint64(self.pointer), C.c_int(size)]
            self.device.launch(self.module, "sha256_chunks", args, shape[0])
        with self.timings.section(f"integrity.gpu{self.device.ordinal}_digest_d2h"):
            if target is None:
                return self.device.download(self.pointer, shape, np.uint8)
            self.device.current()
            self.device.driver.check(self.device.driver.cuMemcpyDtoH_v2(target.ctypes.data, self.pointer, target.nbytes))
            return target

    def verify(self, pointer, expected, target=None, comparison=None):
        if expected.dtype != np.uint8 or expected.shape != self.shape:
            raise ValueError("Original digest table does not identify one complete expert")
        if target is not None and np.shares_memory(expected, target):
            raise ValueError("Audit readback must not overwrite the original digest table")
        if comparison is not None and (np.shares_memory(expected, comparison)
                                        or (target is not None and np.shares_memory(target, comparison))):
            raise ValueError("Audit comparison must not overwrite source or readback digests")
        actual = self.digests(pointer, target=target)
        if comparison is None:
            match = np.array_equal(actual, expected)
        else:
            if comparison.dtype != np.bool_ or comparison.shape != self.shape:
                raise ValueError("Integrity comparison scratch has the wrong layout")
            np.equal(actual, expected, out=comparison)
            match = bool(comparison.all())
        if not match:
            raise ValueError("GPU expert bytes differ from verified original SHA-256 chunks")
        return actual.nbytes

    def close(self):
        if self.pointer is not None:
            self.device.free(self.pointer)
            self.pointer = None
