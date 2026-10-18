"""Small CUDA Driver / NVRTC bindings for probes, without a tensor framework."""

from __future__ import annotations

import ctypes as C
import ctypes.util
import os
from pathlib import Path

import numpy as np

from .resources import GiB, MiB, MemoryBudget


class CudaError(RuntimeError):
    pass


def _bind(lib, name, args):
    function = getattr(lib, name)
    function.argtypes, function.restype = args, C.c_int
    return function


class Driver:
    def __init__(self):
        if os.name == "nt":
            self.lib = C.WinDLL(str(Path(os.environ.get("WINDIR", r"C:\Windows")) / "System32/nvcuda.dll"))
        else:
            self.lib = C.CDLL(ctypes.util.find_library("cuda") or "libcuda.so.1")
        ptr, integer, size, devptr = C.c_void_p, C.c_int, C.c_size_t, C.c_uint64
        signatures = {
            "cuInit": [C.c_uint], "cuDeviceGetCount": [C.POINTER(integer)],
            "cuDeviceGet": [C.POINTER(integer), integer],
            "cuDeviceGetName": [ptr, integer, integer],
            "cuDeviceComputeCapability": [C.POINTER(integer), C.POINTER(integer), integer],
            "cuDeviceTotalMem_v2": [C.POINTER(size), integer],
            "cuDevicePrimaryCtxRetain": [C.POINTER(ptr), integer],
            "cuDevicePrimaryCtxRelease_v2": [integer], "cuCtxSetCurrent": [ptr],
            "cuCtxSynchronize": [], "cuMemGetInfo_v2": [C.POINTER(size), C.POINTER(size)],
            "cuMemAlloc_v2": [C.POINTER(devptr), size], "cuMemFree_v2": [devptr],
            "cuMemHostAlloc": [C.POINTER(ptr), size, C.c_uint], "cuMemFreeHost": [ptr],
            "cuMemcpyHtoD_v2": [devptr, ptr, size], "cuMemcpyDtoH_v2": [ptr, devptr, size],
            "cuDeviceCanAccessPeer": [C.POINTER(integer), integer, integer],
            "cuMemcpyPeer": [devptr, ptr, devptr, ptr, size],
            "cuModuleLoadData": [C.POINTER(ptr), ptr], "cuModuleUnload": [ptr],
            "cuModuleGetFunction": [C.POINTER(ptr), ptr, C.c_char_p],
            "cuLaunchKernel": [ptr, C.c_uint, C.c_uint, C.c_uint, C.c_uint, C.c_uint, C.c_uint,
                               C.c_uint, ptr, C.POINTER(ptr), C.POINTER(ptr)],
            "cuGetErrorName": [integer, C.POINTER(C.c_char_p)],
            "cuGetErrorString": [integer, C.POINTER(C.c_char_p)],
            "cuDriverGetVersion": [C.POINTER(integer)],
        }
        for name, args in signatures.items():
            setattr(self, name, _bind(self.lib, name, args))
        self.check(self.cuInit(0))

    def check(self, result):
        if result:
            name, description = C.c_char_p(), C.c_char_p()
            self.cuGetErrorName(result, C.byref(name))
            self.cuGetErrorString(result, C.byref(description))
            raise CudaError(f"{result}: {name.value!r}: {description.value!r}")

    def inventory(self):
        count, version = C.c_int(), C.c_int()
        self.check(self.cuDeviceGetCount(C.byref(count)))
        self.check(self.cuDriverGetVersion(C.byref(version)))
        devices = []
        for ordinal in range(count.value):
            device, major, minor, total = C.c_int(), C.c_int(), C.c_int(), C.c_size_t()
            name = C.create_string_buffer(256)
            self.check(self.cuDeviceGet(C.byref(device), ordinal))
            self.check(self.cuDeviceGetName(name, len(name), device))
            self.check(self.cuDeviceComputeCapability(C.byref(major), C.byref(minor), device))
            self.check(self.cuDeviceTotalMem_v2(C.byref(total), device))
            devices.append({"ordinal": ordinal, "name": name.value.decode(),
                            "compute_capability": f"{major.value}.{minor.value}", "total_bytes": total.value})
        peers = []
        for src in range(count.value):
            for dst in range(count.value):
                if src != dst:
                    possible = C.c_int()
                    self.check(self.cuDeviceCanAccessPeer(C.byref(possible), src, dst))
                    peers.append({"source": src, "destination": dst, "direct_peer_access": bool(possible.value)})
        return {"driver_api_version": version.value, "devices": devices, "peers": peers}


class Device:
    def __init__(self, driver: Driver, ordinal: int, cap=256 * MiB, free_floor=GiB):
        self.driver, self.ordinal = driver, ordinal
        self.cap, self.free_floor = cap, free_floor
        self.context = C.c_void_p()
        self.allocations = {}
        self.modules = []
        self.peak = 0
        driver.check(driver.cuDevicePrimaryCtxRetain(C.byref(self.context), ordinal))
        self.current()

    def current(self):
        self.driver.check(self.driver.cuCtxSetCurrent(self.context))

    def memory(self):
        self.current()
        free, total = C.c_size_t(), C.c_size_t()
        self.driver.check(self.driver.cuMemGetInfo_v2(C.byref(free), C.byref(total)))
        return {"free": free.value, "total": total.value}

    def allocate(self, size: int) -> int:
        self.current()
        used = sum(self.allocations.values())
        if size <= 0 or used + size > self.cap:
            raise MemoryError("Device probe allocation exceeds its explicit cap")
        if self.memory()["free"] - size < self.free_floor:
            raise MemoryError("Device probe would consume the VRAM reserve")
        pointer = C.c_uint64()
        self.driver.check(self.driver.cuMemAlloc_v2(C.byref(pointer), size))
        self.allocations[pointer.value] = size
        self.peak = max(self.peak, used + size)
        return pointer.value

    def free(self, pointer: int):
        self.current()
        if pointer not in self.allocations:
            raise ValueError("Cannot free an allocation owned by another device")
        self.driver.check(self.driver.cuMemFree_v2(pointer))
        del self.allocations[pointer]

    def upload(self, array: np.ndarray) -> int:
        array = np.ascontiguousarray(array)
        pointer = self.allocate(array.nbytes)
        self.driver.check(self.driver.cuMemcpyHtoD_v2(pointer, array.ctypes.data, array.nbytes))
        return pointer

    def download(self, pointer: int, shape, dtype) -> np.ndarray:
        self.current()
        result = np.empty(shape, dtype=dtype)
        if result.nbytes > self.allocations.get(pointer, 0):
            raise ValueError("Read exceeds owned device allocation")
        self.driver.check(self.driver.cuMemcpyDtoH_v2(result.ctypes.data, pointer, result.nbytes))
        return result

    def load(self, cubin: bytes):
        self.current()
        module = C.c_void_p()
        image = C.create_string_buffer(cubin)
        self.driver.check(self.driver.cuModuleLoadData(C.byref(module), image))
        self.modules.append(module)
        return module

    def launch(self, module, name: str, arguments: list, items: int, threads=128):
        self.current()
        function = C.c_void_p()
        self.driver.check(self.driver.cuModuleGetFunction(C.byref(function), module, name.encode()))
        pointers = (C.c_void_p * len(arguments))(*[C.cast(C.byref(arg), C.c_void_p) for arg in arguments])
        self.driver.check(self.driver.cuLaunchKernel(function, (items + threads - 1) // threads, 1, 1,
                                                   threads, 1, 1, 0, None, pointers, None))
        self.driver.check(self.driver.cuCtxSynchronize())

    def close(self):
        if not self.context:
            return
        self.current()
        try:
            for pointer in list(self.allocations):
                self.free(pointer)
            for module in reversed(self.modules):
                self.driver.check(self.driver.cuModuleUnload(module))
        finally:
            self.driver.check(self.driver.cuDevicePrimaryCtxRelease_v2(self.ordinal))
            self.context = C.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class Pinned:
    def __init__(self, driver, size: int, budget: MemoryBudget):
        self.driver, self.size, self.budget = driver, size, budget
        self.pointer = C.c_void_p()
        budget.acquire(size)
        try:
            driver.check(driver.cuMemHostAlloc(C.byref(self.pointer), size, 1))  # PORTABLE
        except Exception:
            budget.release(size)
            raise

    def array(self):
        return np.ctypeslib.as_array((C.c_uint8 * self.size).from_address(self.pointer.value))

    def __enter__(self):
        return self

    def __exit__(self, *_):
        try:
            self.driver.check(self.driver.cuMemFreeHost(self.pointer))
        finally:
            self.budget.release(self.size)


def compile_probe(architecture: str, *, reduction_trace: bool = False, expert: bool = False,
                  integrity: bool = False, routing: bool = False, shared: bool = False,
                  aggregation: bool = False, kv: bool = False, normalization: bool = False,
                  projections: bool = False, compiler_info: dict | None = None) -> bytes:
    """Compile an actual architecture cubin; no host compiler or GPU required."""
    if os.name == "nt":
        roots = [Path(value) for key in ("CUDA_PATH", "CUDA_HOME") if (value := os.environ.get(key))]
        roots += sorted(Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA").glob("v*"), reverse=True)
        root = next((p for p in roots if list((p / "bin").glob("nvrtc64_*.dll"))), None)
        if root is None:
            raise RuntimeError("CUDA Toolkit / NVRTC was not found")
        dll_directory = os.add_dll_directory(str(root / "bin"))
        lib = C.CDLL(str(next((root / "bin").glob("nvrtc64_*.dll"))))
    else:
        root = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
        dll_directory = None
        lib = C.CDLL(ctypes.util.find_library("nvrtc") or str(root / "lib64/libnvrtc.so"))
    pointer, size = C.c_void_p, C.c_size_t
    create = _bind(lib, "nvrtcCreateProgram", [C.POINTER(pointer), C.c_char_p, C.c_char_p, C.c_int, pointer, pointer])
    compile_ = _bind(lib, "nvrtcCompileProgram", [pointer, C.c_int, C.POINTER(C.c_char_p)])
    destroy = _bind(lib, "nvrtcDestroyProgram", [C.POINTER(pointer)])
    log_size = _bind(lib, "nvrtcGetProgramLogSize", [pointer, C.POINTER(size)])
    get_log = _bind(lib, "nvrtcGetProgramLog", [pointer, pointer])
    get_size = _bind(lib, "nvrtcGetCUBINSize", [pointer, C.POINTER(size)])
    get_cubin = _bind(lib, "nvrtcGetCUBIN", [pointer, pointer])
    source = (Path(__file__).parent / "kernels/format_probe.cu").read_bytes()
    if reduction_trace:
        source += b"\n" + (Path(__file__).parent / "kernels/reduction_probe.cu").read_bytes()
    if expert:
        source += b"\n" + (Path(__file__).parent / "kernels/expert_probe.cu").read_bytes()
    if integrity:
        source += b"\n" + (Path(__file__).parent / "kernels/integrity_probe.cu").read_bytes()
    if routing:
        source += b"\n" + (Path(__file__).parent / "kernels/routing_probe.cu").read_bytes()
    if shared:
        source += b"\n" + (Path(__file__).parent / "kernels/shared_probe.cu").read_bytes()
    if aggregation:
        source += b"\n" + (Path(__file__).parent / "kernels/aggregation_probe.cu").read_bytes()
    if kv or projections:
        source += b"\n" + (Path(__file__).parent / "kernels/kv_probe.cu").read_bytes()
    if normalization:
        source += b"\n" + (Path(__file__).parent / "kernels/normalization_probe.cu").read_bytes()
    if projections:
        source += b"\n" + (Path(__file__).parent / "kernels/projection_probe.cu").read_bytes()
    if kv or normalization or projections:
        source = source.replace(b"\r\n", b"\n")
    if compiler_info is not None:
        import hashlib
        major, minor = C.c_int(), C.c_int()
        version = _bind(lib, "nvrtcVersion", [C.POINTER(C.c_int), C.POINTER(C.c_int)])
        if version(C.byref(major), C.byref(minor)):
            raise CudaError("Cannot query the native NVRTC compiler version")
        compiler_info.update(nvrtc_version=[major.value, minor.value], architecture=architecture,
                             concatenated_source_sha256=hashlib.sha256(source).hexdigest(),
                             options=[f"--gpu-architecture=sm_{architecture}", "--std=c++17", "--fmad=false",
                                      "--ftz=false", "--include-path=<CUDA>/include"], include_path_redacted=True)
    program = pointer()
    if create(C.byref(program), source, b"format_probe.cu", 0, None, None):
        raise CudaError("NVRTC could not create a program")
    try:
        options = [f"--gpu-architecture=sm_{architecture}", "--std=c++17", "--fmad=false",
                   "--ftz=false", f"--include-path={root / 'include'}"]
        encoded = (C.c_char_p * len(options))(*[option.encode() for option in options])
        result = compile_(program, len(options), encoded)
        if result:
            length = size()
            log_size(program, C.byref(length))
            log = C.create_string_buffer(length.value)
            get_log(program, log)
            raise CudaError(f"NVRTC error {result}: {log.value.decode(errors='replace')}")
        length = size()
        if get_size(program, C.byref(length)) or not length.value:
            raise CudaError("NVRTC did not produce a cubin")
        image = C.create_string_buffer(length.value)
        if get_cubin(program, image):
            raise CudaError("NVRTC could not return its cubin")
        return image.raw
    finally:
        destroy(C.byref(program))
        if dll_directory:
            dll_directory.close()
