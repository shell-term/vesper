"""Small cuBLAS binding for the single-row FP32 gate projection."""

import ctypes as C
import ctypes.util
import os
from pathlib import Path

from .cuda import CudaError, _bind, load_toolkit_library

GATE_BLAS_WORKSPACE = 4096 * 1024 * 2 + 16 * 1024 * 8


def find_cublas(library=None):
    if library is not None:
        return str(Path(library).resolve(strict=True))
    if os.name == "nt":
        roots = [Path(value) for name in ("CUDA_PATH", "CUDA_HOME") if (value := os.environ.get(name))]
        roots += sorted(Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA").glob("v*"), reverse=True)
        for root in roots:
            candidates = sorted((root / "bin").glob("cublas64_*.dll"), reverse=True)
            if candidates:
                return str(candidates[0])
        raise RuntimeError("cuBLAS was not found in the CUDA Toolkit")
    return ctypes.util.find_library("cublas") or "libcublas.so"


class Cublas:
    """Own handle, default stream, host scalars, explicit workspace, FP32 math."""

    def __init__(self, library=None):
        if os.environ.get("NVIDIA_TF32_OVERRIDE") == "1":
            raise ValueError("This FP32 gate profile does not allow forced TF32")
        self.library = find_cublas(library)
        self.directory = os.add_dll_directory(str(Path(self.library).parent)) if os.name == "nt" else None
        self.handle = C.c_void_p()
        try:
            self.lib = (load_toolkit_library('cublas') if os.name!='nt' and library is None
                        else C.CDLL(self.library))
            ptr, integer = C.c_void_p, C.c_int
            signatures = {
                "cublasCreate_v2": [C.POINTER(ptr)], "cublasDestroy_v2": [ptr],
                "cublasGetVersion_v2": [ptr, C.POINTER(integer)],
                "cublasSetStream_v2": [ptr, ptr], "cublasSetPointerMode_v2": [ptr, integer],
                "cublasSetMathMode": [ptr, integer], "cublasGetMathMode": [ptr, C.POINTER(integer)],
                "cublasSetWorkspace_v2": [ptr, ptr, C.c_size_t],
                "cublasSgemm_v2": [ptr, integer, integer, integer, integer, integer, C.POINTER(C.c_float),
                                   ptr, integer, ptr, integer, C.POINTER(C.c_float), ptr, integer],
            }
            for name, args in signatures.items():
                setattr(self, name, _bind(self.lib, name, args))
            self.check(self.cublasCreate_v2(C.byref(self.handle)))
            self.check(self.cublasSetStream_v2(self.handle, None))
            self.check(self.cublasSetPointerMode_v2(self.handle, 0))
            self.check(self.cublasSetMathMode(self.handle, 0))
            version, mode = C.c_int(), C.c_int()
            self.check(self.cublasGetVersion_v2(self.handle, C.byref(version)))
            self.check(self.cublasGetMathMode(self.handle, C.byref(mode)))
            self.version, self.math_mode = version.value, mode.value
        except BaseException:
            self.close()
            raise

    @staticmethod
    def check(result):
        if result:
            raise CudaError(f"cuBLAS status {result}")

    def workspace(self, pointer, size):
        self.check(self.cublasSetWorkspace_v2(self.handle, pointer, size))

    def gate(self, weight, activation, output):
        alpha, beta = C.c_float(1), C.c_float(0)
        self.check(self.cublasSgemm_v2(self.handle, 1, 0, 384, 1, 5120, C.byref(alpha),
                                     weight, 5120, activation, 5120, C.byref(beta), output, 384))

    def close(self):
        try:
            handle, self.handle = self.handle, C.c_void_p()
            if handle and hasattr(self, "cublasDestroy_v2"):
                self.check(self.cublasDestroy_v2(handle))
        finally:
            if self.directory is not None:
                self.directory.close()
                self.directory = None
