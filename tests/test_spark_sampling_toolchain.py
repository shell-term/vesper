"""CPU NVRTC mocks check include/options provenance, not sampler arithmetic."""

import ctypes as C
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from vesper import cuda
from vesper.execution_target import compile_execution_kernel, kernel_source_sha256


class Compiler:
    def __init__(self, version):
        self.version, self.source, self.options = version, None, None
        self.binary = b"CPU mock binary, never GPU evidence"
        self.nvrtcCreateProgram = Mock(side_effect=self.create)
        self.nvrtcCompileProgram = Mock(side_effect=self.compile)
        self.nvrtcDestroyProgram = Mock(return_value=0)
        self.nvrtcVersion = Mock(side_effect=self.get_version)
        self.nvrtcGetCUBINSize = Mock(side_effect=self.size)
        self.nvrtcGetCUBIN = Mock(side_effect=self.image)
        self.nvrtcGetProgramLogSize = Mock(return_value=0)
        self.nvrtcGetProgramLog = Mock(return_value=0)

    def create(self, pointer, source, *args):
        C.cast(pointer, C.POINTER(C.c_void_p)).contents.value = 1
        self.source = source
        return 0

    def compile(self, program, count, options):
        self.options = [options[i].decode() for i in range(count)]
        return 0

    def get_version(self, major, minor):
        C.cast(major, C.POINTER(C.c_int)).contents.value = self.version[0]
        C.cast(minor, C.POINTER(C.c_int)).contents.value = self.version[1]
        return 0

    def size(self, program, pointer):
        C.cast(pointer, C.POINTER(C.c_size_t)).contents.value = len(self.binary)
        return 0

    def image(self, program, target):
        C.memmove(target, self.binary, len(self.binary))
        return 0


class SparkSamplingToolchainTests(unittest.TestCase):
    def compile(self, architecture, *, sampling=True, version=(13, 0)):
        compiler, info = Compiler(version), {}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "bin").mkdir()
            (root / "bin/nvrtc64_mock.dll").write_bytes(b"CPU fixture")
            with patch.dict(os.environ, {"CUDA_HOME": str(root), "CUDA_PATH": str(root)}), \
                    patch.object(cuda, "load_toolkit_library", return_value=compiler), \
                    patch.object(cuda.C, "CDLL", return_value=compiler), \
                    patch.object(cuda.os, "add_dll_directory", return_value=Mock(), create=True):
                image = cuda.compile_probe(architecture, sampling=sampling, compiler_info=info)
        return compiler, info, image

    def test_spark_sampling_passes_actual_cccl_and_records_redacted_option(self):
        compiler, info, image = self.compile("121")
        self.assertEqual(image, compiler.binary)
        self.assertEqual(info["nvrtc_version"], [13, 0])
        self.assertEqual(info["options"][-1], "--include-path=<CUDA>/include/cccl")
        self.assertTrue(compiler.options[-1].replace("\\", "/").endswith("/include/cccl"))
        self.assertNotIn("<CUDA>", compiler.options[-1])
        self.assertTrue(info["include_path_redacted"])
        self.assertEqual(info["concatenated_source_sha256"], kernel_source_sha256(sampling=True))
        compiler.nvrtcDestroyProgram.assert_called_once()

    def test_pc_sampler_source_and_five_compile_options_remain_unchanged(self):
        compiler, info, image = self.compile("89", version=(12, 4))
        self.assertEqual(info["options"], ["--gpu-architecture=sm_89", "--std=c++17", "--fmad=false",
                                          "--ftz=false", "--include-path=<CUDA>/include"])
        self.assertEqual(len(compiler.options), 5)
        spark, _, _ = self.compile("121")
        self.assertEqual(compiler.source, spark.source)
        self.assertEqual(info["concatenated_source_sha256"], kernel_source_sha256(sampling=True))
        binary, provenance = compile_execution_kernel(lambda *a, **kw: (kw["compiler_info"].update(info) or image),
            compiler_info={}, sampling=True, pc_cubin_sha256=hashlib.sha256(image).hexdigest())
        self.assertEqual(binary, image)
        self.assertEqual(provenance["execution_target"], "pc-sm89")

    def test_other_spark_operators_do_not_inherit_sampler_include_change(self):
        compiler, info, _ = self.compile("121", sampling=False)
        self.assertEqual(len(compiler.options), 5)
        self.assertNotIn("--include-path=<CUDA>/include/cccl", info["options"])

    def test_target_gate_requires_sampler_cccl_option_and_rejects_extra_options(self):
        _, info, image = self.compile("121")
        def invoke(metadata):
            return compile_execution_kernel(lambda *a, **kw: (kw["compiler_info"].update(metadata) or image),
                "spark-sm121-candidate", compiler_info={}, sampling=True, pc_cubin_sha256="a" * 64)
        binary, provenance = invoke(info)
        self.assertEqual(binary, image)
        self.assertEqual(provenance["compiler"]["options"][-1], "--include-path=<CUDA>/include/cccl")
        for options in (info["options"][:-1], [*info["options"], "--use_fast_math"]):
            with self.assertRaises(ValueError):
                invoke({**info, "options": options})


if __name__ == "__main__":
    unittest.main()
