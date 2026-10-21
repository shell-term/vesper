"""Mock compiler checks for scoped Spark reuse; no CUDA/GPU qualification."""

import ctypes as C
from contextlib import ExitStack
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from vesper import cuda
from vesper.execution_target import spark_kernel_cache
from vesper.memory import MemoryPolicy, RuntimeMemory
from vesper.resources import GiB, MiB


class Compiler:
    def __init__(self, library):
        self._name, self._handle = str(library), 1
        self.version, self.binary = (13, 0), b'CPU mock CUBIN, never a qualification receipt'
        self.nvrtcVersion = Mock(side_effect=self.version_query)
        self.nvrtcCreateProgram = Mock(side_effect=self.create)
        self.nvrtcCompileProgram = Mock(return_value=0)
        self.nvrtcDestroyProgram = Mock(return_value=0)
        self.nvrtcGetCUBINSize = Mock(side_effect=self.size)
        self.nvrtcGetCUBIN = Mock(side_effect=self.image)
        self.nvrtcGetProgramLogSize = Mock(return_value=0)
        self.nvrtcGetProgramLog = Mock(return_value=0)

    def version_query(self, major, minor):
        C.cast(major,C.POINTER(C.c_int)).contents.value = self.version[0]
        C.cast(minor,C.POINTER(C.c_int)).contents.value = self.version[1]
        return 0

    def create(self, pointer, *_):
        C.cast(pointer,C.POINTER(C.c_void_p)).contents.value = 1
        return 0

    def size(self, _program, pointer):
        C.cast(pointer,C.POINTER(C.c_size_t)).contents.value = len(self.binary)
        return 0

    def image(self, _program, target):
        C.memmove(target,self.binary,len(self.binary))
        return 0


class CudaCompileCacheTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        self.temp = self.stack.enter_context(tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]))
        self.root = Path(self.temp)
        (self.root/'bin').mkdir()
        self.library = self.root/'bin/nvrtc64_mock.dll'
        self.library.write_bytes(b'CPU fixture compiler library identity')
        self.compiler = Compiler(self.library)
        self.stack.enter_context(patch.dict(os.environ,{'CUDA_HOME':str(self.root),'CUDA_PATH':str(self.root)}))
        self.stack.enter_context(patch.object(cuda,'load_toolkit_library',return_value=self.compiler))
        self.stack.enter_context(patch.object(cuda.C,'CDLL',return_value=self.compiler))
        self.stack.enter_context(patch.object(cuda.os,'add_dll_directory',return_value=Mock(),create=True))
        self.stack.enter_context(patch.object(RuntimeMemory,'CONTROL_METADATA_BYTES',0))
        self.stack.enter_context(patch.object(RuntimeMemory,'ENABLE_WATCHDOG',False))
        self.policy = self.stack.enter_context(RuntimeMemory(MemoryPolicy(host_cap_bytes=MiB,
            pinned_cap_bytes=128,device_caps={0:1024},unified_device_memory=True),
            available=lambda:128*GiB,process=lambda:{'resident_bytes':0}))

    def test_direct_spark_calls_reuse_exact_build_and_reproduce_redacted_metadata(self):
        with spark_kernel_cache() as cache:
            first, second = {}, {}
            a = cuda.compile_probe('121',projections=True,compiler_info=first)
            b = cuda.compile_probe('121',projections=True,compiler_info=second)
            self.assertEqual(a,b)
            self.assertEqual(first,second)
            self.assertEqual(self.compiler.nvrtcCompileProgram.call_count,1)
            self.assertEqual(self.compiler.nvrtcDestroyProgram.call_count,1)
            self.assertEqual((cache.stats()['hits'],cache.stats()['misses']),(1,1))
            self.assertGreater(self.policy.host,len(a))
            self.assertNotIn(str(self.root),str(first))
            self.assertFalse(cache.stats()['qualification_receipt_issued'])
        self.assertEqual(self.policy.host,0)
        self.assertFalse(self.policy.reclaimers)

    def test_pc_and_calls_outside_scope_keep_compiling(self):
        with spark_kernel_cache() as cache:
            for _ in range(2): cuda.compile_probe('89',expert=True)
            self.assertEqual(cache.stats()['entries'],0)
        for _ in range(2): cuda.compile_probe('121',expert=True)
        self.assertEqual(self.compiler.nvrtcCompileProgram.call_count,4)

    def test_changed_flags_source_compiler_library_and_options_do_not_hit(self):
        with spark_kernel_cache() as cache:
            cuda.compile_probe('121',expert=True)
            cuda.compile_probe('121',routing=True)
            self.compiler._handle = 2
            cuda.compile_probe('121',expert=True)
            real_read = Path.read_bytes
            def changed(path):
                raw=real_read(path)
                return raw+b'\n// changed compile input\n' if path.name=='format_probe.cu' else raw
            with patch.object(Path,'read_bytes',changed): cuda.compile_probe('121',expert=True)
            another = self.root/'another'; (another/'bin').mkdir(parents=True)
            (another/'bin/nvrtc64_mock.dll').write_bytes(b'fixture')
            with patch.dict(os.environ,{'CUDA_HOME':str(another),'CUDA_PATH':str(another)}):
                cuda.compile_probe('121',expert=True)
            self.assertEqual(self.compiler.nvrtcCompileProgram.call_count,5)
            self.assertEqual(cache.stats()['hits'],0)

    def test_changed_compiler_version_is_refused_before_cache_reuse(self):
        with spark_kernel_cache() as cache:
            cuda.compile_probe('121',expert=True)
            self.compiler.version=(13,1)
            with self.assertRaisesRegex(ValueError,'explicit profile'):
                cuda.compile_probe('121',expert=True)
            self.assertEqual(self.compiler.nvrtcCompileProgram.call_count,1)
            self.assertEqual(cache.stats()['entries'],1)

    def test_failed_compile_is_destroyed_and_never_retained(self):
        self.compiler.nvrtcCompileProgram.return_value=1
        with spark_kernel_cache() as cache:
            with self.assertRaises(cuda.CudaError): cuda.compile_probe('121',expert=True)
            self.assertEqual(cache.stats()['entries'],0)
            self.assertEqual(self.policy.host,0)
            self.compiler.nvrtcCompileProgram.return_value=0
            cuda.compile_probe('121',expert=True)
            self.assertEqual(self.compiler.nvrtcCompileProgram.call_count,2)
            self.assertEqual(self.compiler.nvrtcDestroyProgram.call_count,2)

    def test_unidentifiable_library_compiles_fresh_instead_of_reusing_an_entry(self):
        with spark_kernel_cache() as cache:
            with patch.object(cuda,'_compiler_library_identity',return_value=None):
                for _ in range(2): cuda.compile_probe('121',expert=True)
            self.assertEqual(self.compiler.nvrtcCompileProgram.call_count,2)
            self.assertEqual(cache.stats()['entries'],0)


if __name__=='__main__': unittest.main()
