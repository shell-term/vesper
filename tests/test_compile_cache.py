from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch

from vesper.compile_cache import CompilationCache, cached_compilation
from vesper.resources import GiB


class CompilationCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'synthetic-compiler-input';self.path.write_bytes(b'input')
        for p in (patch('vesper.compile_cache.toolchain_files',return_value=[self.path]),
                patch('vesper.resources.physical_memory',return_value={'available':64*GiB})):
            p.start();self.addCleanup(p.stop)

    def test_default_is_uncached_and_opt_in_reuses_exact_immutable_bytes_and_metadata(self):
        calls=[]
        @cached_compilation
        def compile_(architecture,*,expert=False,compiler_info=None):
            calls.append((architecture,expert));compiler_info.update(version=[12,4],source='fixed')
            return (architecture+str(expert)).encode()
        compile_('89',compiler_info={});compile_('89',compiler_info={});self.assertEqual(len(calls),2)
        with CompilationCache() as cache:
            a={};b={};first=compile_('89',expert=True,compiler_info=a);second=compile_('89',expert=True,compiler_info=b)
            self.assertIs(first,second);self.assertEqual(a,b)
            compile_('86',expert=True,compiler_info={});compile_('89',compiler_info={})
            self.assertEqual((cache.hits,cache.misses,len(cache.entries)),(1,3,3))
        self.assertEqual(cache.stats()['host_reserved_bytes'],0);self.assertFalse(cache.entries)

    def test_changed_source_or_toolchain_refuses_and_cleanup_still_runs(self):
        @cached_compilation
        def compile_(architecture,*,compiler_info=None):return b'qualified'
        with self.assertRaisesRegex(ValueError,'changed'):
            with CompilationCache() as cache:
                compile_('89');self.path.write_bytes(b'other')
                compile_('89')
        self.assertEqual(cache.stats()['host_reserved_bytes'],0)

    def test_same_size_restamped_change_is_detected_before_scope_can_qualify(self):
        stamp=self.path.stat()
        with self.assertRaisesRegex(ValueError,'changed'):
            with CompilationCache() as cache:
                self.path.write_bytes(b'other');os.utime(self.path,ns=(stamp.st_atime_ns,stamp.st_mtime_ns))
        self.assertEqual(cache.stats()['host_reserved_bytes'],0)

    def test_failed_compilation_publishes_no_entry_and_retry_can_succeed(self):
        calls=0
        @cached_compilation
        def compile_(architecture,*,compiler_info=None):
            nonlocal calls
            calls+=1
            if calls==1:raise RuntimeError('synthetic compiler refusal')
            return b'unchanged'
        with CompilationCache() as cache:
            with self.assertRaises(RuntimeError):compile_('89')
            self.assertFalse(cache.entries);self.assertEqual(cache.images_bytes,0)
            self.assertEqual(compile_('89'),b'unchanged');self.assertEqual(compile_('89'),b'unchanged')
            self.assertEqual((cache.hits,cache.misses),(1,1))

    def test_nested_scope_and_excessive_entry_count_are_refused(self):
        @cached_compilation
        def compile_(architecture,*,compiler_info=None):return architecture.encode()
        with CompilationCache() as cache:
            with self.assertRaises(RuntimeError):
                with CompilationCache():pass
            for i in range(32):compile_(str(i))
            with self.assertRaises(MemoryError):compile_('33')
            self.assertEqual(len(cache.entries),32)
