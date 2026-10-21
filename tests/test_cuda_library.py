"""Installed toolkit discovery covers ARM64 without hiding other failures."""
import unittest
from pathlib import PurePosixPath
from unittest.mock import patch, sentinel
from vesper.cuda import load_toolkit_library


class ToolkitLibraryTests(unittest.TestCase):
    def setUp(self):
        # The Linux resolver is tested on Windows too, without changing the
        # host pathlib implementation by faking the operating system globally.
        patcher=patch('vesper.cuda.Path',PurePosixPath)
        patcher.start(); self.addCleanup(patcher.stop)

    def test_sbsa_fallback_after_loader_and_legacy_layout(self):
        seen=[]
        def load(path):
            seen.append(path)
            if 'sbsa-linux' in path: return sentinel.library
            raise OSError('not found')
        with patch('vesper.cuda.os.name','posix'),patch('vesper.cuda.ctypes.util.find_library',return_value=None), \
                patch('vesper.cuda.C.CDLL',side_effect=load):
            self.assertIs(load_toolkit_library('nvrtc'),sentinel.library)
        self.assertEqual(len(seen),3)
        self.assertTrue(seen[-1].replace('\\','/').endswith('targets/sbsa-linux/lib/libnvrtc.so'))

    def test_loader_success_avoids_other_libraries(self):
        with patch('vesper.cuda.os.name','posix'),patch('vesper.cuda.ctypes.util.find_library',return_value='libcublas.so.13'), \
                patch('vesper.cuda.C.CDLL',return_value=sentinel.library) as load:
            self.assertIs(load_toolkit_library('cublas'),sentinel.library)
            load.assert_called_once_with('libcublas.so.13')

    def test_reject_unknown_and_sanitize_failure(self):
        with patch('vesper.cuda.os.name','posix'):
            with self.assertRaises(ValueError): load_toolkit_library('../private')
            with patch('vesper.cuda.ctypes.util.find_library',return_value=None), \
                    patch('vesper.cuda.C.CDLL',side_effect=OSError('private host path')):
                with self.assertRaisesRegex(RuntimeError,'Installed CUDA') as error:
                    load_toolkit_library('nvrtc')
                self.assertNotIn('private host path',str(error.exception))


if __name__=='__main__': unittest.main()
