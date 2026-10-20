"""Explicit Spark rounding adaptation leaves PC bytes and other math intact."""
import unittest
from pathlib import Path
from vesper.execution_target import apply_execution_source_profile, kernel_source_sha256


class SparkRopeSourceTests(unittest.TestCase):
    def setUp(self):
        self.source=(Path(__file__).parents[1]/'src/vesper/kernels/normalization_probe.cu').read_bytes().replace(b'\r\n',b'\n')

    def test_only_original_imaginary_fused_multiply_is_reordered(self):
        old=b'float im = fmaf(real, d, __fmul_rn(imag, c));'
        new=b'float im = fmaf(imag, c, __fmul_rn(real, d));'
        actual=apply_execution_source_profile(self.source,'121',has_normalization=True)
        self.assertEqual(actual.replace(new,old),self.source)
        self.assertEqual(actual.count(new),1)
        self.assertIn(b'float r = fmaf(real, c, -__fmul_rn(imag, d));',actual)

    def test_pc_and_non_normalization_sources_are_byte_identical(self):
        self.assertIs(apply_execution_source_profile(self.source,'89',has_normalization=True),self.source)
        self.assertIs(apply_execution_source_profile(self.source,'121',has_normalization=False),self.source)
        self.assertEqual(kernel_source_sha256(normalization=True),kernel_source_sha256(execution_target='pc-sm89',normalization=True))
        self.assertNotEqual(kernel_source_sha256(normalization=True),kernel_source_sha256(execution_target='spark-sm121-candidate',normalization=True))

    def test_changed_or_duplicate_boundary_refuses_requalification(self):
        for source in (b'changed kernel',self.source+self.source):
            with self.assertRaises(ValueError):
                apply_execution_source_profile(source,'121',has_normalization=True)


if __name__=='__main__':unittest.main()
