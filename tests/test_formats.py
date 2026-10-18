import unittest

import numpy as np

from vesper.conformance import compare
from vesper.formats import (bf16_bits, bf16_values, e4m3_values, e8m0_values,
                            encode_e4m3, fp4_values, quantize_activation, unpack_fp4)


class FormatTests(unittest.TestCase):
    def test_fp4_known_values_and_signed_zero(self):
        expected = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32)
        actual = fp4_values(np.arange(16, dtype=np.uint8))
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))

    def test_e8m0_special_and_normal_scales(self):
        values = e8m0_values(np.array([0, 1, 126, 127, 128, 254, 255], np.uint8))
        np.testing.assert_array_equal(values[:6], np.array([2.**-127, 2.**-126, .5, 1, 2, 2.**127], np.float32))
        self.assertTrue(np.isnan(values[-1]))

    def test_fp4_nibble_order_and_scale(self):
        packed = np.full((1, 16), 0x98, np.uint8)
        actual = unpack_fp4(packed, np.array([[128]], np.uint8))
        self.assertTrue(np.signbit(actual[0, 0]))
        self.assertEqual(actual[0, 1], -1)

    def test_e4m3_known_ties_and_saturation(self):
        inputs = np.array([0., -0., 1., 1.0625, 1.1875, 448., 1000., 2.**-10], np.float32)
        np.testing.assert_array_equal(encode_e4m3(inputs), [0, 128, 56, 56, 58, 126, 126, 0])

    def test_all_finite_e4m3_codes_round_trip(self):
        codes = np.arange(256, dtype=np.uint8)
        codes = codes[(codes & 127) != 127]
        np.testing.assert_array_equal(encode_e4m3(e4m3_values(codes)), codes)

    def test_bf16_ties_round_to_even(self):
        values = np.array([0x3F808000, 0x3F818000, 0x80000000], np.uint32).view(np.float32)
        np.testing.assert_array_equal(bf16_bits(values), [0x3F80, 0x3F82, 0x8000])

    def test_zero_activation_uses_reference_amax_floor(self):
        values = np.zeros((2, 32), np.float32)
        values[1] = -0.
        quantized, scales = quantize_activation(values)
        np.testing.assert_array_equal(scales, [[105], [105]])
        self.assertTrue((quantized[0] == 0).all())
        self.assertTrue((quantized[1] == 128).all())

    def test_activation_rejects_nonfinite_and_partial_blocks(self):
        with self.assertRaises(ValueError):
            quantize_activation(np.ones(31, np.float32))
        with self.assertRaises(ValueError):
            quantize_activation(np.full(32, np.inf, np.float32))

    def test_comparison_does_not_hide_a_single_bit_or_signed_zero(self):
        a = np.array([1., 0.], np.float32)
        b = np.array([1., -0.], np.float32)
        self.assertEqual(compare(a, b, "zero")["status"], "fail")
        b = a.copy()
        b.view(np.uint32)[0] += 1
        self.assertEqual(compare(a, b, "rounding")["mismatches"], 1)


if __name__ == "__main__":
    unittest.main()
