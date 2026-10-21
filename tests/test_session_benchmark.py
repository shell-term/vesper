from copy import deepcopy
import json
from pathlib import Path
import unittest

import numpy as np

from vesper.attention_validation import array_digest
from vesper.session_benchmark import benchmark_config, compare_step, timing_summary, validate_images


class SessionBenchmarkTests(unittest.TestCase):
    def test_placement_moves_two_helper_slots_to_ram_with_matched_total_capacity(self):
        a, b = (benchmark_config(mode) for mode in ('retained', 'host-retained'))
        self.assertEqual(a.host_cap_bytes, b.host_cap_bytes)
        self.assertEqual(a.cache.host_cap_bytes, b.cache.host_cap_bytes)
        self.assertEqual(a.cache.primary_slots, b.cache.primary_slots)
        self.assertEqual(a.cache.ram_cache_bytes + 18800640 * a.cache.helper_slots,
                         b.cache.ram_cache_bytes + 18800640 * b.cache.helper_slots)
        self.assertEqual((b.cache.helper_device, b.cache.helper_slots), (None, 0))
        self.assertEqual(a.cache.integrity_mode, b.cache.integrity_mode)
        self.assertEqual(b.cache.integrity_mode, 'gpu-sha256')
        self.assertEqual(a.cache.available_floor, b.cache.available_floor)

    def test_host_only_requires_all_thirteen_qualified_primary_images(self):
        path = Path(__file__).resolve().parents[1] / 'results/pc-heterogeneous/2026-10-21-session-lifetime/retained.json'
        images = json.loads(path.read_bytes())['compiled_images']
        primary = [i for i in images if i['compiler']['architecture'] == '89']
        validate_images(images, 'retained')
        validate_images(primary, 'host-retained')
        for changed in (images, primary[:-1]):
            with self.assertRaises(RuntimeError):
                validate_images(changed, 'host-retained')
        changed = deepcopy(primary)
        changed[0]['sha256'] = '0' * 64
        with self.assertRaises(RuntimeError):
            validate_images(changed, 'host-retained')

    def test_unknown_mode_is_refused_before_model_execution(self):
        with self.assertRaises(ValueError):
            benchmark_config('unknown')

    def test_decode_rate_excludes_prompt_and_first_generated_token(self):
        report = timing_summary([1, 2, 3, 4, 5], 3)
        self.assertEqual(report['prefill_forward_seconds'], 6)
        self.assertEqual(report['decode_tokens_after_first'], 2)
        self.assertEqual(report['decode_forward_seconds'], 9)
        self.assertAlmostEqual(report['decode_tokens_per_second'], 2 / 9)
        self.assertEqual(report['forward_seconds'], 15)

    def test_partial_prompt_cannot_publish_a_decode_rate(self):
        report = timing_summary([1, 2], 3)
        self.assertFalse(report['prompt_complete'])
        self.assertIsNone(report['decode_tokens_per_second'])
        self.assertEqual(report['decode_tokens_after_first'], 0)

    def test_invalid_clock_values_are_refused(self):
        for value in (0, -1, float('nan'), float('inf'), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                timing_summary([value], 3)

    def fixture(self):
        logits = np.array([[0., 3., 1.]], np.float32)
        indices = np.arange(6, dtype=np.int64)
        weights = np.full(6, 1 / 6, np.float32)
        result = {'head': {'logits_fp32': logits}}
        for layer in range(40):
            result[f'layer{layer}'] = {'moe_gate.indices': indices.copy(), 'moe_gate.weights': weights.copy()}
        meta = {'position': 2, 'token_id': 260, 'sampling_requested': True, 'selected_id': 1}
        expected = {'meta': meta.copy(), 'rng_offset': 4,
                    'boundaries': [{'check': 'result/head/logits_fp32/', 'dtype': 'float32',
                                    'shape': [1, 3], 'elements': 3, 'sha256': array_digest(logits)}],
                    'ordered_routes': {str(l): {'indices': indices.tolist(), 'weight_bits': weights.view(np.uint32).tolist()}
                                       for l in range(40)}}
        return meta, result, expected

    def test_short_check_binds_all_logits_even_when_selected_id_is_unchanged(self):
        meta, result, expected = self.fixture()
        self.assertTrue(compare_step(meta, result, 4, expected)['matches_reference'])
        result['head']['logits_fp32'][0, 2] += .125
        self.assertEqual(int(result['head']['logits_fp32'].argmax()), meta['selected_id'])
        with self.assertRaises(RuntimeError):
            compare_step(meta, result, 4, expected)

    def test_short_check_rejects_changed_routes_feedback_and_rng(self):
        for kind in ('route', 'feedback', 'rng'):
            with self.subTest(kind=kind):
                meta, result, expected = self.fixture()
                offset = 4
                if kind == 'route':
                    result['layer39']['moe_gate.indices'][0] = 7
                elif kind == 'feedback':
                    meta['token_id'] = 11
                else:
                    offset = 8
                with self.assertRaises(RuntimeError):
                    compare_step(meta, result, offset, expected)


if __name__ == '__main__':
    unittest.main()
