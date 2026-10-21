import json
from pathlib import Path
import unittest

from vesper.model_artifacts import digest


class PublishedNativeGPQATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = Path(__file__).resolve().parents[1] / 'results/pc-heterogeneous/2026-10-21-gpqa-native'
        cls.summary = json.loads((cls.directory / 'summary.json').read_bytes())
        cls.raw = json.loads((cls.directory / 'attempt.json').read_bytes())

    def test_receipts_are_unchanged_and_preserve_actual_execution_source(self):
        for name, entry in self.summary['files'].items():
            raw = (self.directory / name).read_bytes()
            self.assertEqual(len(raw), entry['bytes'])
            self.assertEqual(digest(raw), entry['sha256'])
        self.assertEqual(self.summary['execution_context'], self.raw['workspace_context'])
        self.assertTrue(self.raw['workspace_context']['git_head'].startswith('2220715'))

    def test_partial_prefill_cannot_become_a_quality_or_decode_result(self):
        row = self.raw['results'][0]
        self.assertEqual(row['stop_reason'], 'budget_exceeded')
        self.assertEqual(row['declared_max_new_tokens'], 262144)
        self.assertEqual(row['timing']['completed_model_calls'], 4)
        self.assertEqual(row['timing']['prompt_tokens'], 286)
        self.assertFalse(row['timing']['prompt_complete'])
        self.assertEqual(row['selected_ids_including_eos'], [])
        self.assertIsNone(row['score']['passed'])
        self.assertIsNone(row['timing']['decode_tokens_per_second'])
        self.assertEqual(self.raw['completed_case_ids'], [])
        self.assertEqual(len(self.raw['pending_case_ids']), 3)
        self.assertFalse(self.summary['quality_scored'])

    def test_original_asset_reuse_and_cleanup_remain_scoped(self):
        backend = self.raw['backend_observations'][0]
        self.assertEqual(backend['model_steps_attempted'], 5)
        self.assertEqual(backend['completed_model_calls'], 4)
        self.assertEqual(backend['reader']['payload_bytes_fetched'], 0)
        self.assertEqual(backend['bank']['expert_imports_this_process'], 0)
        after = self.raw['memory_after_close']
        for key in ('charged_host_bytes','pinned_bytes','live_leases','live_reclaimers','inflight_operations'):
            self.assertEqual(after[key], 0)
        self.assertTrue(all(x == 0 for x in after['device_used_bytes'].values()))
        self.assertTrue(after['monitor_thread_stopped'])
        self.assertIsNone(after['monitor_error'])
