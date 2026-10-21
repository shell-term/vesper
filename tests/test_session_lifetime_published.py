import json
from pathlib import Path
import unittest

from vesper.compile_cache_validation import image_identity, QUALIFIED_CUBIN_IMAGE_SET
from vesper.generation_validation import QUALIFIED_IDENTITY, QUALIFIED_GOLDENS, validate_memory
from vesper.model_artifacts import digest
from vesper.resources import GiB

RUN = Path(__file__).resolve().parents[1] / 'results/pc-heterogeneous/2026-10-21-session-lifetime'


class PublishedSessionLifetimeTests(unittest.TestCase):
    def test_file_bound_execution_versions_and_complete_output_checks_are_preserved(self):
        summary = json.loads((RUN / 'summary.json').read_bytes())
        for name in ('new_model_execution_run_in_summary_writer', 'repeated_measurement_claim',
                     'fully_matched_resource_policy', 'task_quality_scored'):
            self.assertFalse(summary[name])
        for name, receipt in summary['files'].items():
            self.assertFalse(Path(name).is_absolute()); self.assertNotIn('..', Path(name).parts)
            raw = (RUN / name).read_bytes()
            self.assertEqual((len(raw), digest(raw)), (receipt['bytes'], receipt['sha256']))
            self.assertNotRegex(raw, rb'(?im)(?:^|[\s"\x27])[a-z]:[/\\]|physical_directory|GPU-[0-9a-f-]{16,}')
        expected = {'baseline': ('a920fa3f79192452e9533c72c876596139273d47',
                                '74cfcb19878ee1d6f1b9d495889e9e86560cc13ab92b8eeee456082f1e1a2c25'),
                    'retained': ('68bbde5f7410f866ac1d93271de6631b6272fbf2',
                                '1f6783bec8b4ca3c60214407002a0998929364b77ef3dfb614089b4a6a47c5b9')}
        reports = {}
        for mode, (head, sha) in expected.items():
            raw = (RUN / (mode + '.json')).read_bytes(); self.assertEqual(digest(raw), sha)
            r = json.loads(raw); reports[mode] = r; c = r['workspace_context']
            self.assertEqual(c['git_head'], head)
            self.assertEqual(c['dirty_diff_sha256'], '9d908ecfb6b256def8b49a7c504e6c889c4b0e41fe6ce3e01863dd7b61a20aa0')
            self.assertEqual((c['platform'], c['profile']), ('pc-heterogeneous', 'native'))
            self.assertEqual((r['original_identity_sha256'], r['complete_goldens_sha256']),
                             (QUALIFIED_IDENTITY, QUALIFIED_GOLDENS))
            self.assertEqual((r['status'], len(r['calls'])), ('pass', 5))
            self.assertTrue(r['complete_short_reference_match'])
            self.assertEqual(r['completion']['completion_ids'], [1014, 260, 223])
            self.assertEqual(sum(x['logits']['elements'] for x in r['calls']), 646400)
            self.assertEqual(sum(len(x['ordered_routes']) for x in r['calls']), 200)
            self.assertEqual(image_identity(r['compiled_images']), QUALIFIED_CUBIN_IMAGE_SET)
        a, b = reports['baseline'], reports['retained']
        self.assertEqual(a['final_state_sha256'], b['final_state_sha256'])
        for x, y in zip(a['calls'], b['calls']):
            self.assertEqual({k: v for k, v in x.items() if k != 'forward_seconds'},
                             {k: v for k, v in y.items() if k != 'forward_seconds'})

    def test_measured_tradeoff_includes_actual_resources_and_cleanup(self):
        summary = json.loads((RUN / 'summary.json').read_bytes())
        for mode, cap in (('baseline', 64 * GiB), ('retained', 96 * GiB)):
            r = json.loads((RUN / (mode + '.json')).read_bytes()); validate_memory(r, cap)
            self.assertFalse(r['additional_quantization']); self.assertFalse(r['changed_routing_or_sampling'])
            self.assertEqual(r['HTTP_calls'], 0); self.assertEqual(r['weight_payloads_fetched'], 0)
            self.assertFalse(any(any(v.values()) for v in r['native_frameworks'].values()))
            self.assertEqual(r['timing']['decode_tokens_after_first'], 2)
            self.assertAlmostEqual(r['timing']['decode_tokens_per_second'],
                                   2 / sum(c['forward_seconds'] for c in r['calls'][3:]))
            observed = summary['observations'][mode]
            self.assertEqual(observed['direct_read_bytes'], sum(p['direct_read_bytes'] for p in r['providers'].values()))
            self.assertEqual(observed['context_opens'], observed['context_closes'])
            for p in r['providers'].values():
                self.assertFalse(p['opened']); self.assertFalse(p['failed'])
                self.assertFalse(any(p['last_cleanup'].values()))
        a, b = summary['observations']['baseline'], summary['observations']['retained']
        self.assertAlmostEqual(summary['observed_ratios']['decode_rate_retained_over_baseline'],
                               b['timing']['decode_tokens_per_second'] / a['timing']['decode_tokens_per_second'])

    def test_refused_setup_attempts_are_not_completed_execution_or_quality_evidence(self):
        for name in ('initial-retained-refusal.json', 'second-retained-refusal.json'):
            r = json.loads((RUN / name).read_bytes())
            self.assertEqual((r['status'], r['exit_code'], r['completed_model_calls']), ('resource_refused', 1, 0))
            self.assertTrue(r['not_a_stamped_model_execution_receipt'])
            self.assertFalse(r['completed_execution_receipt_produced']); self.assertFalse(r['quality_scored'])
            self.assertNotIn('workspace_context', r)


if __name__ == '__main__':
    unittest.main()
