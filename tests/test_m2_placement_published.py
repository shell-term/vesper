import json
from pathlib import Path
import unittest

from vesper.compile_cache_validation import image_identity,QUALIFIED_CUBIN_IMAGE_SET
from vesper.generation_validation import QUALIFIED_IDENTITY,QUALIFIED_GOLDENS,validate_memory
from vesper.model_artifacts import digest
from vesper.resources import GiB
from vesper.session_benchmark import QUALIFIED_PRIMARY_IMAGES

RUN=Path(__file__).resolve().parents[1]/'results/pc-heterogeneous/2026-10-21-m2-placement'


class PublishedPlacementCycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary=json.loads((RUN/'summary.json').read_bytes())
        cls.reports={name:json.loads((RUN/(name+'.json')).read_bytes()) for name in cls.summary['trial_order']}

    def test_raw_files_preserve_execution_identity_and_exact_short_output_scope(self):
        self.assertEqual(len(self.summary['files']),6)
        for name,receipt in self.summary['files'].items():
            self.assertFalse(Path(name).is_absolute());self.assertNotIn('..',Path(name).parts)
            raw=(RUN/name).read_bytes()
            self.assertEqual((len(raw),digest(raw)),(receipt['bytes'],receipt['sha256']))
            self.assertNotRegex(raw,rb'(?im)(?:^|[\s"\x27])[a-z]:[/\\]|physical_directory|GPU-[0-9a-f-]{16,}')
        expected=None
        for r in self.reports.values():
            c=r['workspace_context']
            self.assertEqual(c['git_head'],'78a9cc00348ddde16afe9998c8dde2a490e85770')
            self.assertEqual(c['registry_sha256'],'62c940bcbae1587ad43f67a2080aa08597abb92071de8844a6eea1810215427e')
            self.assertEqual(c['dirty_diff_sha256'],'9d908ecfb6b256def8b49a7c504e6c889c4b0e41fe6ce3e01863dd7b61a20aa0')
            self.assertEqual((c['platform'],c['profile']),('pc-heterogeneous','native'))
            self.assertEqual((r['original_identity_sha256'],r['complete_goldens_sha256']),(QUALIFIED_IDENTITY,QUALIFIED_GOLDENS))
            self.assertEqual((r['status'],len(r['calls'])),('pass',5))
            self.assertTrue(r['complete_short_reference_match'])
            self.assertEqual(r['completion']['completion_ids'],[1014,260,223])
            self.assertEqual(sum(c['logits']['elements'] for c in r['calls']),646400)
            self.assertEqual(sum(len(c['ordered_routes']) for c in r['calls']),200)
            actual=([{k:v for k,v in c.items() if k!='forward_seconds'} for c in r['calls']],r['final_state_sha256'],r['completion'])
            if expected is None:expected=actual
            self.assertEqual(actual,expected)
            wanted=QUALIFIED_PRIMARY_IMAGES if r['mode']=='host-retained' else QUALIFIED_CUBIN_IMAGE_SET
            self.assertEqual(image_identity(r['compiled_images']),wanted)

    def test_matched_capacity_and_audits_with_actual_headroom_and_zero_cleanup(self):
        for r in self.reports.values():
            validate_memory(r,96*GiB)
            self.assertEqual(r['memory_after_policy_close']['pinned_cap_bytes'],2*GiB)
            self.assertEqual(r['cooperative_budget_seconds'],300)
            self.assertEqual(r['HTTP_calls'],0);self.assertEqual(r['weight_payloads_fetched'],0)
            self.assertFalse(r['additional_quantization']);self.assertFalse(r['changed_routing_or_sampling'])
            policy=r['cache_policy']
            self.assertEqual(policy['integrity_mode'],'gpu-sha256')
            self.assertEqual(policy['primary_slots'],2)
            self.assertEqual(policy['ram_cache_bytes']+18800640*policy['helper_slots'],8*18800640)
            for p in r['providers'].values():
                self.assertEqual((p['calls'],p['context_opens'],p['context_closes']),(5,1,1))
                self.assertTrue(p['retained']);self.assertFalse(p['opened']);self.assertFalse(p['failed'])
                self.assertEqual(p['cache_config'],policy)
                self.assertFalse(any(p['last_cleanup'].values()))
                if r['mode']=='host-retained':
                    self.assertEqual(p['source_tiers'].get('helper_gpu',0),0)
                    self.assertFalse(any(v['calls'] for k,v in p['stage_timings'].items() if k.startswith('transfer.helper_')))
            self.assertEqual(policy['helper_slots'],0 if r['mode']=='host-retained' else 2)

    def test_four_run_decision_and_pooled_rate_keep_target_and_quality_limits_explicit(self):
        s=self.summary
        self.assertEqual(s['trial_order'],['r1-retained','r2-host-retained','r3-host-retained','r4-retained'])
        self.assertEqual(s['trial_count'],4);self.assertEqual(s['planned_model_budget_seconds'],1200)
        self.assertLess(s['actual_total_engine_wall_seconds'],1200)
        self.assertFalse(s['new_model_execution_run_in_summary_writer'])
        self.assertFalse(s['task_quality_scored']);self.assertFalse(s['vision_input_run'])
        self.assertFalse(s['general_or_production_throughput_claim'])
        self.assertEqual(s['twenty_tps_stretch_target'],20)
        self.assertFalse(s['twenty_tps_achieved_on_this_short_input'])
        for mode in ('retained','host-retained'):
            timings=[r['timing'] for r in self.reports.values() if r['mode']==mode]
            self.assertEqual(s['pooled_decode'][mode]['decode_tokens'],4)
            self.assertAlmostEqual(s['pooled_decode'][mode]['decode_tokens_per_second'],4/sum(t['decode_forward_seconds'] for t in timings))
        self.assertAlmostEqual(s['pooled_decode_speedup'],s['pooled_decode']['host-retained']['decode_tokens_per_second']/s['pooled_decode']['retained']['decode_tokens_per_second'])
        for pair in s['paired_ratios']:
            a,b=(self.reports[pair[k]]['timing'] for k in ('baseline','candidate'))
            self.assertAlmostEqual(pair['decode_rate_candidate_over_baseline'],b['decode_tokens_per_second']/a['decode_tokens_per_second'])
        adopt=all(p['decode_rate_candidate_over_baseline']>1 and p['request_TTFT_baseline_over_candidate']>1 for p in s['paired_ratios'])
        self.assertEqual(s['decision'],'adopt_opt_in_host_retained_for_measured_short_profile' if adopt else 'reject_candidate_for_now')


if __name__=='__main__':
    unittest.main()
