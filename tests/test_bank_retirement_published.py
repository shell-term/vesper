import json
from pathlib import Path
import unittest

from vesper.checkpoint import MODEL,REVISION
from vesper.model_artifacts import digest

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/pc-heterogeneous/2026-10-20-expert-bank-retirement'


class PublishedBankRetirementTests(unittest.TestCase):
    def test_actual_retirement_keeps_the_exact_original_copy_census_and_root_identity(self):
        summary=json.loads((RUN/'summary.json').read_bytes())
        for name,entry in summary['files'].items():
            self.assertFalse(Path(name).is_absolute());self.assertNotIn('..',Path(name).parts)
            raw=(RUN/name).read_bytes()
            self.assertEqual((len(raw),digest(raw)),(entry['bytes'],entry['sha256']))
            self.assertNotRegex(raw,rb'(?i)(?:[A-Z]:[/\\]|physical_directory|GPU-[0-9a-f]{8})')
        raw=(RUN/'retirement.json').read_bytes();retired=json.loads(raw)
        self.assertEqual(digest(raw),'78821d9915d40c0745920e799e665a074eae7f4b972a30bc043954fcd14a86e6')
        prior=summary['original_copy_receipt'];praw=(RUN/prior['file']).read_bytes();copied=json.loads(praw)
        self.assertEqual((len(praw),digest(praw)),(prior['bytes'],prior['sha256']))
        self.assertEqual(retired['files'],copied['files'])
        self.assertEqual(retired['plan_sha256'],copied['plan_sha256'])
        self.assertEqual(retired['source_storage'],copied['source_storage'])
        self.assertEqual((retired['model'],retired['revision']),(MODEL,REVISION))
        self.assertEqual((retired['objects'],len(retired['files'])),(3375,3375))
        self.assertTrue(retired['source_retirement_requested'])
        self.assertEqual(retired['retired_source_file_bytes_this_process'],63467411542)
        self.assertEqual(retired['all_files_verified_bytes'],63467411542)
        self.assertEqual(retired['original_source_files_remaining'],0)
        for name in ('weight_bytes_converted','original_index_bytes_changed','copied_file_bytes_this_process'):
            self.assertEqual(retired[name],0,name)
        self.assertFalse(retired['model_execution_run']);self.assertFalse(retired['quality_scored'])
        context=retired['workspace_context']
        self.assertEqual(context['git_head'],'cd0407c4a3a9d92548b86aba1cf169a8ce277f7f')
        self.assertEqual(context['dirty_diff_sha256'],'9d908ecfb6b256def8b49a7c504e6c889c4b0e41fe6ce3e01863dd7b61a20aa0')
        self.assertEqual((context['platform'],context['profile']),('pc-heterogeneous','native'))
        self.assertEqual(summary['original_source_files_retired'],6750)
        self.assertEqual(summary['post_retirement_owned_expert_object_census'],{
            'pc-heterogeneous/expert-bank-v1':3375,'pc-heterogeneous/expert-bank-v2':552})
        self.assertFalse(summary['new_model_execution_run_in_summary_writer'])
        self.assertTrue(summary['reference_restart_required'])

    def test_first_reference_interruption_is_not_a_completed_or_scored_model_run(self):
        raw=(RUN/'reference-interruption.json').read_bytes();failed=json.loads(raw)
        self.assertEqual(digest(raw),'9e55663faafef3c9b32ae418faecb09660feb55586fc998bac3f9a7ecaaceb23')
        self.assertEqual((failed['successful_original_calls'],failed['incomplete_position'],failed['last_observed_layer']),(0,0,26))
        self.assertEqual(failed['observed_source_commit'],'cdee98010b25de132af5358ba6ed88e1030712a3')
        self.assertFalse(failed['complete_original_report_produced']);self.assertFalse(failed['reference_execution_receipt_produced'])
        self.assertFalse(failed['quality_scored']);self.assertFalse(failed['pagefile_growth_demonstrated'])
        self.assertFalse(failed['pagefile_settings_changed']);self.assertNotIn('workspace_context',failed)
        self.assertEqual(failed['explicit_build_cache_measured_file_bytes'],21846018)
        self.assertEqual(failed['cause_of_system_volume_decrease'],
            'not_established; explicit_build_cache_extent_is_insufficient_to_explain_the_observed_decrease')
        self.assertFalse(failed['private_log']['published'])
