import json
from pathlib import Path
import re
import unittest

from vesper.model_artifacts import digest
from vesper.checkpoint import MODEL,REVISION

RUN=Path(__file__).resolve().parents[1]/'results/pc-heterogeneous/2026-10-20-expert-bank-ssd'


class PublishedExpertSsdTests(unittest.TestCase):
    def test_actual_copy_receipt_is_file_bound_retains_originals_and_claims_no_model_run(self):
        summary=json.loads((RUN/'summary.json').read_bytes())
        for name,m in summary['files'].items():
            raw=(RUN/name).read_bytes();self.assertEqual((len(raw),digest(raw)),(m['bytes'],m['sha256']))
            self.assertNotRegex(raw,rb'(?i)(?:[A-Z]:[/\\]|physical_directory|GPU-[0-9a-f]{8})')
        migration=json.loads((RUN/'migration.json').read_bytes())
        self.assertEqual((migration['model'],migration['revision']),(MODEL,REVISION))
        self.assertEqual((migration['objects'],migration['original_source_files_remaining']),(3375,6750))
        self.assertEqual(migration['original_packed_payload_bytes'],63452160000)
        self.assertEqual(sum(e['payload_bytes']+e['manifest_bytes'] for e in migration['files']),63467411542)
        self.assertEqual(len({e['directory'] for e in migration['files']}),3375)
        for e in migration['files']:
            self.assertRegex(e['directory'],r'^layer[0-9]{1,2}/expert-[0-9]{3}$')
            self.assertRegex(e['payload_sha256'],r'^[0-9a-f]{64}$')
        self.assertFalse(migration['source_retirement_requested'])
        self.assertEqual(migration['retired_source_file_bytes_this_process'],0)
        self.assertEqual(migration['weight_bytes_converted'],0)
        self.assertFalse(migration['model_execution_run']);self.assertFalse(migration['quality_scored'])
        self.assertEqual(migration['workspace_context']['git_head'],'461ba64b1bae140c3d0fa57c291fbb8c2bde507c')
        self.assertEqual(migration['source_storage']['active_bank'],'pc-heterogeneous/expert-bank-v2')
        self.assertEqual(migration['source_storage']['per_bank_packed_payload_cap_bytes'],64*1024**3)
        self.assertEqual(migration['source_storage']['disk_available_floor_bytes'],64*1024**3)
        self.assertFalse(summary['new_model_execution_run_in_summary_writer'])

    def test_capacity_failure_is_incomplete_unscored_and_not_retrospectively_stamped_as_gpu_execution(self):
        failure=json.loads((RUN/'capacity-failure.json').read_bytes())
        self.assertEqual((failure['status'],failure['successful_preparation_calls']),('capacity_refused',69))
        self.assertEqual(failure['sampled_ids_including_any_actual_eos'],[2581,1309,3287,16,21017,27438,19075])
        self.assertEqual(failure['decoded_selected_ids_prefix'],'We need answer. Need parse trick')
        for name in ('actual_eos_reached','quality_scored','complete_plan_produced','original_reference_execution_run',
                'independent_native_comparison_run','completed_preparation_receipt_produced','new_model_execution_run_in_this_writer'):
            self.assertFalse(failure[name],name)
        self.assertNotIn('workspace_context',failure)
        self.assertFalse(failure['observed_attempt_source']['execution_context_receipt_produced'])
        self.assertEqual(failure['observed_attempt_source']['git_head'],'dae9b94aefaf49b369dae4ede0db8d6e062e2995')
        self.assertEqual(failure['new_objects_imported'],2942)
        self.assertEqual(failure['private_log']['sha256'],'d59adddd7f9386481457716af0343a702cf8eea6e6bee76579c42fe10706f052')
        self.assertFalse(failure['private_log']['published'])
