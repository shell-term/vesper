import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from vesper.engram_qualification import ROOT,CRITERIA,qualify,validate_manifest,validate_native,validate_storage
from vesper.resources import GiB


class EngramQualificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(__file__).resolve().parents[1]; cls.folder=cls.root/ROOT
        cls.raw=(cls.folder/'original.json').read_bytes(); cls.original=json.loads(cls.raw)
        cls.native=json.loads((cls.folder/'native.json').read_bytes()); cls.io=json.loads((cls.folder/'direct-io.json').read_bytes())
        cls.plans=json.loads((cls.folder/'import-plan.json').read_bytes()); cls.manifest=json.loads((cls.folder/'manifest.json').read_bytes())

    def test_complete_original_issue8_acceptance(self):
        with patch('vesper.resources.physical_memory',return_value={'available':64*GiB}):
            report=qualify(self.root,self.folder/'manifest.json')
        self.assertEqual(report['status'],'pass'); self.assertEqual(len(report['criteria']),5)
        self.assertEqual([(r['id'],r['criterion']) for r in report['criteria']],list(CRITERIA))
        self.assertFalse(report['milestone2_complete']); self.assertEqual(report['remaining_integration_issues'],[9,10])
        self.assertEqual(report['bitwise_boundaries'],6820); self.assertEqual(report['full_shards_registered'],0)

    def test_manifest_rejects_omitted_evidence_criteria_and_notices(self):
        for mutate in (lambda m:m['criteria'].pop(),lambda m:m['evidence'].pop(),lambda m:m['notices'].pop(),
                lambda m:m['criteria'][0].update(criterion='a weaker check'),lambda m:m['evidence'][0].update(sha256='0'*64)):
            changed=copy.deepcopy(self.manifest); mutate(changed)
            with self.assertRaises(ValueError): validate_manifest(changed)

    def test_native_replay_refuses_changed_boundaries_frameworks_or_history(self):
        validate_native(self.native,self.original,self.raw)
        for mutate in (lambda n:n.update(torch_installed=True),lambda n:n.update(torch_imported=True),
                lambda n:n['session_cases'].pop(),lambda n:n['session_cases'][0]['boundaries'].pop(),
                lambda n:n['session_cases'][0]['boundaries'][0].update(actual_sha256='0'*64),
                lambda n:n['session_cases'][0].update(committed_state_sha256='0'*64),
                lambda n:n['memory']['cleanup']['rows']['1'].update(host_reserved_bytes=1)):
            changed=copy.deepcopy(self.native); mutate(changed)
            with self.assertRaises(ValueError): validate_native(changed,self.original,self.raw)

    def test_capacity_and_bounded_io_do_not_accept_weaker_scope(self):
        validate_storage(self.io,self.plans)
        for mutate in (lambda r:r.update(full_shards_registered=48),lambda r:r.update(full_checkpoint_downloaded=True),
                lambda r:r['small_original_tensors'][0].update(file_mapping=True),lambda r:r['small_original_tensors'][0]['ranges'].pop()):
            changed=copy.deepcopy(self.io); mutate(changed)
            with self.assertRaises(ValueError): validate_storage(changed,self.plans)
        for mutate in (lambda r:r['plans'].pop(),lambda r:r['plans'][0].update(required_free_bytes=1),
                lambda r:r['plans'][1].update(capacity_sufficient=False),lambda r:r['plans'][0]['catalog']['shards'][0].update(original_file_sha256='0'*64),
                lambda r:r['plans'][0].update(disk_reserve_bytes=0),lambda r:r['plans'][0]['tensor_bytes_by_category'].update(vision=0)):
            changed=copy.deepcopy(self.plans); mutate(changed)
            with self.assertRaises(ValueError): validate_storage(self.io,changed)


if __name__=='__main__': unittest.main()
