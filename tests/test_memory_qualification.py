import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from vesper.memory_qualification import ROOT,CRITERIA,qualify,validate_manifest,validate_governor,validate_geometry,validate_native,validate_reference,validate_chain_replay
from vesper.model_artifacts import canonical
from vesper.resources import GiB


class MemoryQualificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(__file__).resolve().parents[1]; cls.folder=cls.root/ROOT
        def read(name): return json.loads((cls.folder/name).read_bytes())
        cls.scale=read('scale.json'); cls.chain=read('chain.json'); cls.chain_memory=read('chain-memory.json')
        cls.profile=read('reference-memory.json'); cls.reference_raw=(cls.folder/'reference-original.json').read_bytes()
        cls.reference=json.loads(cls.reference_raw); cls.manifest=read('manifest.json')
        cls.engram_raw=(cls.root/'results/2026-10-19-engram/original.json').read_bytes(); cls.engram=json.loads(cls.engram_raw)
        cls.chain_raw=(cls.root/'results/2026-10-19-refresh-block-chain/chain.json').read_bytes(); cls.original_chain=json.loads(cls.chain_raw)

    def test_all_six_original_issue9_checks_with_complete_evidence(self):
        with patch('vesper.resources.physical_memory',return_value={'available':64*GiB}):
            r=qualify(self.root,self.folder/'manifest.json')
        self.assertEqual([(c['id'],c['criterion']) for c in r['criteria']],list(CRITERIA))
        self.assertEqual(len(r['criteria']),6); self.assertEqual(r['original_expert_identities_visited'],15360)
        self.assertEqual(r['governed_Engram_boundaries'],7192); self.assertEqual(r['governed_chain_boundaries'],59600)
        self.assertFalse(r['all_forty_layers_numerically_executed']); self.assertFalse(r['milestone2_complete'])
        self.assertEqual(r['remaining_integration_issues'],[10])

    def test_manifest_cannot_drop_or_rewrite_original_requirements_or_evidence(self):
        for mutate in (lambda m:m['criteria'].pop(),lambda m:m['evidence'].pop(),
                lambda m:m['criteria'][0].update(criterion='small cache only'),lambda m:m['evidence'][0].update(sha256='0'*64)):
            m=copy.deepcopy(self.manifest); mutate(m)
            with self.assertRaises(ValueError): validate_manifest(m)

    def test_rehashed_geometry_cannot_weaken_expert_or_retained_state_scope(self):
        for mutate in (lambda p:p['expert_blocks'].pop(),lambda p:p['expert_blocks'][0].update(bytes=1),
                lambda p:p['KV_allocations'].pop(),lambda p:p['KV_allocations'][0].update(bytes=1),lambda p:p.update(context=8)):
            p=copy.deepcopy(self.scale['plan']); mutate(p); p['plan_sha256']=canonical({k:v for k,v in p.items() if k!='plan_sha256'})
            with self.assertRaises(ValueError): validate_geometry(p)

    def test_cleanup_cap_headroom_and_process_scope_are_independent_checks(self):
        for mutate in (lambda g:g.update(charged_host_bytes=1),lambda g:g.update(live_reclaimers=1),
                lambda g:g.update(available_floor_bytes=31*GiB),lambda g:g.update(host_cap_bytes=97*GiB),
                lambda g:g.update(reservation_is_RSS_or_OS_limit=True),lambda g:g.update(monitor_thread_stopped=False),
                lambda g:g['device_used_bytes'].update({'0':1}),lambda g:g.update(sampled_process_peaks={})):
            g=copy.deepcopy(self.scale['cleanup_governor']); mutate(g)
            with self.assertRaises((ValueError,KeyError)): validate_governor(g,8*GiB)

    def test_native_replay_requires_every_boundary_history_and_cancel_gate(self):
        for mutate in (lambda n:n.update(torch_installed=True),lambda n:n['session_cases'].pop(),
                lambda n:n['session_cases'][0]['boundaries'][0].update(actual_sha256='0'*64),
                lambda n:n['session_cases'][0].update(committed_state_sha256='0'*64),
                lambda n:n['extra_faults'][0].update(resume_while_inflight_refused=False),lambda n:n['extra_steps'].pop()):
            n=copy.deepcopy(self.scale['native_comparisons']); mutate(n)
            with self.assertRaises(ValueError): validate_native(n,self.engram,self.engram_raw)

    def test_reference_resident_commit_and_unavailable_metrics_cannot_be_conflated(self):
        for mutate in (lambda p:p.update(native_owned_cap_includes_reference_frameworks=True),
                lambda p:p.update(phase_deltas_are_exact_component_attribution=True),
                lambda p:p['sampled_process_peaks'].update(private_resident_bytes=0),
                lambda p:p['VMM_after_close'].update(physical_owned_bytes=1)):
            p=copy.deepcopy(self.profile); mutate(p)
            with self.assertRaises(ValueError): validate_reference(p,self.reference,self.reference_raw,self.engram)

    def test_chain_fingerprint_and_helper_cleanup_cannot_be_omitted(self):
        n=copy.deepcopy(self.chain); n['chain_cases'][0]['actual_layout_sha256']='0'*64
        with self.assertRaises(ValueError): validate_chain_replay(n,self.original_chain,self.chain_raw,self.chain_memory)
        g=copy.deepcopy(self.chain_memory); g['governor']['device_used_bytes']['1']=1
        with self.assertRaises(ValueError): validate_chain_replay(self.chain,self.original_chain,self.chain_raw,g)


if __name__=='__main__': unittest.main()
