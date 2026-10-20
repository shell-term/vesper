from copy import deepcopy
import json
from pathlib import Path
import unittest

from vesper.indexed_validation import validate_indexed
from vesper.model_artifacts import digest,canonical
from vesper.sequence_validation import read_report,validate_reference,validate_native

ROOT=Path(__file__).resolve().parents[1]


class SequenceEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ext=ROOT/'results/2026-10-20-extended-input';cls.idx=ROOT/'results/2026-10-20-indexed-generation'
        cls.original,cls.raw=read_report(cls.ext/'original.json');cls.native,_=read_report(cls.ext/'native.json')
        cls.generation_raw=(ROOT/'results/2026-10-19-generation/original.json').read_bytes();cls.generation=json.loads(cls.generation_raw)
        cls.indexed,_=read_report(cls.idx/'native.json');cls.indices={str(l):(cls.idx/f'indices/layer{l}.json').read_bytes() for l in range(40)}

    def test_complete_original_and_native_longer_component_evidence(self):
        self.assertTrue(validate_reference(self.original));self.assertTrue(validate_native(self.native,self.original,self.raw))
        self.assertEqual(sum(len(c['boundaries']) for c in self.native['cases']),3906)
        self.assertEqual(self.original['original_input_embedding']['elements'],322560)
        self.assertFalse(self.native['scope']['whole_backbone_63_call_comparison_run'])

    def test_actual_indexed_all_layer_generation_recheck(self):
        self.assertTrue(validate_indexed(self.indexed,self.generation,self.generation_raw,self.indices))
        self.assertEqual(sum(s['direct_read_calls'] for s in self.indexed['indexed_sources'].values()),1200)
        self.assertEqual(sum(s['payload_file_opens_and_switches'] for s in self.indexed['indexed_sources'].values()),321)

    def test_all_published_files_remain_bound_to_actual_capture_bytes(self):
        for root in (self.ext,self.idx):
            summary=json.loads((root/'summary.json').read_bytes())
            for name,m in summary['files'].items():
                raw=(root/name).read_bytes();self.assertEqual((len(raw),digest(raw)),(m['bytes'],m['sha256']))

    def test_rehashed_original_is_not_automatically_qualified(self):
        changed=deepcopy(self.original);changed['cases'][-1]['boundaries'][0]['sha256']='0'*64
        changed['complete_goldens_sha256']=canonical(changed['cases'])
        with self.assertRaises(ValueError):validate_reference(changed)
        changed=deepcopy(self.original);changed['original_row_transport_after_close']['physical_owned_bytes']=2097152
        with self.assertRaises(ValueError):validate_reference(changed,require_qualified=False)

    def test_native_missing_math_history_retry_framework_or_cleanup_fails(self):
        for mutate in (lambda r:r['cases'][-1]['boundaries'].pop(),lambda r:r['cases'][32].update(state_sha256='0'*64),
                lambda r:r.update(fault=None),lambda r:r['reference_frameworks']['torch'].update(imported=True),
                lambda r:r.update(HTTP_calls_during_execution=1),lambda r:r['row_caches_after_close']['14'].update(cached_rows=1),
                lambda r:r['memory_after_policy_close'].update(charged_host_bytes=1),
                lambda r:r['scope'].update(whole_backbone_63_call_comparison_run=True)):
            r=deepcopy(self.native);mutate(r)
            with self.assertRaises(ValueError):validate_native(r,self.original,self.raw)

    def test_indexed_missing_source_route_fetch_or_changed_index_fails(self):
        for mutate in (lambda r:r['indexed_sources'].pop('39'),lambda r:r['indexed_sources']['0'].update(direct_read_calls=29),
                lambda r:r['indexed_sources']['14'].update(expert_payloads_fetched_in_forward=1),
                lambda r:r['indexed_sources']['20'].update(original_experts=[])):
            r=deepcopy(self.indexed);mutate(r)
            with self.assertRaises(ValueError):validate_indexed(r,self.generation,self.generation_raw,self.indices)
        indices=dict(self.indices);indices['14']+=b' '
        with self.assertRaises(ValueError):validate_indexed(self.indexed,self.generation,self.generation_raw,indices)
