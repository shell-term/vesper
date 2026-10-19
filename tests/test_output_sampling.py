import copy
import json
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch

import numpy as np

from vesper.memory import RuntimeMemory,MemoryPolicy,RuntimeCancelled
from vesper.model_artifacts import canonical,digest,CATALOG_SHA256
from vesper.output import greedy,validate_hidden,NativeOutput
from vesper.output_assets import QUALIFIED_IDENTITY,TENSOR_SHA256,prepare,exclusive_import
from vesper.output_validation import validate_output,validate_sampling
from vesper.resources import GiB
from vesper.sampling import NativeSampler,SamplerSnapshot,SAMPLING_CUBIN_SHA256


class OutputSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder=Path(__file__).resolve().parents[1]/'results/2026-10-19-output-sampling'
        cls.raw={n:(cls.folder/n).read_bytes() for n in ('output-original.json','sampling-original.json')}
        cls.output=json.loads(cls.raw['output-original.json']); cls.sampling=json.loads(cls.raw['sampling-original.json'])
        cls.native_output=json.loads((cls.folder/'output-native.json').read_bytes()); cls.native_sampling=json.loads((cls.folder/'sampling-native.json').read_bytes())

    def test_complete_original_and_independent_native_component_gates(self):
        self.assertEqual(len(validate_output(self.output)),24); self.assertEqual(len(validate_sampling(self.sampling)),90)
        self.assertEqual(canonical(self.output['fixture']),QUALIFIED_IDENTITY)
        for original,native,name in ((self.output,self.native_output,'output-original.json'),(self.sampling,self.native_sampling,'sampling-original.json')):
            self.assertEqual(native['reference_report_sha256'],digest(self.raw[name])); self.assertEqual(native['status'],'pass')
            self.assertEqual(len(native['cases']),len(original['cases']))
            for n in ('torch','tilelang','transformers','sympy'):
                self.assertFalse(native[n+'_installed']); self.assertFalse(native[n+'_imported'])
            for a,b in zip(native['cases'],original['cases']):
                self.assertEqual(a['meta'],b['meta']); self.assertEqual(len(a['checks']),len(b['checks']))
                for x,y in zip(a['checks'],b['checks']):
                    for key in ('check','dtype','shape','elements','status','expected_sha256','actual_sha256'): self.assertEqual(x[key],y[key])
            g=native['governor_after_close']; self.assertEqual(g['charged_host_bytes'],0); self.assertFalse(any(g['device_used_bytes'].values()))
        for a,b in zip(self.native_output['cases'],self.output['cases']): self.assertEqual(a['sampling'],[{**v,'checks':[{k:x[k] for k in ('check','dtype','shape','elements','expected_sha256','actual_sha256','status')} for x in v['checks']]} for v in b['sampling']])
        self.assertEqual(self.native_sampling['cubin_sha256'],SAMPLING_CUBIN_SHA256)
        self.assertEqual(self.native_sampling['faults'],self.sampling['faults']); self.assertEqual(self.native_sampling['snapshots'],self.sampling['snapshots'])

    def test_output_gate_rejects_lost_vocabulary_boundaries_or_model_claims(self):
        for mutate in (lambda r:r['cases'].pop(),lambda r:r['cases'][0]['checks'].pop(),lambda r:r['cases'][0]['sampling'].pop(),
                lambda r:r.update(full_vocabulary=32000),lambda r:r.update(head_compute_dtype='bfloat16'),
                lambda r:r['cases'][0]['checks'][-2].update(actual_sha256='0'*64),lambda r:r.update(text_generation_available=True),
                lambda r:r['negative_controls'].clear()):
            changed=copy.deepcopy(self.output); mutate(changed)
            with self.assertRaises(ValueError): validate_output(changed)

    def test_sampler_gate_rejects_changed_state_noise_and_incomplete_recovery(self):
        for mutate in (lambda r:r['cases'].pop(),lambda r:r['cases'][1].update(actual_offset_after=8),
                lambda r:r['cases'][1]['checks'][3].update(actual_sha256='0'*64),lambda r:r['faults'].pop(),
                lambda r:r['snapshots'].update(original_second_draw_reproduced=False),lambda r:r['negative_controls'].clear(),
                lambda r:r['governor_after_close'].update(charged_host_bytes=1)):
            changed=copy.deepcopy(self.sampling); mutate(changed)
            with self.assertRaises(ValueError): validate_sampling(changed)

    def test_greedy_full_vocabulary_ties_use_the_first_original_index(self):
        values=np.full((2,129280),-5.,np.float32); values[0,123456]=values[0,129279]=1.; values[1,0]=-0.; values[1,129279]=0.
        np.testing.assert_array_equal(greedy(values),[123456,0])
        for bad in (values[:,:-1].copy(),values.astype(np.float64),np.full_like(values,np.nan),np.full_like(values,np.inf)):
            with self.assertRaises(ValueError): greedy(bad)

    def test_bad_head_inputs_and_weights_are_refused_before_cuda(self):
        validate_hidden(np.zeros((1,5120),np.uint16))
        for bad in (np.zeros((1,5120),np.float32),np.zeros((3,5120),np.uint16),np.full((1,5120),0x7f80,np.uint16)):
            with self.assertRaises(ValueError): validate_hidden(bad)
        with self.assertRaises(ValueError): NativeOutput(np.zeros((128,5120),np.uint16))

    def test_bad_sampling_parameters_do_not_consume_rng_state(self):
        s=NativeSampler(); values=np.zeros((1,129280),np.float32)
        for value in (-1.,float('nan'),float('inf'),True,1001.):
            with self.assertRaises(ValueError): s.sample(values,value)
            self.assertEqual((s.seed,s.offset,s.calls),(0,0,0))
        for seed in (-1,2**64,True,0.):
            with self.assertRaises(ValueError): NativeSampler(seed)

    def test_unregistered_partial_import_is_preserved_before_any_fetch(self):
        with tempfile.TemporaryDirectory() as directory:
            part=Path(directory)/'head-weight.part'; part.write_bytes(b'keep existing data')
            tensor={'dtype':'BF16','shape':[129280,5120],'bytes':1323827200}
            with patch('vesper.output_assets.load_catalog',return_value=({'head.weight':tensor},{})), \
                    patch('vesper.output_assets.shutil.disk_usage',return_value=type('Capacity',(),{'free':100*GiB})()), \
                    patch('vesper.resources.physical_memory',return_value={'available':128*GiB}), \
                    patch('vesper.output_assets.read_range') as fetch:
                with self.assertRaises(FileExistsError): prepare('unused',directory)
                fetch.assert_not_called()
            self.assertEqual(part.read_bytes(),b'keep existing data')

    def test_concurrent_importers_cannot_mutate_the_same_partial(self):
        @exclusive_import
        def second(source,directory): raise AssertionError('Second importer should not enter')
        @exclusive_import
        def first(source,directory):
            with self.assertRaises(OSError): second(source,directory)
        with tempfile.TemporaryDirectory() as directory: first('unused',directory)

    def test_rng_snapshot_identity_consumption_and_seed_reset_are_bounded(self):
        s=NativeSampler(2026); s.opened=True; s.budget.available_floor=0
        h=s.snapshot(); s.offset=12
        for foreign in (None,copy.copy(h),SamplerSnapshot(2026,0)):
            with self.assertRaises(ValueError): s.rollback(foreign)
        with self.assertRaises(RuntimeError): s.snapshot()
        with self.assertRaises(RuntimeError): s.manual_seed(1)
        s.rollback(h); self.assertEqual((s.seed,s.offset),(2026,0))
        with self.assertRaises(ValueError): s.rollback(h)
        s.manual_seed(1); self.assertEqual((s.seed,s.offset),(1,0)); s.__exit__()

    def test_cancelled_rng_restoration_preserves_the_shared_governor_contract(self):
        with RuntimeMemory(MemoryPolicy(host_cap_bytes=2*GiB),available=lambda:128*GiB) as policy:
            s=NativeSampler(7); s.opened=True; s.budget.available_floor=0
            h=s.snapshot(); s.offset=4; policy.cancel()
            with self.assertRaises(RuntimeError): s.manual_seed(9) # A pending snapshot also forbids seed reset.
            s.rollback(h); self.assertEqual((s.seed,s.offset),(7,0))
            with self.assertRaises(RuntimeCancelled): s.manual_seed(9)
            with self.assertRaises(RuntimeCancelled): s.snapshot()
            policy.resume(); s.manual_seed(9); s.__exit__()

    def test_complete_evidence_and_streamed_reference_plan_are_scoped(self):
        summary=json.loads((self.folder/'summary.json').read_bytes())
        self.assertFalse(summary['issue10_complete']); self.assertFalse(summary['full_model_execution_run'])
        for entry in summary['evidence']:
            raw=(self.folder/entry['file']).read_bytes(); self.assertEqual(len(raw),entry['bytes']); self.assertEqual(digest(raw),entry['sha256'])
        p=json.loads((self.folder/'reference-plan.json').read_bytes()); self.assertEqual(p['plan_sha256'],canonical({k:v for k,v in p.items() if k!='plan_sha256'}))
        self.assertEqual(p['catalog_sha256'],CATALOG_SHA256); self.assertEqual([r['layer'] for r in p['layers']],list(range(40)))
        self.assertEqual(p['backbone_control_bytes'],7199083968); self.assertEqual(p['head_FP32_GPU_bytes'],2647654400)
        self.assertFalse(p['whole_checkpoint_required_in_RAM']); self.assertFalse(p['paid_remote_compute_started'])
        self.assertEqual(p['state_transaction'],['window0','window1','all_compressed_KV_and_index_source_state','Engram_token_history','sampler_Philox_seed_offset','model_position'])
        self.assertEqual(len(summary['original_issue10_acceptance_completed']),1); self.assertEqual(len(summary['remaining_original_issue10_acceptance']),4)


if __name__=='__main__': unittest.main()
