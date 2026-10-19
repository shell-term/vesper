import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from vesper.checkpoint import MODEL,REVISION
from vesper.engram import validate_input
from vesper.engram_assets import expected_row_address,MAX_ROWS
from vesper.engram_rows import EngramRows
from vesper.engram_session import EngramSession
from vesper.engram_validation import validate_goldens
from vesper.model_artifacts import CATALOG_SHA256,SHARD_INVENTORY_SHA256,digest,read_checked
from vesper.ngram import NgramState,HashSnapshot,TABLE_ROWS,LAYERS
from vesper.original_engram_rows import OriginalShardRows
from vesper.resources import MiB,GiB


def mapping():
    a=np.arange(129280,dtype=np.int64)%99092
    return a


class EngramTests(unittest.TestCase):
    def setUp(self):
        self.memory=patch('vesper.resources.physical_memory',return_value={'available':64*GiB})
        self.memory.start(); self.addCleanup(self.memory.stop)

    def test_hash_partition_masks_and_three_token_tail(self):
        ids=np.array([[0,2,129279,129264,671,17408],[671,344,16,2,0,129279]],np.int64)
        mask=np.array([[1,1,0,1,1,1],[1,0,1,1,0,1]],np.bool_)
        def run(counts):
            state=NgramState(mapping()); results=[]; looks=[]; pos=0
            for n in counts:
                step=state.prepare(np.ascontiguousarray(ids[:,pos:pos+n]),pos,np.ascontiguousarray(mask[:,pos:pos+n]))
                results.append(step.hash_ids); looks.append(step.lookbacks); state.commit(step); pos+=n
            return np.concatenate(results,1),np.concatenate(looks,1),state
        a,look,s=run([1]*6); b,_,t=run([3,2,1]); np.testing.assert_array_equal(a,b)
        self.assertEqual(s.state_sha256(),t.state_sha256()); self.assertEqual(s.history.shape,(2,3))
        self.assertTrue((look[0,3,1:]==2).all())  # A masked token blocks all older history.
        for i,total in enumerate(TABLE_ROWS): self.assertTrue(((a[:,:,i]>=0)&(a[:,:,i]<total)).all())

    def test_candidate_commit_snapshot_ownership_and_bounded_handles(self):
        state=NgramState(mapping()); ids=np.array([[671]],np.int64)
        step=state.prepare(ids); self.assertEqual(state.position,0); self.assertIsNone(state.history)
        state.commit(step)
        with self.assertRaises(ValueError): state.commit(step)
        handle=state.snapshot(); before=state.state_sha256(); state.commit(state.prepare(ids)); state.rollback(handle)
        self.assertEqual(state.state_sha256(),before)
        copied=state.inspect(); copied['history'].fill(0); self.assertEqual(state.state_sha256(),before)
        for token in (handle,HashSnapshot(),[],None):
            with self.assertRaises(ValueError): state.rollback(token)
        handles=[state.snapshot() for _ in range(16)]
        with self.assertRaises(MemoryError): state.snapshot()
        stale=state.prepare(ids); state.reset()
        with self.assertRaises(ValueError): state.commit(stale)
        with self.assertRaises(ValueError): state.rollback(handles[0])
        self.assertFalse(state.handles)

    def test_invalid_hash_inputs_do_not_mutate_history(self):
        state=NgramState(mapping()); ids=np.array([[671]],np.int64); state.commit(state.prepare(ids)); before=state.state_sha256()
        for x,pos,mask in ((ids.astype(np.int32),1,None),(np.array([[-1]],np.int64),1,None),
                (np.array([[129280]],np.int64),1,None),(np.zeros((1,4),np.int64),1,None),
                (ids,0,None),(ids,True,None),(ids,1,np.ones((1,1),np.uint8)),(ids,1,np.ones((2,1),np.bool_)),
                (np.zeros((2,1),np.int64),1,None)):
            with self.assertRaises(ValueError): state.prepare(x,pos,mask)
            self.assertEqual(state.state_sha256(),before)

    def selected(self,root):
        root=Path(root); raw=b'\x38'*256+b'\x7f'*8; payload=raw+bytes(512-264)+raw+bytes(4096-512-264)
        (root/'rows.bin').write_bytes(payload)
        entries=[{'row':r,'offset':i*512,'sha256':digest(raw),'weight_sha256':digest(raw[:256]),'scale_sha256':digest(raw[256:]),
            'source':expected_row_address(1,r)} for i,r in enumerate((0,TABLE_ROWS[0]-1))]
        manifest={'kind':'original_engram_selected_rows','model':MODEL,'revision':REVISION,'catalog_sha256':CATALOG_SHA256,
            'layer':1,'record_bytes':512,'maximum_rows':MAX_ROWS,'file_bytes':4096,'rows':entries}
        (root/'manifest.json').write_text(json.dumps(manifest)); return manifest

    def test_selected_rows_real_direct_io_lru_corruption_and_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.selected(tmp); rows=EngramRows(tmp,1,cache_rows=1)
            with rows:
                ids=np.full((1,1,24),0,np.int64); codes,scales=rows.get(ids)
                self.assertTrue((codes==0x38).all()); self.assertTrue((scales==127).all())
                self.assertEqual(rows.misses,1); self.assertEqual(rows.hits,23)
                ids.fill(TABLE_ROWS[0]-1); rows.get(ids); self.assertEqual(len(rows.cache),1)
                before=rows.physical_bytes; ids.fill(2)
                with self.assertRaises(ValueError): rows.get(ids)
                self.assertEqual(rows.physical_bytes,before)
                rows.clear(); original=rows.reader.read
                with patch.object(rows.reader,'read',side_effect=lambda *a: bytes(b^1 for b in original(*a))):
                    ids.fill(0)
                    with self.assertRaisesRegex(ValueError,'SHA-256'): rows.get(ids)
                self.assertFalse(rows.cache)
            self.assertEqual(rows.budget.reserved,0); self.assertIsNone(rows.reader)

    def test_wrong_row_provenance_refuses_before_opening_direct_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            m=self.selected(tmp); m['rows'][0]['source'][0]['shard_byte_offset']+=256
            (Path(tmp)/'manifest.json').write_text(json.dumps(m)); rows=EngramRows(tmp,1)
            with patch('vesper.engram_rows.DirectReader',side_effect=AssertionError('No I/O expected')):
                with self.assertRaises(ValueError): rows.__enter__()
            self.assertEqual(rows.budget.reserved,0)

    def test_row_pressure_clears_cache_then_refuses_before_io(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.selected(tmp)
            with EngramRows(tmp,1) as rows:
                ids=np.zeros((1,1,24),np.int64); rows.get(ids)
                with patch('vesper.resources.physical_memory',return_value={'available':GiB}), patch.object(rows.reader,'read',side_effect=AssertionError('No I/O under pressure')):
                    with self.assertRaises(MemoryError): rows.get(ids)
                self.assertFalse(rows.cache)

    def test_full_shard_backend_preserves_first_last_original_addresses(self):
        # Controlled reader checks address ownership, not original numerical parity.
        inventory=[{'file':f'model-{i:05}-of-00048.safetensors','bytes':100,'sha256':'a'*64} for i in range(1,49)]
        with tempfile.TemporaryDirectory() as tmp:
            m={'kind':'registered_original_artifacts','model':MODEL,'revision':REVISION,'catalog_sha256':CATALOG_SHA256,
                'shard_inventory_sha256':SHARD_INVENTORY_SHA256,'entries':inventory}
            (Path(tmp)/'manifest.json').write_text(json.dumps(m))
            requests=[]
            class Reader:
                index_initialization_bytes=4096
                data=SimpleNamespace(aligned_bytes_requested=0); index=SimpleNamespace(aligned_bytes_requested=0)
                def read(self,offset,size):
                    requests.append((offset,size)); self.data.aligned_bytes_requested+=65536
                    return (b'\x38' if size==256 else b'\x7f')*size
                def clear(self): pass
                def close(self): pass
            with patch('vesper.original_engram_rows.shard_inventory',return_value=inventory), patch('vesper.original_engram_rows.VerifiedBlob',return_value=Reader()):
                rows=OriginalShardRows('unused',tmp,tmp,14,cache_rows=1)
                with rows:
                    ids=np.zeros((1,1,24),np.int64); rows.get(ids); ids.fill(TABLE_ROWS[1]-1); rows.get(ids)
                    self.assertEqual(len(rows.cache),1)
                expected=[(e['shard_byte_offset'],e['bytes']) for r in (0,TABLE_ROWS[1]-1) for e in expected_row_address(14,r)]
                self.assertEqual(requests,expected); self.assertEqual(rows.budget.reserved,0)

    def test_session_failure_restores_history_and_parent_refuses_before_backend(self):
        state=NgramState(mapping()); ops={}; calls=[]
        for l in LAYERS:
            device=SimpleNamespace(allocations={l:157521920},cap=256*MiB)
            def allocate(n,device=device):
                calls.append(n); device.allocations[999]=n; return 999
            device.allocate=allocate
            ops[l]=SimpleNamespace(opened=True,ordinal=0,rows=SimpleNamespace(layer=l),device=device,
                forward=lambda h,*args: {'output_bf16':h.copy()})
        inputs={l:np.zeros((1,1,4,5120),np.uint16) for l in LAYERS}; ids=np.array([[671]],np.int64)
        with EngramSession(state,ops) as session:
            session.step(ids,inputs); before=state.state_sha256()
            with patch.object(ops[14],'forward',side_effect=OSError('late failure')):
                with self.assertRaises(OSError): session.step(ids,inputs)
            self.assertEqual(state.state_sha256(),before)
            with self.assertRaises(MemoryError): ops[14].device.allocate(100*MiB)
            self.assertFalse(calls); self.assertEqual(session.combined_cap_refusals,1)
            session.step(ids,inputs); self.assertEqual(state.position,2)
        self.assertEqual(session.budget.reserved,0)

    def test_pinned_metadata_and_arithmetic_inputs_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'config.json').write_bytes(bytes(3311))
            with self.assertRaisesRegex(ValueError,'SHA-256'): read_checked(tmp,'config.json')
        x=np.zeros((1,1,4,5120),np.uint16); ids=np.zeros((1,1,24),np.int64)
        validate_input(x,ids,None)
        for h,i,m in ((x.astype(np.float32),ids,None),(np.full_like(x,0x7f80),ids,None),(x,ids.astype(np.int32),None),
                (x,ids,np.zeros((1,1),np.uint8)),(x[...,::-1],ids,None)):
            with self.assertRaises(ValueError): validate_input(h,i,m)

    def test_published_original_goldens_reject_omission_substitution_and_unchecked_failures(self):
        path=Path(__file__).resolve().parents[1]/'results/2026-10-19-engram/original.json'
        report=json.loads(path.read_bytes()); self.assertEqual(len(validate_goldens(report)),110)
        changes=[lambda r:r['session_cases'].pop(),lambda r:r['session_cases'][0]['boundaries'].pop(),
            lambda r:r['plan']['hash_cases'].pop(),lambda r:r['native_profiles']['1'].update(cubin_sha256='0'*64),
            lambda r:r['faults'].pop(),lambda r:r['faults'][8].update(combined_parent_refused_before_backend=False),
            lambda r:r['negative_controls'].pop(),lambda r:r['negative_controls'][0].update(status='pass'),
            lambda r:r['snapshots'][0].update(history_restored=False),lambda r:r['unchanged_original_controls'].pop(),
            lambda r:r['memory']['cleanup']['operators']['1'].update(device_owned_bytes=1),
            lambda r:r['memory']['active']['VMM'].update(original_row_addresses_remapped=True),
            lambda r:r['memory'].update(minimum_sampled_available_bytes=GiB),lambda r:r.update(full_model_quality='pass')]
        for change in changes:
            candidate=copy.deepcopy(report); change(candidate)
            with self.assertRaises((ValueError,KeyError,IndexError)): validate_goldens(candidate)


if __name__=='__main__': unittest.main()
