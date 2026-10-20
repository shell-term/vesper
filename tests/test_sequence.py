from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from vesper.engram_assets import expected_row_address
from vesper.engram_rows import EngramRows
from vesper.generation import GenerationRequest, TextGeneration
from vesper.generation_assets import PlanningRows
from vesper.model_artifacts import digest
from vesper.resources import GiB
from vesper.sequence import SequenceRequest, MAX_CALLS, MAX_ROWS


class Tokenizer:
    def __init__(self, ids): self.ids=ids
    def encode(self, text): return list(self.ids)
    def decode(self, ids): return " ".join(map(str,ids))


class SequenceTests(unittest.TestCase):
    def setUp(self):
        memory=patch("vesper.resources.physical_memory",return_value={"available":64*GiB})
        memory.start();self.addCleanup(memory.stop)

    def test_all_three_whole_quality_prompts_fit_explicit_profile_only(self):
        root=Path(__file__).resolve().parents[1]
        evidence=json.loads((root/'results/2026-10-20-message-format/original.json').read_bytes())
        for r in evidence['quality_preflight']:
            t=Tokenizer(r['prompt_token_ids']);request=SequenceRequest(r['prompt'],r['requested_max_new_tokens'])
            self.assertEqual(list(request.encode(t)),r['prompt_token_ids'])
            with self.assertRaises(ValueError):GenerationRequest(r['prompt'],r['requested_max_new_tokens']).encode(t)

    def test_exact_512_call_boundary_prefix_and_eos(self):
        t=Tokenizer(range(512));seen=[]
        def step(ids,*,sample):
            seen.append((int(ids[0,0]),sample))
            return {'sampling':{'ids_i64':np.array([1],np.int64)}} if sample else {'sampling':{}}
        s=TextGeneration(t,step,SequenceRequest('whole prefix',1))
        while not s.done:s.advance()
        self.assertEqual(s.position,512);self.assertEqual(s.stop_reason,'eos')
        self.assertEqual(seen,[(i,i==511) for i in range(512)])
        self.assertEqual(s.summary()['completion_ids'],[])
        with self.assertRaises(ValueError):SequenceRequest('too long',2).encode(t)
        for count in (0,513,True):
            with self.assertRaises(ValueError):SequenceRequest('x',count).encode(Tokenizer([0]))

    def test_failed_middle_prefix_keeps_driver_position_for_retry(self):
        t=Tokenizer(range(64));seen=[];fail=True
        def step(ids,*,sample):
            nonlocal fail
            if int(ids[0,0])==32 and fail:fail=False;raise OSError('owned input fault')
            seen.append(int(ids[0,0]));return {'sampling':{'ids_i64':np.array([1],np.int64)}} if sample else {'sampling':{}}
        s=TextGeneration(t,step,SequenceRequest('prefix',1))
        for _ in range(32):s.advance()
        with self.assertRaises(OSError):s.advance()
        self.assertEqual((s.position,s.generated),(32,()))
        while not s.done:s.advance()
        self.assertEqual(seen,list(range(64)))

    def test_1025_aligned_rows_need_explicit_profile_and_keep_lru_bound(self):
        # Synthetic original-format row bytes prove capacity and I/O guards;
        # the separate GPU corpus uses actual released row payloads.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=PlanningRows(root,None,1,max_rows=MAX_ROWS)
            raw=b'\x38'*256+b'\x7f'*8
            for row in range(1025):p.entries[row]=(raw,{'source':expected_row_address(1,row),'sha256':digest(raw),
                'weight_sha256':digest(raw[:256]),'scale_sha256':digest(raw[256:])})
            m=p.freeze();directory=root/'rows/layer1'
            self.assertEqual((m['maximum_rows'],len(m['rows'])),(12288,1025))
            default=EngramRows(directory,1)
            with self.assertRaises(ValueError):
                with default:pass
            self.assertEqual(default.budget.reserved,0)
            with EngramRows(directory,1,max_rows=MAX_ROWS,cache_rows=8) as rows:
                for base in (0,1001):
                    ids=np.arange(base,base+24,dtype=np.int64).reshape(1,1,24)
                    codes,scales=rows.get(ids)
                    self.assertTrue((codes==0x38).all());self.assertTrue((scales==127).all())
                    self.assertLessEqual(len(rows.cache),8)
                before=rows.physical_bytes
                with self.assertRaises(ValueError):rows.get(np.full((1,1,24),1025,np.int64))
                self.assertEqual(rows.physical_bytes,before)
            self.assertEqual(rows.budget.reserved,0)

    def test_metadata_growth_bound_precedes_row_io(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'manifest.json'
            with path.open('wb') as f:f.truncate(12*1024**2+1)
            rows=EngramRows(root,1,max_rows=MAX_ROWS)
            with patch.object(Path,'open',side_effect=AssertionError('Reject oversized metadata before reading')):
                with self.assertRaises(ValueError):
                    with rows:pass
            self.assertEqual(rows.budget.reserved,0)
