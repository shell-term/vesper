from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from vesper.bandwidth import EXPERT_BYTES
from vesper.expert_bank import ExpertBank
from vesper.resources import GiB
from vesper.storage import tensor_layout
import test_indexed_store


class ExpertBankTests(unittest.TestCase):
    def setUp(self):
        p=patch('vesper.resources.physical_memory',return_value={'available':64*GiB});p.start();self.addCleanup(p.stop)
        p=patch('vesper.expert_bank.reserve_disk');p.start();self.addCleanup(p.stop)

    def sources(self,root,count=1):return test_indexed_store.IndexedStoreTests().sources(root,count)

    def importer(self,template):
        def fetch(directory,layer,expert,*,reader):
            m=deepcopy(template['entries'][0]['provenance']);m['expert']=expert;m['weights_fetched_bytes']=EXPERT_BYTES
            for t in m['tensors']:t['tensor']=t['tensor'].replace('experts.0.',f'experts.{expert}.')
            directory.mkdir();np.savez(directory/'native-expert.npz',**{n:np.zeros(v['shape'],np.uint8) for n,v in tensor_layout().items()})
            (directory/'manifest.json').write_text(json.dumps(m),encoding='utf-8');return m
        return fetch

    def test_reuse_index_and_offline_exact_array_reads_without_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);sources=self.sources(root)
            with ExpertBank(root,root/'bank',sources) as bank,patch('vesper.expert_bank.fetch_expert_fixture',side_effect=AssertionError('No fetch')):
                bank.prepare(0,[0],None,None);bank.index(0,[0],root/'index')
                arrays,m=bank.arrays(0,0);self.assertEqual(m['expert'],0);self.assertTrue(all(not a.any() for a in arrays.values()))
                self.assertEqual(bank.new_bytes,0)
            self.assertEqual(bank.stats()['host_reserved_bytes'],0)

    def test_new_expert_is_packed_once_and_reopens_without_NPZ_or_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);sources=self.sources(root);template=json.loads((sources[0]/'manifest.json').read_bytes())
            with patch('vesper.expert_bank.prefetch_expert',side_effect=lambda *a:nullcontext(None)),patch('vesper.expert_bank.fetch_expert_fixture',side_effect=self.importer(template)) as fetch:
                with ExpertBank(root,root/'bank',sources) as bank:
                    bank.prepare(0,[0,1],None,None);bank.prepare(0,[1],None,None)
                    self.assertEqual(fetch.call_count,1);self.assertEqual(bank.new_bytes,EXPERT_BYTES)
                    self.assertEqual(bank.arrays(0,1)[1]['expert'],1)
            self.assertFalse(list((root/'bank').rglob('*.npz')))
            with ExpertBank(root,root/'bank',sources) as bank:self.assertTrue(bank.has(0,1));self.assertEqual(bank.new_bytes,EXPERT_BYTES)

    def test_concurrent_import_cap_refuses_without_exceeding_persistent_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);sources=self.sources(root);template=json.loads((sources[0]/'manifest.json').read_bytes())
            with patch('vesper.expert_bank.prefetch_expert',side_effect=lambda *a:nullcontext(None)),patch('vesper.expert_bank.fetch_expert_fixture',side_effect=self.importer(template)):
                with ExpertBank(root,root/'bank',sources,new_payload_cap=EXPERT_BYTES) as bank:
                    with self.assertRaises(MemoryError):bank.prepare(0,[1,2],None,None)
                    self.assertEqual(bank.pending_bytes,0);self.assertLessEqual(bank.new_bytes,EXPERT_BYTES)

    def test_failed_import_cleans_owned_staging_and_publishes_no_expert(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);sources=self.sources(root)
            with ExpertBank(root,root/'bank',sources) as bank,patch('vesper.expert_bank.prefetch_expert',side_effect=lambda *a:nullcontext(None)),patch('vesper.expert_bank.fetch_expert_fixture',side_effect=OSError('synthetic broken transfer')):
                with self.assertRaises(OSError):bank.prepare(0,[1],None,None)
                self.assertFalse(bank.has(0,1));self.assertEqual(bank.pending_bytes,0);self.assertFalse(list((root/'bank/staging').iterdir()))

    def test_source_corruption_and_path_escape_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);sources=self.sources(root)
            with ExpertBank(root,root/'bank',sources) as bank:
                path=sources[0]/'manifest.json';path.write_bytes(path.read_bytes()+b' ')
                with self.assertRaisesRegex(ValueError,'changed'):bank.arrays(0,0)
            with self.assertRaises(ValueError):ExpertBank(root,root.parent/'escape')
