from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from vesper.bank_migration import make_plan,execute,_publish
from vesper.bandwidth import EXPERT_BYTES
from vesper.indexed_store import build_index,IndexedStore
from vesper.model_artifacts import canonical
from vesper.resources import GiB,MiB,MemoryBudget
from vesper.source_roots import BANKS,CONFIG,SourceRoots,source_scope
import test_indexed_store


class BankMigrationTests(unittest.TestCase):
    def setUp(self):
        p=patch('vesper.resources.physical_memory',return_value={'available':64*GiB});p.start();self.addCleanup(p.stop)
        p=patch('vesper.bank_migration.shutil.disk_usage',return_value=SimpleNamespace(free=512*GiB));p.start();self.addCleanup(p.stop)

    def setup(self,base,count=1):
        root=base/'artifacts';root.mkdir();bank=root/BANKS[0];bank.mkdir(parents=True)
        sources=test_indexed_store.IndexedStoreTests().sources(bank,count);(bank/'layer0').mkdir()
        paths=[]
        for i,source in enumerate(sources):
            target=bank/f'layer0/expert-{i:03d}';source.rename(target);paths.append(target)
        (bank/'synthetic.bin').unlink()
        build_index(paths,root/'index',root,layer=0)
        return root,base/'ssd',paths

    def test_atomic_publication_never_replaces_an_existing_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);pending=base/'pending';target=base/'final'
            pending.write_bytes(b'new');target.write_bytes(b'preserve')
            with self.assertRaises(OSError):_publish(pending,target)
            self.assertEqual(target.read_bytes(),b'preserve');self.assertEqual(pending.read_bytes(),b'new')

    def read_index(self,root,count):
        budget=MemoryBudget(cap=96*MiB,available_floor=0);target=np.empty(EXPERT_BYTES,np.uint8)
        with IndexedStore(root/'index',root,budget,layer=0) as source:
            for i in range(count):source.read_into(f'0:{i}',target);self.assertFalse(target.any())
        self.assertEqual(budget.reserved,0)

    def test_copy_verify_activate_retire_preserves_old_index_and_all_exact_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,target,sources=self.setup(Path(tmp));index=(root/'index/manifest.json').read_bytes()
            plan=make_plan(root,target)
            self.assertFalse(target.exists());self.assertTrue(sources[0].exists())
            result=execute(root,plan)
            self.assertEqual(result['original_packed_payload_bytes'],EXPERT_BYTES)
            self.assertEqual(result['all_files_verified_bytes'],plan['total_file_bytes'])
            self.assertEqual(result['retired_source_file_bytes_this_process'],plan['total_file_bytes'])
            self.assertEqual(result['weight_bytes_converted'],0);self.assertFalse(result['model_execution_run'])
            self.assertFalse(sources[0].exists());self.assertEqual((root/'index/manifest.json').read_bytes(),index)
            self.assertEqual(SourceRoots(root).active_bank,BANKS[1]);self.read_index(root,1)
            self.assertNotIn(str(target),json.dumps(result))

    def test_interruption_after_one_retirement_resumes_without_recopy_or_lost_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,target,sources=self.setup(Path(tmp),2);plan=make_plan(root,target)
            def interrupt(value):
                if value['stage']=='retired' and value['objects']==1:raise RuntimeError('synthetic operator interruption')
            with self.assertRaisesRegex(RuntimeError,'operator interruption'):execute(root,plan,interrupt)
            self.assertFalse(sources[0].exists());self.assertTrue(sources[1].exists());self.assertTrue((root/CONFIG).exists())
            self.read_index(root,2)
            result=execute(root,plan);self.assertEqual(result['copied_file_bytes_this_process'],0)
            self.assertFalse(sources[1].exists());self.read_index(root,2)

    def test_corrupted_copy_is_rejected_before_activation_or_source_retirement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,target,sources=self.setup(Path(tmp));plan=make_plan(root,target)
            def corrupt(value):
                if value['stage']=='copied':
                    file=target/'expert-bank-v1/layer0/expert-000/experts.bin'
                    with file.open('r+b') as stream:stream.write(b'\x01')
            with self.assertRaisesRegex(ValueError,'bytes differ'):execute(root,plan,corrupt)
            self.assertFalse((root/CONFIG).exists());self.assertTrue((sources[0]/'experts.bin').exists())
            with self.assertRaisesRegex(ValueError,'bytes differ'):execute(root,plan)

    def test_partial_copy_retries_only_after_the_complete_original_is_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,target,sources=self.setup(Path(tmp));plan=make_plan(root,target)
            payload=sources[0]/'experts.bin'
            with payload.open('r+b') as stream:stream.write(b'\x01')
            with self.assertRaisesRegex(ValueError,'Source payload changed'):execute(root,plan)
            self.assertFalse((root/CONFIG).exists());self.assertTrue(payload.exists())
            with payload.open('r+b') as stream:stream.write(b'\x00')
            execute(root,plan);self.read_index(root,1)

    def test_capacity_invalid_plan_and_unowned_destination_refuse_without_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,target,sources=self.setup(Path(tmp));plan=make_plan(root,target)
            with patch('vesper.bank_migration.shutil.disk_usage',return_value=SimpleNamespace(free=128*GiB)):
                with self.assertRaisesRegex(OSError,'reserve'):execute(root,plan)
            self.assertFalse(target.exists());self.assertTrue(sources[0].exists())
            changed=deepcopy(plan);changed['entries'][0]['directory']='../outside';changed['plan_sha256']=canonical({k:v for k,v in changed.items() if k!='plan_sha256'})
            with self.assertRaises(ValueError):execute(root,changed)
            self.assertFalse(target.exists())
            target.mkdir();(target/'unrelated.txt').write_text('preserve',encoding='utf-8')
            with self.assertRaises(FileExistsError):execute(root,plan)
            self.assertEqual((target/'unrelated.txt').read_text(),'preserve');self.assertTrue(sources[0].exists())

    def test_live_capture_cannot_race_migration_and_source_addition_is_not_silently_lost(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,target,sources=self.setup(Path(tmp));plan=make_plan(root,target)
            with source_scope(root):
                with self.assertRaisesRegex(RuntimeError,'Another PC'):execute(root,plan)
            def add_file(value):
                if value['stage']=='copied':(sources[0]/'unowned.txt').write_text('keep',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Unexpected files'):execute(root,plan,add_file)
            self.assertFalse((root/CONFIG).exists());self.assertTrue(sources[0].exists())
