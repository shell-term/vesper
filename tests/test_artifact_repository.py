import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from vesper.artifact_repository import BLOCK,DirectBytes,register_blob,register_existing,VerifiedBlob
from vesper.resources import MemoryBudget,MiB


class ArtifactRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.memory=patch('vesper.resources.physical_memory',return_value={'available':64*1024**3})
        self.memory.start(); self.addCleanup(self.memory.stop)

    def budget(self): return MemoryBudget(cap=8*MiB,available_floor=0)

    def create(self,root,size=3*BLOCK+123):
        raw=bytes(range(256))*(size//256)+bytes(range(size%256))
        file=Path(root)/'original.bin'; file.write_bytes(raw)
        expected={'file':file.name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
        return file,raw,expected

    def test_real_direct_io_tail_and_cross_block_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            file,raw,expected=self.create(tmp); budget=self.budget()
            with DirectBytes(file,budget) as reader:
                self.assertEqual(reader.read(3*BLOCK,BLOCK),raw[3*BLOCK:])
                self.assertEqual(reader.valid_bytes_read,123)
                for offset,size in ((1,4096),(0,10),(4*BLOCK,BLOCK),(0,2*BLOCK)):
                    with self.assertRaises(ValueError): reader.read(offset,size)
            receipt=register_blob(file,expected,Path(tmp)/(file.name+'.sha256'),budget)
            blob=VerifiedBlob(file,tmp,receipt,expected,budget)
            try:
                for offset,size in ((0,256),(BLOCK-13,264),(2*BLOCK+7,256),(3*BLOCK+2,121),(0,8)):
                    self.assertEqual(blob.read(offset,size),raw[offset:offset+size])
                    self.assertLessEqual(len(blob.pages),2)
                self.assertEqual(receipt['blocks'],4)
                self.assertEqual(receipt['index_bytes'],4096)
                for offset,size in ((-1,8),(0,4097),(len(raw),8),(True,1)):
                    with self.assertRaises(ValueError): blob.read(offset,size)
                blob.clear(); self.assertFalse(blob.pages)
            finally: blob.close()
            self.assertEqual(budget.reserved,0)
            self.assertLess(budget.peak,MiB)

    def test_wrong_original_hash_cleans_only_created_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            file,raw,expected=self.create(tmp); budget=self.budget(); index=Path(tmp)/'original.bin.sha256'
            changed={**expected,'sha256':'0'*64}
            with self.assertRaises(ValueError): register_blob(file,changed,index,budget)
            self.assertFalse(index.exists()); self.assertEqual(file.read_bytes(),raw)
            index.write_bytes(b'owned-by-someone-else')
            with self.assertRaises(ValueError): register_blob(file,expected,index,budget)
            self.assertEqual(index.read_bytes(),b'owned-by-someone-else'); self.assertEqual(budget.reserved,0)

    def test_corrupted_data_and_index_refuse_before_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            file,raw,expected=self.create(tmp); budget=self.budget(); index=Path(tmp)/(file.name+'.sha256')
            receipt=register_blob(file,expected,index,budget)
            blob=VerifiedBlob(file,tmp,receipt,expected,budget)
            try:
                with file.open('r+b',buffering=0) as out: out.write(bytes([raw[0]^1])); out.flush()
                with self.assertRaisesRegex(ValueError,'block SHA'): blob.read(0,8)
                self.assertFalse(blob.pages)
                with index.open('r+b',buffering=0) as out: out.write(b'\xff'*32); out.flush()
                with self.assertRaisesRegex(ValueError,'index page changed'): blob.read(0,8)
            finally: blob.close()
            with self.assertRaisesRegex(ValueError,'index SHA'): VerifiedBlob(file,tmp,receipt,expected,budget)
            self.assertEqual(budget.reserved,0)

    def test_receipt_and_pressure_refuse_without_data_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            file,raw,expected=self.create(tmp); budget=self.budget(); index=Path(tmp)/(file.name+'.sha256')
            receipt=register_blob(file,expected,index,budget)
            for key,value in (('sha256','0'*64),('bytes',1),('block_bytes',4096),('index_file','../other'),('index_bytes',8192)):
                changed={**receipt,key:value}
                with patch('vesper.artifact_repository.DirectBytes',side_effect=AssertionError('Must refuse before I/O')):
                    with self.assertRaises(ValueError): VerifiedBlob(file,tmp,changed,expected,budget)
            with patch.object(budget,'acquire',side_effect=MemoryError('Pressure')):
                with self.assertRaises(MemoryError): register_blob(file,expected,Path(tmp)/'new.sha256',budget)
            self.assertFalse((Path(tmp)/'new.sha256').exists()); self.assertEqual(budget.reserved,0)

    def test_full_registration_requires_explicit_execution(self):
        with patch('vesper.artifact_repository.shard_inventory',side_effect=AssertionError('No I/O before opt-in')):
            with self.assertRaises(ValueError): register_existing('metadata','originals','integrity')

    def test_full_registration_accounts_for_scratch_and_refuses_replacement(self):
        # Small controlled originals exercise orchestration, not checkpoint parity.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); file,raw,expected=self.create(root); target=root/'integrity'
            with patch('vesper.artifact_repository.shard_inventory',return_value=[expected]), patch('vesper.artifact_repository.shutil.disk_usage') as usage:
                usage.return_value.free=2**40
                result=register_existing('unused',root,target,execute=True)
                self.assertEqual(result['host_allowance_bytes'],32*MiB)
                self.assertGreater(result['host_peak_reserved_bytes'],8*MiB)
                self.assertEqual(json.loads((target/'manifest.json').read_bytes()),result)
                with self.assertRaises(FileExistsError): register_existing('unused',root,target,execute=True)
            self.assertEqual(file.read_bytes(),raw)


if __name__=='__main__': unittest.main()
