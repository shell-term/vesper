import copy
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from vesper.artifact_repository import BLOCK,MAX_VERIFIED_SPAN,DirectBytes,register_blob,register_existing,VerifiedBlob
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

    def test_opt_in_span_uses_bounded_bulk_io_and_verifies_unaligned_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            file,raw,expected=self.create(tmp,size=2*MAX_VERIFIED_SPAN+123)
            budget=MemoryBudget(cap=64*MiB,available_floor=0)
            receipt=register_blob(file,expected,Path(tmp)/(file.name+'.sha256'),budget)
            blob=VerifiedBlob(file,tmp,receipt,expected,budget,max_span_bytes=MAX_VERIFIED_SPAN)
            try:
                target=bytearray(MAX_VERIFIED_SPAN)
                with patch.object(blob.data,'_read_view',wraps=blob.data._read_view) as data_read, \
                     patch.object(blob.index,'read',wraps=blob.index.read) as index_read, \
                     patch.object(budget,'acquire',wraps=budget.acquire) as acquire:
                    blob.read_into(17,target)
                    self.assertEqual(target,raw[17:17+len(target)])
                    self.assertEqual([call.args for call in data_read.call_args_list],
                                     [(0,MAX_VERIFIED_SPAN),(MAX_VERIFIED_SPAN,BLOCK)])
                    self.assertEqual(index_read.call_count,2)
                    self.assertEqual(acquire.call_count,2)
                tail=bytearray(130)
                blob.read_into(len(raw)-len(tail),tail)
                self.assertEqual(tail,raw[-len(tail):])
                self.assertFalse(blob.pages)
                self.assertLess(budget.peak,MAX_VERIFIED_SPAN+MiB)
            finally: blob.close()
            self.assertEqual(budget.reserved,0)

    def test_direct_scratch_lease_is_read_only_exclusive_and_invalidated_on_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            file,raw,expected=self.create(tmp); budget=self.budget()
            replacement=Path(tmp)/'replacement.bin'; replacement.write_bytes(b'X'*BLOCK)
            with DirectBytes(file,budget) as reader:
                with reader._read_view(0,BLOCK) as borrowed:
                    self.assertTrue(borrowed.readonly)
                    self.assertEqual(hashlib.sha256(borrowed).digest(),hashlib.sha256(raw[:BLOCK]).digest())
                    with self.assertRaises(TypeError): borrowed[0]=255
                    with self.assertRaisesRegex(RuntimeError,'live read lease'): reader.read(BLOCK,BLOCK)
                    with self.assertRaisesRegex(RuntimeError,'live read lease'):
                        with reader._read_view(BLOCK,BLOCK): pass
                    with self.assertRaisesRegex(RuntimeError,'live read lease'): reader.close()
                    with self.assertRaisesRegex(RuntimeError,'live read lease'): reader.switch_file(replacement)
                with self.assertRaises(ValueError): borrowed.tobytes()
                owned=reader.read(0,BLOCK)
                reader.switch_file(replacement)
                self.assertEqual(reader.read(0,BLOCK),b'X'*BLOCK)
                self.assertEqual(owned,raw[:BLOCK])
                self.assertFalse(reader._scratch_borrowed)
            self.assertEqual(budget.reserved,0)

    def test_bulk_uses_existing_scratch_without_owned_payload_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            file,raw,expected=self.create(tmp,size=MAX_VERIFIED_SPAN+BLOCK)
            budget=MemoryBudget(cap=32*MiB,available_floor=0)
            receipt=register_blob(file,expected,Path(tmp)/(file.name+'.sha256'),budget)
            blob=VerifiedBlob(file,tmp,receipt,expected,budget,max_span_bytes=MAX_VERIFIED_SPAN)
            try:
                held=budget.reserved; target=bytearray(MAX_VERIFIED_SPAN)
                with patch.object(blob.data,'read',side_effect=AssertionError('No owned payload copy')), \
                     patch.object(budget,'acquire',wraps=budget.acquire) as acquire:
                    blob.read_into(0,target)
                self.assertEqual(target,raw[:MAX_VERIFIED_SPAN])
                self.assertEqual([call.args for call in acquire.call_args_list],[(4096,)])
                self.assertEqual(budget.reserved,held)
                self.assertEqual(budget.peak,held+4096)
                self.assertFalse(blob.data._scratch_borrowed)
            finally: blob.close()
            self.assertEqual(budget.reserved,0)

    def test_cancelled_verification_releases_read_lease_and_retry_checks_source_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            file,raw,expected=self.create(tmp); budget=self.budget()
            receipt=register_blob(file,expected,Path(tmp)/(file.name+'.sha256'),budget)
            blob=VerifiedBlob(file,tmp,receipt,expected,budget,max_span_bytes=MiB)
            try:
                held=budget.reserved; target=bytearray(b'Z'*BLOCK)
                with patch.object(blob.index,'read',side_effect=KeyboardInterrupt('Cancelled index verification')):
                    with self.assertRaises(KeyboardInterrupt): blob.read_into(0,target)
                self.assertEqual(target,b'Z'*BLOCK)
                self.assertFalse(blob.data._scratch_borrowed)
                self.assertEqual(budget.reserved,held)
                # A retry must re-read and authenticate the source rather than
                # using any prefix, digest result or copy from the cancelled call.
                with file.open('r+b',buffering=0) as out: out.write(b'X'); out.flush()
                with self.assertRaisesRegex(ValueError,'block SHA'): blob.read_into(0,target)
                self.assertEqual(target,b'Z'*BLOCK)
                self.assertFalse(blob.data._scratch_borrowed)
                with file.open('r+b',buffering=0) as out: out.write(raw[:1]); out.flush()
                blob.read_into(0,target)
                self.assertEqual(target,raw[:BLOCK])
                self.assertEqual(budget.reserved,held)
            finally: blob.close()
            self.assertEqual(budget.reserved,0)

    def test_short_direct_read_releases_lease_before_healthy_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            file,raw,expected=self.create(tmp); budget=self.budget()
            with DirectBytes(file,budget) as reader:
                with file.open('r+b',buffering=0) as out: out.truncate(1)
                with self.assertRaisesRegex(OSError,'short direct artifact read'):
                    with reader._read_view(0,BLOCK): self.fail('Short read published a view')
                self.assertFalse(reader._scratch_borrowed)
                file.write_bytes(raw)
                self.assertEqual(reader.read(0,BLOCK),raw[:BLOCK])
            self.assertEqual(budget.reserved,0)

    def test_concurrent_reader_waits_until_scratch_lease_ends(self):
        with tempfile.TemporaryDirectory() as tmp:
            file,raw,expected=self.create(tmp); budget=self.budget()
            with DirectBytes(file,budget) as reader:
                attempted=threading.Event(); done=threading.Event(); result=[]
                def next_read():
                    attempted.set()
                    try: result.append(reader.read(BLOCK,BLOCK))
                    except BaseException as error: result.append(error)
                    finally: done.set()
                worker=threading.Thread(target=next_read)
                with reader._read_view(0,BLOCK) as borrowed:
                    worker.start(); self.assertTrue(attempted.wait(2))
                    self.assertFalse(done.wait(0.05))
                    self.assertEqual(borrowed.tobytes(),raw[:BLOCK])
                worker.join(2)
                self.assertFalse(worker.is_alive())
                self.assertEqual(result,[raw[BLOCK:2*BLOCK]])
            self.assertEqual(budget.reserved,0)

    def test_span_opt_in_preserves_legacy_row_limit_and_rejects_before_io(self):
        with tempfile.TemporaryDirectory() as tmp:
            file,raw,expected=self.create(tmp); budget=self.budget()
            receipt=register_blob(file,expected,Path(tmp)/(file.name+'.sha256'),budget)
            blob=VerifiedBlob(file,tmp,receipt,expected,budget)
            try:
                with patch.object(blob.data,'_read_view',side_effect=AssertionError('Invalid span must not read payload')):
                    for offset,target in ((0,bytearray(4097)),(-1,bytearray(2)),(True,bytearray(2)),
                            (len(raw),bytearray(2)),(0,b'readonly'),(0,bytearray()),
                            (0,memoryview(bytearray(16))[::2])):
                        with self.subTest(offset=offset,size=len(target)),self.assertRaises(ValueError):
                            blob.read_into(offset,target)
                    with self.assertRaises(ValueError): blob.read(0,4097)
            finally: blob.close()
            for capacity in (0,True,4095,4097,MAX_VERIFIED_SPAN+4096):
                with patch('vesper.artifact_repository.DirectBytes',side_effect=AssertionError('No I/O')):
                    with self.subTest(capacity=capacity),self.assertRaises(ValueError):
                        VerifiedBlob(file,tmp,receipt,expected,budget,max_span_bytes=capacity)
            self.assertEqual(budget.reserved,0)

    def test_bulk_corruption_and_short_read_do_not_publish_or_leak(self):
        with tempfile.TemporaryDirectory() as tmp:
            file,raw,expected=self.create(tmp,size=8*BLOCK+123)
            budget=MemoryBudget(cap=32*MiB,available_floor=0)
            index=Path(tmp)/(file.name+'.sha256')
            receipt=register_blob(file,expected,index,budget)
            for fault in ('data','index','short'):
                file.write_bytes(raw)
                # Preserve the authenticated index before each independent fault.
                index_raw=index.read_bytes()
                blob=VerifiedBlob(file,tmp,receipt,expected,budget,max_span_bytes=MAX_VERIFIED_SPAN)
                target=bytearray(b'Z'*(4*BLOCK))
                try:
                    if fault=='data':
                        with file.open('r+b',buffering=0) as out:
                            out.seek(3*BLOCK+7); out.write(b'X'); out.flush()
                        pattern='block SHA'
                    elif fault=='index':
                        with index.open('r+b',buffering=0) as out:
                            out.write(b'\xff'*32); out.flush()
                        pattern='index page changed'
                    else:
                        with file.open('r+b',buffering=0) as out: out.truncate(BLOCK)
                        pattern='short direct artifact read'
                    held=budget.reserved
                    with self.subTest(fault=fault),self.assertRaisesRegex((ValueError,OSError),pattern):
                        blob.read_into(0,target)
                    self.assertEqual(target,b'Z'*len(target))
                    self.assertEqual(budget.reserved,held)
                finally:
                    blob.close(); index.write_bytes(index_raw)
                self.assertEqual(budget.reserved,0)

    def test_bulk_pressure_refuses_before_payload_and_releases_transient_reservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            file,raw,expected=self.create(tmp); budget=MemoryBudget(cap=32*MiB,available_floor=0)
            receipt=register_blob(file,expected,Path(tmp)/(file.name+'.sha256'),budget)
            blob=VerifiedBlob(file,tmp,receipt,expected,budget,max_span_bytes=MAX_VERIFIED_SPAN)
            try:
                held=budget.reserved
                with patch.object(budget,'acquire',side_effect=MemoryError('Pressure')), \
                     patch.object(blob.data,'_read_view',side_effect=AssertionError('Must refuse before payload')):
                    with self.assertRaisesRegex(MemoryError,'Pressure'): blob.read_into(0,bytearray(BLOCK))
                self.assertEqual(budget.reserved,held)
            finally: blob.close()
            self.assertEqual(budget.reserved,0)

    def test_bulk_checks_whole_blocks_and_failed_later_batch_keeps_only_verified_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            file,raw,expected=self.create(tmp,size=MAX_VERIFIED_SPAN+BLOCK)
            budget=MemoryBudget(cap=64*MiB,available_floor=0)
            receipt=register_blob(file,expected,Path(tmp)/(file.name+'.sha256'),budget)
            blob=VerifiedBlob(file,tmp,receipt,expected,budget,max_span_bytes=MAX_VERIFIED_SPAN)
            try:
                with file.open('r+b',buffering=0) as out: out.write(b'X'); out.flush()
                tiny=bytearray(b'Z'*8)
                with self.assertRaisesRegex(ValueError,'block SHA'): blob.read_into(17,tiny)
                self.assertEqual(tiny,b'Z'*8)
                with file.open('r+b',buffering=0) as out:
                    out.write(raw[:1]); out.seek(MAX_VERIFIED_SPAN+5); out.write(b'X'); out.flush()
                target=bytearray(b'Z'*MAX_VERIFIED_SPAN); held=budget.reserved
                with self.assertRaisesRegex(ValueError,'block SHA'): blob.read_into(17,target)
                self.assertEqual(target[:-17],raw[17:MAX_VERIFIED_SPAN])
                self.assertEqual(target[-17:],b'Z'*17)
                self.assertEqual(budget.reserved,held)
            finally: blob.close()
            self.assertEqual(budget.reserved,0)

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
