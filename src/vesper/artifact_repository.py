"""Register existing original shards and read verified blocks without file mapping.

Registration streams originals once, validates their released SHA-256 and writes
small block-digest sidecars. It neither downloads nor duplicates checkpoint data.
"""

from collections import OrderedDict
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import shutil

from .checkpoint import MODEL,REVISION
from .disk import DirectReader
from .model_artifacts import shard_inventory,CATALOG_SHA256,SHARD_INVENTORY_SHA256,digest
from .resources import MemoryBudget,MiB,GiB

BLOCK=65536
MAX_VERIFIED_SPAN=8*MiB


class DirectBytes(DirectReader):
    """One aligned anonymous scratch buffer; exact EOF tails, no buffered fallback."""
    def __init__(self,path,budget,capacity=BLOCK):
        self.file_bytes=Path(path).stat().st_size; self.valid_bytes_read=0; self.aligned_bytes_requested=0
        super().__init__(path,capacity,budget)

    def read(self,offset,size):
        if offset<0 or offset%4096 or size<=0 or size%4096 or size>self.capacity or offset>=self.file_bytes:
            raise ValueError('Unaligned, excessive or out-of-file direct artifact read')
        expected=min(size,self.file_bytes-offset)
        if os.name=='nt':
            count=C.c_uint32()
            if not self.lib.SetFilePointerEx(self.handle,offset,None,0): raise C.WinError(C.get_last_error())
            if not self.lib.ReadFile(self.handle,self.buffer,size,C.byref(count),None): raise C.WinError(C.get_last_error())
            actual=count.value
        else:
            view=memoryview(self.buffer)[:size]
            try: actual=os.preadv(self.handle,[view],offset)
            finally: view.release()
        if actual!=expected: raise OSError('Unexpected short direct artifact read')
        self.valid_bytes_read+=actual; self.aligned_bytes_requested+=size
        return self.array()[:actual].tobytes()


def register_blob(path,expected,index_path,budget):
    """Bounded streaming primitive; also exercised with small original tensors."""
    if (type(expected.get('bytes')) is not int or expected['bytes']<=0 or type(expected.get('sha256')) is not str
            or len(expected['sha256'])!=64 or any(c not in '0123456789abcdef' for c in expected['sha256'])):
        raise ValueError('Expected an immutable original byte count and SHA-256')
    path=Path(path); index_path=Path(index_path)
    before=path.stat()
    if before.st_size!=expected['bytes'] or index_path.exists(): raise ValueError('Original artifact size differs or integrity index already exists')
    whole,index_hash=hashlib.sha256(),hashlib.sha256(); count=0; created=False
    try:
        with index_path.open('xb',buffering=0) as output:
            created=True
            with DirectBytes(path,budget) as source:
                for offset in range(0,before.st_size,BLOCK):
                    budget.acquire(0); raw=source.read(offset,BLOCK); whole.update(raw)
                    block=hashlib.sha256(raw).digest(); index_hash.update(block)
                    if output.write(block)!=32: raise OSError('Short integrity-index write')
                    count+=1
            padding=(-output.tell())%4096
            if padding:
                pad=bytes(padding); index_hash.update(pad)
                if output.write(pad)!=len(pad): raise OSError('Short integrity-index padding')
            index_bytes=output.tell(); os.fsync(output.fileno())
        after=path.stat()
        if whole.hexdigest()!=expected['sha256'] or (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
            raise ValueError('Released original SHA-256 differs or source changed during registration')
        return {**expected,'block_bytes':BLOCK,'blocks':count,'index_file':index_path.name,'index_bytes':index_bytes,'index_sha256':index_hash.hexdigest()}
    except BaseException:
        if created: index_path.unlink(missing_ok=True)
        raise


def register_existing(source,originals,integrity,*,execute=False):
    if execute is not True: raise ValueError('Full original registration requires explicit execution; use the import plan first')
    original_root=Path(originals); target=Path(integrity)
    inventory=shard_inventory(source)
    if target.exists(): raise FileExistsError('Refusing to replace an original-artifact integrity repository')
    for row in inventory:
        if (original_root/row['file']).stat().st_size!=row['bytes']: raise ValueError('All 48 originals must exist before registration')
    index_bytes=sum((((r['bytes']+BLOCK-1)//BLOCK)*32+4095)//4096*4096 for r in inventory)
    if shutil.disk_usage(target.parent).free<index_bytes+64*GiB: raise OSError('Integrity import would consume the disk reserve')
    # Include Python metadata, streaming copies and digest objects, while leaving
    # room for the direct reader's separately charged aligned scratch buffer.
    budget=MemoryBudget(cap=32*MiB); overhead=8*MiB; budget.acquire(overhead)
    try:
        target.mkdir(); entries=[]
        for row in inventory:
            entries.append(register_blob(original_root/row['file'],row,target/(row['file']+'.sha256'),budget))
        result={'schema_version':1,'kind':'registered_original_artifacts','model':MODEL,'revision':REVISION,
            'catalog_sha256':CATALOG_SHA256,'shard_inventory_sha256':SHARD_INVENTORY_SHA256,'entries':entries,
            'original_weights_duplicated':False,'full_checkpoint_downloaded':False,'host_allowance_bytes':budget.cap,
            'host_peak_reserved_bytes':budget.peak,'reservation_is_RSS_limit':False}
        (target/'manifest.json').write_bytes((json.dumps(result,indent=2)+'\n').encode()); return result
    finally: budget.release(overhead)


class VerifiedBlob:
    def __init__(self,path,index_root,receipt,expected,budget,*,max_span_bytes=4096):
        if (type(max_span_bytes) is not int or not 4096<=max_span_bytes<=MAX_VERIFIED_SPAN
                or max_span_bytes%4096):
            raise ValueError('Verified span capacity must be 4 KiB aligned and at most 8 MiB')
        self.path=Path(path); self.index_path=Path(index_root)/receipt['index_file']; self.receipt=receipt; self.budget=budget
        self.max_span_bytes=max_span_bytes
        self.data=self.index=None; self.pages=OrderedDict(); self.closed=False
        self.metadata_reserved=0
        if any(receipt.get(k)!=v for k,v in expected.items()) or receipt.get('block_bytes')!=BLOCK or receipt.get('blocks')!=(expected['bytes']+BLOCK-1)//BLOCK:
            raise ValueError('Original artifact receipt differs from released identity')
        if receipt['index_file']!=expected['file']+'.sha256' or receipt['index_bytes']!=((receipt['blocks']*32+4095)//4096)*4096:
            raise ValueError('Original artifact integrity-index layout differs')
        if receipt['index_bytes']>64*MiB: raise ValueError('Excessive original integrity-index metadata')
        if self.path.stat().st_size!=expected['bytes'] or self.index_path.stat().st_size!=receipt['index_bytes']:
            raise ValueError('Original artifact or integrity-index size differs')
        try:
            metadata=(receipt['index_bytes']//4096)*96
            budget.acquire(metadata,category='integrity'); self.metadata_reserved=metadata
            self.data=DirectBytes(self.path,budget,capacity=max(BLOCK,max_span_bytes)); self.index=DirectBytes(self.index_path,budget)
            sha=hashlib.sha256(); self.index_page_hashes=[]
            for offset in range(0,receipt['index_bytes'],BLOCK):
                budget.acquire(0); raw=self.index.read(offset,BLOCK); sha.update(raw)
                self.index_page_hashes.extend(hashlib.sha256(raw[i:i+4096]).digest() for i in range(0,len(raw),4096))
            if sha.hexdigest()!=receipt['index_sha256']: raise ValueError('Original artifact integrity-index SHA-256 differs')
            self.index_initialization_bytes=self.index.valid_bytes_read
        except BaseException: self.close(); raise

    def read(self,offset,size):
        if self.closed or type(offset) is not int or type(size) is not int or offset<0 or not 0<size<=4096 or offset+size>self.receipt['bytes']:
            raise ValueError('Original artifact row range differs')
        self.budget.acquire(0); result=[]
        for number in range(offset//BLOCK,(offset+size-1)//BLOCK+1):
            if number in self.pages:
                raw=self.pages.pop(number); self.pages[number]=raw
            else:
                index_offset=number*32; page=(index_offset//4096)*4096
                index_page=self.index.read(page,4096)
                if hashlib.sha256(index_page).digest()!=self.index_page_hashes[page//4096]:
                    raise ValueError('Integrity-index page changed after original registration')
                expected=index_page[index_offset%4096:index_offset%4096+32]
                raw=self.data.read(number*BLOCK,BLOCK)
                if hashlib.sha256(raw).digest()!=expected: raise ValueError('Original artifact block SHA-256 differs before row publication')
                if len(self.pages)==2: self.pages.popitem(last=False); self.budget.release(BLOCK,category='cache')
                self.budget.acquire(BLOCK,category='cache'); self.pages[number]=raw
            begin=max(offset-number*BLOCK,0); end=min(offset+size-number*BLOCK,len(raw)); result.append(raw[begin:end])
        output=b''.join(result)
        if len(output)!=size: raise OSError('Original artifact row range is incomplete')
        return output

    def read_into(self,offset,target):
        """Fill a bounded caller-owned span, verifying all source blocks first.

        The larger capacity is explicit opt-in; legacy row reads retain their
        4 KiB limit. Direct reads include complete 64 KiB integrity blocks and
        use the existing aligned scratch. Each batch additionally reserves its
        temporary byte copy and one digest-index page before any payload I/O.
        No payload or index-page cache is added by this path. A failed later
        batch may leave a verified prefix in the caller's target, just as a
        sequence of row reads does; callers publish only after full success.
        """
        view=memoryview(target)
        if view.readonly or not view.c_contiguous:
            raise ValueError('Expected a caller-owned writable contiguous verified span')
        view=view.cast('B'); size=view.nbytes
        if (self.closed or type(offset) is not int or offset<0 or not 0<size<=self.max_span_bytes
                or offset+size>self.receipt['bytes']):
            raise ValueError('Original artifact verified span range differs')
        aligned_start=(offset//BLOCK)*BLOCK
        aligned_end=((offset+size+BLOCK-1)//BLOCK)*BLOCK
        # At least one whole integrity block fits even for the legacy capacity.
        batch_capacity=(self.data.capacity//BLOCK)*BLOCK
        for start in range(aligned_start,aligned_end,batch_capacity):
            request=min(batch_capacity,aligned_end-start)
            temporary=request+4096
            self.budget.acquire(temporary,category='scratch')
            raw=index_page=data_view=block_view=None
            try:
                raw=self.data.read(start,request); data_view=memoryview(raw)
                current_page=None
                for local in range(0,len(raw),BLOCK):
                    number=(start+local)//BLOCK; index_offset=number*32
                    page=(index_offset//4096)*4096
                    if page!=current_page:
                        # Drop the previous page before creating its replacement;
                        # the transient reservation includes exactly one page.
                        index_page=None
                        index_page=self.index.read(page,4096)
                        if hashlib.sha256(index_page).digest()!=self.index_page_hashes[page//4096]:
                            raise ValueError('Integrity-index page changed after original registration')
                        current_page=page
                    expected=index_page[index_offset%4096:index_offset%4096+32]
                    block_view=data_view[local:min(local+BLOCK,len(raw))]
                    if hashlib.sha256(block_view).digest()!=expected:
                        raise ValueError('Original artifact block SHA-256 differs before span publication')
                    block_view.release(); block_view=None
                begin=max(offset,start); end=min(offset+size,start+len(raw))
                if end<=begin:
                    raise OSError('Original artifact verified span is incomplete')
                view[begin-offset:end-offset]=data_view[begin-start:end-start]
            finally:
                if block_view is not None: block_view.release()
                if data_view is not None: data_view.release()
                raw=index_page=None
                self.budget.release(temporary,category='scratch')

    def close(self):
        self.clear()
        for reader in (self.data,self.index):
            if reader: reader.close()
        self.data=self.index=None; self.closed=True
        self.index_page_hashes=[]
        self.budget.release(self.metadata_reserved,category='integrity'); self.metadata_reserved=0

    def clear(self):
        self.budget.release(len(self.pages)*BLOCK,category='cache'); self.pages.clear()
