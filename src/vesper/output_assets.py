"""Bounded original output matrix, final norm and selected embedding rows."""

import hashlib
import json
import os
from pathlib import Path
import shutil
from functools import wraps

import numpy as np

from .checkpoint import MODEL,REVISION,LIMIT,read_range
from .model_artifacts import load_catalog,CATALOG_SHA256,canonical,digest
from .resources import MemoryBudget,MiB,GiB

VOCAB,WIDTH=129280,5120
EMBED_IDS=(0,1,2,42,1000,4096,128000,129279)
LAYOUT={'head.weight':('BF16',[VOCAB,WIDTH]),'norm.weight':('BF16',[WIDTH])}
PAYLOAD_BYTES=VOCAB*WIDTH*2+WIDTH*2+len(EMBED_IDS)*WIDTH*2
QUALIFIED_IDENTITY='132292679ef545c838969beddeefdd3e37b0c2bb724d61ae72e2c8208b658271'
TENSOR_SHA256={'head.weight':'12f45f83b1a1cd9f440f48f2d5a8b7b38b0a71fc9b35b7f9b7de7c4d81673508',
    'norm.weight':'22bcf3d1a2a739308b1a21bcc8137555725701ac8fdcd9be980a9e161be92a34',
    'embedding_rows':'631a1b12f5d93bb2580cf0997dc9ddcae03ed76bf1e8804ac842f4bfb6cdc9c4'}


def file_digest(path,size):
    if path.stat().st_size!=size: raise ValueError('Original output fixture byte count differs')
    sha=hashlib.sha256()
    with path.open('rb',buffering=0) as file:
        while raw:=file.read(LIMIT): sha.update(raw)
    return sha.hexdigest()


def exclusive_import(function):
    @wraps(function)
    def run(source,directory,*args,**kwargs):
        root=Path(directory); root.mkdir(parents=True,exist_ok=True)
        with (root/'.output-import.lock').open('a+b') as guard:
            guard.seek(0,2)
            if not guard.tell(): guard.write(b'\0'); guard.flush()
            guard.seek(0)
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(guard.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(guard.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            # Closing the descriptor releases the OS lock after a crash too.
            return function(source,root,*args,**kwargs)
    return run


@exclusive_import
def prepare(source,directory,progress=None):
    root=Path(directory); root.mkdir(parents=True,exist_ok=True)
    if (root/'manifest.json').exists(): return load(root,arrays=False)[1]
    if shutil.disk_usage(root).free<PAYLOAD_BYTES+64*GiB: raise OSError('Output probe needs its original payload plus the 64 GiB disk reserve')
    budget=MemoryBudget(cap=640*MiB,owner='output_fixture_transfer_and_catalog'); budget.acquire(budget.cap)
    records=[]; fetched=0
    try:
        catalog,_=load_catalog(source)
        for name,(dtype,shape) in LAYOUT.items():
            t=catalog[name]
            if t['dtype']!=dtype or t['shape']!=shape: raise ValueError('Original output layout differs')
            path=root/(name.replace('.','-')+'.bin'); part=path.with_suffix('.part')
            if path.exists():
                sha=file_digest(path,t['bytes'])
                if sha!=TENSOR_SHA256[name]: raise ValueError('Existing original output tensor differs')
                records.append({**t,'file':path.name,'sha256':sha}); continue
            # Resume bounded completed chunks only after rechecking every byte.
            receipt=part.with_suffix('.chunks.json'); chunks=[]
            if part.exists() and not receipt.exists(): raise FileExistsError('Unregistered output partial; preserve it for inspection')
            if receipt.exists():
                if receipt.stat().st_size>65536: raise ValueError('Excessive output partial metadata')
                saved=json.loads(receipt.read_bytes())
                if saved['tensor']!=t or len(saved['chunks'])>(t['bytes']+LIMIT-1)//LIMIT: raise ValueError('Output partial source/extent differs')
                with part.open('rb' if part.exists() else 'xb') as file:
                    for i,c in enumerate(saved['chunks']):
                        size=min(LIMIT,t['bytes']-i*LIMIT); raw=file.read(size)
                        if c!={'bytes':size,'sha256':digest(raw)} or len(raw)!=size: raise ValueError('Output partial chunk differs')
                        chunks.append(c)
            else:
                receipt.write_bytes((json.dumps({'tensor':t,'chunks':[]})+'\n').encode())
            with part.open('r+b' if part.exists() else 'xb',buffering=0) as file:
                file.truncate(sum(c['bytes'] for c in chunks)); file.seek(0,2)
                for offset in range(sum(c['bytes'] for c in chunks),t['bytes'],LIMIT):
                    budget.acquire(0); raw=read_range(t['shard'],t['shard_byte_offset']+offset,min(LIMIT,t['bytes']-offset))
                    if file.write(raw)!=len(raw): raise OSError('Short output fixture write')
                    chunks.append({'bytes':len(raw),'sha256':digest(raw)}); fetched+=len(raw)
                    os.fsync(file.fileno())
                    temporary=receipt.with_suffix('.tmp'); temporary.write_bytes((json.dumps({'tensor':t,'chunks':chunks})+'\n').encode()); os.replace(temporary,receipt)
                    if progress and (len(chunks)%8==0 or offset+len(raw)==t['bytes']):
                        progress({'tensor':name,'bytes_ready':offset+len(raw),'bytes_total':t['bytes']})
            sha=file_digest(part,t['bytes'])
            if sha!=TENSOR_SHA256[name]: raise ValueError('Fetched original output tensor differs from its qualified bytes')
            os.replace(part,path); receipt.unlink(missing_ok=True)
            records.append({**t,'file':path.name,'sha256':sha})
        t=catalog['embed.weight']; rows=[]
        if (t['dtype'],t['shape'])!=('BF16',[VOCAB,WIDTH]): raise ValueError('Original embedding layout differs')
        for token in EMBED_IDS:
            budget.acquire(0); raw=read_range(t['shard'],t['shard_byte_offset']+token*WIDTH*2,WIDTH*2); fetched+=len(raw)
            rows.append(raw)
        data=b''.join(rows); path=root/'embedding-rows.bin'
        if digest(data)!=TENSOR_SHA256['embedding_rows']: raise ValueError('Fetched original embedding rows differ')
        if path.exists():
            if path.read_bytes()!=data: raise ValueError('Existing original embedding rows differ')
        else:
            with path.open('xb') as file: file.write(data)
        records.append({'name':'embedding_rows','source_tensor':t,'ids':list(EMBED_IDS),'dtype':'BF16','shape':[len(EMBED_IDS),WIDTH],
            'bytes':len(data),'file':path.name,'sha256':digest(data)})
        manifest={'schema_version':1,'kind':'original_output_fixture','model':MODEL,'revision':REVISION,'catalog_sha256':CATALOG_SHA256,
            'tensor_bytes':PAYLOAD_BYTES,'tensors':records,'maximum_range_bytes':LIMIT,'full_checkpoint_downloaded':False}
        (root/'manifest.json').write_bytes((json.dumps(manifest,indent=2)+'\n').encode())
        if progress: progress({'status':'pass','payload_bytes':PAYLOAD_BYTES,'new_weight_bytes':fetched,'fixture_identity':canonical(manifest)})
        return manifest
    finally: budget.release(budget.cap)


def load(directory,*,arrays=True,require_qualified=True):
    root=Path(directory); path=root/'manifest.json'
    if not 0<path.stat().st_size<=65536: raise ValueError('Excessive output fixture metadata')
    m=json.loads(path.read_bytes())
    if (m.get('kind'),m.get('model'),m.get('revision'),m.get('catalog_sha256'),m.get('tensor_bytes'))!=(
            'original_output_fixture',MODEL,REVISION,CATALOG_SHA256,PAYLOAD_BYTES): raise ValueError('Original output fixture identity differs')
    if require_qualified and canonical(m)!=QUALIFIED_IDENTITY: raise ValueError('Unqualified original output fixture')
    if [t['name'] for t in m['tensors']]!=['head.weight','norm.weight','embedding_rows']: raise ValueError('Incomplete output fixture')
    result={}
    for t in m['tensors']:
        expected=LAYOUT.get(t['name'],('BF16',[len(EMBED_IDS),WIDTH]))
        if (t['dtype'],t['shape'])!=expected or t['bytes']!=int(np.prod(t['shape']))*2:
            raise ValueError('Original output tensor layout/extent differs')
        if t['name']=='embedding_rows' and t['ids']!=list(EMBED_IDS): raise ValueError('Original embedding-row addresses differ')
        if t['sha256']!=TENSOR_SHA256[t['name']]: raise ValueError('Original output payload identity differs')
        p=(root/t['file']).resolve()
        if not p.is_relative_to(root.resolve()) or file_digest(p,t['bytes'])!=t['sha256']: raise ValueError('Output fixture tensor differs')
        if arrays: result[t['name']]=np.fromfile(p,dtype=np.uint16).reshape(t['shape'])
    return result,m
