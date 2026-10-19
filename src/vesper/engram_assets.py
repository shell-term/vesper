"""Bounded original Engram control tensors and selected-row import."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .checkpoint import MODEL, REVISION, LIMIT, read_range
from .model_artifacts import load_catalog, CATALOG_SHA256, canonical, digest
from .ngram import LAYERS, TABLE_ROWS
from .resources import MemoryBudget, MiB

CONTROL_LAYOUT={"q_weight":("BF16",[4,5120]),"k_weight":("BF16",[4,5120]),
    "wkv.weight":("F8_E4M3",[25600,6144]),"wkv.scale":("F8_E8M0",[800,192])}
CONTROL_BYTES=157521920
MAX_ROWS=1024
CONTROL_IDENTITIES={1:'da10e23e4832d733cb9862c4417ea5683c7ee1a8c225680944c520af5f3abaa2',
    14:'98e380c1a01ddb17eca7284ae46df22c63ddd119b73d20f3bd0e4b015b6c58d3'}
TABLE_BASES={1:('model-00047-of-00048.safetensors',664,98305579672),
    14:('model-00048-of-00048.safetensors',672,98308271264)}


def expected_row_address(layer,row):
    if type(layer) is not int or layer not in LAYERS or type(row) is not int or not 0<=row<TABLE_ROWS[LAYERS.index(layer)]:
        raise ValueError('Invalid original Engram row')
    shard,weight,scale=TABLE_BASES[layer]
    return [{'tensor':f'layers.{layer}.engram.embed.{name}','dtype':dtype,'shard':shard,'shard_byte_offset':base+row*width,'bytes':width}
        for name,width,dtype,base in (('weight',256,'F8_E4M3',weight),('scale',8,'F8_E8M0',scale))]


def control_file(name): return name.replace(".","-")+".bin"


def fetch_controls(source, destination, layer):
    if type(layer) is not int or layer not in LAYERS: raise ValueError("Unregistered original Engram layer")
    catalog,_=load_catalog(source); root=Path(destination); manifest=root/'manifest.json'
    if manifest.exists():
        arrays,m=load_controls(root,layer); del arrays
        return m
    root.mkdir(parents=True,exist_ok=True)
    budget=MemoryBudget(cap=32*MiB); budget.acquire(budget.cap)
    entries=[]
    try:
        for short,(dtype,shape) in CONTROL_LAYOUT.items():
            t=catalog[f"layers.{layer}.engram.{short}"]
            if (t['dtype'],t['shape'])!=(dtype,shape): raise ValueError("Original Engram control layout differs")
            path=root/control_file(short)
            if path.exists(): raise FileExistsError("Partial control import requires inspection; refusing replacement")
            part=path.with_suffix('.part'); sha=hashlib.sha256()
            created=False
            try:
                with part.open('xb',buffering=0) as file:
                    created=True
                    for offset in range(0,t['bytes'],LIMIT):
                        budget.acquire(0); raw=read_range(t['shard'],t['shard_byte_offset']+offset,min(LIMIT,t['bytes']-offset))
                        sha.update(raw)
                        if file.write(raw)!=len(raw): raise OSError("Short original control write")
                    os.fsync(file.fileno())
                os.replace(part,path)
            except BaseException:
                if created: part.unlink(missing_ok=True)
                raise
            entries.append({**t,'sha256':sha.hexdigest(),'file':path.name})
        result={'kind':'original_engram_controls','model':MODEL,'revision':REVISION,'layer':layer,'catalog_sha256':CATALOG_SHA256,
            'tensors':entries,'tensor_bytes':sum(t['bytes'] for t in entries),'maximum_range_bytes':LIMIT}
        if result['tensor_bytes']!=CONTROL_BYTES: raise ValueError("Original control import exceeded its declared scope")
        manifest.write_bytes((json.dumps(result,indent=2)+'\n').encode())
        return result
    finally: budget.release(budget.cap)


def load_controls(root,layer):
    if type(layer) is not int or layer not in LAYERS: raise ValueError("Unregistered original Engram layer")
    root=Path(root); path=root/'manifest.json'
    if path.stat().st_size>16384: raise ValueError("Excessive original Engram control manifest")
    m=json.loads(path.read_bytes()); entries=m.get('tensors',[])
    if canonical(m)!=CONTROL_IDENTITIES[layer]: raise ValueError('Unqualified original Engram control manifest')
    if (m.get('kind'),m.get('model'),m.get('revision'),m.get('layer'),m.get('catalog_sha256'),m.get('tensor_bytes'),len(entries))!=(
            'original_engram_controls',MODEL,REVISION,layer,CATALOG_SHA256,CONTROL_BYTES,4):
        raise ValueError("Original Engram control identity differs")
    arrays={}
    for (name,(dtype,shape)),t in zip(CONTROL_LAYOUT.items(),entries):
        size=int(np.prod(shape))*(2 if dtype=='BF16' else 1)
        if (t.get('name'),t.get('dtype'),t.get('shape'),t.get('bytes'),t.get('file'))!=(f'layers.{layer}.engram.{name}',dtype,shape,size,control_file(name)):
            raise ValueError("Original Engram control provenance differs")
        path=root/control_file(name)
        if path.stat().st_size!=size: raise ValueError("Original Engram control size differs")
        raw=path.read_bytes()
        if digest(raw)!=t['sha256']: raise ValueError("Original Engram control SHA-256 differs")
        a=np.frombuffer(raw,np.uint16 if dtype=='BF16' else np.uint8).reshape(shape)
        for offset in range(0,a.size,4*MiB):
            part=a.reshape(-1)[offset:offset+4*MiB]
            if np.any((part&0x7F80)==0x7F80) if dtype=='BF16' else np.any(part==255) if dtype=='F8_E8M0' else np.any((part&127)==127):
                raise ValueError("Original Engram controls must be finite")
        arrays[name]=a
    return arrays,m


def row_address(catalog,layer,row):
    if type(layer) is not int or layer not in LAYERS or type(row) is not int or not 0<=row<TABLE_ROWS[LAYERS.index(layer)]:
        raise ValueError("Invalid original Engram row address")
    entries=[]
    for suffix,width,dtype in (('weight',256,'F8_E4M3'),('scale',8,'F8_E8M0')):
        t=catalog[f'layers.{layer}.engram.embed.{suffix}']
        if (t['dtype'],t['shape'])!=(dtype,[TABLE_ROWS[LAYERS.index(layer)],width]): raise ValueError("Original table layout differs")
        entries.append({'tensor':t['name'],'dtype':dtype,'shard':t['shard'],'shard_byte_offset':t['shard_byte_offset']+row*width,'bytes':width})
    return entries


def prepare_rows(source,root,required,progress=None):
    if set(required)!={'1','14'} or any(not isinstance(v,list) or not 1<=len(v)<=MAX_ROWS or sorted(set(v))!=v for v in required.values()):
        raise ValueError("Expected two bounded sorted Engram row sets")
    catalog,_=load_catalog(source)
    for key,ids in required.items():
        for row in ids: row_address(catalog,int(key),row)
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    cache=root/'downloaded-rows'; cache.mkdir(exist_ok=True)
    budget=MemoryBudget(cap=32*MiB); budget.acquire(budget.cap)
    def fetch(item):
        layer,row=item; path=cache/f'{layer}-{row}.bin'; meta=path.with_suffix('.json'); expected=row_address(catalog,layer,row)
        if path.exists() or meta.exists():
            if not path.is_file() or not meta.is_file() or path.stat().st_size!=264 or meta.stat().st_size>4096: raise ValueError("Incomplete cached row")
            m=json.loads(meta.read_bytes()); raw=path.read_bytes()
            if m.get('source')!=expected or digest(raw)!=m.get('sha256'): raise ValueError("Cached original row identity differs")
            return layer,row,raw,m,0
        values=[read_range(e['shard'],e['shard_byte_offset'],e['bytes']) for e in expected]
        raw=b''.join(values)
        if any((x&127)==127 for x in values[0]) or 255 in values[1]: raise ValueError("Original Engram row is nonfinite")
        m={'source':expected,'sha256':digest(raw),'weight_sha256':digest(values[0]),'scale_sha256':digest(values[1])}
        with path.open('xb') as file: file.write(raw)
        meta.write_bytes(json.dumps(m).encode())
        return layer,row,raw,m,264
    fetched,prepared=0,{}
    try:
        items=[(int(k),row) for k,rows in required.items() for row in rows]
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures=[pool.submit(fetch,item) for item in items]
            try:
                for n,future in enumerate(as_completed(futures),1):
                    layer,row,raw,m,count=future.result(); prepared[(layer,row)]=(raw,m); fetched+=count
                    if progress and (n%48==0 or n==len(items)): progress({'prepared_rows':n,'total_rows':len(items)})
            except BaseException:
                for f in futures: f.cancel()
                raise
        results={}
        for layer in LAYERS:
            directory=root/f'layer{layer}'
            if directory.exists(): raise FileExistsError("Refusing to replace a completed Engram row store")
            directory.mkdir(); entries=[]
            with (directory/'rows.bin').open('xb',buffering=0) as file:
                for n,row in enumerate(required[str(layer)]):
                    raw,m=prepared[(layer,row)]; payload=raw+bytes(512-264)
                    if file.write(payload)!=512: raise OSError("Short selected-row store write")
                    entries.append({'row':row,'offset':n*512,**m})
                padding=(-file.tell())%4096
                if padding: file.write(bytes(padding))
                size=file.tell(); os.fsync(file.fileno())
            m={'kind':'original_engram_selected_rows','model':MODEL,'revision':REVISION,'catalog_sha256':CATALOG_SHA256,'layer':layer,
                'rows':entries,'record_bytes':512,'file_bytes':size,'original_payload_bytes':len(entries)*264,'logical_rows_per_record':1,
                'physical_alignment_bytes':4096,'maximum_rows':MAX_ROWS}
            (directory/'manifest.json').write_bytes((json.dumps(m,indent=2)+'\n').encode()); results[str(layer)]=m
        return {'status':'pass','new_original_row_bytes':fetched,'verified_reused_original_row_bytes':len(items)*264-fetched,
            'stores':results,'maximum_fetch_workers':8,'host_allowance_bytes':budget.peak,'full_table_downloaded':False}
    finally: budget.release(budget.cap)
