"""Check bounded original-tensor registration and direct verified row reads.

Uses six small already-downloaded Engram control tensors, never full shards.
Original inputs are read-only. Only a new temporary integrity directory is made.
"""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from vesper.artifact_repository import BLOCK,register_blob,VerifiedBlob
from vesper.attention import read_bounded
from vesper.engram_assets import CONTROL_IDENTITIES
from vesper.model_artifacts import canonical,digest
from vesper.resources import MemoryBudget,MiB


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('fixtures','scratch','output'): p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args(); scratch=args.scratch.resolve(); scratch.mkdir(parents=True,exist_ok=True)
    budget=MemoryBudget(cap=16*MiB); budget.acquire(8*MiB); results=[]
    try:
        for layer in (1,14):
            root=args.fixtures/f'controls{layer}'
            m=json.loads(read_bounded(root/'manifest.json',16384))
            if canonical(m)!=CONTROL_IDENTITIES[layer]: raise ValueError('Unqualified original small tensors')
            for t in m['tensors']:
                if t['name'].endswith('wkv.weight'): continue
                file=root/t['file']; expected={'file':t['file'],'bytes':t['bytes'],'sha256':t['sha256']}
                # This bounded read supplies comparison bytes, not runtime paging.
                original=read_bounded(file,MiB)
                if len(original)!=expected['bytes'] or digest(original)!=expected['sha256']: raise ValueError('Original small tensor differs')
                with tempfile.TemporaryDirectory(prefix='vesper-integrity-',dir=scratch) as tmp:
                    if Path(tmp).resolve().parent!=scratch: raise ValueError('Unexpected temporary integrity location')
                    receipt=register_blob(file,expected,Path(tmp)/(file.name+'.sha256'),budget)
                    blob=VerifiedBlob(file,tmp,receipt,expected,budget)
                    try:
                        rng=np.random.default_rng(812+layer); ranges=[(0,264),(len(original)-264,264)]
                        ranges.extend((min(n,len(original)-264),264) for n in (BLOCK-13,2*BLOCK-13))
                        ranges.extend((int(n),264) for n in rng.integers(0,len(original)-264+1,28))
                        checks=[]
                        for offset,count in ranges:
                            actual=blob.read(offset,count)
                            if actual!=original[offset:offset+count]: raise RuntimeError('Verified original small-tensor range differs')
                            checks.append({'offset':offset,'bytes':count,'sha256':digest(actual),'status':'pass'})
                        proof={'source_tensor':t,'receipt':receipt,'ranges':checks,'status':'pass',
                            'data_valid_bytes_read':blob.data.valid_bytes_read,'data_aligned_bytes_requested':blob.data.aligned_bytes_requested,
                            'index_initialization_bytes':blob.index_initialization_bytes,'maximum_cached_blocks':2,'file_mapping':False,'buffered_read_fallback':False}
                    finally: blob.close()
                proof['temporary_integrity_directory_removed']=not Path(tmp).exists(); results.append(proof)
    finally: budget.release(8*MiB)
    result={'schema_version':1,'probe':'bounded_original_artifact_direct_IO','status':'pass','small_original_tensors':results,
        'original_input_bytes':sum(t['source_tensor']['bytes'] for t in results),'integrity_block_bytes':BLOCK,
        'host_cap_bytes':budget.cap,'host_peak_reserved_bytes':budget.peak,'host_reserved_after_close':budget.reserved,
        'system_available_floor_bytes':budget.available_floor,'reservation_is_RSS_limit':False,
        'full_shards_registered':0,'full_checkpoint_downloaded':False,'weight_payload_bytes_fetched':0,
        'scope':'Direct-I/O registration and requested-range integrity for six original small tensors; full-shard registration not run'}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_bytes((json.dumps(result,indent=2)+'\n').encode())
    print(json.dumps({'status':'pass','tensors':len(results),'ranges':sum(len(t['ranges']) for t in results),'reserved_after_close':budget.reserved}))


if __name__=='__main__': main()
