"""Small deterministic token/history cases, separate from full-model inputs."""

import numpy as np
from .formats import bf16_bits

TEXTS=("The moon is bright.", " THE the The ", "ＡＢＣ café 日本語🙂", "\t \r\n", "e\u0301 É é", "<｜begin▁of▁sentence｜>hello<｜end▁of▁sentence｜>")


def streams(tokenizer):
    english=tokenizer.encode(TEXTS[0])
    mixed=tokenizer.encode(TEXTS[2])[:6]
    return [
        {"id":"english","ids":[english],"mask":None},
        {"id":"special-and-image","ids":[[0,2,129279,129264,671,17408,16,2]],"mask":[[True,True,True,False,False,True,True,True]]},
        {"id":"unicode","ids":[mixed],"mask":[[i!=2 for i in range(len(mixed))]]},
        {"id":"batch2","ids":[[671,17408,344,11533,16,2],[0,2,129279,129264,671,16]],
            "mask":[[True,True,False,True,True,True],[True,False,True,False,True,True]]},
        {"id":"all-masked","ids":[[129264]*4],"mask":[[False]*4]},
    ]


def chunks(length, kind):
    if kind=="decode": return [1]*length
    result=[]
    while length:
        size=min(length,3 if len(result)%2==0 else 2); result.append(size); length-=size
    return result


def hidden(stream_index, start, batch, sequence, layer):
    # Independent of chunk partition, retaining signed zeros and finite BF16s.
    rows=[]
    for b in range(batch):
        rows.append(np.stack([np.random.default_rng(881+stream_index*1009+b*17+(start+s)*31+layer).normal(0,.5,(4,5120)).astype(np.float32)
            for s in range(sequence)]))
    result=bf16_bits(np.stack(rows))
    result[...,0]=0; result[...,1]=0x8000
    return np.ascontiguousarray(result)


def row_plan(source):
    from .tokenizer import PinnedTokenizer,compressed_token_map
    from .ngram import NgramState,LAYERS
    from .checkpoint import MODEL,REVISION
    from .model_artifacts import CATALOG_SHA256
    tokenizer=PinnedTokenizer(source); mapping,identity=compressed_token_map(tokenizer)
    state=NgramState(mapping); cases=[]; rows={str(l):set() for l in LAYERS}
    for i,stream in enumerate(streams(tokenizer)):
        ids=np.array(stream['ids'],np.int64); masks=None if stream['mask'] is None else np.array(stream['mask'],np.bool_)
        for kind in ('decode','chunk'):
            state.reset(); position=0
            for count in chunks(ids.shape[1],kind):
                x=np.ascontiguousarray(ids[:,position:position+count]); mask=None if masks is None else np.ascontiguousarray(masks[:,position:position+count])
                candidate=state.prepare(x,position,mask); state.commit(candidate)
                for k,l in enumerate(LAYERS): rows[str(l)].update(map(int,candidate.hash_ids[:,:,k,:].reshape(-1)))
                cases.append({'id':stream['id']+'/'+kind+'/'+str(position),'stream_index':i,'position':position,'ids':x.tolist(),
                    'mask':None if mask is None else mask.tolist(),'hash_ids':candidate.hash_ids.tolist(),
                    'compressed':candidate.compressed.tolist(),'lookbacks':candidate.lookbacks.tolist(),
                    'committed_history':state.history.tolist(),'state_sha256':state.state_sha256()})
                position+=count
    return {'schema_version':1,'probe':'bounded_original_Engram_row_plan','model':MODEL,'revision':REVISION,'catalog_sha256':CATALOG_SHA256,
        'tokenizer':identity,'hash_cases':cases,'required_rows':{k:sorted(v) for k,v in rows.items()},'maximum_rows_per_layer':1024,
        'hash_state_history_tokens':3,'full_table_downloaded':False,'original_row_payload_bytes':sum(len(v)*264 for v in rows.values())}
