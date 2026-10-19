"""Original full-backbone allocation geometry for resource-only scale tests."""

import json
import re

from .checkpoint import MODEL,REVISION
from .model_artifacts import load_catalog,read_checked,CATALOG_SHA256,canonical
from .resources import MiB


def full_backbone_plan(source,*,context=131072,batch=1):
    if type(context) is not int or context not in (65536,131072,1048576) or type(batch) is not int or batch not in (1,2):
        raise ValueError('Expected an explicit representative original context/batch')
    catalog,identity=load_catalog(source); config=json.loads(read_checked(source,'inference/config.json'))
    experts={}
    for name,t in catalog.items():
        match=re.fullmatch(r'layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)',name)
        if match:
            layer,expert=map(int,match.group(1,2)); experts.setdefault((layer,expert),[]).append(t)
    if (len(experts)!=40*384 or set(experts)!={(l,e) for l in range(40) for e in range(384)}
            or any(len(ts)!=6 or sum(t['bytes'] for t in ts)!=18800640 for ts in experts.values())):
        raise ValueError('Original complete-backbone expert inventory differs')
    blocks=[{'layer':l,'expert':e,'bytes':sum(t['bytes'] for t in ts),'source_tensor_names':sorted(t['name'] for t in ts)}
        for (l,e),ts in sorted(experts.items())]
    kv=[]
    for layer in range(40):
        kv.append({'owner':f'window.{layer}','shape':[batch,128,512],'dtype':'uint16','bytes':batch*128*512*2})
    for layer in config['kv_source_layers']:
        ratio=config['compress_ratios'][layer]; count=context//ratio
        kv.extend([{'owner':f'compressed_KV.{layer}','shape':[batch,count,512],'dtype':'uint16','bytes':batch*count*512*2},
            {'owner':f'index_keys.{layer}','shape':[batch,count,128],'dtype':'uint16','bytes':batch*count*128*2}])
        if ratio==2:
            for name in ('pool_values','pool_scores'):
                kv.append({'owner':f'{name}.{layer}','shape':[batch,2,512],'dtype':'float32','bytes':batch*2*512*4})
    # These buffers retain sharing-source state, never one duplicate per consumer.
    kv.append({'owner':'source20_candidates','shape':[batch,(context+7)//8],'dtype':'bool','bytes':batch*((context+7)//8)})
    nonexpert={l:max((t for n,t in catalog.items() if n.startswith(f'layers.{l}.') and '.ffn.experts.' not in n and '.engram.embed.' not in n),
        key=lambda t:t['bytes']) for l in range(40)}
    result={'schema_version':1,'probe':'original_backbone_memory_geometry','model':MODEL,'revision':REVISION,
        'catalog_sha256':CATALOG_SHA256,'catalog_tensor_count':len(catalog),'original_tensor_bytes':identity['original_tensor_bytes'],
        'context':context,'batch':batch,'layers':40,'experts_per_layer':384,'expert_count':len(blocks),'expert_blocks':blocks,
        'expert_bytes':18800640,'all_expert_tensor_bytes':sum(b['bytes'] for b in blocks),'KV_allocations':kv,
        'KV_bytes':sum(t['bytes'] for t in kv),'engram_layers':[1,14],'engram_history_tokens':3,
        'largest_nonexpert_tensor':max((t for n,t in catalog.items() if '.ffn.experts.' not in n and '.engram.embed.' not in n),key=lambda t:t['bytes']),
        'layer_nonexpert_working_weights':{str(l):t for l,t in nonexpert.items()},'retained_metadata_admission_bytes':512*MiB,
        'scope':'All backbone expert identities and original allocation shapes; resource tests use synthetic payload bytes, not model inference or full-checkpoint loading'}
    result['plan_sha256']=canonical(result); return result
