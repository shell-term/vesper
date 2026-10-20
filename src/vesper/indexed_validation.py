"""Original five-call full-backbone recheck through indexed immutable payloads.

The corpus and original math goldens remain unchanged. This storage variant
does not turn the longer input/Engram gate into a model-quality evaluation.
"""

import json
from pathlib import Path
import re

from .bandwidth import EXPERT_BYTES
from .generation_validation import run_native, validate_native as validate_generation
from .indexed_moe import IndexedMoE
from .indexed_store import build_index, KIND
from .checkpoint import MODEL, REVISION
from .model_artifacts import canonical, digest
from .storage import tensor_layout


def prepare_indices(root,run,progress=None):
    root,run=Path(root),Path(run)
    if not run.resolve().is_relative_to((root/'sequence').resolve()):raise ValueError("Expected a separate indexed-source run")
    run.mkdir(parents=True,exist_ok=True)
    if (run/'indices.json').exists():raise FileExistsError("Refusing to replace prepared original-source indexes")
    prior=json.loads((root/'backbone-plan.json').read_bytes())['experts']
    story=json.loads((root/'generation/story-v1/plan.json').read_bytes())['experts'];reports={}
    for layer in range(40):
        p=story[str(layer)]
        reports[str(layer)]=build_index([root/prior[str(layer)]['store'],root/p['store']],run/f'indices/layer{layer}',root,
            layer=layer,experts=p['ids'])
        if progress:progress({'layer':layer,**reports[str(layer)]})
    result={'schema_version':1,'probe':'indexed_story_preparation','status':'pass','indices':reports,'payload_bytes_copied':0,'weight_payloads_fetched':0}
    result['manifest_sha256']=canonical(result)
    with (run/'indices.json').open('xb') as f:f.write((json.dumps(result,indent=2)+'\n').encode())
    return result


def run_indexed(root,run,original,raw,progress=None):
    root,run=Path(root),Path(run);providers={}
    indices=json.loads((run/'indices.json').read_bytes())
    def factory(root,bindings,layer,prepared,provenance):
        p=IndexedMoE(root,bindings,layer,prepared,provenance,index_directory=run/f'indices/layer{layer}');providers[layer]=p;return p
    report=run_native(root,root/'generation/story-v1',original,progress,provider_factory=factory)
    report['reference_report_sha256']=digest(raw)
    report['indexed_sources']={str(l):p.stats() for l,p in providers.items()}
    for l,r in report['indexed_sources'].items():
        if r['index_sha256']!=indices['indices'][l]['manifest_sha256']:raise ValueError("Executed expert index differs from preparation")
    return report


def validate_indexed(report,original,raw,manifests):
    validate_generation(report,original,raw)
    names=set(map(str,range(40)))
    if set(report.get('indexed_sources',{}))!=names or set(manifests)!=names:raise ValueError("Missing actual all-layer indexed sources")
    for l in map(str,range(40)):
        r,index_raw=report['indexed_sources'][l],manifests[l]
        if type(index_raw) is not bytes or len(index_raw)>4*1024**2 or digest(index_raw)!=r['index_sha256']:raise ValueError("Executed index bytes differ")
        m=json.loads(index_raw)
        if (r['layer'],r['calls'],r['original_experts'],r['direct_read_calls'],r['direct_read_bytes'],r['payload_bytes_copied_to_build_index'],r['expert_payloads_fetched_in_forward'])!=(
                int(l),5,original['expert_plan'][l]['ids'],30,30*EXPERT_BYTES,0,0):raise ValueError("Indexed source execution does not cover every actual original route")
        if not 5<=r['payload_file_opens_and_switches']<=30:raise ValueError("Unbounded or missing actual original-file reads")
        if (m['kind'],m['model'],m['revision'],m['layer'],m['maximum_experts'],m['payload_bytes_copied'],m['weight_payloads_fetched'])!=(
                KIND,MODEL,REVISION,int(l),384,0,0):raise ValueError("Different indexed-source identity/capacity")
        if m['layout']!=tensor_layout() or not 1<=len(m['sources'])<=384:raise ValueError("Unbounded indexed metadata/layout")
        names=[s['directory'] for s in m['sources']]
        if names!=sorted(set(names)) or any(Path(n).is_absolute() or '..' in Path(n).parts or '\\' in n or re.match(r'^[A-Za-z]:',n) for n in names):
            raise ValueError("Unanchored indexed-source path")
        if any(not re.fullmatch(r'[0-9a-f]{64}',s['manifest_sha256']) or type(s['file_bytes']) is not int or not 1<=s['file_bytes']//EXPERT_BYTES<=48
                or s['file_bytes']%EXPERT_BYTES for s in m['sources']):raise ValueError("Invalid packed-source identity/extent")
        if [e['key'] for e in m['entries']]!=[f'{l}:{e}' for e in r['original_experts']]:raise ValueError("Missing actual original expert index entries")
        if any(e['provenance']!=original['expert_provenance'][e['key']] for e in m['entries']):raise ValueError("Indexed original expert provenance differs")
        for e in m['entries']:
            if (type(e['source']) is not int or not 0<=e['source']<len(m['sources']) or type(e['offset']) is not int or e['offset']<0
                    or e['offset']%EXPERT_BYTES or e['offset']+EXPERT_BYTES>m['sources'][e['source']]['file_bytes']
                    or e['bytes']!=EXPERT_BYTES or not re.fullmatch(r'[0-9a-f]{64}',e['sha256'])):raise ValueError("Invalid original expert source address")
    return True
