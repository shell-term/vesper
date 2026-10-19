"""Replay the complete original Engram corpus with NumPy and CUDA Driver only."""

import argparse
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from vesper.attention import read_bounded
from vesper.attention_validation import array_digest
from vesper.engram_validation import open_session,exercise,STAGES,validate_goldens,validate_faults,validate_memory
from vesper.model_artifacts import digest
from vesper.ngram import LAYERS
from vesper.resources import MiB,GiB


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('reference-report','metadata','fixtures','output'): parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    raw=read_bounded(args.reference_report,32*MiB); reference=json.loads(raw); goldens=validate_goldens(reference)
    # Authenticate the complete selected-row manifests before opening any GPU.
    for l in LAYERS:
        row=json.loads(read_bounded(args.fixtures/'rows'/f'layer{l}'/'manifest.json',MiB))
        if row!=reference['fixtures'][str(l)]['rows']: raise ValueError('Original selected-row fixture differs')
    with ExitStack() as stack:
        session,fixtures,tokenizer=open_session(stack,args.metadata,args.fixtures)
        profiles={str(l):{'compiler':o.compiler,'cubin_sha256':o.cubin_sha256} for l,o in session.operators.items()}
        if fixtures!=reference['fixtures'] or profiles!=reference['native_profiles'] or tokenizer!=reference['plan']['tokenizer']:
            raise ValueError('Original Engram source/fixture/compiler identity differs')
        def checker(meta,inputs,mask,actual):
            golden=goldens[meta['id']]
            if meta!=golden['meta'] or {str(l):array_digest(a) for l,a in inputs.items()}!=golden['input_sha256']:
                raise ValueError('Engram replay input identity differs')
            checks=[]
            for l in LAYERS:
                for n in STAGES:
                    a=actual['layers'][l][n]; expected=golden['boundaries'][len(checks)]
                    sha=array_digest(a)
                    passed=sha==expected['expected_sha256'] and str(a.dtype)==expected['dtype'] and list(a.shape)==expected['shape']
                    checks.append({'check':f'layer{l}/{n}','status':'pass' if passed else 'fail','elements':a.size,
                        'dtype':str(a.dtype),'shape':list(a.shape),'expected_sha256':expected['expected_sha256'],'actual_sha256':sha})
            for name in ('hash_ids','compressed','lookbacks'):
                if array_digest(actual[name])!=golden[name+'_sha256']: raise ValueError('Engram replay address identity differs')
            if actual['state_sha256']!=golden['committed_state_sha256']: raise ValueError('Engram replay history differs')
            return {'status':'pass' if all(c['status']=='pass' for c in checks) else 'fail','boundaries':checks}
        cases,faults,snapshots,minimum=exercise(session,reference['plan'],checker)
        active={'session':session.stats(),'operators':{str(l):o.stats() for l,o in session.operators.items()},
            'rows':{str(l):o.rows.stats() for l,o in session.operators.items()}}
    cleanup={'session':session.stats(),'operators':{str(l):o.stats() for l,o in session.operators.items()},
        'rows':{str(l):o.rows.stats() for l,o in session.operators.items()}}
    memory={'active':active,'cleanup':cleanup,'minimum_sampled_available_bytes':minimum,'native_host_allowances_bytes':2096*MiB,
        'system_available_floor_bytes':32*GiB,'reservation_is_RSS_limit':False}
    validate_faults(faults,snapshots); validate_memory(memory,reference=False)
    frameworks={n+'_installed':importlib.util.find_spec(n) is not None for n in ('torch','tilelang','transformers','sympy')}
    frameworks.update({n+'_imported':any(m==n or m.startswith(n+'.') for m in sys.modules) for n in ('torch','tilelang','transformers','sympy')})
    if any(frameworks[n+'_imported'] for n in ('torch','tilelang','transformers','sympy')): raise RuntimeError('Native replay imported an original framework')
    result={'schema_version':1,'probe':'native_Engram_complete_recheck','status':'pass','reference_report_sha256':digest(raw),
        'qualified_identity_sha256':reference['source_fixture_compiler_identity_sha256'],'complete_goldens_sha256':reference['complete_goldens_sha256'],
        'native_profiles':profiles,**frameworks,'session_cases':cases,'faults':faults,'snapshots':snapshots,'memory':memory,
        'weight_payload_bytes_fetched_during_recheck':0,'full_model_quality':'not_run','end_to_end_speed':'not_measured'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_bytes((json.dumps(result,indent=2,allow_nan=False)+'\n').encode())
    print(json.dumps({'status':'pass','cases':len(cases),'boundaries':sum(len(c['boundaries']) for c in cases),**frameworks}))


if __name__=='__main__': main()
