"""Replay every original output/connected-sampling golden without Torch installed."""

import argparse
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import sys

from vesper.attention import read_bounded
from vesper.attention_validation import array_digest
from vesper.hc import NativeHC
from vesper.memory import RuntimeMemory,MemoryPolicy
from vesper.model_artifacts import digest
from vesper.normalization import NativeNormRoPE
from vesper.output import NativeOutput
from vesper.output_assets import load
from vesper.output_corpus import cases,inputs
from vesper.output_validation import validate_output
from vesper.prefix import load_prefix_fixture
from vesper.resources import MemoryBudget,GiB,MiB
from vesper.sampling import NativeSampler


def check(goldens,actual):
    for g in goldens:
        a=actual[g['check']]
        if str(a.dtype)!=g['dtype'] or list(a.shape)!=g['shape'] or array_digest(a)!=g['expected_sha256']:
            raise ValueError('Original output/connected sampling complete golden differs')
    return [{'check':g['check'],'dtype':g['dtype'],'shape':g['shape'],'elements':g['elements'],
        'expected_sha256':g['expected_sha256'],'actual_sha256':array_digest(actual[g['check']]),'status':'pass'} for g in goldens]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('fixtures','prefix','reference-report','output'): p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args(); raw=read_bounded(a.reference_report,32*MiB); original=json.loads(raw); goldens=validate_output(original); rows=[]
    policy=RuntimeMemory(MemoryPolicy(host_cap_bytes=16*GiB))
    with policy,ExitStack() as stack:
        budget=MemoryBudget(cap=2*GiB,owner='output_replay_fixture'); budget.acquire(budget.cap); stack.callback(budget.release,budget.cap)
        arrays,manifest=load(a.fixtures); old,_=load_prefix_fixture(a.prefix); stack.callback(arrays.clear); stack.callback(old.clear)
        if manifest!=original['fixture']: raise ValueError('Original output fixture differs from reference')
        native=stack.enter_context(NativeOutput(arrays['head.weight'])); hc=stack.enter_context(NativeHC(old)); norm=stack.enter_context(NativeNormRoPE())
        sampler=stack.enter_context(NativeSampler())
        for c in cases():
            x,pre=inputs(c,arrays['embedding_rows']); expected=goldens[c['id']]
            if array_digest(x)!=expected['input_sha256'] or array_digest(pre)!=expected['incoming_pre_sha256']: raise ValueError('Output replay inputs differ')
            collapsed=hc.pre(x,pre); normalized=norm.norm(collapsed['output_bf16'],arrays['norm.weight'])
            actual={**{'pre_'+n:v for n,v in collapsed.items()},**{'norm_'+n:v for n,v in normalized.items()},
                **native.project(normalized['output_bf16'][:,-1].copy())}
            sampling=[]
            for g in expected['sampling']:
                sampler.manual_seed(g['seed']); sampled=sampler.sample(actual['logits_fp32'],g['temperature'])
                if (sampled['offset_before'],sampled['offset_after'])!=(g['offset_before'],g['offset_after']): raise ValueError('Connected output RNG advancement differs')
                sampling.append({'seed':g['seed'],'temperature':g['temperature'],'offset_before':sampled['offset_before'],
                    'offset_after':sampled['offset_after'],'checks':check(g['checks'],sampled),'status':'pass'})
            rows.append({'meta':c,'checks':check(expected['checks'],actual),'sampling':sampling,'status':'pass'})
            print(json.dumps({'case':c['id'],'status':'pass'}),flush=True)
        active=native.stats(); policy.sample('after_output_native_replay')
    frameworks={n+'_'+s:(importlib.util.find_spec(n) is not None if s=='installed' else any(m==n or m.startswith(n+'.') for m in sys.modules))
        for n in ('torch','tilelang','transformers','sympy') for s in ('installed','imported')}
    if any(frameworks.values()): raise RuntimeError('Output native replay requires the independent framework-free environment')
    result={'schema_version':1,'probe':'native_output_complete_recheck','status':'pass','reference_report_sha256':digest(raw),
        'cases':rows,'native_active':active,'native_cleanup':native.stats(),'governor_after_close':policy.snapshot(),**frameworks,
        'full_model_quality':'not_run','text_generation_available':False,'end_to_end_speed':'not_measured'}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_bytes((json.dumps(result,indent=2)+'\n').encode())
    print(json.dumps({'status':'pass','cases':len(rows),'independent_environment':True}))


if __name__=='__main__': main()
