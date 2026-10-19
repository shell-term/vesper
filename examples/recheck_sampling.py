"""Replay complete original sampling probabilities, RNG and failure goldens."""

import argparse
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import sys

from vesper.attention import read_bounded
from vesper.memory import RuntimeMemory,MemoryPolicy
from vesper.model_artifacts import digest
from vesper.output_validation import validate_sampling
from vesper.resources import GiB,MiB
from vesper.sampling import NativeSampler
from vesper.sampling_reference import sampling_logits
from vesper.sampling_validation import compare_golden,fault_matrix


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-report',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    raw=read_bounded(a.reference_report,32*MiB); original=json.loads(raw); goldens=validate_sampling(original); rows=[]
    policy=RuntimeMemory(MemoryPolicy(host_cap_bytes=2*GiB))
    with policy,ExitStack() as stack:
        native=stack.enter_context(NativeSampler()); key=None
        for g in goldens:
            meta=g['meta']; stream=(meta['seed'],meta['batch'])
            if stream!=key: native.manual_seed(meta['seed']); key=stream
            actual=native.sample(sampling_logits(meta),meta['temperature']); checks=compare_golden(actual,g)
            rows.append({'meta':meta,'checks':checks,'status':'pass','expected_offset_before':g['expected_offset_before'],
                'expected_offset_after':g['expected_offset_after'],'actual_offset_before':actual['offset_before'],'actual_offset_after':actual['offset_after']})
        faults,retries,snapshots=fault_matrix(native,policy,goldens); policy.sample('after_sampling_native_replay')
    frameworks={n+'_'+s:(importlib.util.find_spec(n) is not None if s=='installed' else any(m==n or m.startswith(n+'.') for m in sys.modules))
        for n in ('torch','tilelang','transformers','sympy') for s in ('installed','imported')}
    if any(frameworks.values()): raise RuntimeError('Sampler native replay requires the independent framework-free environment')
    result={'schema_version':1,'probe':'native_sampling_complete_recheck','status':'pass','reference_report_sha256':digest(raw),
        'compiler':native.compiler,'cubin_sha256':native.cubin_sha256,'cases':rows,'faults':faults,'retries':retries,'snapshots':snapshots,
        'governor_after_close':policy.snapshot(),**frameworks,'full_model_quality':'not_run','text_generation_available':False,'end_to_end_speed':'not_measured'}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_bytes((json.dumps(result,indent=2)+'\n').encode())
    print(json.dumps({'status':'pass','cases':len(rows),'independent_environment':True}))


if __name__=='__main__': main()
