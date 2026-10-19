"""Original numerical replay and allocator/cancellation gates under shared caps."""

from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

from .attention import read_bounded
from .attention_validation import array_digest
from .cuda import CudaError,Pinned
from .engram_corpus import hidden
from .engram_validation import open_session,exercise,STAGES,validate_goldens
from .memory import OwnedBuffer,RuntimeCancelled
from .model_artifacts import digest
from .ngram import LAYERS
from .resources import MemoryBudget,MiB


def replay_case(golden,inputs,actual):
    if {str(l):array_digest(a) for l,a in inputs.items()}!=golden['input_sha256']: raise ValueError('Governed native input differs')
    checks=[]
    for l in LAYERS:
        for name in STAGES:
            a=actual['layers'][l][name]; g=golden['boundaries'][len(checks)]; sha=array_digest(a)
            passed=(sha==g['expected_sha256'] and str(a.dtype)==g['dtype'] and list(a.shape)==g['shape'])
            checks.append({'check':g['check'],'status':'pass' if passed else 'fail','shape':list(a.shape),'dtype':str(a.dtype),
                'elements':a.size,'expected_sha256':g['expected_sha256'],'actual_sha256':sha})
    for name in ('hash_ids','compressed','lookbacks'):
        if array_digest(actual[name])!=golden[name+'_sha256']: raise ValueError('Governed native addresses differ')
    if actual['state_sha256']!=golden['committed_state_sha256']: raise ValueError('Governed native history differs')
    if any(c['status']!='pass' for c in checks): raise ValueError('Governed original Engram arithmetic differs')
    return {'status':'pass','boundaries':checks}


def governed_engram(policy,source,fixtures,reference_report):
    raw=read_bounded(reference_report,32*MiB); reference=json.loads(raw); goldens=validate_goldens(reference)
    with ExitStack() as stack:
        session,manifests,tokenizer=open_session(stack,source,fixtures)
        profiles={str(l):{'compiler':o.compiler,'cubin_sha256':o.cubin_sha256} for l,o in session.operators.items()}
        if manifests!=reference['fixtures'] or profiles!=reference['native_profiles'] or tokenizer!=reference['plan']['tokenizer']:
            raise ValueError('Governed original source/fixture/compiler identity differs')
        def checker(meta,inputs,mask,actual):
            golden=goldens[meta['id']]
            if meta!=golden['meta']: raise ValueError('Governed replay case differs')
            return replay_case(golden,inputs,actual)
        cases,faults,snapshots,minimum=exercise(session,reference['plan'],checker)
        by_id={c['id']:c for c in reference['plan']['hash_cases']}; extra=[]; extra_steps=[]
        def step(case_id,key):
            case=by_id[case_id]; ids=np.array(case['ids'],np.int64)
            inputs={l:hidden(case['stream_index'],case['position'],ids.shape[0],ids.shape[1],l) for l in LAYERS}
            actual=session.step(ids,inputs,None,case['position'])
            record={'id':key,'source_golden':'repeat0/'+case_id,**replay_case(goldens['repeat0/'+case_id],inputs,actual)}
            extra_steps.append(record)
        for mode in ('cancel_after_layer1_output','cancel_after_layer14_output','GPU_allocator_failure'):
            session.hash.reset(); step('english/decode/0',mode+'/warm')
            state=session.hash.state_sha256(); allocations={l:dict(o.device.allocations) for l,o in session.operators.items()}
            gpu=dict(policy.device_used); o1,o14=[session.operators[l] for l in LAYERS]
            first,last,allocate=o1.forward,o14.forward,o1.driver.cuMemAlloc_v2
            def cancelled_result(function,*a,**k):
                result=function(*a,**k); policy.cancel()
                try: policy.resume()
                except RuntimeError: pass
                else: raise RuntimeError('Cancellation was cleared while native work was still active')
                return result
            def cancel_first(*a,**k): return cancelled_result(first,*a,**k)
            def cancel_last(*a,**k): return cancelled_result(last,*a,**k)
            def fail(*a): raise CudaError('Injected GPU allocator failure before backend allocation')
            if mode=='cancel_after_layer1_output': o1.forward=cancel_first
            elif mode=='cancel_after_layer14_output': o14.forward=cancel_last
            else: o1.driver.cuMemAlloc_v2=fail
            caught=None
            try:
                ids=np.array(by_id['english/decode/1']['ids'],np.int64)
                session.step(ids,{l:hidden(0,1,1,1,l) for l in LAYERS},None,1)
            except (RuntimeCancelled,CudaError) as error: caught=type(error).__name__
            finally: o1.forward,o14.forward,o1.driver.cuMemAlloc_v2=first,last,allocate
            if (caught is None or session.hash.state_sha256()!=state or policy.device_used!=gpu
                    or any(session.operators[l].device.allocations!=allocations[l] for l in LAYERS)):
                raise RuntimeError('Cancelled/failed native work changed history or leaked buffers')
            policy.resume(); step('english/decode/1',mode+'/retry')
            extra.append({'name':mode,'status':'pass','error_type':caught,'history_unchanged':True,'temporary_GPU_buffers_released':True,
                'retry_all_boundaries_equal':True,'resume_while_inflight_refused':mode.startswith('cancel_')})
        before=policy.host
        try:
            with patch('vesper.memory.np.empty',side_effect=MemoryError('Injected host allocator failure')):
                OwnedBuffer(policy,MiB,'scratch','injected_host_allocator')
        except MemoryError: pass
        else: raise RuntimeError('Injected host allocator failure was not detected')
        if policy.host!=before: raise RuntimeError('Host allocation failure leaked an admission')
        extra.append({'name':'host_allocator_failure','status':'pass','no_allocation_published':True})
        budget=MemoryBudget(cap=4*MiB,owner='injected_pinned_allocator'); before=policy.host; pinned_before=policy.pinned
        original=o1.driver.cuMemHostAlloc
        try:
            o1.driver.cuMemHostAlloc=lambda *args:2
            try: Pinned(o1.driver,MiB,budget)
            except CudaError: pass
            else: raise RuntimeError('Injected pinned allocator failure was not detected')
        finally: o1.driver.cuMemHostAlloc=original
        if policy.host!=before or policy.pinned!=pinned_before or budget.reserved: raise RuntimeError('Pinned failure leaked ownership')
        extra.append({'name':'pinned_allocator_failure','status':'pass','no_allocation_published':True})
        policy.sample('real_Engram_with_model_scale_state')
    frameworks={n+'_'+s:(importlib.util.find_spec(n) is not None if s=='installed' else any(m==n or m.startswith(n+'.') for m in sys.modules))
        for n in ('torch','tilelang','transformers','sympy') for s in ('installed','imported')}
    if any(frameworks.values()): raise RuntimeError('Native governed replay requires its independent framework-free environment')
    return {'schema_version':1,'probe':'governed_original_Engram_parity','status':'pass','reference_report_sha256':digest(raw),
        'qualified_identity_sha256':reference['source_fixture_compiler_identity_sha256'],'complete_goldens_sha256':reference['complete_goldens_sha256'],
        'native_profiles':profiles,'session_cases':cases,'faults':faults,'snapshots':snapshots,'extra_faults':extra,'extra_steps':extra_steps,
        'minimum_sampled_available_bytes':minimum,**frameworks,'full_model_quality':'not_run','end_to_end_speed':'not_measured'}
