"""Native sampling corpus replay, atomic RNG recovery and negative controls."""

from pathlib import Path
from unittest.mock import patch

import numpy as np

from .attention_validation import array_digest,check_arrays
from .cuda import compile_probe
from .memory import RuntimeCancelled
from .sampling_reference import sampling_logits


def compare_golden(actual,golden):
    checks=[]
    for g in golden['checks']:
        name=g['check']; a=actual[name]; sha=array_digest(a)
        checks.append({'check':name,'status':'pass' if sha==g['expected_sha256'] and list(a.shape)==g['shape'] and str(a.dtype)==g['dtype'] else 'fail',
            'elements':a.size,'mismatches':0 if sha==g['expected_sha256'] else a.size,'dtype':str(a.dtype),'shape':list(a.shape),'expected_sha256':g['expected_sha256'],'actual_sha256':sha})
    if any(c['status']!='pass' for c in checks): raise ValueError('Native sampler differs from original complete typed/shaped goldens')
    if (actual['offset_before'],actual['offset_after'])!=(golden['expected_offset_before'],golden['expected_offset_after']):
        raise ValueError('Native Philox advancement differs from original')
    return checks


def fault_matrix(native,policy,goldens):
    golden=next(g for g in goldens if g['meta']=={'seed':0,'batch':1,'pattern':'flat','temperature':1.})
    x=sampling_logits(golden['meta']); faults=[]; retries=[]
    for mode in ('host_pressure','third_GPU_allocation','probability_download','cancel_after_scores'):
        native.manual_seed(0); before=dict(native.device.allocations); gpu=dict(policy.device_used)
        acquire,allocate,download,launch=native.budget.acquire,native.device.allocate,native.device.download,native._launch
        calls=0
        def fail_allocate(size):
            nonlocal calls
            calls+=1
            if calls==3: raise MemoryError('Injected third sampling allocation failure')
            return allocate(size)
        def fail_acquire(*a,**k): raise MemoryError('Injected sampling reserve pressure')
        def fail_download(*a,**k): raise OSError('Injected probability readback failure')
        def cancel(name,*a,**k):
            result=launch(name,*a,**k)
            if name=='sample_scores': policy.cancel()
            return result
        if mode=='host_pressure': native.budget.acquire=fail_acquire
        elif mode=='third_GPU_allocation': native.device.allocate=fail_allocate
        elif mode=='probability_download': native.device.download=fail_download
        else: native._launch=cancel
        caught=None
        try: native.sample(x,1.)
        except (MemoryError,OSError,RuntimeCancelled) as e: caught=type(e).__name__
        finally: native.budget.acquire,native.device.allocate,native.device.download,native._launch=acquire,allocate,download,launch
        if caught is None or native.offset or native.device.allocations!=before or policy.device_used!=gpu:
            raise RuntimeError('Failed sampling changed committed RNG state or leaked a buffer')
        policy.resume(); actual=native.sample(x,1.); checks=compare_golden(actual,golden)
        retries.append({'meta':golden['meta'],'fault':mode,'checks':checks,'status':'pass',
            'expected_offset_before':0,'expected_offset_after':4,'actual_offset_before':actual['offset_before'],'actual_offset_after':actual['offset_after']})
        faults.append({'name':mode,'status':'pass','error_type':caught,'committed_seed_offset_unchanged':True,
            'temporary_buffers_released':True,'retry_all_original_boundaries_equal':True})
    # Compare the second original draw, then restore its precise starting state.
    second=next(g for g in goldens if g['meta']=={'seed':0,'batch':1,'pattern':'flat','temperature':.7})
    handle=native.snapshot(); first=native.sample(x,.7); compare_golden(first,second); native.rollback(handle)
    again=native.sample(x,.7); compare_golden(again,second)
    if any(not np.array_equal(first[k].view(np.uint8),again[k].view(np.uint8)) for k in first if isinstance(first[k],np.ndarray)):
        raise RuntimeError('Sampler snapshot did not reproduce original draw bits')
    return faults,retries,{'name':'whole_sampler_snapshot','status':'pass','original_second_draw_reproduced':True,
        'offset_before':4,'offset_after':8,'output_sha256':array_digest(again['ids_i64']),
        'all_boundary_sha256':{k:array_digest(v) for k,v in again.items() if isinstance(v,np.ndarray)}}


def precise_log_negative(native,oracle_noise):
    """More accurate logf is still a different operation than the original __logf."""
    read=Path.read_bytes
    def changed(path):
        raw=read(path)
        return raw.replace(b':__logf(u)',b':logf(u)') if path.name=='sampling_probe.cu' else raw
    info={}
    with patch.object(Path,'read_bytes',changed): cubin=compile_probe('89',sampling=True,compiler_info=info)
    old_module,seed,offset=native.module,native.seed,native.offset
    try:
        native.module=native.device.load(cubin); native.manual_seed(0)
        actual=native.sample(sampling_logits({'batch':1,'pattern':'flat'}),1.)
        row=check_arrays(oracle_noise,actual['exponential_fp32'],'replace_original_fast_log_with_logf'); row['candidate_compiler']=info
        if row['status']!='fail' or row['mismatches']<=0: raise RuntimeError('Changed-log sampling control was not detected')
        return row
    finally: native.module=old_module; native.seed,native.offset=seed,offset
