"""Original sampling controls, all probabilities/noise/scores and RNG offsets."""

import ast
from contextlib import ExitStack
import hashlib
from pathlib import Path
import urllib.request

import numpy as np

from .attention_storage_reference import _numpy
from .attention_validation import array_digest,check_arrays
from .checkpoint import MODEL,REVISION
from .memory import RuntimeMemory,MemoryPolicy
from .prefix_reference import fetch_prefix_reference
from .sampling import NativeSampler
from .resources import GiB

TORCH_REVISION='5c4886908584029761b579af026dcfb627c84070'
PYTORCH_SOURCES={
    'SoftMax.cu':('aten/src/ATen/native/cuda/SoftMax.cu','17bf86e61f8c0f0bc2a37eed314bf412a1992826141bdcb6d9202c5066765b90'),
    'DistributionTemplates.h':('aten/src/ATen/native/cuda/DistributionTemplates.h','ffc06850b55eb836e38936d56f15da4f65763c91567aee6d224e49b97bf17514'),
    'TransformationHelper.h':('aten/src/ATen/core/TransformationHelper.h','adbb8d572ae7bea53e491622733e617cf8a08a4476b00604ec54f5525c8f6cd2'),
    'Distributions.cu':('aten/src/ATen/native/cuda/Distributions.cu','cef5ee8525ae2fd2e6a68eca69eeaf1b0a479cc8b7668c4d81f8cd92acf18719'),
    'block_reduce.cuh':('aten/src/ATen/native/cuda/block_reduce.cuh','31ce6e510f68f23de06647114cf4eb4face8812778866b4d23e471b138c07bae'),
    'NumericUtils.h':('aten/src/ATen/NumericUtils.h','5574c6007151c46f48b149c80fcbc4036800afa011989887807e18668f8e9a37')}


def reference_sources(reference):
    root=Path(reference)/'pytorch-sampling'; root.mkdir(exist_ok=True); result={}
    for name,(upstream,expected) in PYTORCH_SOURCES.items():
        path=root/name
        if path.exists():
            if path.stat().st_size>256*1024: raise ValueError('Excessive original sampling source')
            raw=path.read_bytes()
        else:
            with urllib.request.urlopen(f'https://raw.githubusercontent.com/pytorch/pytorch/{TORCH_REVISION}/{upstream}',timeout=45) as response:
                raw=response.read(256*1024+1)
            if len(raw)>256*1024 or hashlib.sha256(raw).hexdigest()!=expected: raise ValueError('Original sampling CUDA source differs')
            with path.open('xb') as file: file.write(raw)
        if hashlib.sha256(raw).hexdigest()!=expected: raise ValueError('Original sampling CUDA source differs')
        result[name]={'upstream':upstream,'sha256':expected,'bytes':len(raw)}
    return {'revision':TORCH_REVISION,'files':result}


def sampling_cases():
    return [{'seed':seed,'batch':batch,'pattern':pattern,'temperature':temperature}
        for seed in (0,2026,2**64-1) for batch in (1,2) for pattern in ('flat','banded','large')
        for temperature in (0.,1.,.7,1e-6,2.)]


def sampling_logits(meta):
    indices=np.arange(meta['batch']*129280,dtype=np.int64).reshape(meta['batch'],129280)
    if meta['pattern']=='flat': return np.zeros(indices.shape,np.float32)
    if meta['pattern']=='banded': return np.ascontiguousarray(((indices*37%2049)-1024).astype(np.float32)/64)
    return np.ascontiguousarray((indices%129280-64640).astype(np.float32)/4)


def original_sample(torch,function,values,temperature):
    logits=torch.from_numpy(values).to('cuda:0'); generator=torch.cuda.default_generators[0]
    before=torch.cuda.get_rng_state(); offset_before=generator.get_offset()
    selected=function(logits,temperature); after=torch.cuda.get_rng_state(); offset_after=generator.get_offset()
    torch.cuda.set_rng_state(before); expected={'ids_i64':_numpy(selected)}
    if temperature!=0:
        scaled=logits/max(temperature,1e-5); probabilities=torch.softmax(scaled,dim=-1,dtype=torch.float32)
        noise=torch.empty_like(probabilities).exponential_(1); scores=probabilities/noise
        if not torch.equal(scores.argmax(-1),selected): raise RuntimeError('Original sampling diagnostics changed the selected token')
        expected.update(scaled_fp32=_numpy(scaled),probabilities_fp32=_numpy(probabilities),exponential_fp32=_numpy(noise),scores_fp32=_numpy(scores))
    if not torch.equal(torch.cuda.get_rng_state(),after): raise RuntimeError('Original sampling diagnostics changed generator advancement')
    return expected,offset_before,offset_after


def compare_sampling(reference,progress=None):
    import torch
    if torch.__version__!='2.14.1+cu130' or torch.version.git_version!='5c4886908584029761b579af026dcfb627c84070':
        raise ValueError('Unregistered original sampling reference framework')
    source=fetch_prefix_reference(Path(reference)); cuda_source=reference_sources(reference); tree=ast.parse((Path(reference)/'model.py').read_bytes())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='sample'); scope={'torch':torch}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<pinned-original-sampling>','exec'),scope)
    rows=[]
    with RuntimeMemory(MemoryPolicy(host_cap_bytes=2*GiB)) as policy,ExitStack() as stack,torch.inference_mode():
        torch.cuda.init()
        native=stack.enter_context(NativeSampler()); generator=torch.cuda.default_generators[0]
        key=None; negative_noise=None
        for meta in sampling_cases():
            stream=(meta['seed'],meta['batch'])
            if stream!=key:
                torch.cuda.manual_seed(meta['seed']); native.manual_seed(meta['seed']); key=stream
            values=sampling_logits(meta); expected,offset_before,offset_after=original_sample(torch,scope['sample'],values,meta['temperature'])
            if meta=={'seed':0,'batch':1,'pattern':'flat','temperature':1.}: negative_noise=expected['exponential_fp32'].copy()
            actual=native.sample(values,meta['temperature']); checks=[]
            for name,want in expected.items():
                check=check_arrays(want,actual[name],name); check['shape']=list(want.shape); checks.append(check)
            matching_offsets=actual['offset_before']==offset_before and actual['offset_after']==offset_after
            row={'meta':meta,'input_sha256':array_digest(values),'status':'pass' if matching_offsets and all(v['status']=='pass' for v in checks) else 'fail',
                'checks':checks,'expected_offset_before':offset_before,'expected_offset_after':offset_after,
                'actual_offset_before':actual['offset_before'],'actual_offset_after':actual['offset_after']}
            rows.append(row)
            if progress: progress({'case':meta,'status':row['status'],'offsets_match':matching_offsets,'failures':[c for c in checks if c['status']!='pass']})
        from .sampling_validation import fault_matrix,precise_log_negative
        faults,retries,snapshots=fault_matrix(native,policy,rows)
        negatives=[precise_log_negative(native,negative_noise)]
        compiler,cubin=native.compiler,native.cubin_sha256
        profile={'multiprocessors':native.multiprocessors,'threads_per_sm':native.threads_per_sm}
        policy.sample('after_complete_original_sampling'); observed=policy.snapshot()
    return {'schema_version':1,'probe':'original_full_vocabulary_sampling','status':'pass' if all(c['status']=='pass' for c in rows) else 'fail',
        'model':MODEL,'revision':REVISION,'original_source':source,'sample_AST_sha256':hashlib.sha256(ast.dump(node).encode()).hexdigest(),
        'torch_version':torch.__version__,'torch_revision':torch.version.git_version,'pytorch_sources':cuda_source,
        'cases':rows,'compiler':compiler,'cubin_sha256':cubin,
        'device_profile':profile,'faults':faults,'retries':retries,'snapshots':snapshots,'negative_controls':negatives,
        'governor_before_close':observed,'governor_after_close':policy.snapshot(),
        'full_model_quality':'not_run','text_generation_available':False,'end_to_end_speed':'not_measured',
        'scope':'Complete 129280-token stochastic/greedy sampling on generated finite full-vocabulary logits; original RNG state/offset transitions; no model forward'}
