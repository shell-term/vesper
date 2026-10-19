"""Original full-vocabulary head methods and bounded independent replay inputs."""

import ast
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .attention_storage_reference import _numpy
from .attention_validation import check_arrays,array_digest
from .checkpoint import MODEL,REVISION
from .hc import NativeHC
from .memory import RuntimeMemory,MemoryPolicy
from .normalization import NativeNormRoPE,NORM_STAGES,NORM_EPS
from .output import NativeOutput
from .output_assets import load
from .output_corpus import cases,inputs
from .prefix import load_prefix_fixture
from .prefix_reference import fetch_prefix_reference
from .model_artifacts import read_checked
from .resources import MemoryBudget,GiB,physical_memory
from .sampling import NativeSampler
from .sampling_reference import original_sample


class OutputOracle:
    def __init__(self,reference,arrays,ordinal=0):
        import torch
        self.torch=torch; self.ordinal=ordinal; self.source=fetch_prefix_reference(Path(reference))
        self.scope={'torch':torch,'F':torch.nn.functional,'world_size':1}; self.methods={}
        tree=ast.parse((Path(reference)/'model.py').read_bytes())
        for parent,name in (('Block','hc_pre'),('RMSNorm','forward'),('ParallelHead','forward'),(None,'sample')):
            nodes=tree.body if parent is None else next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==parent).body
            node=next(n for n in nodes if isinstance(n,ast.FunctionDef) and n.name==name)
            key=(parent+'.' if parent else '')+name; self.methods[key]=hashlib.sha256(ast.dump(node).encode()).hexdigest()
            exec(compile(ast.Module(body=[node],type_ignores=[]),'<pinned-output-method>','exec'),self.scope)
            setattr(self,key.replace('.','_'),self.scope[name])
        self.weight=torch.from_numpy(arrays['head.weight']).view(torch.bfloat16).to(f'cuda:{ordinal}',dtype=torch.float32)
        self.norm=torch.from_numpy(arrays['norm.weight']).view(torch.bfloat16).to(f'cuda:{ordinal}')

    def evaluate(self,x,pre):
        torch=self.torch; device=f'cuda:{self.ordinal}'
        x=torch.from_numpy(x).view(torch.bfloat16).to(device); pre=torch.from_numpy(pre).to(device)
        products=pre.unsqueeze(-1)*x.float(); sums=torch.sum(products,dim=2)
        collapsed=self.Block_hc_pre(None,x,pre)
        if not torch.equal(sums.bfloat16().view(torch.uint16),collapsed.view(torch.uint16)): raise RuntimeError('Original collapse diagnostics differ')
        output=self.RMSNorm_forward(SimpleNamespace(weight=self.norm,eps=NORM_EPS),collapsed)
        square=collapsed.float().square(); mean=square.mean(-1,keepdim=True); shifted=mean+NORM_EPS
        rstd=torch.rsqrt(shifted); normalized=collapsed.float()*rstd; weighted=self.norm*normalized
        if not torch.equal(weighted.bfloat16().view(torch.uint16),output.view(torch.uint16)): raise RuntimeError('Original final norm diagnostics differ')
        logits=self.ParallelHead_forward(SimpleNamespace(weight=self.weight),output)
        ids=self.sample(logits,0.)
        result={'pre_products_fp32':products,'pre_sum_fp32':sums,'pre_output_bf16':collapsed,
            **{'norm_'+n:v for n,v in zip(NORM_STAGES,(square,mean,shifted,rstd,normalized,weighted,output))},
            'promoted_fp32':output[:,-1].float(),'logits_fp32':logits,'greedy_ids_i64':ids}
        return {n:_numpy(v) for n,v in result.items()}


def compare_output(source,fixtures,prefix,reference,progress=None):
    import torch
    if torch.__version__!='2.14.1+cu130' or torch.version.git_version!='5c4886908584029761b579af026dcfb627c84070':
        raise ValueError('Unregistered original output reference framework')
    read_checked(source,'inference/config.json')
    previous=torch.get_default_dtype(); torch.set_default_dtype(torch.bfloat16)
    rows=[]; minimum=physical_memory()['available']
    try:
        # Explicit small comparison envelopes; the default full-runtime ceiling
        # stays 96 GiB. Optional-framework overhead is observed separately.
        with RuntimeMemory(MemoryPolicy(host_cap_bytes=16*GiB)) as policy,ExitStack() as stack,torch.inference_mode():
            fixture_budget=MemoryBudget(cap=2*GiB,owner='output_reference_fixture'); fixture_budget.acquire(fixture_budget.cap)
            stack.callback(fixture_budget.release,fixture_budget.cap)
            arrays,manifest=load(fixtures); old,_=load_prefix_fixture(prefix)
            stack.callback(arrays.clear); stack.callback(old.clear)
            oracle=OutputOracle(reference,arrays)
            native=stack.enter_context(NativeOutput(arrays['head.weight']))
            hc=stack.enter_context(NativeHC(old)); norm=stack.enter_context(NativeNormRoPE()); sampler=stack.enter_context(NativeSampler())
            for index,c in enumerate(cases()):
                x,pre=inputs(c,arrays['embedding_rows']); expected=oracle.evaluate(x,pre)
                collapsed=hc.pre(x,pre); normalized=norm.norm(collapsed['output_bf16'],arrays['norm.weight'])
                actual={**{'pre_'+n:v for n,v in collapsed.items()},**{'norm_'+n:v for n,v in normalized.items()},
                    **native.project(normalized['output_bf16'][:,-1].copy())}
                checks=[]
                for name in expected:
                    check=check_arrays(expected[name],actual[name],name); check['shape']=list(expected[name].shape); checks.append(check)
                sampling=[]
                for temperature in (0.,.7):
                    seed=2026+index; torch.cuda.manual_seed(seed); sampler.manual_seed(seed)
                    want,before,after=original_sample(torch,oracle.sample,expected['logits_fp32'],temperature)
                    sampled=sampler.sample(actual['logits_fp32'],temperature); sample_checks=[]
                    for name,value in want.items():
                        check=check_arrays(value,sampled[name],name); check['shape']=list(value.shape); sample_checks.append(check)
                    if (sampled['offset_before'],sampled['offset_after'])!=(before,after): raise RuntimeError('Connected output/sampler RNG differs')
                    sampling.append({'temperature':temperature,'seed':seed,'offset_before':before,'offset_after':after,'checks':sample_checks,
                        'status':'pass' if all(v['status']=='pass' for v in sample_checks) else 'fail'})
                row={'meta':c,'input_sha256':array_digest(x),'incoming_pre_sha256':array_digest(pre),'checks':checks,
                    'sampling':sampling,'status':'pass' if all(v['status']=='pass' for v in checks+sampling) else 'fail'}
                rows.append(row); minimum=min(minimum,physical_memory()['available'])
                if progress: progress({'case':c['id'],'status':row['status'],'failures':[v for v in checks if v['status']!='pass']})
            negatives=[]
            # Reducing the original FP32 head to a BF16 GEMM must be detected.
            # Use a nonzero input rather than the final all-zero corpus condition.
            c=cases()[2]; hx,pp=inputs(c,arrays['embedding_rows']); gold=oracle.evaluate(hx,pp)
            bad_input=torch.from_numpy(gold['norm_output_bf16']).view(torch.bfloat16).to('cuda:0')[:,-1]
            bad=torch.nn.functional.linear(bad_input,oracle.weight.bfloat16()).float()
            negative=check_arrays(gold['logits_fp32'],_numpy(bad),'replace_original_FP32_head_with_BF16_GEMM')
            if negative['status']!='fail' or not negative['mismatches']: raise RuntimeError('Reduced-precision head control was not detected')
            negatives.append(negative); del bad,bad_input
            active=native.stats(); source_identity=oracle.source; methods=oracle.methods
            policy.sample('after_original_output_comparison'); observed=policy.snapshot()
            del oracle
        return {'schema_version':1,'probe':'original_output_complete_parity','status':'pass' if all(r['status']=='pass' for r in rows) else 'fail',
            'model':MODEL,'revision':REVISION,'fixture':manifest,'original_source':source_identity,'original_methods':methods,
            'cases':rows,'negative_controls':negatives,'native_active':active,'native_cleanup':native.stats(),'governor_before_close':observed,'governor_after_close':policy.snapshot(),
            'minimum_sampled_available_bytes':minimum,'temperature':0.,'full_vocabulary':129280,'head_storage_dtype':'bfloat16','head_compute_dtype':'float32',
            'torch_version':torch.__version__,'torch_revision':torch.version.git_version,
            'scope':'Original full output matrix and final norm, generated HC or selected original embedding inputs; no preceding forty-layer execution',
            'stochastic_sampling':'original_full_logits_temperatures_0_and_0.7','full_model_quality':'not_run','text_generation_available':False,'end_to_end_speed':'not_measured'}
    finally: torch.set_default_dtype(previous)
