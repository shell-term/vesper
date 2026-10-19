"""Unchanged original full-shape Engram methods and separately checked traces."""

import ast
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .attention_storage_reference import _numpy
from .checkpoint import REVISION
from .engram_assets import load_controls
from .engram_rows import EngramRows
from .model_artifacts import read_checked,digest
from .ngram import LAYERS,TABLE_ROWS
from .prefix_reference import fetch_prefix_reference
from .reference_runtime import configure_reference_runtime


class EngramOracle:
    def __init__(self,reference,source,fixtures,virtual,ordinal=0):
        import torch
        import tilelang
        if torch.get_default_dtype()!=torch.bfloat16:
            raise ValueError('Original Engram linear requires the released BF16 default dtype')
        if torch.__version__!='2.14.1+cu130' or torch.version.git_version!='5c4886908584029761b579af026dcfb627c84070' or tilelang.__version__!='0.1.12':
            raise ValueError('Unqualified original Engram reference framework profile')
        self.torch=torch; self.ordinal=ordinal; self.virtual=virtual; self.controls=[]; self.methods={}
        self.source=fetch_prefix_reference(Path(reference))
        raw=read_checked(source,'inference/engram.py')
        self.engram_source_sha256=digest(raw)
        spec=importlib.util.spec_from_file_location('vesper_original_engram_kernel',Path(reference)/'kernel.py')
        self.kernel=importlib.util.module_from_spec(spec); spec.loader.exec_module(self.kernel)
        self.workarounds=configure_reference_runtime(self.kernel,tilelang.__version__)
        self.scope={'torch':torch,'F':torch.nn.functional,'world_size':1,'fp8_block_size':32,'scale_fmt':'ue8m0',
            'scale_dtype':torch.float8_e8m0fnu,'act_quant':self.kernel.act_quant,'fp8_gemm':self.kernel.fp8_gemm,'fp4_gemm':self.kernel.fp4_gemm}
        self.tree=ast.parse((Path(reference)/'model.py').read_bytes())
        node=next(n for n in self.tree.body if isinstance(n,ast.FunctionDef) and n.name=='linear')
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<pinned-engram-linear>','exec'),self.scope)
        self.methods['linear']=hashlib.sha256(ast.dump(node).encode()).hexdigest()
        self.embed_forward=self.method('ParallelEngramEmbedding','forward'); self.forward=self.method('Engram','forward')
        self.weights={}; self.embeds={}; self.manifests={}; self.transport=[]; self.allowed={}
        for layer,total in zip(LAYERS,TABLE_ROWS):
            arrays,m=load_controls(Path(fixtures)/f'controls{layer}',layer); self.manifests[str(layer)]={'controls':m}
            weights={n:torch.from_numpy(a.copy()).to(f'cuda:{ordinal}').view(torch.bfloat16 if a.dtype==np.uint16 else
                torch.float8_e8m0fnu if n.endswith('scale') else torch.float8_e4m3fn) for n,a in arrays.items()}
            weights['wkv.weight'].scale=weights['wkv.scale']; self.weights[layer]=weights
            # Original addresses and raw full shapes, with only observed pages backed.
            with EngramRows(Path(fixtures)/'rows'/f'layer{layer}',layer) as rows:
                self.manifests[str(layer)]['rows']=rows.manifest
                self.allowed[layer]=set(rows.entries)
                codes,scales={},{}
                for row,e in rows.entries.items():
                    page=rows.reader.read((e['offset']//4096)*4096,4096); raw=page[e['offset']%4096:e['offset']%4096+264]
                    if digest(raw)!=e['sha256']: raise ValueError('Original oracle row differs before VMM admission')
                    codes[row]=raw[:256]; scales[row]=raw[256:]
                weight=virtual.table((total,256),codes).view(torch.float8_e4m3fn)
                scale=virtual.table((total,8),scales).view(torch.float8_e8m0fnu)
                ids=sorted(codes); index=torch.tensor(ids,dtype=torch.int64,device=f'cuda:{ordinal}')
                for table,values in ((weight,codes),(scale,scales)):
                    actual=torch.nn.functional.embedding(index,table).view(torch.uint8).cpu().numpy().tobytes()
                    expected=b''.join(values[i] for i in ids)
                    if actual!=expected: raise RuntimeError('Full-address original row bit transport differs')
                    self.transport.append({'layer':layer,'rows':len(ids),'bytes':len(expected),'sha256':digest(expected),'status':'pass'})
                self.embeds[layer]=SimpleNamespace(vocab_start_idx=0,vocab_end_idx=total,block_size=32,weight=weight,scale=scale)
        self.fp32=self.kernel.fp8_gemm_kernel(25600,6144,block_size=32,out_dtype='float32',scale_dtype='float8_e8m0fnu')

    def method(self,cls,name):
        parent=next(n for n in self.tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
        node=next(n for n in parent.body if isinstance(n,ast.FunctionDef) and n.name==name)
        self.methods[cls+'.'+name]=hashlib.sha256(ast.dump(node).encode()).hexdigest()
        local=dict(self.scope); exec(compile(ast.Module(body=[node],type_ignores=[]),'<pinned-engram-method>','exec'),local)
        return local[name]

    def step(self,layer,hidden,ids,mask=None):
        if layer not in self.allowed or any(int(i) not in self.allowed[layer] for i in ids.reshape(-1)):
            raise ValueError('Unbacked original reference row refused before GPU access')
        torch=self.torch; device=f'cuda:{self.ordinal}'
        x=torch.from_numpy(hidden.copy()).to(device).view(torch.bfloat16)
        indices=torch.from_numpy(ids.copy()).to(device); token_mask=None if mask is None else torch.from_numpy(mask.copy()).to(device)
        w=self.weights[layer]; embedding=lambda i:self.embed_forward(self.embeds[layer],i)
        linear=lambda x:self.scope['linear'](x,w['wkv.weight'])
        stub=SimpleNamespace(embed=embedding,wkv=linear,hc_mult=4,dim=5120,eps=1e-20,clamp_value=1e-6,q_weight=w['q_weight'],k_weight=w['k_weight'])
        unchanged=self.forward(stub,x,indices,token_mask)
        codes=torch.nn.functional.embedding(indices,self.embeds[layer].weight)
        scales=torch.nn.functional.embedding(indices,self.embeds[layer].scale)
        embedded=(codes.float().unflatten(-1,(-1,32))*scales.float().unsqueeze(-1)).flatten(-2).flatten(-2)
        bf16=embedding(indices).flatten(-2)
        if not torch.equal(bf16.view(torch.uint16),embedded.bfloat16().view(torch.uint16)): raise RuntimeError('Original embedding forward control differs')
        q,s=self.kernel.act_quant(bf16,32,'ue8m0',torch.float8_e8m0fnu)
        kv=linear(bf16); trace=torch.empty(kv.shape,dtype=torch.float32,device=device)
        self.fp32(q.view(-1,6144),w['wkv.weight'],trace.view(-1,25600),s.view(-1,192),w['wkv.scale'])
        if not torch.equal(trace.bfloat16().view(torch.uint16),kv.view(torch.uint16)): raise RuntimeError('Original FP32 projection diagnostic differs')
        key,value=kv.split([20480,5120],dim=-1); key=key.float().unflatten(-1,(4,5120)); h=x.float()
        weight=w['q_weight'].float()*w['k_weight'].float(); hw=h*weight; product=hw*key
        hs=h.square(); ks=key.square(); hm=hs.mean(-1); km=ks.mean(-1); he=hm+1e-20; ke=km+1e-20
        hr=torch.rsqrt(he); kr=torch.rsqrt(ke); rstd=hr*kr; sums=product.sum(-1); normalized=sums*rstd; dot=normalized*5120**-.5
        clamped=dot.abs().clamp_min(1e-6); root=clamped.sqrt(); signed=torch.copysign(root,dot); sigmoid=torch.sigmoid(signed)
        gate=sigmoid if token_mask is None else sigmoid.masked_fill(~token_mask.unsqueeze(-1),0)
        delta=gate.unsqueeze(-1)*value.float().unsqueeze(-2); residual=h+delta; output=residual.bfloat16()
        if not torch.equal(output.view(torch.uint16),unchanged.view(torch.uint16)): raise RuntimeError('Unchanged original Engram.forward control differs')
        self.controls.append({'layer':layer,'status':'pass','embedding_forward_bitwise':True,'projection_FP32_rounds_to_original_linear':True,
            'Engram_forward_bitwise':True,'output_sha256':digest(_numpy(unchanged).tobytes())})
        tensors={'rows_codes_fp8':codes.view(torch.uint8),'rows_scales_e8m0':scales.view(torch.uint8),
            'embedding_fp32':embedded,'embedding_bf16':bf16,'projection_input_fp8':q.view(torch.uint8),'projection_scale_e8m0':s.view(torch.uint8),
            'projection_output_fp32':trace,'projection_output_bf16':kv,'hidden_square_fp32':hs,'key_square_fp32':ks,
            'weight_product_fp32':weight,'hidden_weight_fp32':hw,'dot_product_fp32':product,'hidden_mean_fp32':hm,'key_mean_fp32':km,'dot_sum_fp32':sums,
            'hidden_mean_eps_fp32':he,'key_mean_eps_fp32':ke,'hidden_rstd_fp32':hr,'key_rstd_fp32':kr,'rstd_fp32':rstd,'normalized_dot_fp32':normalized,
            'dot_fp32':dot,'clamped_abs_dot_fp32':clamped,'sqrt_abs_dot_fp32':root,'signed_sqrt_fp32':signed,'sigmoid_fp32':sigmoid,'gate_fp32':gate,
            'delta_fp32':delta,'residual_fp32':residual,'output_bf16':output}
        return {n:_numpy(a) for n,a in tensors.items()}


def negative_controls(oracle,source,plan):
    from .attention_validation import check_arrays
    from .engram_corpus import hidden
    from .ngram import NgramState
    from .tokenizer import PinnedTokenizer,compressed_token_map
    torch=oracle.torch; device=f'cuda:{oracle.ordinal}'; by_id={c['id']:c for c in plan['hash_cases']}
    def tensor(a): return torch.from_numpy(a.copy()).to(device)
    def evaluate(case,zero=False):
        ids=np.array(case['hash_ids'],np.int64); x=hidden(case['stream_index'],case['position'],ids.shape[0],ids.shape[1],1)
        if zero: x.fill(0); x[...,1]=0x8000
        mask=None if case['mask'] is None else np.array(case['mask'],np.bool_)
        return oracle.step(1,x,np.ascontiguousarray(ids[:,:,0,:]),mask)
    expected=evaluate(by_id['english/decode/1']); controls=[]
    hm=tensor(expected['hidden_square_fp32']).mean((-1,-2)); km=tensor(expected['key_square_fp32']).mean((-1,-2))
    wrong_dot=tensor(expected['dot_sum_fp32'])*(torch.rsqrt(hm+1e-20)*torch.rsqrt(km+1e-20)).unsqueeze(-1)*5120**-.5
    wrong=torch.sigmoid(torch.copysign(wrong_dot.abs().clamp_min(1e-6).sqrt(),wrong_dot))
    controls.append(check_arrays(expected['sigmoid_fp32'],_numpy(wrong),'normalize_across_HC_copies'))
    controls.append(check_arrays(expected['sigmoid_fp32'],_numpy(torch.sigmoid(tensor(expected['dot_fp32']))),'remove_signed_sqrt'))
    unscaled=tensor(expected['rows_codes_fp8']).view(torch.float8_e4m3fn).float().flatten(-2)
    controls.append(check_arrays(expected['embedding_fp32'],_numpy(unscaled),'discard_original_row_scales'))
    zero=evaluate(by_id['english/decode/0'],True); dot=tensor(zero['dot_fp32'])
    controls.append(check_arrays(zero['sigmoid_fp32'],_numpy(torch.sigmoid(torch.copysign(dot.abs().sqrt(),dot))),'drop_signed_sqrt_clamp_at_zero'))
    masked=evaluate(by_id['special-and-image/decode/3'])
    controls.append(check_arrays(masked['gate_fp32'],masked['sigmoid_fp32'],'omit_token_mask'))
    mapping,_=compressed_token_map(PinnedTokenizer(source))
    for name,ignore_mask in (('ignore_masked_history',True),('clear_history_between_chunks',False)):
        state=NgramState(mapping); correct=[]; changed=[]
        for case in plan['hash_cases']:
            if case['position']==0 or not ignore_mask: state.reset()
            ids=np.array(case['ids'],np.int64); mask=None if ignore_mask or case['mask'] is None else np.array(case['mask'],np.bool_)
            step=state.prepare(ids,None,mask); state.commit(step)
            correct.append(np.array(case['hash_ids'],np.int64).reshape(-1)); changed.append(step.hash_ids.reshape(-1))
        controls.append(check_arrays(np.concatenate(correct),np.concatenate(changed),name))
    if any(c['status']!='fail' or c['mismatches']<=0 for c in controls): raise RuntimeError('Changed Engram arithmetic/history control was not detected')
    return controls


def compare_engram(source,fixtures,reference,progress=None):
    from contextlib import ExitStack
    from .engram_hash_reference import compare_hashes
    from .engram_reference_storage import VirtualTables
    from .engram_validation import open_session,exercise,compare_outputs,report_identity,golden_digest
    from .resources import MemoryBudget,GiB
    from .checkpoint import MODEL
    import torch
    previous=torch.get_default_dtype(); torch.set_default_dtype(torch.bfloat16)
    try:
        with ExitStack() as stack,torch.inference_mode():
            budget=MemoryBudget(cap=2*GiB); budget.acquire(budget.cap); stack.callback(budget.release,budget.cap)
            hashes=compare_hashes(source); virtual=stack.enter_context(VirtualTables())
            oracle=EngramOracle(reference,source,fixtures,virtual)
            session,manifests,token_identity=open_session(stack,source,fixtures)
            def checker(meta,inputs,mask,actual):
                expected={l:oracle.step(l,inputs[l],actual['hash_ids'][:,:,i,:].copy(),mask) for i,l in enumerate(LAYERS)}
                checks=compare_outputs(expected,actual)
                for check in checks:
                    layer,name=check['check'].split('/'); check['shape']=list(expected[int(layer[5:])][name].shape)
                if progress and any(c['status']!='pass' for c in checks): progress({'case':meta,'failed_checks':[c for c in checks if c['status']!='pass']})
                return {'status':'pass' if all(c['status']=='pass' for c in checks) else 'fail','boundaries':checks}
            cases,faults,snapshots,minimum=exercise(session,hashes['plan'],checker,progress)
            controls=list(oracle.controls)  # Negative-sensitivity probes are separate from the successful matrix.
            negatives=negative_controls(oracle,source,hashes['plan'])
            active={'session':session.stats(),'operators':{str(l):o.stats() for l,o in session.operators.items()},
                'rows':{str(l):o.rows.stats() for l,o in session.operators.items()},'VMM':virtual.stats()}
            result={'schema_version':1,'probe':'original_Engram_complete_parity','status':'pass','model':MODEL,'revision':REVISION,
                'plan':hashes['plan'],'fixtures':manifests,'native_profiles':{str(l):{'compiler':o.compiler,'cubin_sha256':o.cubin_sha256} for l,o in session.operators.items()},
                'hash_reference':hashes,'original_source':oracle.source,'original_methods':oracle.methods,'VMM_transport':oracle.transport,
                'session_cases':cases,'faults':faults,'snapshots':snapshots,'unchanged_original_controls':controls,'negative_controls':negatives,
                'memory':{'active':active,'minimum_sampled_available_bytes':minimum,'native_host_allowances_bytes':2096*1024**2,
                    'reference_host_allowance_bytes':budget.cap,'system_available_floor_bytes':32*GiB,'reservation_is_RSS_limit':False},
                'reference_runtime_workarounds':oracle.workarounds,'reference_default_dtype':'bfloat16',
                'scope':'Independent original Engram layers 1/14; generated HC hidden inputs; original tokenization, hashes, masks and three-token carried history; batch1/2, chunks1/2/3, context<=8, world1',
                'full_model_quality':'not_run','end_to_end_speed':'not_measured','weight_payload_bytes_fetched_during_comparison':0}
        result['memory']['cleanup']={'session':session.stats(),'operators':{str(l):o.stats() for l,o in session.operators.items()},
            'rows':{str(l):o.rows.stats() for l,o in session.operators.items()},'VMM':virtual.stats()}
        result['source_fixture_compiler_identity_sha256']=report_identity(result); result['complete_goldens_sha256']=golden_digest(result)
        return result
    finally: torch.set_default_dtype(previous)
