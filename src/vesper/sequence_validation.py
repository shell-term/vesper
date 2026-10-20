"""Longer input-address/Engram gate; explicitly not whole-model generation."""

from contextlib import ExitStack
import gc
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from .attention_validation import array_digest
from .checkpoint import MODEL, REVISION
from .engram import NativeEngram
from .engram_assets import load_controls, expected_row_address
from .engram_rows import EngramRows
from .engram_session import EngramSession
from .engram_validation import STAGES
from .formats import bf16_bits
from .generation_assets import embedding_address
from .model_artifacts import canonical, digest, load_catalog
from .ngram import NgramState
from .resources import GiB, MemoryBudget
from .sequence import MAX_CALLS, MAX_ROWS, scope as profile_scope
from .sequence_assets import request
from .tokenizer import PinnedTokenizer, compressed_token_map

QUALIFIED_IDENTITY = "64a1c54c777885661cb174f08b4257342c9e5c3c392929b4cd1ac4bc1e43e976"
QUALIFIED_GOLDENS = "e5a193ae67e448e146bc9219ab8afdf2b1e7fea70defb90e09cc073aa4e3c7e6"


def scope():
    return {**profile_scope(), "model": MODEL, "revision": REVISION, "case_id": "reasoning-integer", "prompt_tokens": 63,
        "Engram_layers": [1,14], "healthy_prefix_calls": 63, "original_rows_per_layer": 1496,
        "original_input_embedding_rows": 49, "hidden_inputs": "synthetic_finite_BF16_with_signed_zeros",
        "reference_row_transport": "paged_original_virtual_addresses", "reference_physical_row_cap_bytes":512*1024**2,
        "original_Ngram_hash_run": True, "original_full_address_embedding_run": True, "original_Engram_forward_run": True,
        "model_inference_run": False, "quality_scored": False, "whole_backbone_63_call_comparison_run": False,
        "generation_or_sampling_run": False, "speed": "not_measured"}


def read_report(path, limit=16*1024**2):
    path=Path(path)
    if not 1<=path.stat().st_size<=limit: raise ValueError("Excessive bounded-sequence metadata")
    with path.open('rb') as f:raw=f.read(limit+1)
    if len(raw)>limit:raise ValueError("Sequence metadata grew beyond its bound")
    return json.loads(raw),raw


def plan_identity(plan):
    return canonical({k:plan[k] for k in ('model','revision','scope','case','request','prompt_token_ids','original_hash_ids','hash_state','tokenizer','assets')})


def load_input(root,run):
    root,run=Path(root),Path(run)
    if not run.resolve().is_relative_to((root/'sequence').resolve()):raise ValueError("Expected an anchored bounded-sequence input")
    plan,_=read_report(run/'input-plan.json')
    if plan.get('plan_sha256')!=canonical({k:v for k,v in plan.items() if k!='plan_sha256'}):raise ValueError("Sequence input plan checksum differs")
    case,req=request('reasoning-integer')
    if (plan.get('probe'),plan.get('status'),plan.get('model'),plan.get('revision'),plan.get('scope'),plan.get('case'),plan.get('request'),
            plan.get('model_inference_run'),plan.get('quality_scored'),plan.get('new_expert_weight_payloads_fetched'))!=(
            'bounded_sequence_input_plan','pass',MODEL,REVISION,profile_scope(),case,
            {'prompt':req.prompt,'max_new_tokens':req.max_new_tokens,'eos_id':req.eos_id},False,False,0):raise ValueError("Different bounded input scope")
    ids=plan['prompt_token_ids']
    if len(ids)!=63 or any(type(i) is not int or not 0<=i<129280 for i in ids):raise ValueError("Different complete prompt tokens")
    hashes=np.asarray(plan['original_hash_ids'],np.int64)
    if hashes.shape!=(1,63,2,24):raise ValueError("Incomplete original input hashes")
    catalog,_=load_catalog(root/'import-source');rows={}
    entries=plan['assets']['embeddings']
    if [e['id'] for e in entries]!=sorted(set(ids)) or len(entries)!=49:raise ValueError("Incomplete original input embedding set")
    for e in entries:
        token=e['id']
        if e['file']!=f'embeddings/{token}.bin' or e['source']!=embedding_address(catalog,token):raise ValueError("Original input embedding address differs")
        p=run/e['file']
        if p.stat().st_size!=10240:raise ValueError("Original input embedding extent differs")
        raw=p.read_bytes()
        if digest(raw)!=e['sha256'] or len(raw)!=10240:raise ValueError("Original input embedding bytes differ")
        a=np.frombuffer(raw,np.uint16)
        if np.any((a&0x7f80)==0x7f80):raise ValueError("Nonfinite original input embedding")
        rows[token]=a
    for i,l in enumerate((1,14)):
        m,_=read_report(run/f'rows/layer{l}/manifest.json',limit=12*1024**2)
        if (m!=plan['assets']['Engram_rows'][str(l)] or m.get('maximum_rows')!=MAX_ROWS or len(m['rows'])!=1496
                or [e['row'] for e in m['rows']]!=sorted(set(map(int,hashes[:,:,i,:].reshape(-1))))
                or any(e['source']!=expected_row_address(l,e['row']) for e in m['rows'])):raise ValueError("Long original Engram row binding differs")
    return plan,hashes,rows


def hidden(position,layer):
    a=((np.arange(20480,dtype=np.int64)*17+position*31+layer*13)%127-63).astype(np.float32)/np.float32(64)
    result=bf16_bits(a.reshape(1,1,4,5120));result[...,0]=0;result[...,1]=0x8000
    return np.ascontiguousarray(result)


def fingerprint(result):
    if set(result)!={1,14} or any(set(result[l])!=set(STAGES) for l in (1,14)):raise ValueError("Incomplete actual Engram math boundaries")
    return [{'check':f'layer{l}/{name}','elements':int(a.size),'shape':list(a.shape),'dtype':str(a.dtype),'sha256':array_digest(a)}
        for l in (1,14) for name in STAGES for a in (result[l][name],)]


def identity(report):
    return canonical({k:report[k] for k in ('scope','input_plan_sha256','assets','original_source','original_methods','original_ngram_source_sha256',
        'original_input_embedding','tokenizer','Engram_controls_manifests','original_row_transport')})


def capture_reference(root,run,progress=None):
    import torch
    from .attention_storage_reference import _numpy
    from .engram_reference import EngramOracle
    from .paged_tables import PagedVirtualTables
    from .model_artifacts import read_checked
    from .token_prefix_reference import original_hash
    root,run=Path(root),Path(run)
    previous=torch.get_default_dtype();torch.set_default_dtype(torch.bfloat16)
    try:
        with ExitStack() as stack:
            allowance=MemoryBudget(cap=2*GiB,owner='sequence_original_fixtures_framework');allowance.acquire(allowance.cap)
            stack.callback(allowance.release,allowance.cap)
            http=stack.enter_context(patch('urllib.request.urlopen',side_effect=RuntimeError('Sequence reference must remain offline')))
            plan,expected_hashes,rows=load_input(root,run)
            if torch.cuda.is_initialized():raise ValueError("Expected fresh reference process before the independent CPU hash")
            from transformers import AutoTokenizer
            tokenizer=AutoTokenizer.from_pretrained(str(root/'import-source'),local_files_only=True,trust_remote_code=False)
            if tokenizer.encode(plan['request']['prompt'])!=plan['prompt_token_ids']:raise ValueError("Independent original prompt tokenization differs")
            h=original_hash(root/'import-source',max_seq_len=MAX_CALLS);hashes=[];histories=[]
            if digest(h.token_map.numpy().tobytes())!=plan['tokenizer']['compressed_map_sha256']:raise ValueError("Independent original complete compressed vocabulary differs")
            for pos,token in enumerate(plan['prompt_token_ids']):
                hashes.append(h(torch.tensor([[token]],dtype=torch.int64,device='cpu'),pos).numpy().copy())
                histories.append(h.cache[:1,max(0,pos-2):pos+1].numpy().copy().tolist())
            hashes=np.concatenate(hashes,axis=1)
            if not np.array_equal(hashes,expected_hashes):raise ValueError("Independent longer original Ngram hashes differ")
            if torch.cuda.is_initialized():raise ValueError("Original input hashing unexpectedly initialized CUDA")
            baseline=0;torch.cuda.init();baseline=torch.cuda.memory_allocated(0);torch.cuda.reset_peak_memory_stats(0)
            with torch.inference_mode(),torch.cuda.device(0),torch.device('cuda:0'):
                virtual=stack.enter_context(PagedVirtualTables(0,max_rows=MAX_ROWS))
                oracle=EngramOracle(root/'reference',root/'import-source',root/'engram-fixture',virtual,rows_root=run/'rows',max_rows=MAX_ROWS,paged_rows=True)
                table=virtual.table((129280,10240),{i:a.tobytes() for i,a in rows.items()}).view(torch.bfloat16)
                stub=SimpleNamespace(weight=table,vocab_start_idx=0,vocab_end_idx=129280)
                embed=oracle.method('ParallelEmbedding','forward')(stub,torch.tensor([plan['prompt_token_ids']],dtype=torch.int64))
                original_embedding=_numpy(embed).copy()
                expected_embedding=np.stack([rows[i] for i in plan['prompt_token_ids']])[None,:,:]
                if not np.array_equal(original_embedding,expected_embedding):raise ValueError("Original full-address input embedding differs")
                cases=[]
                for pos,token in enumerate(plan['prompt_token_ids']):
                    values={l:oracle.step(l,hidden(pos,l),np.ascontiguousarray(hashes[:,pos:pos+1,i,:])) for i,l in enumerate((1,14))}
                    cases.append({'position':pos,'token_id':token,'hash_ids':hashes[:,pos:pos+1].tolist(),'history':histories[pos],
                        'hash_state_sha256':canonical({'position':pos+1,'history':histories[pos]}),
                        'hidden_input_sha256':{str(l):array_digest(hidden(pos,l)) for l in (1,14)},'boundaries':fingerprint(values)})
                    if progress and (pos%8==7 or pos==62):progress({'original_prefix':pos+1,'boundaries':sum(len(c['boundaries']) for c in cases)})
                report={'schema_version':1,'probe':'original_extended_input_Engram_goldens','status':'pass','scope':scope(),
                    'input_plan_sha256':plan_identity(plan),'assets':plan['assets'],'original_source':oracle.source,'original_methods':oracle.methods,
                    'original_ngram_source_sha256':digest(read_checked(root/'import-source','inference/engram.py')),
                    'original_input_embedding':{'dtype':str(original_embedding.dtype),'shape':list(original_embedding.shape),'elements':int(original_embedding.size),'sha256':array_digest(original_embedding)},
                    'tokenizer':plan['tokenizer'],'Engram_controls_manifests':{l:m['controls'] for l,m in oracle.manifests.items()},
                    'original_controls':oracle.controls,'original_row_transport':virtual.stats(),'cases':cases,
                    'CPU_hash_initialized_CUDA':False,'python_http_disabled_during_execution':True,'HTTP_calls_during_execution':http.call_count,
                    'model_weight_payloads_fetched_in_reference':0}
                report['identity_sha256']=identity(report);report['complete_goldens_sha256']=canonical(cases)
                del oracle,table,stub,embed
        gc.collect();torch.cuda.synchronize();torch._C._cuda_clearCublasWorkspaces();gc.collect();torch.cuda.empty_cache()
        report['reference_device_peak_allocated_bytes']=torch.cuda.max_memory_allocated(0)
        report['reference_device_allocated_baseline_bytes']=baseline
        report['reference_device_allocated_after_cleanup_bytes']=torch.cuda.memory_allocated(0)
        report['original_row_transport_after_close']=virtual.stats()
        if torch.cuda.memory_allocated(0)!=baseline:raise RuntimeError('Reference retained GPU tensors after extended-row execution')
        return report
    finally:torch.set_default_dtype(previous)


def validate_reference(report,*,require_qualified=True):
    if (report.get('probe'),report.get('status'),report.get('scope'))!=('original_extended_input_Engram_goldens','pass',scope()):raise ValueError("Different extended component scope")
    if report['identity_sha256']!=identity(report) or report['complete_goldens_sha256']!=canonical(report['cases']):raise ValueError("Changed original extended goldens")
    if require_qualified and (not QUALIFIED_IDENTITY or (report['identity_sha256'],report['complete_goldens_sha256'])!=(QUALIFIED_IDENTITY,QUALIFIED_GOLDENS)):
        raise ValueError("Extended original capture has not been independently qualified")
    if len(report['cases'])!=63 or [c['position'] for c in report['cases']]!=list(range(63)):
        raise ValueError("Incomplete original longer prefix")
    if len(report['original_controls'])!=126 or any(c['status']!='pass' for c in report['original_controls']):raise ValueError("Missing original Engram execution controls")
    for c in report['cases']:
        expected=[f'layer{l}/{n}' for l in (1,14) for n in STAGES]
        if [b['check'] for b in c['boundaries']]!=expected or any(b['elements']!=int(np.prod(b['shape'])) for b in c['boundaries']):raise ValueError("Missing actual Engram math boundary")
        if c['hash_state_sha256']!=canonical({'position':c['position']+1,'history':c['history']}):raise ValueError("Original longer-prefix hash history differs")
    if (report['CPU_hash_initialized_CUDA'] is not False or report['python_http_disabled_during_execution'] is not True or report['HTTP_calls_during_execution']
            or report['model_weight_payloads_fetched_in_reference'] or report['reference_device_allocated_after_cleanup_bytes']!=report['reference_device_allocated_baseline_bytes']):
        raise ValueError("Original execution fetched payloads, initialized CUDA during CPU hashing or leaked tensors")
    after=report['original_row_transport_after_close']
    if after['physical_owned_bytes'] or after['virtual_reserved_bytes'] or after['owned_context_active']:raise ValueError("Original virtual rows were not fully released")
    transport=report['original_row_transport']
    if (transport['physical_cap_bytes']!=512*1024**2 or transport['peak_physical_owned_bytes']>512*1024**2
            or transport['maximum_active_rows_per_table']!=24 or transport['table_close_callbacks']!=4
            or sum(t['mapped_physical_pages'] for t in transport['tables'])*transport['page_bytes']!=transport['physical_owned_bytes']
            or after['total_pages_mapped']!=after['total_pages_evicted'] or any(t['mapped_physical_pages'] for t in after['tables'])):
        raise ValueError("Original row residency or page/callback cleanup is not bounded")
    from .generation_validation import validate_memory
    validate_memory(report,8*GiB)
    return True


def run_native(root,run,original,original_raw,progress=None):
    validate_reference(original)
    frameworks={n:{'installed':importlib.util.find_spec(n) is not None,'imported':n in sys.modules} for n in ('torch','tilelang','transformers','sympy')}
    if any(any(v.values()) for v in frameworks.values()):raise ValueError("Expected an independent framework-free native environment")
    root,run=Path(root),Path(run)
    with ExitStack() as stack:
        allowance=MemoryBudget(cap=2*GiB,owner='sequence_native_fixture');allowance.acquire(allowance.cap);stack.callback(allowance.release,allowance.cap)
        http=stack.enter_context(patch('urllib.request.urlopen',side_effect=RuntimeError('Native extended row execution must stay offline')))
        plan,hashes,rows=load_input(root,run)
        if plan_identity(plan)!=original['input_plan_sha256']:raise ValueError("Different actual input plan")
        tokenizer=PinnedTokenizer(root/'import-source');mapping,map_identity=compressed_token_map(tokenizer)
        if map_identity!=original['tokenizer'] or list(request('reasoning-integer')[1].encode(tokenizer))!=plan['prompt_token_ids']:raise ValueError("Different native tokenizer/whole prompt")
        embeddings=np.stack([rows[i] for i in plan['prompt_token_ids']])[None,:,:]
        if array_digest(embeddings)!=original['original_input_embedding']['sha256']:raise ValueError("Original input embeddings differ")
        ops={};row_objects={}
        for l in (1,14):
            arrays,m=load_controls(root/f'engram-fixture/controls{l}',l)
            if m!=original['Engram_controls_manifests'][str(l)]:raise ValueError("Different original Engram controls")
            row_objects[l]=stack.enter_context(EngramRows(run/f'rows/layer{l}',l,max_rows=MAX_ROWS))
            ops[l]=stack.enter_context(NativeEngram(arrays,row_objects[l]))
        session=stack.enter_context(EngramSession(NgramState(mapping),ops));cases=[];fault=None
        for pos,token in enumerate(plan['prompt_token_ids']):
            ids=np.array([[token]],np.int64);inputs={l:hidden(pos,l) for l in (1,14)}
            if pos==32:
                before=session.hash.state_sha256();rows14=row_objects[14];rows14.clear();reader=rows14.reader;actual_read=reader.read
                with patch.object(reader,'read',side_effect=lambda *a:bytes(b^1 for b in actual_read(*a))):
                    try:session.step(ids,inputs)
                    except ValueError as e:
                        if 'SHA-256' not in str(e):raise
                    else:raise RuntimeError("Actual row corruption was accepted")
                if session.hash.state_sha256()!=before:raise RuntimeError("Failed longer-prefix read published hash history")
                fault={'position':pos,'error':'ValueError','guard':'original_row_SHA256','history_unchanged':True,'retry_compared_to_original':True}
            result=session.step(ids,inputs);expected=original['cases'][pos];boundaries=fingerprint(result['layers'])
            if (result['hash_ids'].tolist()!=expected['hash_ids'] or session.hash.inspect()['history'].tolist()!=expected['history']
                    or result['state_sha256']!=expected['hash_state_sha256'] or boundaries!=expected['boundaries']
                    or {str(l):array_digest(inputs[l]) for l in (1,14)}!=expected['hidden_input_sha256']):raise RuntimeError("Longer native hash/history or complete Engram math differs")
            cases.append({'position':pos,'token_id':token,'boundaries':boundaries,'state_sha256':result['state_sha256'],'status':'pass'})
            if progress and (pos%8==7 or pos==62):progress({'native_prefix':pos+1,'boundaries':sum(len(c['boundaries']) for c in cases)})
        report={'schema_version':1,'probe':'native_extended_input_Engram_comparison','status':'pass','scope':scope(),
            'reference_report_sha256':digest(original_raw),'original_identity_sha256':original['identity_sha256'],
            'complete_goldens_sha256':original['complete_goldens_sha256'],'reference_frameworks':frameworks,
            'cases':cases,'fault':fault,'operators_before_close':{str(l):o.stats() for l,o in ops.items()},
            'row_caches_before_close':{str(l):r.stats() for l,r in row_objects.items()},'session_before_close':session.stats(),
            'python_http_disabled_during_execution':True,'HTTP_calls_during_execution':http.call_count,'model_weight_payloads_fetched_in_native':0}
    report['operators_after_close']={str(l):o.stats() for l,o in ops.items()};report['row_caches_after_close']={str(l):r.stats() for l,r in row_objects.items()}
    report['session_after_close']=session.stats()
    del result,arrays,ops,session;gc.collect()
    return report


def validate_native(report,original,raw):
    validate_reference(original)
    if (report.get('probe'),report.get('status'),report.get('scope'),report.get('reference_report_sha256'),report.get('original_identity_sha256'),report.get('complete_goldens_sha256'))!=(
            'native_extended_input_Engram_comparison','pass',scope(),digest(raw),QUALIFIED_IDENTITY,QUALIFIED_GOLDENS):raise ValueError("Different native extended gate or reference binding")
    if len(report['cases'])!=63:raise ValueError("Incomplete native extended prefix")
    for a,b in zip(report['cases'],original['cases']):
        if (a['position'],a['token_id'],a['status'],a['boundaries'],a['state_sha256'])!=(b['position'],b['token_id'],'pass',b['boundaries'],b['hash_state_sha256']):raise ValueError("Native prefix omits original math or hash state")
    if report['fault']!={'position':32,'error':'ValueError','guard':'original_row_SHA256','history_unchanged':True,'retry_compared_to_original':True}:raise ValueError("Missing real row-corruption failure and retry")
    if report['reference_frameworks']!={n:{'installed':False,'imported':False} for n in ('torch','tilelang','transformers','sympy')}:raise ValueError("Native component used a reference framework")
    if report['HTTP_calls_during_execution'] or report['model_weight_payloads_fetched_in_native'] or report['python_http_disabled_during_execution'] is not True:raise ValueError("Native component fetched weights")
    if report['session_after_close']['host_reserved_bytes'] or any(o['host_reserved_bytes'] or o['device_owned_bytes'] or o['owned_context_active'] for o in report['operators_after_close'].values()):raise ValueError("Native component retained owned allocations")
    if any(r['host_reserved_bytes'] or r['cached_rows'] or r['cache_capacity_rows']!=32 for r in report['row_caches_after_close'].values()):raise ValueError("Native row LRU was not bounded/released")
    from .generation_validation import validate_memory
    validate_memory(report,8*GiB)
    return True
