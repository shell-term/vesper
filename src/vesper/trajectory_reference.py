"""Actual unchanged original generation over a prepared longer whole-model run."""

from contextlib import ExitStack
import ast
import gc
from pathlib import Path
from types import SimpleNamespace
from typing import List
from unittest.mock import patch

from .attention_storage_reference import _numpy
from .backbone import LAYERS
from .backbone_assets import load as load_bindings
from .backbone_reference import BackboneOracle
from .evaluation import score_completion
from .expert_bank import ExpertBank, existing_sources
from .generation_validation import record
from .memory import RuntimeMemory, MemoryPolicy
from .model_artifacts import canonical, digest
from .output_assets import load as load_output
from .resources import GiB, MemoryBudget
from .sequence import MAX_CALLS, MAX_ROWS
from .sequence_assets import request
from .tokenizer import PinnedTokenizer
from .trajectory_evidence import load_plan, load_tokens, plan_identity, expert_plan, scope, identity, write_case


def capture_original(root,run,progress=None):
    import torch
    import json
    root,run=Path(root),Path(run);plan=load_plan(root,run);case,req=request(plan['case']['id'])
    previous=torch.get_default_dtype();torch.set_default_dtype(torch.bfloat16)
    try:
        with RuntimeMemory(MemoryPolicy(host_cap_bytes=32*GiB)) as policy:
            torch.cuda.init();baseline=torch.cuda.memory_allocated(0);torch.cuda.reset_peak_memory_stats(0)
            with ExitStack() as stack,torch.inference_mode(),torch.cuda.device(0),torch.device('cuda:0'):
                http=stack.enter_context(patch('urllib.request.urlopen',side_effect=RuntimeError('Original whole-model execution must remain offline')))
                allowance=MemoryBudget(cap=20*GiB,owner='longer_original_fixture_and_framework');allowance.acquire(allowance.cap)
                stack.callback(allowance.release,allowance.cap)
                bindings=json.loads((root/'backbone-controls.json').read_bytes());arrays=load_bindings(root,bindings)
                head_arrays,head_manifest=load_output(root/'output-fixture');stack.callback(arrays.clear);stack.callback(head_arrays.clear)
                if bindings['bindings_sha256']!=plan['bindings_sha256'] or head_manifest!=plan['output_fixture']:
                    raise ValueError('Original whole-model control/output preparation differs')
                bank=stack.enter_context(ExpertBank(root,root/'expert-bank-v1',existing_sources(root)))
                tokens=load_tokens(root,run,plan)
                oracle=BackboneOracle(stack,root,arrays,head_arrays,progress,token_rows=tokens,rows_root=run/'rows',
                    max_seq_len=MAX_CALLS,max_rows=MAX_ROWS,paged_rows=True,expert_loader=bank.arrays)
                closed=[False]
                def close_oracle(owner=oracle):
                    if not closed[0]:
                        owner.close();closed[0]=True
                stack.callback(close_oracle)
                oracle.bindings=bindings
                if digest(_numpy(oracle.hash.token_map).astype('int64').tobytes())!=plan['tokenizer']['compressed_map_sha256']:
                    raise ValueError('Independent original compressed tokenizer differs')
                raw=(root/'reference/generate.py').read_bytes()
                if digest(raw)!=oracle.p[0].source['files']['generate.py']:raise ValueError('Pinned original generation source differs')
                node=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=='generate')
                method_sha=digest(ast.dump(node).encode());namespace={'torch':torch,'List':List,'Transformer':torch.nn.Module}
                exec(compile(ast.Module(body=[node],type_ignores=[]),'<pinned-original-generate>','exec'),namespace)
                from transformers import AutoTokenizer
                tokenizer=AutoTokenizer.from_pretrained(str(root/'import-source'),local_files_only=True,trust_remote_code=False)
                prompt=tokenizer.encode(req.prompt)
                if tokenizer.eos_token_id!=req.eos_id or prompt!=list(req.encode(PinnedTokenizer(root/'import-source'))):raise ValueError('Independent original complete prompt differs')
                cases=[];selected=[]
                def serial_forward(values,start_pos,images=None,token_types=None):
                    if images is not None or token_types is not None or start_pos!=oracle.position:raise ValueError('Unexpected original generation input')
                    for i in range(values.shape[1]):
                        ids=_numpy(values[:,i:i+1]).copy();sample=i==values.shape[1]-1
                        result,state=oracle.step(ids,sample=sample)
                        choice=int(result['sampling']['ids_i64'][0]) if sample else None
                        if sample:selected.append(choice)
                        meta={'position':start_pos+i,'token_id':int(ids[0,0]),'sampling_requested':sample,'selected_id':choice}
                        captured=record(meta,result,state);captured['original_auxiliary_main_hidden']=oracle.last_auxiliary
                        cases.append(write_case(run,'original',captured))
                        if progress:progress({'original_token':meta,'boundaries':len(captured['boundaries'])})
                    return torch.from_numpy(result['sampling']['ids_i64'].copy()).to('cuda:0'),None,None
                # The upstream loop needs one output slot beyond the last
                # admitted model-call input; its token array is distinct from
                # NgramHash's 512-input-position buffer.
                adapter=SimpleNamespace(max_seq_len=MAX_CALLS+1,forward=serial_forward)
                ids=namespace['generate'](adapter,[prompt],req.max_new_tokens,req.eos_id)[0]
                completion={'prompt':req.prompt,'prompt_ids':prompt,'max_new_tokens':req.max_new_tokens,'eos_id':req.eos_id,
                    'calls':len(cases),'selected_ids_including_eos':selected,'completion_ids':ids,'completion_text':tokenizer.decode(ids),
                    'stop_reason':'eos' if selected[-1]==req.eos_id else 'max_new_tokens'}
                decoded=tokenizer.decode(selected);score=score_completion(case,decoded,stop_reason=completion['stop_reason'])
                report={'schema_version':1,'probe':'original_whole_model_trajectory_goldens','status':'pass','scope':scope(case['id']),
                    'plan_sha256':plan_identity(plan),'bindings_sha256':bindings['bindings_sha256'],'output_fixture':head_manifest,
                    'expert_plan':expert_plan(plan),'expert_provenance':oracle.expert_provenance,'assets':plan['assets'],'tokenizer':plan['tokenizer'],
                    'case_files':cases,'completion':completion,'decoded_selected_ids':decoded,'quality_score':score,
                    'reference_methods':oracle.methods,'original_source':oracle.p[0].source,'original_generate_method_sha256':method_sha,
                    'Engram_controls':oracle.engram.controls,'Engram_fixtures':oracle.engram.manifests,'original_embedding_transport':oracle.virtual.stats(),
                    'original_attention_adapter':oracle.qualification,'minimum_sampled_reference_device_free_bytes':oracle.minimum_gpu_free,
                    'original_grouped_conversion':{str(l):(oracle.p[l] if l<2 else oracle.sharing.p[l]).converted_sha256 for l in LAYERS},
                    'weight_payloads_fetched_in_reference':0,'python_http_disabled_during_reference_execution':True,'HTTP_calls_during_reference':http.call_count}
                virtual=oracle.virtual;report['identity_sha256']=identity(report);report['complete_goldens_sha256']=canonical(cases)
                close_oracle();del oracle,adapter,serial_forward
            gc.collect();torch.cuda.synchronize();report['reference_torch_device_allocated_before_workspace_cleanup_bytes']=torch.cuda.memory_allocated(0)
            torch._C._cuda_clearCublasWorkspaces();gc.collect();torch.cuda.empty_cache()
            report['reference_torch_device_peak_allocated_bytes']=torch.cuda.max_memory_allocated(0)
            report['reference_torch_device_allocated_baseline_bytes']=baseline;report['reference_torch_device_allocated_after_cleanup_bytes']=torch.cuda.memory_allocated(0)
            report['original_embedding_transport_after_close']=virtual.stats()
            if torch.cuda.memory_allocated(0)!=baseline:raise RuntimeError('Original whole-model trajectory retained framework GPU allocations')
            report['memory_before_policy_close']=policy.snapshot()
        report['memory_after_policy_close']=policy.snapshot()
        return report
    finally:torch.set_default_dtype(previous)
