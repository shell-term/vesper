"""Pinned allocation/import boundaries for a local streamed model oracle."""

import json
import re

from .attention_topology import validate_config
from .checkpoint import MODEL,REVISION
from .model_artifacts import load_catalog,read_checked,CATALOG_SHA256,canonical
from .resources import GiB


def reference_plan(source):
    catalog,identity=load_catalog(source); config=json.loads(read_checked(source,'inference/config.json'))
    roles=validate_config(config); controls=[]; experts=[]; engram=[]
    for name,t in catalog.items():
        match=re.match(r'^layers\.(\d+)\.',name)
        if match is None or not 0<=int(match[1])<40: continue
        if '.ffn.experts.' in name: experts.append(t)
        elif '.engram.embed.' in name: engram.append(t)
        else: controls.append(t)
    if len(experts)!=40*384*6 or sum(t['bytes'] for t in experts)!=288777830400: raise ValueError('Original backbone expert inventory differs')
    layers=[]
    for role in roles:
        entries=sorted((t for t in controls if t['name'].startswith(f'layers.{role.layer}.')),key=lambda t:t['name'])
        layers.append({'layer':role.layer,'KV_owner':role.kv_owner,'index_owner':role.index_owner,'candidate_owner':role.candidate_owner,
            'compression_ratio':role.ratio,'engram_before_block':role.layer in (1,14),'controls':entries,'control_bytes':sum(t['bytes'] for t in entries)})
    head=catalog['head.weight']; head_gpu=head['bytes']*2
    result={'schema_version':1,'probe':'streamed_original_model_reference_plan','model':MODEL,'revision':REVISION,'catalog_sha256':CATALOG_SHA256,
        'original_tensor_count':identity['tensor_count'],'original_tensor_bytes':identity['original_tensor_bytes'],
        'layers':layers,'backbone_control_tensor_count':len(controls),'backbone_control_bytes':sum(t['bytes'] for t in controls),
        'all_backbone_expert_bytes':sum(t['bytes'] for t in experts),'original_experts':15360,'active_experts_per_layer':6,
        'six_original_expert_packed_bytes':6*18800640,'all_original_Engram_table_bytes':sum(t['bytes'] for t in engram),
        'Engram_table_access':'original requested row addresses only; no remapping or table-wide host mapping',
        'head':head,'head_FP32_GPU_bytes':head_gpu,'final_norm':catalog['norm.weight'],'embedding':catalog['embed.weight'],
        'minimum_all_controls_head_norm_disk_bytes':sum(t['bytes'] for t in controls)+head['bytes']+catalog['norm.weight']['bytes'],
        'host_cap_bytes':96*GiB,'system_available_floor_bytes':32*GiB,'whole_checkpoint_required_in_RAM':False,
        'oracle_execution':'separate local process; original methods; per-layer/current-expert residency; complete boundary goldens before native replay',
        'reference_compute_endpoint':'local CUDA device only','paid_remote_compute_started':False,
        'initial_execution_scope':{'rank':1,'batch':1,'queries_per_call':1,'maximum_context':1152,'text_only':True,
            'prompt_prefill':'serial single-token original forward calls; compare this exact call shape, not a claim of batch-prefill equality'},
        'state_transaction':['window0','window1','all_compressed_KV_and_index_source_state','Engram_token_history','sampler_Philox_seed_offset','model_position'],
        'full_checkpoint_downloaded':False,'full_model_execution_run':False,'full_model_quality':'not_run','text_generation_available':False,
        'remaining_implementation':['all40_original_layer_artifact_bindings','window1_and_all_compressed_block_connection',
            'whole_model_state_publication_and_recovery','independent_streamed_original_forward_oracle',
            'per_layer_route_logit_sampling_gates','deterministic_decode_and_scoped_reasoning_coding_tool_evaluations']}
    result['plan_sha256']=canonical(result); return result
