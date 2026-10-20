"""Bounded, file-bound evidence for actual longer whole-model continuations."""

import json
from pathlib import Path
import re

import numpy as np

from .backbone import LAYERS
from .backbone_validation import LAYER_KEYS, SHARING_KEYS
from .checkpoint import MODEL, REVISION, validate_expert_provenance
from .compile_cache_validation import QUALIFIED_CUBIN_IMAGE_SET,image_identity
from .generation_assets import embedding_address
from .generation_validation import ORIGINAL_GENERATE_METHOD_SHA256, validate_memory
from .model_artifacts import canonical, digest, load_catalog
from .resources import GiB, MiB
from .sequence import MAX_CALLS, MAX_ROWS, MAX_EMBEDDINGS, scope as preparation_scope
from .sequence_assets import request
from .storage import tensor_layout

# Discovery records cannot qualify themselves by recalculating their checksums.
QUALIFIED = {}
ORIGINAL_FORMAT_RAW_SHA256='cc278155dfa3edfee96e2c7603b4bd14eea00edfd18a6595133d2cb4dd040df0'
# Complete inputs already independently checked by the original CPU encoder.
INPUT_PINS={
    'coding-python-expression':(30,'11b2502c987c78e8e4ea923303c286b8e7fa627bc6568dbc77420daa5502ee39',
        'a2638c963603b459dcfa14e7c00faecb4c76e57548c1f18258f457d9233e07f7','be8053d5f05b36e3fe80dff9432b45d07c3c6a6fbf2bd65a693ec98bb136573a'),
    'reasoning-integer':(63,'12e259f59f1d069473686a5496a391fb5866b5939ea24b190359c90df145cf2c',
        '7ed36ad446c2357a13e9bfed16e3cc55656810f87ab83281673d3d02649b915b','5d11c75a8d7eec2925ffd16a306024997bdd7db6550757a675b9f3d4426db5bb'),
    'tool-add':(293,'41afd7b9959ff328b3f708fdc8e6e97813418a896aff59b0bce8e97e574b38ed',
        '469978547708dae6ec46e13f524bb2e5a3427858c26955b39a8aafab4d688223','eb07db19578707dd1434360b12f84cbaaa83c6cf6fc0292f8c4a308e54aac902')}


def read(path,limit=64*MiB):
    path=Path(path)
    if not 1<=path.stat().st_size<=limit:raise ValueError('Excessive or empty trajectory metadata')
    with path.open('rb') as stream:raw=stream.read(limit+1)
    if len(raw)>limit:raise ValueError('Trajectory metadata grew beyond its bound')
    return json.loads(raw),raw


def scope(case_id):
    case,req=request(case_id)
    count,prompt_sha,ids_sha,case_sha=INPUT_PINS[case_id]
    if digest(req.prompt.encode())!=prompt_sha or canonical(case)!=case_sha:raise ValueError('Declared task differs from the qualified original input format')
    return {**preparation_scope(),'model':MODEL,'revision':REVISION,'case_id':case_id,'category':case['category'],
        'original_format_raw_report_sha256':ORIGINAL_FORMAT_RAW_SHA256,'original_prompt_tokens':count,'original_prompt_ids_sha256':ids_sha,
        'layers':list(LAYERS),'batch':1,'rank':1,'head_attached_after_layer':39,'temperature':.7,'seed':2026,
        'request':{'prompt':req.prompt,'max_new_tokens':req.max_new_tokens,'eos_id':req.eos_id},
        'original_generate_loop':True,'serial_prompt_ingestion':True,'original_prompt_chunk_serialized_by_adapter':True,
        'generated_ids_fed_back':True,'prompt_intermediate_sampling':False,'whole_model_execution_run':True,
        'hidden_inputs':'actual_original_embedding_and_all_layer_handoffs','full_model_quality':'one_declared_task_only',
        'other_declared_quality_tasks':'not_run','speed':'not_measured','retraining':False,
        'original_trained_router_correction_bias':True,'additional_cache_dependent_router_bias':False,
        'speculative_MTP_run':False,'vision_run':False,'original_auxiliary_main_hidden_comparison_run':False}


def load_plan(root,run):
    root,run=Path(root),Path(run)
    if not run.resolve().is_relative_to((root/'trajectories').resolve()):raise ValueError('Unanchored trajectory plan')
    plan,_=read(run/'plan.json');case,req=request(plan['case']['id'])
    if (plan.get('probe'),plan.get('status'),plan.get('model'),plan.get('revision'),plan.get('scope'),plan.get('case'),
            plan.get('request'),plan.get('layers'),plan.get('temperature'),plan.get('seed'),plan.get('quality_scored'),
            plan.get('original_reference_execution_run'),plan.get('independent_model_comparison_run'))!=(
            'whole_model_trajectory_artifact_preparation','pass',MODEL,REVISION,preparation_scope(),case,
            scope(case['id'])['request'],list(LAYERS),.7,2026,False,False,False):raise ValueError('Different whole-model preparation scope')
    if plan['plan_sha256']!=canonical({k:v for k,v in plan.items() if k!='plan_sha256'}):raise ValueError('Whole-model preparation checksum differs')
    prompt=plan['completion']['prompt_ids'];count,_,sha,_=INPUT_PINS[case['id']]
    if len(prompt)!=count or canonical(prompt)!=sha:raise ValueError('Prepared input IDs differ from the independently qualified original format')
    if set(plan['experts'])!=set(map(str,LAYERS)):raise ValueError('Incomplete trajectory expert plan')
    for l,p in plan['experts'].items():
        if (p['store']!=(run.resolve().relative_to(root.resolve())/f'indices/layer{l}').as_posix()
                or not 1<=len(p['ids'])<=384 or p['ids']!=sorted(set(p['ids']))
                or any(type(e) is not int or not 0<=e<384 for e in p['ids'])):raise ValueError('Unbounded or unanchored expert trajectory')
        _,raw=read(root/p['store']/'manifest.json',4*MiB)
        if digest(raw)!=p['manifest_sha256']:raise ValueError('Prepared immutable trajectory index changed')
    return plan


def expert_plan(plan):
    return {l:{'ids':p['ids'],'index':f'indices/layer{l}.json','index_sha256':p['manifest_sha256']} for l,p in plan['experts'].items()}


def plan_identity(plan):
    stable={k:plan[k] for k in ('model','revision','scope','case','request','temperature','seed','layers','bindings_sha256',
        'output_fixture','tokenizer','assets','calls','completion')}
    stable['experts']=expert_plan(plan)
    return canonical(stable)


def load_tokens(root,run,plan):
    catalog,_=load_catalog(Path(root)/'import-source');entries=plan['assets']['embeddings'];rows={}
    if not 1<=len(entries)<=MAX_EMBEDDINGS or [e['id'] for e in entries]!=sorted(set(e['id'] for e in entries)):
        raise ValueError('Unbounded original trajectory embeddings')
    for e in entries:
        if e['file']!=f"embeddings/{e['id']}.bin" or e['source']!=embedding_address(catalog,e['id']):raise ValueError('Original embedding address differs')
        path=Path(run)/e['file']
        if path.stat().st_size!=10240:raise ValueError('Original embedding extent differs')
        raw=path.read_bytes();a=np.frombuffer(raw,np.uint16)
        if digest(raw)!=e['sha256'] or np.any((a&0x7f80)==0x7f80):raise ValueError('Original trajectory embedding bytes differ')
        rows[e['id']]=a
    for l in (1,14):
        m,_=read(Path(run)/f'rows/layer{l}/manifest.json',12*MiB)
        if m!=plan['assets']['Engram_rows'][str(l)] or m['maximum_rows']!=MAX_ROWS:raise ValueError('Original trajectory row binding differs')
    return rows


def write_case(run,kind,case):
    pos=case['meta']['position']
    if kind not in ('original','native') or type(pos) is not int or not 0<=pos<MAX_CALLS:raise ValueError('Unanchored or out-of-scope model-call output')
    name=f'{kind}/call-{pos:04d}.json';path=Path(run)/name
    path.parent.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(case,indent=2)+'\n').encode()
    if len(raw)>4*MiB:raise ValueError('One model-call evidence file exceeds its bound')
    with path.open('xb') as stream:stream.write(raw)
    return {'position':pos,'file':name,'bytes':len(raw),'sha256':digest(raw),'typed_boundaries':len(case['boundaries']),
        'storage_elements':sum(b['elements'] for b in case['boundaries'])}


def load_case(run,descriptor,kind):
    if kind not in ('original','native') or type(descriptor['position']) is not int or not 0<=descriptor['position']<MAX_CALLS:
        raise ValueError('Unanchored or out-of-scope model-call evidence')
    if descriptor['file']!=f"{kind}/call-{descriptor['position']:04d}.json":raise ValueError('Unanchored model-call evidence')
    c,raw=read(Path(run)/descriptor['file'],4*MiB)
    if (len(raw),digest(raw),len(c['boundaries']),sum(b['elements'] for b in c['boundaries']))!=(
            descriptor['bytes'],descriptor['sha256'],descriptor['typed_boundaries'],descriptor['storage_elements']):raise ValueError('Model-call evidence bytes differ')
    return c,raw


def identity(report):
    stable={k:report[k] for k in ('scope','plan_sha256','bindings_sha256','output_fixture','expert_plan','expert_provenance',
        'assets','tokenizer','reference_methods','original_source','original_generate_method_sha256','Engram_fixtures',
        'original_embedding_transport','original_grouped_conversion','Engram_controls','completion','decoded_selected_ids','quality_score')}
    stable['original_attention_adapter']={k:report['original_attention_adapter'][k] for k in (
        'model','revision','reference','profiles','torch_version','torch_git_version','torch_cuda_version','tilelang_version','official_arithmetic_memory_adapter_gate')}
    return canonical(stable)


def check_trajectory(report,run,kind):
    completion=report['completion'];prompt=completion['prompt_ids'];descriptors=report['case_files'];selected=[];routes={str(l):set() for l in LAYERS}
    req=report['scope']['request']
    if any(completion[n]!=req[n] for n in ('prompt','max_new_tokens','eos_id')):raise ValueError('Changed declared quality request')
    if not prompt or not len(prompt)<=len(descriptors)<=MAX_CALLS or [d['position'] for d in descriptors]!=list(range(len(descriptors))):
        raise ValueError('Incomplete whole-model trajectory')
    offset=0
    for pos,d in enumerate(descriptors):
        c,_=load_case(run,d,kind);sample=pos>=len(prompt)-1
        token=prompt[pos] if pos<len(prompt) else selected[-1]
        choice=c['meta']['selected_id']
        if c['meta']!={'position':pos,'token_id':token,'sampling_requested':sample,'selected_id':choice} or (not sample and choice is not None):
            raise ValueError('A generated ID was not independently fed back')
        if sample:
            if type(choice) is not int or not 0<=choice<129280 or (selected and selected[-1]==completion['eos_id']):raise ValueError('Invalid sampling/EOS trajectory')
            selected.append(choice);offset+=4
        if c['rng_offset']!=offset:raise ValueError('Prompt ingestion changed original RNG progression')
        checks=c['boundaries'];names={b['check'] for b in checks}
        required={f'layers/{l}/{n}/' for l in LAYERS for n in (*LAYER_KEYS,'attention_output_bf16')}
        required|={f'attention/{l}/{n}/' for l in range(2,40) for n in SHARING_KEYS}
        if len(checks)!=(5252 if sample else 5247) or len(names)!=len(checks) or not required.issubset(names):raise ValueError('Missing actual whole-model math boundaries')
        if any(b['elements']!=int(np.prod(b['shape'])) or len(b['sha256'])!=64 or any(x not in '0123456789abcdef' for x in b['sha256']) for b in checks):raise ValueError('Invalid typed whole-model fingerprint')
        if set(c['ordered_routes'])!=set(routes):raise ValueError('Missing actual original routes')
        for l,r in c['ordered_routes'].items():
            if len(r['indices'])!=6 or len(set(r['indices']))!=6 or len(r['weight_bits'])!=6:raise ValueError('Incomplete top-six route')
            routes[l].update(r['indices'])
    reason='eos' if selected[-1]==completion['eos_id'] else 'max_new_tokens'
    expected=selected[:-1] if reason=='eos' else selected
    if (completion['calls'],completion['selected_ids_including_eos'],completion['completion_ids'],completion['stop_reason'])!=(len(descriptors),selected,expected,reason):raise ValueError('Incomplete generated completion')
    if reason=='max_new_tokens' and len(selected)!=completion['max_new_tokens']:raise ValueError('Premature generation cannot pass')
    if any(sorted(ids)!=report['expert_plan'][l]['ids'] for l,ids in routes.items()):raise ValueError('Expert inventory does not match actual whole-model routes')
    return True


def validate_original(report,run,*,require_qualified=True):
    case_id=report['scope']['case_id']
    if (report.get('probe'),report.get('status'),report['scope'])!=('original_whole_model_trajectory_goldens','pass',scope(case_id)):
        raise ValueError('Different whole-model scope')
    if report['identity_sha256']!=identity(report) or report['complete_goldens_sha256']!=canonical(report['case_files']):raise ValueError('Changed original whole-model identity/goldens')
    if require_qualified and (report['identity_sha256'],report['complete_goldens_sha256'])!=QUALIFIED.get(case_id):raise ValueError('Original longer trajectory has not been independently qualified')
    if report['original_generate_method_sha256']!=ORIGINAL_GENERATE_METHOD_SHA256:raise ValueError('Original generation method changed')
    for key,sha in (('reference_methods','6b1f52cf9f58c3a218c0b3ceefc7ff8cc37e975a30adef23c051a7b02a70f5ae'),
            ('original_source','345a311dcad3b7e515d8b1e8c57f3a3a2698e0df751362cb0807108a2a6fbdf5'),
            ('original_grouped_conversion','cc48760fe0ba3b8ab6c00c2f63b513be75d1a07270c5b700020a7987fcd90036')):
        if canonical(report[key])!=sha:raise ValueError('Previously qualified original source/method/conversion identity differs')
    check_trajectory(report,run,'original')
    count,_,sha,_=INPUT_PINS[case_id]
    if len(report['completion']['prompt_ids'])!=count or canonical(report['completion']['prompt_ids'])!=sha:
        raise ValueError('Original model did not process the independently qualified complete input')
    validate_indices(report,run)
    expected_keys={f'{l}:{i}' for l,p in report['expert_plan'].items() for i in p['ids']}
    if set(report['expert_provenance'])!=expected_keys:raise ValueError('Incomplete original expert provenance census')
    for key,m in report['expert_provenance'].items():
        validate_expert_provenance(m)
        if key!=f"{m['layer']}:{m['expert']}":raise ValueError('Original expert provenance key differs')
    if len(report['Engram_controls'])!=2*len(report['case_files']) or any(c['status']!='pass' for c in report['Engram_controls']):raise ValueError('Missing unchanged original Engram controls')
    after=report['original_embedding_transport_after_close']
    active=report['original_embedding_transport']
    if (active['physical_cap_bytes']!=512*MiB or active['peak_physical_owned_bytes']>512*MiB
            or active['table_close_callbacks']!=5 or active['maximum_active_rows_per_table']!=24
            or after['total_pages_mapped']!=after['total_pages_evicted']):raise ValueError('Original row transport is not bounded')
    if after['physical_owned_bytes'] or after['virtual_reserved_bytes'] or after['owned_context_active'] or after['current_paged_table_pages'] or after['current_logical_active_rows']:
        raise ValueError('Original paged tables retained allocations')
    if report['reference_torch_device_allocated_after_cleanup_bytes']!=report['reference_torch_device_allocated_baseline_bytes']:
        raise ValueError('Original framework retained tensors')
    if report['weight_payloads_fetched_in_reference'] or report['HTTP_calls_during_reference'] or report['python_http_disabled_during_reference_execution'] is not True:
        raise ValueError('Original whole-model execution fetched weights')
    from .evaluation import score_completion
    case,_=request(case_id)
    if report['quality_score']!=score_completion(case,report['decoded_selected_ids'],stop_reason=report['completion']['stop_reason']):
        raise ValueError('Changed recorded quality score')
    validate_memory(report,32*GiB)
    return True


def validate_indices(report,run):
    for layer,p in report['expert_plan'].items():
        if p['index']!=f'indices/layer{layer}.json':raise ValueError('Unanchored published trajectory index')
        m,raw=read(Path(run)/p['index'],4*MiB)
        if digest(raw)!=p['index_sha256'] or (m['schema_version'],m['kind'],m['model'],m['revision'],m['layer'],m['maximum_experts'],m['layout'])!=(
                1,'vesper_indexed_native_experts',MODEL,REVISION,int(layer),384,tensor_layout()):raise ValueError('Original trajectory index changed')
        if [e['key'] for e in m['entries']]!=[f'{layer}:{i}' for i in p['ids']] or len(p['ids'])>384:raise ValueError('Different indexed original experts')
        if not 1<=len(m['sources'])<=384:raise ValueError('Unbounded original index sources')
        for s in m['sources']:
            name=s['directory']
            if (Path(name).is_absolute() or '..' in Path(name).parts or '\\' in name or re.match(r'^[A-Za-z]:',name)
                    or not re.fullmatch(r'[0-9a-f]{64}',s['manifest_sha256'])):raise ValueError('Unanchored original source')
        for e in m['entries']:
            if e['provenance']!=report['expert_provenance'][e['key']]:raise ValueError('Original expert index provenance differs')


def validate_native(report,run,original,original_raw,reference_run):
    validate_original(original,reference_run)
    case_id=original['scope']['case_id']
    if (report.get('probe'),report.get('status'),report.get('scope'),report.get('reference_report_sha256'),report.get('original_identity_sha256'),
            report.get('complete_goldens_sha256'),report.get('plan_sha256'),report.get('expert_plan'))!=(
            'native_whole_model_trajectory_recheck','pass',scope(case_id),digest(original_raw),original['identity_sha256'],
            original['complete_goldens_sha256'],original['plan_sha256'],original['expert_plan']):raise ValueError('Different native whole-model reference binding')
    if any(report.get(k)!=original[k] for k in ('completion','decoded_selected_ids','quality_score')):raise ValueError('Different actual native completion/score')
    check_trajectory(report,run,'native')
    if len(report['case_files'])!=len(original['case_files']):raise ValueError('Incomplete native model calls')
    for actual,expected in zip(report['case_files'],original['case_files']):
        a,_=load_case(run,actual,'native');e,raw=load_case(reference_run,expected,'original')
        if a.get('status')!='pass' or a.get('reference_case_sha256')!=digest(raw):raise ValueError('Native model call lost its original binding')
        if any(a[k]!=e[k] for k in ('meta','boundaries','state_sha256','rng_offset','ordered_routes')):raise ValueError('Native whole-model math/state differs')
    f=report['fault']
    if (f is None or f['position']!=len(original['completion']['prompt_ids'])-1 or f['stage']!='after_final_logits_and_sampling'
            or f['error']!='InjectedWholeTokenFailure' or f['before_state_sha256']!=f['after_state_sha256'] or f['healthy_retry_compared'] is not True):
        raise ValueError('Missing actual whole-token failure, state restoration and retry')
    if report['native_frameworks']!={n:{'installed':False,'imported':False} for n in ('torch','tilelang','transformers','sympy')}:
        raise ValueError('Native longer trajectory used a reference framework')
    if report['weight_payloads_fetched_in_recheck'] or report['HTTP_calls_during_native'] or report['python_http_disabled_during_native_execution'] is not True:
        raise ValueError('Native whole-model execution fetched weights')
    calls=len(report['case_files'])
    if set(report['indexed_sources'])!=set(map(str,LAYERS)):raise ValueError('Missing actual offline indexed providers')
    for l,p in report['indexed_sources'].items():
        if (p['calls'],p['original_experts'],p['index_sha256'],p['direct_read_calls'],p['direct_read_bytes'],p['payload_bytes_copied_to_build_index'],p['expert_payloads_fetched_in_forward'])!=(
                calls+1,original['expert_plan'][l]['ids'],original['expert_plan'][l]['index_sha256'],6*(calls+1),6*(calls+1)*18800640,0,0):raise ValueError('Native route reads or immutable expert identity differ')
    for name in ('transaction_active','transaction_cleanup'):
        t=report[name]
        if ((t['position'],t['committed_tokens'],t['failed_attempts'])!=(calls,calls,1)
                or t['pending_cleanup_owners'] or t['pending_recovery_owners']):raise ValueError('Native whole-token transaction retained unfinished state')
    if report['transaction_cleanup']['host_reserved_bytes'] or report['compiler_cache_after_close']['host_reserved_bytes'] or report['compiler_cache_after_close']['entries']:
        raise ValueError('Native longer trajectory retained reservations')
    if any(r['host_reserved_bytes'] or r['cached_rows'] or r['cache_capacity_rows']!=32 for r in report['Engram_row_caches_after_close'].values()):
        raise ValueError('Native Engram row cache was not bounded/released')
    if not report['compiler_cache']['hits'] or not 1<=report['compiler_cache']['misses']==len(report['compiled_images'])<=32:
        raise ValueError('Missing bounded compiled-image evidence')
    if image_identity(report['compiled_images'])!=QUALIFIED_CUBIN_IMAGE_SET:raise ValueError('Native cubin bytes differ from the original-checked full replay')
    validate_memory(report,64*GiB)
    return True
