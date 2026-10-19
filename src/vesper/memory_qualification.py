"""Audit the six original Issue #9 checks against complete, sealed evidence.

Resource geometry uses touched synthetic payloads. Original numerical replays
are separate gates; neither is a full forty-layer numerical/model-quality run.
"""

import hashlib
import json
import struct
from pathlib import Path

from .checkpoint import MODEL,REVISION
from .model_artifacts import CATALOG_SHA256,canonical,digest
from .resources import MemoryBudget,GiB,MiB
from .engram_validation import validate_goldens as validate_engram,validate_faults,QUALIFIED_IDENTITY,QUALIFIED_GOLDENS
from .block_chain_validation import validate_goldens as validate_chain

ROOT='results/2026-10-19-memory-policy'
EVIDENCE=(ROOT+'/scale.json',ROOT+'/chain.json',ROOT+'/chain-memory.json',ROOT+'/default-boundaries.json',
    ROOT+'/reference-memory.json',ROOT+'/reference-original.json',
    'results/2026-10-19-engram/original.json','results/2026-10-19-refresh-block-chain/chain.json')
QUALIFIED_MANIFEST_SHA256='a9d9507d3cb531a241365ce499cf911d50310863ccb760543d1264164171a7ef'
CRITERIA=(
    ('all_owned_resources','Account for all runtime-owned caches, weights, activations, KV, Engram, metadata, pinned buffers, integrity tables and scratch.'),
    ('explicit_cap_floor','Enforce the provisional 96 GiB owned-host cap and 32 GiB available-memory floor; keep configuration explicit and revise only with evidence.'),
    ('pressure_reclaim_refuse','Reclaim/shrink eligible resources under pressure and refuse work safely if the reserve cannot be restored.'),
    ('process_external_measurements','Measure process memory and driver/compiler/reference exclusions separately; do not present a reservation counter as an OS limit.'),
    ('failure_cancel_other_work','Test failure/cancellation/retry/cleanup alongside other workloads without stopping them, resetting GPUs or changing the page file.'),
    ('representative_model_scale','Validate resource behavior at representative model scale, not solely the three-expert fixture.'),
)


def validate_manifest(m):
    if ((m.get('probe'),m.get('model'),m.get('revision'))!=('issue9_memory_evidence_manifest',MODEL,REVISION)
            or canonical(m)!=QUALIFIED_MANIFEST_SHA256 or [e['file'] for e in m.get('evidence',[])]!=list(EVIDENCE)
            or m.get('criteria')!=[{'id':n,'criterion':c} for n,c in CRITERIA]):
        raise ValueError('Incomplete or substituted original Issue #9 acceptance manifest')


def validate_governor(g,cap,*,clean=True):
    if (g['host_cap_bytes']!=cap or g['available_floor_bytes']!=32*GiB or not 0<g['charged_host_peak_bytes']<=cap
            or g['pinned_cap_bytes']!=GiB or not 0<=g['pinned_peak_bytes']<=GiB
            or g['device_caps']!={'0':20*GiB,'1':10*GiB}
            or any(v>g['device_caps'][k] for k,v in g['device_peak_bytes'].items())
            or g['control_metadata_admission_bytes']!=64*MiB or g['maximum_live_allocation_records']!=16384
            or g['maximum_reclaimers']!=512 or g['monitor_interval_seconds']!=.25 or g['monitor_error'] is not None
            or g['reservation_is_RSS_or_OS_limit'] is not False): raise ValueError('Global memory policy or measurement scope differs')
    if clean and (g['charged_host_bytes'] or g['explicit_host_bytes'] or g['pinned_bytes'] or any(g['categories'].values())
            or g['live_leases'] or g['live_reclaimers'] or g['inflight_operations'] or any(g['device_used_bytes'].values())
            or g['monitor_thread_stopped'] is not True): raise ValueError('Global cleanup is incomplete')
    if not g['samples'] or not g['process_baseline']['resident_bytes'] or not g['sampled_process_peaks']['resident_bytes']:
        raise ValueError('Actual process observations are missing')
    if any(r['charged_host_bytes']>cap or r['explicit_buffer_bytes']>r['charged_host_bytes'] for r in g['samples']):
        raise ValueError('Observed runtime admission exceeds its cap')


def validate_geometry(p):
    body={k:v for k,v in p.items() if k!='plan_sha256'}
    if (p['plan_sha256']!=canonical(body) or p['model']!=MODEL or p['revision']!=REVISION or p['catalog_sha256']!=CATALOG_SHA256
            or p['catalog_tensor_count']!=96085 or p['original_tensor_bytes']!=510286023000
            or (p['layers'],p['experts_per_layer'],p['expert_count'],p['expert_bytes'],p['context'],p['batch'])!=(40,384,15360,18800640,131072,1)
            or p['all_expert_tensor_bytes']!=288777830400 or p['retained_metadata_admission_bytes']!=512*MiB):
        raise ValueError('Representative original model inventory differs')
    blocks=p['expert_blocks']
    if len(blocks)!=15360: raise ValueError('Incomplete original expert traversal plan')
    for i,b in enumerate(blocks):
        l,e=divmod(i,384)
        names=sorted(f'layers.{l}.ffn.experts.{e}.w{w}.{part}' for w in (1,2,3) for part in ('weight','scale'))
        if b!={'layer':l,'expert':e,'bytes':18800640,'source_tensor_names':names}: raise ValueError('Original expert extents/identities differ')
    expected={f'window.{l}':([1,128,512],'uint16',131072) for l in range(40)}
    for layer,count in ((2,65536),(8,65536),(14,65536),(20,131072)):
        for name,width in (('compressed_KV',512),('index_keys',128)):
            expected[f'{name}.{layer}']=([1,count,width],'uint16',count*width*2)
        if layer!=20:
            for name in ('pool_values','pool_scores'): expected[f'{name}.{layer}']=([1,2,512],'float32',4096)
    expected['source20_candidates']=([1,16384],'bool',16384)
    if ({r['owner']:(r['shape'],r['dtype'],r['bytes']) for r in p['KV_allocations']}!=expected
            or len(p['KV_allocations'])!=len(expected) or p['KV_bytes']!=sum(r[2] for r in expected.values())
            or p['largest_nonexpert_tensor']['name']!='embed.weight' or p['largest_nonexpert_tensor']['bytes']!=1323827200
            or p['largest_nonexpert_tensor']['shape']!=[129280,5120]
            or set(p['layer_nonexpert_working_weights'])!={str(l) for l in range(40)}):
        raise ValueError('Full original retained-state/working-weight allocation geometry differs')


def compare_boundaries(actual,golden):
    if (len(actual)!=len(golden) or any(a.get(k)!=g.get(k) for a,g in zip(actual,golden)
            for k in ('check','status','elements','dtype','shape','expected_sha256','actual_sha256'))
            or any(a['status']!='pass' or a['actual_sha256']!=a['expected_sha256'] for a in actual)):
        raise ValueError('Original typed/shaped numerical boundaries differ')


def validate_native(n,original,raw):
    if ((n['probe'],n['status'],n['reference_report_sha256'],n['qualified_identity_sha256'],n['complete_goldens_sha256'])!=
            ('governed_original_Engram_parity','pass',digest(raw),QUALIFIED_IDENTITY,QUALIFIED_GOLDENS)
            or n['native_profiles']!=original['native_profiles'] or len(n['session_cases'])!=110
            or any(n[k+'_'+s] is not False for k in ('torch','tilelang','transformers','sympy') for s in ('installed','imported'))):
        raise ValueError('Governed native original identity or independent environment differs')
    for a,g in zip(n['session_cases'],original['session_cases']):
        if any(a[k]!=g[k] for k in ('meta','status','input_sha256','hash_ids_sha256','compressed_sha256','lookbacks_sha256','committed_state_sha256')):
            raise ValueError('Governed Engram input/history differs')
        compare_boundaries(a['boundaries'],g['boundaries'])
    validate_faults(n['faults'],n['snapshots'])
    names=['cancel_after_layer1_output','cancel_after_layer14_output','GPU_allocator_failure','host_allocator_failure','pinned_allocator_failure']
    if ([f['name'] for f in n['extra_faults']]!=names or len(n['extra_steps'])!=6
            or any(f['status']!='pass' for f in n['extra_faults'])): raise ValueError('Incomplete governed allocator/cancellation gates')
    for f in n['extra_faults'][:3]:
        if not all(f[k] is True for k in ('history_unchanged','temporary_GPU_buffers_released','retry_all_boundaries_equal')):
            raise ValueError('Failed native work changed history or leaked buffers')
    if any(f['resume_while_inflight_refused'] is not True for f in n['extra_faults'][:2]): raise ValueError('Cancellation was cleared during active work')
    by_id={c['meta']['id']:c for c in original['session_cases']}
    for r in n['extra_steps']:
        if r['status']!='pass': raise ValueError('Native retry did not pass')
        compare_boundaries(r['boundaries'],by_id[r['source_golden']]['boundaries'])


def validate_scale(s,original,raw):
    validate_geometry(s['plan']); g=s['cleanup_governor']; validate_governor(g,8*GiB)
    trace=hashlib.sha256()
    for i in range(15360): trace.update(struct.pack('<QQ',i,18800640))
    if (s['probe']!='representative_model_memory_policy' or s['status']!='pass' or s['complete_backbone_expert_pass'] is not True
            or s['expert_entries_visited']!=15360 or s['synthetic_weight_bytes_touched']!=288777830400 or s['synthetic_trace_sha256']!=trace.hexdigest()
            or [(r['layer'],r['expert_entries'],r['status']) for r in s['layers']]!=[(l,384,'pass') for l in range(40)]
            or any(r['live_cache_bytes']>6*GiB for r in s['layers']) or s['KV_unchanged_across_all_faults'] is not True
            or s['physical_available_min_sampled_bytes']<32*GiB or g['reclaim_events']<=0 or g['reclaimed_charge_bytes']<=0
            or not g['monitor_checks'] or g['pinned_peak_bytes']<2*18800640): raise ValueError('Live representative memory/reclaim gate differs')
    if [f['name'] for f in s['faults']]!=['over_global_cap','partial_source_read','cancel_before_publication','retry_after_failure_and_cancel',
            'injected_pressure_restored','injected_unrestorable_reserve'] or any(f['status']!='pass' for f in s['faults']):
        raise ValueError('Incomplete live failure/retry/pressure evidence')
    companion=s['companion']; progress=s['other_work_progress']
    if (companion['owned_by_probe'] is not True or companion['exit_code']!=0 or companion['reader_error'] is not None
            or companion['stderr_empty'] is not True or companion['exited'] is not True or companion['last']['done'] is not True
            or companion['existing_workloads_stopped'] is not False or progress['counter_after_faults']<=progress['counter_before_faults']
            or s['GPU_resets'] or s['pagefile_changes'] is not False or s['full_checkpoint_downloaded'] is not False
            or s['weights_fetched_during_probe'] or s['actual_96GiB_allocation_attempted'] is not False
            or s['full_model_quality']!='not_run' or s['end_to_end_speed']!='not_measured'):
        raise ValueError('Companion progress, cleanup or resource-only scope differs')
    required={'driver_initialization','driver_context_initialization','native_compiler','device_module_load'}
    if (not required<=set(g['external_phase_counts']) or not required<={r['phase'] for r in g['external_phase_samples']}
            or any(r['delta_is_exact_component_attribution'] is not False for r in g['external_phase_samples'])):
        raise ValueError('Driver/compiler process observations are missing or misattributed')
    validate_native(s['native_comparisons'],original,raw)


def validate_chain_replay(n,original,raw,g):
    validate_governor(g['governor'],16*GiB)
    if (n['probe']!='complete_refresh_chain_recheck' or n['status']!='pass' or n['reference_report_sha256']!=digest(raw)
            or n['native_profiles']!=original['native_profiles'] or n['hc_profiles']!=original['hc_profiles']
            or len(n['chain_cases'])!=40 or n['faults']!=original['faults'] or n['snapshot_events']!=original['snapshot_events']
            or g['all_runtime_charges_released'] is not True or any(n[k+'_'+s] is not False for k in ('torch','tilelang') for s in ('installed','imported'))):
        raise ValueError('Governed complete-chain numerical or cleanup identity differs')
    for a,b in zip(n['chain_cases'],original['chain_cases']):
        fields=('meta','status','expected_boundary_root_sha256','actual_boundary_root_sha256','expected_layout_sha256',
            'actual_layout_sha256','failures','decisions','handoffs','committed_state_sha256','committed_position','index_query_layers',
            'reused_index_owners','all_shared_KV_from_layer20','candidate_owner','final_index_owner')
        if (any(a[k]!=b[k] for k in fields) or a['boundary_comparisons']!=len(b['boundaries'])
                or a['compared_storage_elements']!=sum(x[1] for x in b['boundaries'])): raise ValueError('Governed original chain boundary/state differs')
    if not {'0','1'}<=set(g['governor']['device_peak_bytes']): raise ValueError('Combined two-device ownership was not observed')


def validate_default(d):
    validate_governor(d['cleanup_governor'],96*GiB); validate_governor(d['at_cap_governor'],96*GiB,clean=False)
    if (d['probe']!='default_memory_admission_boundaries' or d['status']!='pass'
            or d['available_observation_is_injected'] is not True or d['physical_payload_allocation_attempted'] is not False
            or d['default_cap_refuses_one_more_byte'] is not True or d['default_floor_refuses_one_more_byte'] is not True
            or d['at_cap_governor']['charged_host_bytes']!=96*GiB or d['at_cap_governor']['explicit_host_bytes']
            or d['cleanup_governor']['refusals']<2): raise ValueError('Default policy counter boundary/scope differs')


def validate_reference(p,r,raw,original):
    validate_engram(r)
    if (p['probe']!='separate_optional_reference_memory' or p['status']!='pass' or p['reference_report_sha256']!=digest(raw)
            or p['qualified_identity_sha256']!=QUALIFIED_IDENTITY or p['complete_goldens_sha256']!=QUALIFIED_GOLDENS
            or r['complete_goldens_sha256']!=original['complete_goldens_sha256'] or p['successful_session_cases']!=110 or p['bitwise_boundaries']!=6820
            or p['measurement_process']!='separate_optional_reference_process' or p['native_owned_cap_includes_reference_frameworks'] is not False
            or p['phase_deltas_are_exact_component_attribution'] is not False or p['reservation_is_RSS_or_OS_limit'] is not False
            or p['minimum_sampled_available_bytes']<32*GiB or len(p['phases'])<12
            or p['VMM_after_close']['physical_owned_bytes'] or p['VMM_after_close']['virtual_reserved_bytes']
            or p['VMM_after_close']['owned_context_active']): raise ValueError('Separate optional reference measurement differs')
    for key,value in p['sampled_process_peaks'].items():
        expected=max((v['process'][key] for v in p['phases'] if v['process'].get(key) is not None),default=None)
        if value!=expected: raise ValueError('Process metrics confuse unavailable, resident or private-commit observations')


def qualify(root,manifest_path):
    root=Path(root).resolve(); path=Path(manifest_path)
    if not 0<path.stat().st_size<=65536: raise ValueError('Excessive memory acceptance manifest')
    m=json.loads(path.read_bytes()); validate_manifest(m); reports=[]; raws=[]
    budget=MemoryBudget(cap=512*MiB,owner='memory_acceptance_audit'); budget.acquire(budget.cap)
    try:
        for e in m['evidence']:
            file=(root/e['file']).resolve()
            if not file.is_relative_to(root) or not 0<e['bytes']<=32*MiB or file.stat().st_size!=e['bytes']: raise ValueError('Evidence path or size differs')
            raw=file.read_bytes()
            if digest(raw)!=e['sha256']: raise ValueError('Original memory evidence SHA-256 differs')
            raws.append(raw); reports.append(json.loads(raw))
        scale,chain,chain_memory,default,profile,reference,engram,original_chain=reports
        validate_engram(engram); validate_chain(original_chain)
        validate_scale(scale,engram,raws[6]); validate_chain_replay(chain,original_chain,raws[7],chain_memory)
        validate_default(default); validate_reference(profile,reference,raws[5],engram)
        return {'schema_version':1,'probe':'issue9_memory_acceptance_audit','status':'pass','model':MODEL,'revision':REVISION,
            'manifest_sha256':canonical(m),'criteria':[{'id':n,'criterion':c,'status':'pass'} for n,c in CRITERIA],
            'evidence':m['evidence'],'default_host_cap_bytes':96*GiB,'available_floor_bytes':32*GiB,
            'resource_probe_host_cap_bytes':8*GiB,'resource_probe_host_peak_bytes':scale['cleanup_governor']['charged_host_peak_bytes'],
            'resource_probe_sampled_process_peaks':scale['cleanup_governor']['sampled_process_peaks'],
            'physical_available_min_sampled_bytes':scale['physical_available_min_sampled_bytes'],
            'original_expert_identities_visited':15360,'synthetic_expert_bytes_touched':288777830400,
            'retained_KV_bytes_per_copy':scale['plan']['KV_bytes'],'governed_Engram_steps':116,'governed_Engram_boundaries':7192,
            'governed_chain_steps':40,'governed_chain_boundaries':59600,'reservation_is_RSS_or_OS_limit':False,
            'all_forty_layers_numerically_executed':False,'full_checkpoint_downloaded':False,'full_model_quality':'not_run',
            'end_to_end_speed':'not_measured','milestone2_complete':False,'remaining_integration_issues':[10],
            'scope':'Cooperative process-wide runtime admission and representative touched resource geometry, with independent original numerical replay gates',
            'audit_host_allowance_bytes':budget.cap}
    finally: budget.release(budget.cap)
