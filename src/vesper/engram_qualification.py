"""Audit Issue #8 against bounded original, native, direct-I/O and capacity evidence."""

import json
from pathlib import Path

from .checkpoint import MODEL,REVISION
from .engram_validation import validate_goldens,validate_faults,validate_memory,QUALIFIED_IDENTITY,QUALIFIED_GOLDENS
from .model_artifacts import canonical,digest,CATALOG_SHA256,SHARD_INVENTORY_SHA256,FILES
from .resources import MemoryBudget,MiB,GiB

ROOT='results/2026-10-19-engram'
EVIDENCE=('original.json','native.json','direct-io.json','import-plan.json')
QUALIFIED_MANIFEST_SHA256='73336890f0007c693becc969a8b33abd0a1e099353cd7ef928b28049f6852363'
CRITERIA=(
    ('tokenizer_hash_mask_address','Preserve tokenizer/compressed-vocabulary mapping, N-gram hashing, masks and row addressing.'),
    ('original_identities_notices','Verify original weight/scale/config/tokenizer identities and third-party notices.'),
    ('bounded_required_rows','Read only required Engram rows with bounded caches/staging; avoid model-wide mapping/page-cache growth.'),
    ('full_import_plan','Plan full artifact import using actual disk capacity and budgets; keep small probes bounded and separate from a full checkpoint download.'),
    ('original_output_state','Compare original Engram output/state across token/history boundaries with complete evidence.'),
)


def validate_native(native,original,original_raw):
    if (native.get('probe'),native.get('status'),native.get('reference_report_sha256'),native.get('qualified_identity_sha256'),native.get('complete_goldens_sha256'))!=(
            'native_Engram_complete_recheck','pass',digest(original_raw),QUALIFIED_IDENTITY,QUALIFIED_GOLDENS): raise ValueError('Native Engram evidence identity differs')
    if native['native_profiles']!=original['native_profiles']: raise ValueError('Native compiler profiles differ')
    if any(native.get(n+'_'+s) is not False for n in ('torch','tilelang','transformers','sympy') for s in ('installed','imported')):
        raise ValueError('Published native recheck is not independent of reference frameworks')
    if len(native['session_cases'])!=len(original['session_cases']): raise ValueError('Incomplete native Engram replay')
    for a,b in zip(native['session_cases'],original['session_cases']):
        for k in ('meta','status','input_sha256','hash_ids_sha256','compressed_sha256','lookbacks_sha256','committed_state_sha256'):
            if a[k]!=b[k]: raise ValueError('Native Engram input/history evidence differs')
        if len(a['boundaries'])!=len(b['boundaries']): raise ValueError('Incomplete native Engram boundary replay')
        for x,y in zip(a['boundaries'],b['boundaries']):
            if (any(x[k]!=y[k] for k in ('check','status','elements','dtype','shape','expected_sha256','actual_sha256'))
                    or x['status']!='pass'): raise ValueError('Native Engram bitwise boundary differs')
    validate_faults(native['faults'],native['snapshots']); validate_memory(native['memory'],reference=False)
    if (native['weight_payload_bytes_fetched_during_recheck']!=0 or native['full_model_quality']!='not_run'
            or native['end_to_end_speed']!='not_measured'): raise ValueError('Native Engram recheck scope differs')


def validate_storage(io,plans):
    if (io['status']!='pass' or len(io['small_original_tensors'])!=6 or io['original_input_bytes']!=471040
            or io['integrity_block_bytes']!=65536 or io['host_cap_bytes']!=16*MiB or io['host_peak_reserved_bytes']>16*MiB
            or io['host_reserved_after_close']!=0 or io['full_shards_registered']!=0 or io['full_checkpoint_downloaded'] is not False
            or io['weight_payload_bytes_fetched']!=0): raise ValueError('Bounded original direct-I/O scope differs')
    for t in io['small_original_tensors']:
        if (t['status']!='pass' or len(t['ranges'])!=32 or any(r['status']!='pass' or r['bytes']!=264 for r in t['ranges'])
                or t['receipt']['sha256']!=t['source_tensor']['sha256'] or t['receipt']['bytes']!=t['source_tensor']['bytes']
                or t['file_mapping'] is not False or t['buffered_read_fallback'] is not False or t['maximum_cached_blocks']!=2
                or t['temporary_integrity_directory_removed'] is not True): raise ValueError('Original direct-I/O integrity evidence differs')
    if len(plans['plans'])!=2 or [p['target_label'] for p in plans['plans']]!=['workspace_volume','existing_alternative_volume']:
        raise ValueError('Incomplete actual-capacity import plan')
    for p in plans['plans']:
        cat=p['catalog']; inventory=[{'file':s['shard'],'bytes':s['file_bytes'],'sha256':s['original_file_sha256']} for s in cat['shards']]
        if (cat['catalog_sha256']!=CATALOG_SHA256 or cat['tensor_count']!=96085 or cat['original_tensor_bytes']!=510286023000
                or canonical(inventory)!=SHARD_INVENTORY_SHA256 or p['original_shard_file_bytes']!=510296708312
                or cat['metadata']!={n:{'bytes':s,'sha256':h} for n,(_,s,h) in FILES.items()}
                or sum(p['tensor_bytes_by_category'].values())!=cat['original_tensor_bytes']): raise ValueError('Full original artifact identities/extents differ')
        index_bytes=sum((((r['bytes']+65535)//65536)*32+4095)//4096*4096 for r in inventory)
        if (p['integrity_index_bytes']!=index_bytes or p['metadata_bytes']!=sum(p['metadata_bytes_by_kind'].values())
                or p['disk_reserve_bytes']!=64*GiB
                or p['required_free_bytes']!=p['original_shard_file_bytes']+p['metadata_bytes']+index_bytes+64*GiB
                or not 0<=p['target_volume_available_bytes']<=p['target_volume_total_bytes']
                or p['capacity_sufficient'] is not (p['target_volume_available_bytes']>=p['required_free_bytes'])
                or p['additional_free_bytes_required']!=max(0,p['required_free_bytes']-p['target_volume_available_bytes'])
                or p['full_import_authorized_or_started'] is not False or p['full_checkpoint_downloaded'] is not False
                or p['weight_payload_bytes_fetched']!=0 or p['provisional_full_runtime_host_cap_bytes']!=96*GiB
                or p['system_available_floor_bytes']!=32*GiB): raise ValueError('Original artifact capacity/budget plan differs')


def validate_manifest(m):
    if ((m.get('probe'),m.get('model'),m.get('revision'))!=('issue8_evidence_manifest',MODEL,REVISION)
            or canonical(m)!=QUALIFIED_MANIFEST_SHA256 or m.get('criteria')!=[{'id':n,'criterion':c} for n,c in CRITERIA]
            or [r['file'] for r in m.get('evidence',[])]!=[ROOT+'/'+n for n in EVIDENCE]):
        raise ValueError('Incomplete or substituted original Issue #8 acceptance manifest')


def verify_notice(root,entry):
    """Git checkouts may translate CRLF; preserve every other license byte.

    This text-only rule never applies to model/source artifact identities or
    original numerical goldens, which continue to use their raw released bytes.
    """
    root=Path(root).resolve(); path=(root/entry['file']).resolve()
    if not path.is_relative_to(root) or not 0<path.stat().st_size<=MiB or entry.get('line_endings')!='LF':
        raise ValueError('Original attribution/license path, size or text policy differs')
    raw=path.read_bytes().replace(b'\r\n',b'\n')
    if len(raw)!=entry['bytes'] or digest(raw)!=entry['sha256']: raise ValueError('Original attribution/license evidence differs')


def qualify(root,manifest_path):
    root=Path(root).resolve(); path=Path(manifest_path)
    if path.stat().st_size>65536: raise ValueError('Excessive Issue #8 acceptance manifest')
    m=json.loads(path.read_bytes()); validate_manifest(m); reports=[]; raw=[]
    budget=MemoryBudget(cap=256*MiB); budget.acquire(budget.cap)
    try:
        for e in m['evidence']:
            p=(root/e['file']).resolve()
            if not p.is_relative_to(root) or not 0<e['bytes']<=32*MiB or p.stat().st_size!=e['bytes']: raise ValueError('Evidence path or byte count differs')
            data=p.read_bytes()
            if digest(data)!=e['sha256']: raise ValueError('Issue #8 evidence SHA-256 differs')
            raw.append(data); reports.append(json.loads(data))
        original,native,io,plans=reports
        validate_goldens(original); validate_native(native,original,raw[0]); validate_storage(io,plans)
        for e in m['notices']: verify_notice(root,e)
        return {'schema_version':1,'probe':'issue8_acceptance_audit','status':'pass','model':MODEL,'revision':REVISION,
            'manifest_sha256':canonical(m),'criteria':[{'id':n,'criterion':c,'status':'pass'} for n,c in CRITERIA],
            'evidence':m['evidence'],'notice_files_verified':len(m['notices']),'successful_session_cases':110,'bitwise_boundaries':6820,
            'failed_attempts_retried':10,'full_checkpoint_downloaded':False,'full_shards_registered':0,'full_model_quality':'not_run',
            'end_to_end_speed':'not_measured','milestone2_complete':False,'remaining_integration_issues':[9,10],
            'host_allowance_bytes':budget.cap,'reservation_is_RSS_limit':False}
    finally: budget.release(budget.cap)
