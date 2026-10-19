"""Bind complete output/sampling component evidence, distinct from model fidelity."""

from .checkpoint import MODEL,REVISION
from .model_artifacts import canonical
from .output_assets import QUALIFIED_IDENTITY
from .output_corpus import cases
from .sampling import SAMPLING_CUBIN_SHA256
from .sampling_reference import sampling_cases

OUTPUT_IDENTITY='c944f4b3b1bb5c45ed6ea52fb2501e87942785c6e7cf09eda67df960f6c94751'
OUTPUT_GOLDENS='ff2440434e88c029c6d5f8de5eab8e814b0c7f6390d3b16e8719e0cc9350054f'
SAMPLING_IDENTITY='456b9e6d5dc868778ff430d01e788a4398414ee02bc58cb14ac66ea5ab828f5e'
SAMPLING_GOLDENS='6a3f230e47a833622ee8e9d9f11e3ed8c88870905fb2c2ac3e658fd015e1c371'
STAGES=('pre_products_fp32','pre_sum_fp32','pre_output_bf16','norm_square_fp32','norm_mean_fp32','norm_mean_eps_fp32',
    'norm_rstd_fp32','norm_normalized_fp32','norm_weighted_fp32','norm_output_bf16','promoted_fp32','logits_fp32','greedy_ids_i64')


def output_identity(r):
    return canonical({k:r[k] for k in ('model','revision','fixture','original_source','original_methods','torch_version','torch_revision')})


def output_goldens(r):
    return canonical([{k:c[k] for k in ('meta','input_sha256','incoming_pre_sha256','checks','sampling')} for c in r['cases']])


def sampling_identity(r):
    return canonical({k:r[k] for k in ('model','revision','original_source','sample_AST_sha256','torch_version','torch_revision','pytorch_sources','compiler','cubin_sha256','device_profile')})


def sampling_goldens(r): return canonical(r['cases'])


def valid_checks(checks,names):
    if (tuple(c['check'] for c in checks)!=tuple(names) or any(c['status']!='pass' or c['mismatches'] or c['elements']<=0
            or c['expected_sha256']!=c['actual_sha256'] or len(c['expected_sha256'])!=64 or not c['shape'] for c in checks)):
        raise ValueError('Incomplete or differing typed/shaped original output/sampling boundaries')


def check_scope(r,probe):
    if ((r.get('probe'),r.get('status'),r.get('model'),r.get('revision'))!=(probe,'pass',MODEL,REVISION)
            or r.get('torch_version')!='2.14.1+cu130' or r.get('torch_revision')!='5c4886908584029761b579af026dcfb627c84070'
            or r.get('text_generation_available') is not False or r.get('full_model_quality')!='not_run'
            or r.get('end_to_end_speed')!='not_measured'): raise ValueError('Unqualified original component scope/profile')
    g=r['governor_after_close']
    if (g['charged_host_bytes'] or any(g['device_used_bytes'].values()) or g['live_leases'] or g['live_reclaimers']
            or g['inflight_operations'] or g['monitor_error'] is not None or g['monitor_thread_stopped'] is not True
            or g['available_floor_bytes']!=32*1024**3 or g['reservation_is_RSS_or_OS_limit'] is not False):
        raise ValueError('Original component cleanup or resource scope differs')


def validate_output(r):
    check_scope(r,'original_output_complete_parity')
    if (output_identity(r)!=OUTPUT_IDENTITY or output_goldens(r)!=OUTPUT_GOLDENS or canonical(r['fixture'])!=QUALIFIED_IDENTITY
            or [c['meta'] for c in r['cases']]!=cases() or r['full_vocabulary']!=129280 or r['head_compute_dtype']!='float32'
            or r['native_active']['cublas_version']!=120402 or r['native_active']['cublas_math_mode']!=0
            or r['native_cleanup']['device_owned_bytes'] or r['native_cleanup']['host_reserved_bytes']):
        raise ValueError('Original output source, complete corpus, FP32 head or cleanup differs')
    for c in r['cases']:
        valid_checks(c['checks'],STAGES)
        if len(c['sampling'])!=2: raise ValueError('Connected output sampling corpus is incomplete')
        for sample,temperature in zip(c['sampling'],(0.,.7)):
            names=('ids_i64',) if temperature==0 else ('ids_i64','scaled_fp32','probabilities_fp32','exponential_fp32','scores_fp32')
            valid_checks(sample['checks'],names)
            if sample['status']!='pass' or sample['temperature']!=temperature: raise ValueError('Connected output sampling differs')
    negative=r['negative_controls']
    if len(negative)!=1 or negative[0]['check']!='replace_original_FP32_head_with_BF16_GEMM' or negative[0]['status']!='fail' or negative[0]['mismatches']<=0:
        raise ValueError('Reduced-precision head control is missing')
    return {c['meta']['id']:c for c in r['cases']}


def validate_sampling(r):
    check_scope(r,'original_full_vocabulary_sampling')
    if (sampling_identity(r)!=SAMPLING_IDENTITY or sampling_goldens(r)!=SAMPLING_GOLDENS
            or r['cubin_sha256']!=SAMPLING_CUBIN_SHA256 or [c['meta'] for c in r['cases']]!=sampling_cases()):
        raise ValueError('Original sampler identity or complete goldens differ')
    for c in r['cases']:
        names=('ids_i64',) if c['meta']['temperature']==0 else ('ids_i64','scaled_fp32','probabilities_fp32','exponential_fp32','scores_fp32')
        valid_checks(c['checks'],names)
        if ((c['expected_offset_before'],c['expected_offset_after'])!=(c['actual_offset_before'],c['actual_offset_after'])
                or c['status']!='pass'): raise ValueError('Original Philox state advancement differs')
    if (len(r['faults'])!=4 or len(r['retries'])!=4 or any(f['status']!='pass' or not all(f[k] for k in
            ('committed_seed_offset_unchanged','temporary_buffers_released','retry_all_original_boundaries_equal')) for f in r['faults'])
            or r['snapshots']['status']!='pass' or r['snapshots']['original_second_draw_reproduced'] is not True):
        raise ValueError('Original sampler failure/retry/state evidence is incomplete')
    for c in r['retries']: valid_checks(c['checks'],('ids_i64','scaled_fp32','probabilities_fp32','exponential_fp32','scores_fp32'))
    negative=r['negative_controls']
    if len(negative)!=1 or negative[0]['check']!='replace_original_fast_log_with_logf' or negative[0]['status']!='fail' or not negative[0]['mismatches']:
        raise ValueError('Changed sampling arithmetic was not detected')
    return r['cases']
