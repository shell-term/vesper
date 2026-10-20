"""Exact cubin reuse plus the complete, unchanged five-call model corpus."""

from .generation_validation import validate_memory
from .indexed_validation import validate_indexed
from .model_artifacts import canonical,digest
from .resources import GiB,MiB

QUALIFIED_CUBIN_IMAGE_SET='677c1c6d72ab5d38ac0bfcff04d1457232921e15cb8cc5310eefe0c265c82f5a'


def image_identity(images):
    return canonical(sorted((i['compiler']['architecture'],i['compiler']['concatenated_source_sha256'],i['sha256']) for i in images))


def validate_reuse(native,native_raw,compiler,original,original_raw,indices):
    validate_indexed(native,original,original_raw,indices)
    if (compiler.get('probe'),compiler.get('status'),compiler.get('GPU_execution_performed'),compiler.get('whole_model_replay_reference_sha256'))!=(
            'sealed_fast_cache_exact_image_recheck','pass',False,digest(native_raw)):raise ValueError('Different compile-only/full-model reference binding')
    for report,name in ((native,'compiler_cache'),(compiler,'cache')):
        cache=report[name];after=report[name+'_after_close'] if name=='compiler_cache' else report['cache_after_close']
        if (cache['entries']!=14 or cache['misses']!=14 or cache['hits']<14 or cache['cubin_bytes']>32*MiB
                or after['entries'] or after['cubin_bytes'] or after['host_reserved_bytes']):raise ValueError('Compiled-image reuse is not bounded/released')
        if image_identity(report['compiled_images'])!=QUALIFIED_CUBIN_IMAGE_SET:raise ValueError('Unqualified cubin image bytes/source identity')
        if sum(i['bytes'] for i in report['compiled_images'])!=cache['cubin_bytes']:raise ValueError('Missing actual compiled images')
        for i in report['compiled_images']:
            c=i['compiler'];arch=c['architecture']
            if c['nvrtc_version']!=[12,4] or arch not in ('89','86') or c['options']!=[
                    f'--gpu-architecture=sm_{arch}','--std=c++17','--fmad=false','--ftz=false','--include-path=<CUDA>/include']:
                raise ValueError('Different qualified native compiler profile')
    validate_memory(compiler,8*GiB)
    return True
