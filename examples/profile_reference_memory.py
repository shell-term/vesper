"""Measure the optional original reference in its own short-lived process.

Run with the pinned reference environment, separately from native-only probes.
Phase deltas are observations, not exact attribution to a library or OS limits.
"""

import argparse
import ctypes
import json
import os
from pathlib import Path

from vesper.memory import process_memory,current_policy
from vesper.resources import GiB,physical_memory
from vesper.model_artifacts import digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('metadata','fixtures','reference','output','original-output'): p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--reference-nvrtc',type=Path)
    a=p.parse_args(); phases=[]; directory=None; library=None; version=None; torch=None
    if current_policy() is not None: raise RuntimeError('Profile the optional reference in a separate process')
    def sample(label):
        available=physical_memory()['available']
        if available<32*GiB: raise MemoryError('Reference observation is below the physical reserve; stop this probe')
        row={'phase':label,'process':process_memory(),'available_bytes':available}
        if torch is not None and torch.cuda.is_initialized():
            torch.cuda.synchronize()
            row['torch_cuda']={'allocated_bytes':torch.cuda.memory_allocated(0),'reserved_bytes':torch.cuda.memory_reserved(0),
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(0),'peak_reserved_bytes':torch.cuda.max_memory_reserved(0),
                'includes_direct_driver_or_VMM_allocations':False}
        phases.append(row)
    sample('before_optional_framework_imports')
    try:
        if a.reference_nvrtc is not None:
            if os.name!='nt': raise ValueError('Explicit reference DLL preloading is Windows only')
            dll=a.reference_nvrtc.resolve(strict=True); directory=os.add_dll_directory(str(dll.parent)); library=ctypes.CDLL(str(dll))
            major,minor=ctypes.c_int(),ctypes.c_int()
            if library.nvrtcVersion(ctypes.byref(major),ctypes.byref(minor)): raise OSError('Cannot read reference NVRTC version')
            version=[major.value,minor.value]
        import torch
        import tilelang
        sample('after_optional_framework_imports')
        from vesper.engram_reference import compare_engram
        from vesper.engram_validation import validate_goldens
        steps=0
        def progress(record):
            nonlocal steps
            if record.get('status')=='pass':
                steps+=1
                if steps%10==0: sample('reference_steps_'+str(steps))
            print(json.dumps(record),flush=True)
        original=compare_engram(a.metadata,a.fixtures,a.reference,progress)
        validate_goldens(original); sample('after_complete_original_comparison_and_cleanup')
        original['reference_nvrtc_version']=version
        raw=(json.dumps(original,indent=2,allow_nan=False)+'\n').encode()
        a.original_output.parent.mkdir(parents=True,exist_ok=True); a.original_output.write_bytes(raw)
        result={'schema_version':1,'probe':'separate_optional_reference_memory','status':'pass',
            'phases':phases,'torch_version':torch.__version__,'tilelang_version':tilelang.__version__,
            'reference_nvrtc_version':version,'reference_report_sha256':digest(raw),
            'qualified_identity_sha256':original['source_fixture_compiler_identity_sha256'],
            'complete_goldens_sha256':original['complete_goldens_sha256'],
            'successful_session_cases':len(original['session_cases']),'bitwise_boundaries':6820,
            'VMM':original['memory']['active']['VMM'],'VMM_after_close':original['memory']['cleanup']['VMM'],
            'sampled_process_peaks':{k:max((r['process'][k] for r in phases if r['process'].get(k) is not None),default=None)
                for k in ('resident_bytes','private_commit_bytes','private_resident_bytes')},
            'minimum_sampled_available_bytes':min(r['available_bytes'] for r in phases),
            'measurement_process':'separate_optional_reference_process','native_owned_cap_includes_reference_frameworks':False,
            'phase_deltas_are_exact_component_attribution':False,'reservation_is_RSS_or_OS_limit':False,
            'GPU_reset':False,'pagefile_changes':False,'full_checkpoint_downloaded':False,
            'full_model_quality':'not_run','end_to_end_speed':'not_measured'}
        a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_bytes((json.dumps(result,indent=2)+'\n').encode())
        print(json.dumps({'status':'pass','separate_reference_process':True,'process_peaks':result['sampled_process_peaks']}))
    finally:
        if directory: directory.close()


if __name__=='__main__': main()
