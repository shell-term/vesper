"""Explicit opt-in extended input/Engram and indexed-storage checks."""

import argparse
import ctypes
import json
import os
from pathlib import Path
import re

from .memory import MemoryPolicy,RuntimeMemory
from .resources import GiB
from .sequence_validation import read_report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation',choices=('prepare-input','reference','native','check','indexed-prepare','indexed-native'))
    p.add_argument('--artifacts',type=Path,default=Path('artifacts'))
    p.add_argument('--run',default='reasoning-v1')
    p.add_argument('--reference',type=Path)
    p.add_argument('--nvrtc-bin',type=Path)
    a=p.parse_args()
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,63}',a.run):p.error('Expected a simple sequence run name')
    run=a.artifacts/'sequence'/a.run
    reference=a.reference or run/'original.json'
    def progress(r):print(json.dumps(r),flush=True)
    if a.operation=='check':
        from .sequence_validation import validate_native
        original,raw=read_report(reference);native,_=read_report(run/'native.json');validate_native(native,original,raw)
        progress({'status':'pass','gate':'input_Engram_component_only','model_inference_run':False});return
    name={'prepare-input':'input-plan.json','reference':'original.json','native':'native.json','indexed-prepare':'indices.json','indexed-native':'native.json'}[a.operation]
    if (run/name).exists():raise FileExistsError('Refusing to replace a complete sequence capture')
    cap=2*GiB if a.operation in ('prepare-input','indexed-prepare') else 64*GiB if a.operation=='indexed-native' else 8*GiB
    handle=nvrtc=None
    if a.operation=='reference' and a.nvrtc_bin is not None:
        if os.name!='nt':p.error('Explicit NVRTC DLL preload is Windows-only')
        handle=os.add_dll_directory(str(a.nvrtc_bin.resolve()));nvrtc=ctypes.CDLL(str(a.nvrtc_bin.resolve()/'nvrtc64_130_0.dll'))
    try:
        with RuntimeMemory(MemoryPolicy(host_cap_bytes=cap)) as policy:
            if a.operation=='prepare-input':
                from .sequence_assets import prepare_input
                report=prepare_input(a.artifacts,run,progress=progress)
            elif a.operation=='indexed-prepare':
                from .indexed_validation import prepare_indices
                report=prepare_indices(a.artifacts,run,progress)
            elif a.operation=='indexed-native':
                from .indexed_validation import run_indexed
                original,raw=read_report(reference,limit=64*1024**2);report=run_indexed(a.artifacts,run,original,raw,progress)
            elif a.operation=='reference':
                from .sequence_validation import capture_reference
                report=capture_reference(a.artifacts,run,progress)
            else:
                from .sequence_validation import run_native
                original,raw=read_report(reference);report=run_native(a.artifacts,run,original,raw,progress)
            report['memory_before_policy_close']=policy.snapshot()
        report['memory_after_policy_close']=policy.snapshot()
        if a.operation in ('reference','native'):
            from .sequence_validation import validate_reference,validate_native
            if a.operation=='reference':validate_reference(report,require_qualified=False)
            else:validate_native(report,original,raw)
        # Preparation functions already sealed their data plan separately.
        if a.operation in ('prepare-input','indexed-prepare'):
            report={'schema_version':1,'probe':'bounded_sequence_preparation_resources','status':'pass',
                'prepared_identity_sha256':report.get('plan_sha256',report.get('manifest_sha256')),
                'model_inference_run':False,'memory_before_policy_close':report['memory_before_policy_close'],
                'memory_after_policy_close':report['memory_after_policy_close']}
            target=run/'preparation-resources.json'
        else:target=run/name
        with target.open('xb') as f:f.write((json.dumps(report,indent=2)+'\n').encode())
        progress({'status':'pass','probe':report['probe']})
    finally:
        if handle is not None:handle.close()


if __name__=='__main__':main()
