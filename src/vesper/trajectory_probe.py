"""Actual longer whole-model trajectory preparation, capture and comparison."""

import argparse
import ctypes
import json
import os
from pathlib import Path
import re

from .memory import RuntimeMemory,MemoryPolicy
from .resources import GiB
from .trajectory_evidence import read,validate_original,validate_native


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation',choices=('prepare','reference','native','check'))
    p.add_argument('--artifacts',type=Path,default=Path('artifacts'));p.add_argument('--run',default='coding-v1')
    p.add_argument('--case');p.add_argument('--reference-run',type=Path)
    p.add_argument('--reference',type=Path);p.add_argument('--nvrtc-bin',type=Path)
    a=p.parse_args()
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,63}',a.run):p.error('Expected a simple whole-model run name')
    if a.nvrtc_bin is not None and a.operation!='reference':p.error('--nvrtc-bin is reference-only')
    if a.case is not None and a.operation!='prepare':p.error('--case is preparation-only; execution uses the sealed plan')
    run=a.artifacts/'trajectories'/a.run;reference_run=a.reference_run or run;reference=a.reference or reference_run/'original.json'
    def progress(value):print(json.dumps(value),flush=True)
    if a.operation=='check':
        original,raw=read(reference);native,_=read(run/'native.json');validate_native(native,run,original,raw,reference_run)
        progress({'status':'pass','gate':'whole_model_declared_task'});return
    name={'prepare':'preparation.json','reference':'original.json','native':'native.json'}[a.operation]
    if (run/name).exists():raise FileExistsError('Refusing to replace a completed whole-model capture')
    handle=None
    try:
        if a.operation=='reference':
            if a.nvrtc_bin is not None:
                if os.name!='nt':p.error('Explicit NVRTC DLL preload is Windows-only')
                handle=os.add_dll_directory(str(a.nvrtc_bin.resolve()));nvrtc=ctypes.CDLL(str(a.nvrtc_bin.resolve()/'nvrtc64_130_0.dll'))
            from .trajectory_reference import capture_original
            report=capture_original(a.artifacts,run,progress);validate_original(report,run,require_qualified=False)
        else:
            with RuntimeMemory(MemoryPolicy(host_cap_bytes=64*GiB)) as policy:
                if a.operation=='prepare':
                    from .trajectory_prepare import prepare
                    report=prepare(a.artifacts,run,case_id=a.case or 'coding-python-expression',progress=progress)
                else:
                    from .trajectory_native import run_native
                    original,raw=read(reference);report=run_native(a.artifacts,run,original,raw,reference_run,progress)
                report['memory_before_policy_close']=policy.snapshot()
            report['memory_after_policy_close']=policy.snapshot()
            if a.operation=='native':validate_native(report,run,original,raw,reference_run)
        with (run/name).open('xb') as stream:stream.write((json.dumps(report,indent=2)+'\n').encode())
        progress({'status':'pass','probe':report['probe'],'completion':report.get('completion'),'score':report.get('quality_score')})
    finally:
        if handle is not None:handle.close()


if __name__=='__main__':main()
