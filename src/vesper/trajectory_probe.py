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
from .trajectory_workspace import run_path,migrate_input,own_evidence
from .model_artifacts import digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation',choices=('prepare','reference','native','check','migrate-input'))
    p.add_argument('--artifacts',type=Path,default=Path('artifacts'));p.add_argument('--run',default='coding-v1')
    p.add_argument('--case');p.add_argument('--reference-run',type=Path)
    p.add_argument('--reference',type=Path);p.add_argument('--nvrtc-bin',type=Path)
    a=p.parse_args()
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,63}',a.run):p.error('Expected a simple whole-model run name')
    if a.nvrtc_bin is not None and a.operation!='reference':p.error('--nvrtc-bin is reference-only')
    if a.case is not None and a.operation!='prepare':p.error('--case is preparation-only; execution uses the sealed plan')
    workspace,run=run_path(a.artifacts,a.run)
    workspace.check_runtime_module(__file__,'src/vesper/trajectory_probe.py')
    reference_run=own_evidence(a.artifacts,a.reference_run or run)
    reference=own_evidence(a.artifacts,a.reference or reference_run/'original.json')
    def progress(value):print(json.dumps(value),flush=True)
    if a.operation=='migrate-input':progress(migrate_input(a.artifacts,a.run));return
    if a.operation=='check':
        original,raw=read(reference);native,_=read(run/'native.json');validate_native(native,run,original,raw,reference_run)
        progress({'status':'pass','gate':'whole_model_declared_task'});return
    name={'prepare':'preparation.json','reference':'original.json','native':'native.json'}[a.operation]
    if (run/name).exists():raise FileExistsError('Refusing to replace a completed whole-model capture')
    context=workspace.context('pc-'+a.run+'-'+a.operation)
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
        raw=(json.dumps(report,indent=2)+'\n').encode()
        receipt=workspace.stamp_result({'probe':'pc_trajectory_execution_receipt','status':'pass','operation':a.operation,
            'raw_report':{'file':name,'bytes':len(raw),'sha256':digest(raw)},'scope':report['scope']},initial_context=context)
        with (run/name).open('xb') as stream:stream.write(raw)
        with (run/(a.operation+'-context.json')).open('xb') as stream:stream.write((json.dumps(receipt,indent=2)+'\n').encode())
        progress({'status':'pass','probe':report['probe'],'completion':report.get('completion'),'score':report.get('quality_score')})
    finally:
        if handle is not None:handle.close()


if __name__=='__main__':main()
