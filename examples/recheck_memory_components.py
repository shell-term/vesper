"""Run existing complete numerical replays under a single process-wide governor."""

import argparse
import json
from pathlib import Path
import runpy
import sys

from vesper.memory import RuntimeMemory,MemoryPolicy
from vesper.resources import GiB


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--component',choices=('engram','block_chain'),required=True)
    p.add_argument('--memory-output',type=Path,required=True)
    p.add_argument('arguments',nargs=argparse.REMAINDER)
    a=p.parse_args(); args=a.arguments
    if args and args[0]=='--': args=args[1:]
    script=Path(__file__).parent/('recheck_'+a.component+'.py')
    previous=list(sys.argv); policy=RuntimeMemory(MemoryPolicy(host_cap_bytes=16*GiB))
    try:
        sys.argv=[str(script),*args]
        with policy:
            try: runpy.run_path(str(script),run_name='__main__')
            except SystemExit as error:
                if error.code not in (None,0): raise
            policy.sample('after_complete_native_replay')
    finally: sys.argv=previous
    result={'schema_version':1,'probe':'governed_complete_native_replay','component':a.component,'status':'pass',
        'governor':policy.snapshot(),'all_runtime_charges_released':policy.host==0 and not any(policy.device_used.values()),
        'full_model_quality':'not_run','end_to_end_speed':'not_measured'}
    a.memory_output.parent.mkdir(parents=True,exist_ok=True)
    a.memory_output.write_bytes((json.dumps(result,indent=2)+'\n').encode())
    print(json.dumps({'component':a.component,'global_host_peak_bytes':policy.host_peak,'remaining_host_bytes':policy.host,'device_bytes':policy.device_used}))


if __name__=='__main__': main()
