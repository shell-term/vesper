"""Plan a local streamed original model comparison from pinned metadata only."""

import argparse
import json
from pathlib import Path
from vesper.decode_plan import reference_plan

p=argparse.ArgumentParser(description=__doc__); p.add_argument('--metadata',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
a=p.parse_args(); r=reference_plan(a.metadata); a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_bytes((json.dumps(r,indent=2)+'\n').encode())
print(json.dumps({'status':'plan_only','control_bytes':r['backbone_control_bytes'],'head_FP32_GPU_bytes':r['head_FP32_GPU_bytes'],
    'whole_checkpoint_in_RAM':False,'full_model_execution_run':False}))
