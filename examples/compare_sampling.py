"""Compare original stochastic sampling over the complete vocabulary."""

import argparse
import json
from pathlib import Path
from vesper.sampling_reference import compare_sampling

p=argparse.ArgumentParser(description=__doc__); p.add_argument('--reference',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
a=p.parse_args(); r=compare_sampling(a.reference,lambda r:print(json.dumps(r),flush=True))
a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_bytes((json.dumps(r,indent=2)+'\n').encode())
print(json.dumps({'status':r['status'],'cases':len(r['cases'])}))
if r['status']!='pass': raise SystemExit(2)
