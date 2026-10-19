"""Compare complete original output logits and deterministic token selection."""

import argparse
import json
from pathlib import Path
from vesper.output_reference import compare_output

p=argparse.ArgumentParser(description=__doc__)
for n in ('metadata','fixtures','prefix','reference','output'): p.add_argument('--'+n,type=Path,required=True)
a=p.parse_args(); result=compare_output(a.metadata,a.fixtures,a.prefix,a.reference,lambda r:print(json.dumps(r),flush=True))
a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps({'status':result['status'],'cases':len(result['cases'])}))
if result['status']!='pass': raise SystemExit(2)
