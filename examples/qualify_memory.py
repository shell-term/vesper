"""Audit the sealed evidence for all six original Issue #9 acceptance checks."""

import argparse
import json
from pathlib import Path

from vesper.memory_qualification import ROOT,qualify


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--manifest',type=Path,default=Path(ROOT)/'manifest.json'); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); result=qualify(a.root,a.manifest)
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_bytes((json.dumps(result,indent=2)+'\n').encode())
    print(json.dumps({'status':result['status'],'criteria':len(result['criteria']),'model_scale_experts':result['original_expert_identities_visited']}))


if __name__=='__main__': main()
