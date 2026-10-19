"""Audit all five original Issue #8 criteria from complete sanitized evidence."""

import argparse
import json
from pathlib import Path
from vesper.engram_qualification import ROOT,qualify


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--manifest',type=Path,default=Path(ROOT)/'manifest.json'); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); r=qualify(a.root,a.manifest); a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_bytes((json.dumps(r,indent=2)+'\n').encode()); print(json.dumps({'status':r['status'],'criteria':len(r['criteria'])}))


if __name__=='__main__': main()
