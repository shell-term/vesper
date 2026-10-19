"""Plan capacity, or explicitly register already-existing original shards.

Neither operation downloads the full checkpoint. Registration streams all 48
existing files once and produces integrity sidecars in a NEW directory.
"""

import argparse
import json
from pathlib import Path

from vesper.artifact_repository import register_existing
from vesper.model_artifacts import full_import_plan


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    plan=sub.add_parser('plan'); plan.add_argument('--target',type=Path,required=True)
    register=sub.add_parser('register'); register.add_argument('--originals',type=Path,required=True)
    register.add_argument('--integrity',type=Path,required=True); register.add_argument('--execute',action='store_true')
    for command in (plan,register):
        command.add_argument('--metadata',type=Path,required=True); command.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    result=(full_import_plan(a.metadata,a.target) if a.command=='plan' else
        register_existing(a.metadata,a.originals,a.integrity,execute=a.execute))
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_bytes((json.dumps(result,indent=2)+'\n').encode())
    print(json.dumps({k:v for k,v in result.items() if k in ('status','kind','required_free_bytes','capacity_sufficient','full_checkpoint_downloaded')}))


if __name__=='__main__': main()
