"""Explicit bounded import of pinned metadata, two controls and <=1024 rows/layer."""

import argparse
import json
from pathlib import Path

from vesper.engram_assets import fetch_controls,prepare_rows
from vesper.engram_corpus import row_plan
from vesper.model_artifacts import fetch_headers,load_catalog


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata',type=Path,required=True); p.add_argument('--fixtures',type=Path,required=True)
    a=p.parse_args(); fetch_headers(a.metadata,lambda r:print(json.dumps(r),flush=True))
    _,catalog=load_catalog(a.metadata); plan=row_plan(a.metadata)
    a.fixtures.mkdir(parents=True,exist_ok=True)
    for layer in (1,14): fetch_controls(a.metadata,a.fixtures/f'controls{layer}',layer)
    rows=prepare_rows(a.metadata,a.fixtures/'rows',plan['required_rows'],lambda r:print(json.dumps(r),flush=True))
    (a.fixtures/'row-plan.json').write_bytes((json.dumps(plan,indent=2)+'\n').encode())
    print(json.dumps({'status':'prepared','original_control_bytes':2*157521920,'original_row_payload_bytes':plan['original_row_payload_bytes'],
        'full_checkpoint_downloaded':False,'catalog_sha256':catalog['catalog_sha256']}))


if __name__=='__main__': main()
