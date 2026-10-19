"""Run actual full-backbone resource geometry with bounded synthetic payloads."""

import argparse
import json
from pathlib import Path
from vesper.memory_scale import scale_probe
from vesper.resources import GiB
from vesper.memory_validation import governed_engram


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--experts-per-layer',type=int,default=384,help='Smaller values are smoke checks, not the full representative gate')
    p.add_argument('--cache-gib',type=int,default=6); p.add_argument('--host-cap-gib',type=int,default=8)
    p.add_argument('--engram-fixtures',type=Path); p.add_argument('--engram-reference-report',type=Path,default=Path('results/2026-10-19-engram/original.json'))
    a=p.parse_args(); native=None if a.engram_fixtures is None else lambda policy:governed_engram(policy,a.metadata,a.engram_fixtures,a.engram_reference_report)
    r=scale_probe(a.metadata,experts_per_layer=a.experts_per_layer,cache_bytes=a.cache_gib*GiB,host_cap=a.host_cap_gib*GiB,
        progress=lambda r:print(json.dumps(r),flush=True),native_check=native)
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_bytes((json.dumps(r,indent=2)+'\n').encode())
    print(json.dumps({'status':r['status'],'experts':r['expert_entries_visited'],'complete_backbone':r['complete_backbone_expert_pass'],
        'host_peak':r['cleanup_governor']['charged_host_peak_bytes'],'after_close':r['cleanup_governor']['charged_host_bytes']}))


if __name__=='__main__': main()
