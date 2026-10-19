"""Test the default ceiling/floor gates without allocating 96 GiB of payload."""

import argparse
import json
from pathlib import Path

from vesper.memory import RuntimeMemory
from vesper.resources import GiB


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    observation={'available':128*GiB}; policy=RuntimeMemory(available=lambda:observation['available'])
    with policy:
        # A symbolic admission is distinct from the touched-buffer scale probe.
        lease=policy.reserve(96*GiB-policy.host,'admission','default_cap_boundary',kind='boundary_admission_only')
        policy.sample('default_cap_counter_boundary'); at_cap=policy.snapshot()
        try: policy.reserve(1,'scratch','one_byte_over_cap')
        except MemoryError: pass
        else: raise RuntimeError('Default cap admitted another byte')
        lease.release(); observation['available']=32*GiB
        try: policy.reserve(1,'scratch','one_byte_into_floor')
        except MemoryError: pass
        else: raise RuntimeError('Default floor admitted another byte')
        observation['available']=128*GiB; policy.checkpoint(); policy.sample('default_gates_recovered')
    r={'schema_version':1,'probe':'default_memory_admission_boundaries','status':'pass',
        'available_observation_is_injected':True,'physical_payload_allocation_attempted':False,
        'default_cap_refuses_one_more_byte':True,'default_floor_refuses_one_more_byte':True,
        'at_cap_governor':at_cap,'cleanup_governor':policy.snapshot(),
        'scope':'Counter gates only; the separate full-backbone probe touches actual bounded payloads'}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_bytes((json.dumps(r,indent=2)+'\n').encode())
    print(json.dumps({'status':'pass','cap_gib':96,'floor_gib':32,'payload_allocated':False}))


if __name__=='__main__': main()
