"""Compare the complete bounded Engram matrix with unchanged original methods."""

import argparse
import ctypes
import json
import os
from pathlib import Path

from vesper.engram_reference import compare_engram
from vesper.engram_validation import validate_goldens


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('metadata','fixtures','reference','output'): p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--reference-nvrtc',type=Path,help='Windows: preload the matching reference NVRTC DLL; native compilation still uses its registered 12.4 library')
    a=p.parse_args(); directory=None; library=None; version=None
    if a.reference_nvrtc is not None:
        if os.name!='nt': raise ValueError('Explicit NVRTC DLL preloading is a Windows-only reference option')
        dll=a.reference_nvrtc.resolve(strict=True); directory=os.add_dll_directory(str(dll.parent)); library=ctypes.CDLL(str(dll))
        major,minor=ctypes.c_int(),ctypes.c_int()
        if library.nvrtcVersion(ctypes.byref(major),ctypes.byref(minor)): raise OSError('Cannot read original reference NVRTC version')
        version=[major.value,minor.value]
    try: result=compare_engram(a.metadata,a.fixtures,a.reference,lambda r:print(json.dumps(r),flush=True))
    finally:
        if directory: directory.close()
    result['reference_nvrtc_version']=version
    validate_goldens(result); a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_bytes((json.dumps(result,indent=2,allow_nan=False)+'\n').encode())
    print(json.dumps({'status':'pass','cases':len(result['session_cases']),'boundaries':6820}))


if __name__=='__main__': main()
