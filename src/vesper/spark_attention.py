"""Fresh GB10 Attention candidate bundles; PC qualification stays unchanged.

Only the pinned Query storage transformation is reused. CUDA generated on GB10
is separately compiled, recorded and loaded. Building or running this candidate
does not issue an original-kernel numerical qualification receipt.
"""
import ctypes as C
import hashlib
import json
from pathlib import Path
import re

from .attention import (NativeAttention, CANDIDATES, argument_names,
                        kernel_filename, read_bounded)
from .attention_memory import FUNCTION
from .attention_storage import variant_source
from .checkpoint import MODEL, REVISION, REFERENCE_HASHES
from .cuda import Driver, Device, _bind
from .execution_target import resolve_execution_target
from .resources import GiB, MiB, MemoryBudget


def sha(data): return hashlib.sha256(data).hexdigest()


def load_candidate(directory):
    """Validate bounded local provenance and payloads before any CUDA load."""
    directory=Path(directory)
    raw=read_bounded(directory/'manifest.json',128*1024)
    manifest=json.loads(raw)
    expected={'schema_version':1,'kind':'vesper_spark_attention_candidate',
        'model':MODEL,'revision':REVISION,'compute_capability':'12.1',
        'execution_target':'spark-sm121-candidate','qualification':'unqualified_candidate',
        'minimum_driver_api':13000,'nvrtc_version':[13,0],'tilelang_version':'0.1.12',
        'kernel_source_sha256':REFERENCE_HASHES['kernel.py'],
        'original_kernel_parity_qualified':False}
    if not isinstance(manifest,dict) or any(manifest.get(k)!=v for k,v in expected.items()):
        raise ValueError('Foreign or unregistered Spark Attention candidate')
    if sha(read_bounded(directory/'DeepSeek-LICENSE',64*1024))!=REFERENCE_HASHES['LICENSE']:
        raise ValueError('Missing pinned Attention source license')
    original=read_bounded(directory/'kernel.py',MiB)
    if sha(original)!=REFERENCE_HASHES['kernel.py']: raise ValueError('Pinned Attention source differs')
    profiles=manifest.get('profiles')
    if not isinstance(profiles,list) or len(profiles)!=6: raise ValueError('Expected all six Attention profiles')
    binaries={}; total=0
    for profile in profiles:
        if not isinstance(profile,dict): raise ValueError('Invalid Attention profile')
        key=tuple(profile.get(k) for k in ('heads','dimension','diagnostics'))
        if (type(key[0]) is not int or type(key[1]) is not int or type(key[2]) is not bool
                or key not in CANDIDATES or key in binaries): raise ValueError('Unexpected Attention shape')
        filename=kernel_filename(*key)
        if (profile.get('file')!=filename or profile.get('function')!=FUNCTION
                or profile.get('argument_names')!=argument_names(key[2])
                or type(profile.get('dynamic_shared_bytes')) is not int
                or not 0<profile['dynamic_shared_bytes']<=128*1024
                or profile.get('binary_reproduction') is not True):
            raise ValueError('Attention launch contract differs')
        rendered,audit=variant_source(original,'register-query',key[2])
        if profile.get('storage_transformation')!=audit:
            raise ValueError('Attention Query-only AST transformation differs')
        if read_bounded(directory/(filename+'.py'),MiB)!=rendered:
            raise ValueError('Attention generated Python source differs')
        cuda=read_bounded(directory/(filename+'.cu'),MiB)
        host=read_bounded(directory/(filename+'.host.py'),MiB)
        binary=read_bounded(directory/filename,2*MiB)
        if (sha(cuda)!=profile.get('generated_cuda_sha256') or sha(binary)!=profile.get('cubin_sha256')
                or sha(host)!=profile.get('generated_host_sha256')):
            raise ValueError('Attention payload hash differs')
        from .prefix_reference import required_shared_bytes
        if required_shared_bytes(host.decode('utf-8'))!=profile['dynamic_shared_bytes']:
            raise ValueError('Attention dynamic shared-memory contract differs from generated launch source')
        signature=re.search(r'void '+FUNCTION+r'\(([^)]*)\)',cuda.decode('utf-8'))
        if (signature is None or [p.strip().split()[-1].lstrip('*') for p in signature[1].split(',')]
                !=profile['argument_names']):
            raise ValueError('Attention generated CUDA launch signature differs')
        if len(binary)<20 or binary[:4]!=b'\x7fELF' or int.from_bytes(binary[18:20],'little')!=190:
            raise ValueError('Expected an ELF CUDA CUBIN')
        total+=len(binary)
        if total>8*MiB: raise ValueError('Combined Attention bundle exceeds its allowance')
        binaries[key]=binary
    return manifest,binaries,sha(raw)


def build_candidate(workspace,relative,*,ordinal=0):
    """Reference-environment build only. No Torch dependency in native loading."""
    workspace.check()
    workspace.check_runtime_module(__file__,'src/vesper/spark_attention.py')
    if workspace.context()['platform']!='dgx-spark': raise ValueError('Expected bound Spark workspace')
    budget=MemoryBudget(cap=3*GiB,owner='spark_attention_builder'); budget.acquire(budget.cap)
    try:
        target=resolve_execution_target('spark-sm121-candidate')
        driver=Driver(); inventory=driver.inventory()
        device=next(d for d in inventory['devices'] if d['ordinal']==ordinal)
        observed=target.validate_device(device,inventory)
        import torch
        import tilelang
        from tilelang.contrib.nvrtc import compile_cuda,get_nvrtc_version
        from tilelang.env import TILELANG_TEMPLATE_PATH,CUTLASS_INCLUDE_DIR,CUDA_HOME
        from .attention_storage import load_variant
        from .attention_storage_reference import _host_source
        from .checkpoint import fetch_reference
        from .prefix_reference import required_shared_bytes
        if tilelang.__version__!='0.1.12' or tuple(get_nvrtc_version())!=(13,0):
            raise ValueError('Spark candidate builder requires TileLang 0.1.12 and NVRTC 13.0')
        directory=workspace.artifact_path(relative,create_parent=True)
        if directory.exists(): raise FileExistsError('Use a new immutable Attention build directory')
        directory.mkdir()
        fetch_reference(directory)
        raw=read_bounded(directory/'kernel.py',MiB)
        query=_bind(driver.lib,'cuDeviceGetAttribute',[C.POINTER(C.c_int),C.c_int,C.c_int])
        limit=C.c_int(); driver.check(query(C.byref(limit),97,ordinal))
        options=[f'-I{TILELANG_TEMPLATE_PATH}',f'-I{CUTLASS_INCLUDE_DIR}',f'-I{CUDA_HOME}/include',
                 f'-I{CUDA_HOME}/include/cccl','-D__CUDACC_VER_MAJOR__=13']
        profiles=[]
        with torch.cuda.device(ordinal):
            for heads,dimension,diagnostics in CANDIDATES:
                budget.acquire(0)
                factory,audit=load_variant(raw,directory/'variants','register-query',diagnostics)
                kernel=factory(heads,dimension,dimension**-.5)
                source=kernel.get_kernel_source()
                host=_host_source(kernel)
                shared=required_shared_bytes(host)
                if shared>limit.value: raise ValueError('Spark candidate exceeds measured shared-memory capacity')
                signature=re.search(r'void '+FUNCTION+r'\(([^)]*)\)',source)
                if signature is None: raise ValueError('Missing Attention launch signature')
                args=[part.strip().split()[-1].lstrip('*') for part in signature[1].split(',')]
                if args!=argument_names(diagnostics): raise ValueError('Attention generated ABI differs')
                flags=kernel.adapter.lib_generator.compile_flags or []
                extra=[item for flag in flags for item in flag.split() if item not in options]
                binary=bytes(compile_cuda(source,'cubin',121,options+extra))
                cached=read_bounded(kernel.adapter.lib_generator.libpath,2*MiB)
                if binary!=cached: raise ValueError('Spark Attention binary reproduction failed')
                filename=kernel_filename(heads,dimension,diagnostics)
                (directory/filename).write_bytes(binary)
                (directory/(filename+'.cu')).write_bytes(source.encode())
                (directory/(filename+'.host.py')).write_bytes(host.encode())
                rendered,_=variant_source(raw,'register-query',diagnostics)
                (directory/(filename+'.py')).write_bytes(rendered)
                profiles.append({'heads':heads,'dimension':dimension,'diagnostics':diagnostics,
                    'file':filename,'function':FUNCTION,'argument_names':args,'dynamic_shared_bytes':shared,
                    'generated_cuda_sha256':sha(source.encode()),'generated_host_sha256':sha(host.encode()),'cubin_sha256':sha(binary),
                    'binary_reproduction':True,'storage_transformation':audit})
        (directory/'DeepSeek-LICENSE').write_bytes(read_bounded(directory/'LICENSE',64*1024))
        manifest={'schema_version':1,'kind':'vesper_spark_attention_candidate','model':MODEL,'revision':REVISION,
            'compute_capability':'12.1','execution_target':target.name,'qualification':'unqualified_candidate',
            'minimum_driver_api':13000,'nvrtc_version':[13,0],'tilelang_version':'0.1.12',
            'kernel_source_sha256':REFERENCE_HASHES['kernel.py'],'original_kernel_parity_qualified':False,
            'device':observed,'measured_optin_shared_bytes':limit.value,'profiles':profiles}
        (directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
        load_candidate(directory)
        return manifest
    finally: budget.release(budget.cap)


class SparkAttentionCandidate(NativeAttention):
    def __init__(self,bundle,ordinal=0,*,profile='baseline',execution_target='spark-sm121-candidate'):
        self.target=resolve_execution_target(execution_target)
        if not self.target.is_candidate: raise ValueError('Use NativeAttention for the existing PC profile')
        super().__init__(bundle,ordinal,profile=profile)

    def __enter__(self):
        with self.lock:
            if self.opened: raise RuntimeError('Spark Attention candidate is already open')
            try:
                self.budget.acquire(self.host_allowance)
                self.stack.callback(self.budget.release,self.host_allowance)
                self.manifest,binaries,self.manifest_sha256=load_candidate(self.bundle)
                self.driver=Driver(); inventory=self.driver.inventory()
                self.device_info=next(d for d in inventory['devices'] if d['ordinal']==self.ordinal)
                self.target.validate_device(self.device_info,inventory)
                self.device=self.stack.enter_context(Device(self.driver,self.ordinal,cap=self.device_cap,free_floor=2*GiB))
                query=_bind(self.driver.lib,'cuDeviceGetAttribute',[C.POINTER(C.c_int),C.c_int,C.c_int])
                limit=C.c_int(); self.driver.check(query(C.byref(limit),97,self.ordinal))
                attribute=_bind(self.driver.lib,'cuFuncSetAttribute',[C.c_void_p,C.c_int,C.c_int])
                self.kernels={}
                for p in self.manifest['profiles']:
                    if p['dynamic_shared_bytes']>limit.value: raise ValueError('Attention shared-memory capacity changed')
                    key=p['heads'],p['dimension'],p['diagnostics']
                    module=self.device.load(binaries[key]); function=C.c_void_p()
                    self.driver.check(self.driver.cuModuleGetFunction(C.byref(function),module,FUNCTION.encode()))
                    self.driver.check(attribute(function,8,p['dynamic_shared_bytes']))
                    self.kernels[key]=function,p
                self.opened=True; return self
            except BaseException: self.stack.close(); raise

    def stats(self): return {**super().stats(),**self.target.describe(),'original_kernel_parity_qualified':False}
