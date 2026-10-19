"""Original Engram arithmetic with the native CUDA Driver and bounded rows."""

from contextlib import ExitStack
import ctypes as C
import hashlib
import threading

import numpy as np

from .cuda import Driver,Device,compile_probe
from .engram_assets import CONTROL_LAYOUT
from .normalization import mean_threads
from .resources import MemoryBudget,MiB,GiB

ENGRAM_CUBIN_SHA256="4a293882f82d9558c3b88bc1848b4163eab891d4cba91e931231efef17d48c75"
GATE_STAGES=('hidden_mean_eps_fp32','key_mean_eps_fp32','hidden_rstd_fp32','key_rstd_fp32','rstd_fp32',
    'normalized_dot_fp32','dot_fp32','clamped_abs_dot_fp32','sqrt_abs_dot_fp32','signed_sqrt_fp32','sigmoid_fp32','gate_fp32')


def validate_input(hidden,ids,mask):
    if (not isinstance(hidden,np.ndarray) or hidden.dtype!=np.uint16 or hidden.ndim!=4 or hidden.shape[-2:]!=(4,5120)
            or hidden.shape[0] not in (1,2) or not 1<=hidden.shape[1]<=3 or not hidden.flags.c_contiguous
            or np.any((hidden&0x7f80)==0x7f80) or not isinstance(ids,np.ndarray) or ids.dtype!=np.int64
            or ids.shape!=(*hidden.shape[:2],24) or not ids.flags.c_contiguous):
        raise ValueError("Expected bounded original Engram hidden states and hash IDs")
    if mask is not None and (not isinstance(mask,np.ndarray) or mask.dtype!=np.bool_ or mask.shape!=hidden.shape[:2] or not mask.flags.c_contiguous):
        raise ValueError("Expected original Engram text-token mask")
    return hidden.shape[0]*hidden.shape[1]


class NativeEngram:
    def __init__(self,arrays,rows,ordinal=0):
        self.arrays,self.rows,self.ordinal=arrays,rows,ordinal
        self.budget=MemoryBudget(cap=512*MiB); self.stack=ExitStack(); self.lock=threading.RLock(); self.opened=False
        self.calls_started=0; self.calls_completed=0

    def __enter__(self):
        with self.lock:
            if self.opened: raise RuntimeError("Engram arithmetic is already open")
            try:
                self.budget.acquire(self.budget.cap); self.stack.callback(self.budget.release,self.budget.cap)
                for name,(dtype,shape) in CONTROL_LAYOUT.items():
                    a=self.arrays.get(name)
                    if not isinstance(a,np.ndarray) or a.shape!=tuple(shape) or a.dtype!=(np.uint16 if dtype=='BF16' else np.uint8) or not a.flags.c_contiguous:
                        raise ValueError("Original Engram arithmetic control layout differs")
                self.driver=Driver(); self.device=self.stack.enter_context(Device(self.driver,self.ordinal,cap=256*MiB,free_floor=2*GiB))
                info=next(d for d in self.driver.inventory()['devices'] if d['ordinal']==self.ordinal)
                if info['compute_capability']!='8.9': raise ValueError("Engram arithmetic requires the registered SM89 profile")
                self.compiler={}; binary=compile_probe('89',engram=True,compiler_info=self.compiler); self.cubin_sha256=hashlib.sha256(binary).hexdigest()
                if self.compiler['nvrtc_version']!=[12,4] or self.cubin_sha256!=ENGRAM_CUBIN_SHA256:
                    raise ValueError("Unregistered native Engram compiler or binary")
                self.module=self.device.load(binary)
                self.weights={n:self._upload(self.stack,a) for n,a in self.arrays.items()}
                self.opened=True; return self
            except BaseException: self.stack.close(); raise

    def _allocate(self,stack,size):
        p=self.device.allocate(size); stack.callback(self.device.free,p); return p

    def _upload(self,stack,array):
        p=self._allocate(stack,array.nbytes); self.driver.check(self.driver.cuMemcpyHtoD_v2(p,array.ctypes.data,array.nbytes)); return p

    def _launch(self,name,pointers,ints,items,threads=128,floats=()):
        args=[C.c_uint64(p) for p in pointers]+[C.c_int(v) for v in ints]+[C.c_float(v) for v in floats]
        self.device.launch(self.module,name,args,items,threads=threads)

    def forward(self,hidden,ids,mask=None):
        with self.lock:
            if not self.opened: raise RuntimeError("Engram arithmetic is not open")
            tokens=validate_input(hidden,ids,mask); self.budget.acquire(0)
            self.calls_started+=1
            codes,scales=self.rows.get(ids)
            with ExitStack() as stack:
                cp,sp=[self._upload(stack,a) for a in (codes,scales)]
                embed_f=self._allocate(stack,tokens*6144*4); embed=self._allocate(stack,tokens*6144*2)
                self._launch('eg_embed',(cp,sp,embed_f,embed),(tokens*6144,),tokens*6144)
                xfp=self._allocate(stack,tokens*6144*4); q=self._allocate(stack,tokens*6144); s=self._allocate(stack,tokens*192)
                self._launch('projection_input',(embed,xfp),(tokens*6144,),tokens*6144)
                self._launch('activation32',(xfp,q,s),(tokens*192,),tokens*192)
                kvf=self._allocate(stack,tokens*25600*4); kv=self._allocate(stack,tokens*25600*2)
                self._launch('projection_fp8',(q,s,self.weights['wkv.weight'],self.weights['wkv.scale'],kvf,kv),(25600,6144,tokens),tokens*25600*4)
                hp=self._upload(stack,hidden); mp=self._upload(stack,np.ones(hidden.shape[:2],np.uint8) if mask is None else mask.view(np.uint8))
                large=[self._allocate(stack,n*4) for n in (tokens*20480,tokens*20480,20480,tokens*20480,tokens*20480)]
                self._launch('eg_prepare',(hp,kv,self.weights['q_weight'],self.weights['k_weight'],*large),(tokens*20480,),tokens*20480)
                reductions=[self._allocate(stack,tokens*4*4) for _ in range(3)]; threads=mean_threads(tokens*4,5120)
                for p,o,factor in zip((large[0],large[1],large[4]),reductions,(1/5120,1/5120,1.0)):
                    self._launch('norm_mean',(p,o),(5120,),tokens*4*threads,threads=threads,floats=(factor,))
                gates=[self._allocate(stack,tokens*4*4) for _ in GATE_STAGES]
                self._launch('eg_gate',(*reductions,mp,*gates),(tokens*4,),tokens*4)
                delta,residual=[self._allocate(stack,tokens*20480*4) for _ in range(2)]; output=self._allocate(stack,hidden.nbytes)
                self._launch('eg_post',(hp,kv,gates[-1],delta,residual,output),(tokens*20480,),tokens*20480)
                result={'rows_codes_fp8':codes,'rows_scales_e8m0':scales,
                    'embedding_fp32':self.device.download(embed_f,(*hidden.shape[:2],6144),np.float32),
                    'embedding_bf16':self.device.download(embed,(*hidden.shape[:2],6144),np.uint16),
                    'projection_input_fp8':self.device.download(q,(*hidden.shape[:2],6144),np.uint8),
                    'projection_scale_e8m0':self.device.download(s,(*hidden.shape[:2],192),np.uint8),
                    'projection_output_fp32':self.device.download(kvf,(*hidden.shape[:2],25600),np.float32),
                    'projection_output_bf16':self.device.download(kv,(*hidden.shape[:2],25600),np.uint16)}
                for name,p in zip(('hidden_square_fp32','key_square_fp32','weight_product_fp32','hidden_weight_fp32','dot_product_fp32'),large):
                    result[name]=self.device.download(p,(4,5120) if name=='weight_product_fp32' else hidden.shape,np.float32)
                for name,p in zip(('hidden_mean_fp32','key_mean_fp32','dot_sum_fp32'),reductions):
                    result[name]=self.device.download(p,(*hidden.shape[:2],4),np.float32)
                for name,p in zip(GATE_STAGES,gates): result[name]=self.device.download(p,(*hidden.shape[:2],4),np.float32)
                result['delta_fp32']=self.device.download(delta,hidden.shape,np.float32)
                result['residual_fp32']=self.device.download(residual,hidden.shape,np.float32)
                result['output_bf16']=self.device.download(output,hidden.shape,np.uint16)
                self.budget.acquire(0); self.calls_completed+=1; return result

    def stats(self):
        d=getattr(self,'device',None)
        return {'host_allowance_bytes':self.budget.cap,'host_reserved_bytes':self.budget.reserved,
            'device_peak_owned_bytes':d.peak if d else 0,'device_owned_bytes':sum(d.allocations.values()) if d else 0,
            'owned_context_active':bool(d.context) if d else False,'reservation_is_RSS_limit':False,
            'calls_started':self.calls_started,'calls_completed':self.calls_completed}

    def __exit__(self,*_):
        with self.lock:
            self.opened=False
            try: self.stack.close()
            finally: self.weights={}; self.arrays={}
