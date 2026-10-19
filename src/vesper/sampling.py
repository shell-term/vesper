"""Explicit transactional Philox state for the original full-vocabulary sampler."""

from contextlib import ExitStack
import ctypes as C
import hashlib
import math
import threading
from dataclasses import dataclass

import numpy as np

from .cuda import Driver,Device,compile_probe,_bind
from .memory import runtime_operation
from .output import greedy
from .resources import MemoryBudget,MiB

SAMPLING_CUBIN_SHA256='0f2e42e4c986b5b8d6a80735d8ce338f7f94be102f41e3425f4c373441725645'


@dataclass(frozen=True)
class SamplerSnapshot:
    seed:int
    offset:int


class NativeSampler:
    def __init__(self,seed=0,ordinal=0):
        if type(seed) is not int or not 0<=seed<2**64: raise ValueError('Expected an explicit uint64 Philox seed')
        self.seed,self.offset,self.ordinal=seed,0,ordinal
        self.stack,self.lock,self.budget=ExitStack(),threading.RLock(),MemoryBudget(cap=128*MiB,owner='original_sampling')
        self.opened=False; self.calls=0; self.handle=None

    def __enter__(self):
        with self.lock: return self._open()

    def _open(self):
        if self.opened: raise RuntimeError('Sampler is already open')
        try:
            self.budget.acquire(self.budget.cap); self.stack.callback(self.budget.release,self.budget.cap)
            self.driver=Driver(); self.device=self.stack.enter_context(Device(self.driver,self.ordinal,cap=16*MiB))
            info=next(d for d in self.driver.inventory()['devices'] if d['ordinal']==self.ordinal)
            if info['compute_capability']!='8.9': raise ValueError('Unregistered original sampling device profile')
            query=_bind(self.driver.lib,'cuDeviceGetAttribute',[C.POINTER(C.c_int),C.c_int,C.c_int])
            sm,threads=C.c_int(),C.c_int()
            self.driver.check(query(C.byref(sm),16,self.ordinal)); self.driver.check(query(C.byref(threads),39,self.ordinal))
            self.multiprocessors,self.threads_per_sm=sm.value,threads.value
            if (sm.value,threads.value)!=(128,1536): raise ValueError('Unregistered original sampling execution policy')
            self.compiler={}; cubin=compile_probe('89',sampling=True,compiler_info=self.compiler)
            self.cubin_sha256=hashlib.sha256(cubin).hexdigest()
            if self.cubin_sha256!=SAMPLING_CUBIN_SHA256 or self.compiler['nvrtc_version']!=[12,4]: raise ValueError('Unqualified original sampler compiler/binary')
            self.module=self.device.load(cubin)
            self.opened=True; return self
        except BaseException: self.stack.close(); raise

    def _allocate(self,stack,size):
        p=self.device.allocate(size); stack.callback(self.device.free,p); return p

    def _launch(self,name,pointers,extra,count,threads=128):
        self.device.launch(self.module,name,[C.c_uint64(p) for p in pointers]+list(extra),count,threads=threads)

    def manual_seed(self,seed):
        if type(seed) is not int or not 0<=seed<2**64: raise ValueError('Expected an explicit uint64 Philox seed')
        with self.lock:
            if not self.opened or self.handle is not None: raise RuntimeError('Seed reset requires an open sampler without pending snapshots')
            with self.budget.publication(): self.seed,self.offset=seed,0

    def snapshot(self):
        with self.lock:
            if not self.opened or self.handle is not None: raise RuntimeError('Sampler requires one bounded snapshot at a time')
            with self.budget.publication(): self.handle=SamplerSnapshot(self.seed,self.offset)
            return self.handle

    def rollback(self,handle):
        with self.lock,self.budget.restoration():
            if not self.opened or self.handle is None or self.handle is not handle: raise ValueError('Foreign or consumed sampler snapshot')
            with self.budget.publication(): self.seed,self.offset,self.handle=handle.seed,handle.offset,None

    def discard(self,handle):
        with self.lock:
            if not self.opened or self.handle is None or self.handle is not handle: raise ValueError('Foreign or consumed sampler snapshot')
            self.handle=None

    @runtime_operation
    def sample(self,logits,temperature=1.):
        selected=greedy(logits) # Validate the complete finite vocabulary before any RNG change.
        if type(temperature) not in (float,int) or not math.isfinite(temperature) or not 0<=temperature<=1000:
            raise ValueError('Expected an explicit finite original sampling temperature in [0,1000]')
        with self.lock:
            if not self.opened: raise RuntimeError('Sampler is not open')
            self.budget.acquire(0)
            if temperature==0:
                with self.budget.publication(): self.calls+=1
                return {'ids_i64':selected,'offset_before':self.offset,'offset_after':self.offset}
            count=logits.size; grid=min((count+255)//256,self.multiprocessors*(self.threads_per_sm//256))
            increment=((count-1)//(256*grid*4)+1)*4
            if self.offset+increment>=2**64: raise OverflowError('Philox offset exhausted')
            with ExitStack() as stack:
                buffers=[self._allocate(stack,logits.nbytes) for _ in range(5)]; source,scaled,prob,noise,scores=buffers
                self.driver.check(self.driver.cuMemcpyHtoD_v2(source,logits.ctypes.data,logits.nbytes))
                reciprocal=float(np.float32(1.)/np.float32(max(temperature,1e-5)))
                self._launch('sample_scale',(source,scaled),(C.c_float(reciprocal),C.c_int(count)),count)
                self._launch('sample_softmax',(scaled,prob),(),logits.shape[0]*1024,threads=1024)
                self._launch('sample_exponential',(noise,),(C.c_uint64(self.seed),C.c_uint64(self.offset),C.c_int(count)),grid*256,threads=256)
                self._launch('sample_scores',(prob,noise,scores),(C.c_int(count),),count)
                result={n:self.device.download(p,logits.shape,np.float32) for n,p in
                    (('scaled_fp32',scaled),('probabilities_fp32',prob),('exponential_fp32',noise),('scores_fp32',scores))}
                result['ids_i64']=greedy(result['scores_fp32']); result['offset_before']=self.offset; result['offset_after']=self.offset+increment
            with self.budget.publication(): self.offset+=increment; self.calls+=1
            return result

    def __exit__(self,*_):
        with self.lock: self.opened=False; self.handle=None; self.stack.close()
