"""Native full-vocabulary FP32 output projection; no sampled/partial vocabulary."""

from contextlib import ExitStack
import ctypes as C
import threading

import numpy as np

from .blas import Cublas,GATE_BLAS_WORKSPACE
from .cuda import Driver,Device
from .formats import bf16_values
from .memory import runtime_operation
from .output_assets import VOCAB,WIDTH
from .resources import MemoryBudget,GiB,MiB


def validate_hidden(values):
    if (not isinstance(values,np.ndarray) or values.dtype!=np.uint16 or values.ndim!=2 or values.shape[1]!=WIDTH
            or values.shape[0] not in (1,2) or not values.flags.c_contiguous or np.any((values&0x7f80)==0x7f80)):
        raise ValueError('Output projection expects one/two finite contiguous BF16 rows of width 5120')


def greedy(logits):
    if (not isinstance(logits,np.ndarray) or logits.dtype!=np.float32 or logits.ndim!=2 or logits.shape[1]!=VOCAB
            or logits.shape[0] not in (1,2) or not logits.flags.c_contiguous or not np.isfinite(logits).all()):
        raise ValueError('Greedy sampling expects complete finite FP32 logits over all 129280 tokens')
    # NumPy and the original argmax select the first index at a finite tie.
    return np.argmax(logits,axis=-1).astype(np.int64)


class NativeOutput:
    """One original BF16 matrix promoted exactly to the original FP32 head dtype.

    This candidate needs the scoped original numerical gate before use in a
    qualified model runtime. It never substitutes BF16/TF32 output GEMM.
    """
    def __init__(self,weight,ordinal=0,*,library=None):
        if (not isinstance(weight,np.ndarray) or weight.dtype!=np.uint16 or weight.shape!=(VOCAB,WIDTH) or not weight.flags.c_contiguous):
            raise ValueError('Output projection requires the complete original BF16 matrix')
        for start in range(0,VOCAB,512):
            if np.any((weight[start:start+512]&0x7f80)==0x7f80): raise ValueError('Original output weights must be finite')
        self.weight,self.ordinal,self.library=weight,ordinal,library
        self.stack,self.lock,self.budget=ExitStack(),threading.RLock(),MemoryBudget(cap=6*GiB,owner='original_output_projection')
        self.opened=False; self.calls=0; self.initialized=False

    def __enter__(self):
        with self.lock:
            if self.opened or self.initialized: raise RuntimeError('Output projection objects open once; construct a new operator')
            try:
                self.budget.acquire(self.budget.cap); self.stack.callback(self.budget.release,self.budget.cap)
                self.driver=Driver(); info=next(d for d in self.driver.inventory()['devices'] if d['ordinal']==self.ordinal)
                if info['compute_capability']!='8.9': raise ValueError('Unregistered original output device profile')
                self.device=self.stack.enter_context(Device(self.driver,self.ordinal,cap=3*GiB,free_floor=2*GiB))
                self.blas=Cublas(self.library); self.stack.callback(self.blas.close)
                if self.blas.version!=120402 or self.blas.math_mode!=0: raise ValueError('Unregistered output cuBLAS candidate profile')
                self.weight_pointer=self.device.allocate(self.weight.size*4)
                # Bounded promotion chunks avoid a second full host FP32 matrix.
                flat=self.weight.reshape(-1)
                for start in range(0,flat.size,4*MiB):
                    self.budget.acquire(0); lifted=bf16_values(flat[start:start+4*MiB])
                    self.driver.check(self.driver.cuMemcpyHtoD_v2(self.weight_pointer+start*4,lifted.ctypes.data,lifted.nbytes))
                del lifted
                self.input=self.device.allocate(2*WIDTH*4); self.output=self.device.allocate(2*VOCAB*4)
                work=self.device.allocate(GATE_BLAS_WORKSPACE); self.blas.workspace(work,GATE_BLAS_WORKSPACE)
                self.weight=None; self.initialized=True; self.opened=True; return self
            except BaseException: self.stack.close(); raise

    @runtime_operation
    def project(self,values):
        validate_hidden(values)
        with self.lock:
            if not self.opened: raise RuntimeError('Output projection is not open')
            self.budget.acquire(0); self.device.current(); x=bf16_values(values); count=values.shape[0]
            self.driver.check(self.driver.cuMemcpyHtoD_v2(self.input,x.ctypes.data,x.nbytes))
            alpha,beta=C.c_float(1),C.c_float(0)
            self.blas.check(self.blas.cublasSgemm_v2(self.blas.handle,1,0,VOCAB,count,WIDTH,C.byref(alpha),
                self.weight_pointer,WIDTH,self.input,WIDTH,C.byref(beta),self.output,VOCAB))
            self.driver.check(self.driver.cuCtxSynchronize())
            logits=self.device.download(self.output,(count,VOCAB),np.float32)
            ids=greedy(logits)
            with self.budget.publication(): self.calls+=1
            return {'promoted_fp32':x,'logits_fp32':logits,'greedy_ids_i64':ids}

    def stats(self):
        device=getattr(self,'device',None)
        return {'host_allowance_bytes':self.budget.cap,'host_reserved_bytes':self.budget.reserved,'calls':self.calls,
            'device_owned_bytes':sum(device.allocations.values()) if device else 0,'device_peak_owned_bytes':device.peak if device else 0,
            'cublas_version':getattr(getattr(self,'blas',None),'version',None),'cublas_math_mode':getattr(getattr(self,'blas',None),'math_mode',None),
            'full_vocabulary_rows':VOCAB,'head_compute_dtype':'float32','TF32_enabled':False,'reservation_is_RSS_limit':False}

    def __exit__(self,*_):
        with self.lock:
            self.opened=False
            if getattr(self,'device',None) is not None and self.device.context: self.device.current()
            self.stack.close()
