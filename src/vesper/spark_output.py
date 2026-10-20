"""Explicit, unqualified GB10 head candidate with bounded original row staging.

Original BF16 rows are lifted exactly through bounded host chunks into one
retained device matrix. Each projection uses the original full-vocabulary
SGEMM shape, requiring its own same-device numerical comparison. It is never a
PC qualification fallback.
"""
from contextlib import ExitStack
import ctypes as C
import threading

import numpy as np

from .blas import Cublas, GATE_BLAS_WORKSPACE
from .cuda import Device, Driver
from .execution_target import resolve_execution_target
from .formats import bf16_values
from .memory import runtime_operation
from .output import greedy, validate_hidden
from .output_assets import VOCAB, WIDTH
from .resources import GiB, MiB, MemoryBudget, physical_memory

HEAD_DEVICE_CAP = 3 * GiB


def validate_rows(rows, count):
    if (not isinstance(rows, np.ndarray) or rows.dtype != np.uint16
            or rows.shape != (count, WIDTH) or not rows.flags.c_contiguous
            or np.any((rows & 0x7f80) == 0x7f80)):
        raise ValueError('Expected exact, finite, contiguous original BF16 head rows')


def row_blocks(block_rows):
    if type(block_rows) is not int or not 1 <= block_rows <= 1024:
        raise ValueError('Head staging must contain one to 1024 complete rows')
    return tuple((start, min(block_rows, VOCAB-start)) for start in range(0, VOCAB, block_rows))


class SparkOutputCandidate:
    """Read original rows through a verified local provider; no network/ mmap.

    The provider owns the pinned tensor/revision and integrity checks. It must
    return only a requested bounded span. Initialization retains the complete
    FP32 device head while holding only one host staging block. A failed build
    closes all owned resources; construct a new operator to retry. Forward calls
    never reread weights or publish partial logits after a failed launch.
    """
    def __init__(self, row_reader, ordinal=0, *, block_rows=512, library=None,
                 execution_target='spark-sm121-candidate'):
        if not callable(row_reader): raise ValueError('Expected a verified local head row reader')
        self.target=resolve_execution_target(execution_target)
        if not self.target.is_candidate: raise ValueError('Use the existing qualified PC output path for PC')
        self.blocks=row_blocks(block_rows)
        self.row_reader,self.ordinal,self.block_rows,self.library=row_reader,ordinal,block_rows,library
        self.lock,self.stack=threading.RLock(),ExitStack()
        self.budget=MemoryBudget(cap=64*MiB,owner='spark_device_cached_output')
        self.opened=False; self.initialized=False; self.calls=0; self.rows_read=0

    def __enter__(self):
        with self.lock:
            if self.opened or self.initialized: raise RuntimeError('Head candidates open once')
            self.initialized=True
            try:
                self.budget.acquire(self.budget.cap)
                self.stack.callback(self.budget.release,self.budget.cap)
                # Admission is conservative; the shared runtime still charges
                # actual allocations and enforces its physical-memory floor.
                self.budget.acquire(0,physical_memory()['available']-HEAD_DEVICE_CAP)
                self.driver=Driver(); inventory=self.driver.inventory()
                info=next(d for d in inventory['devices'] if d['ordinal']==self.ordinal)
                self.provenance=self.target.validate_device(info,inventory)
                self.device=self.stack.enter_context(Device(self.driver,self.ordinal,cap=HEAD_DEVICE_CAP,free_floor=2*GiB))
                self.blas=Cublas(self.library); self.stack.callback(self.blas.close)
                self.blas_provenance=self.target.validate_blas(self.blas.version,self.blas.math_mode)
                self.weight_pointer=self.device.allocate(VOCAB*WIDTH*4)
                self.input=self.device.allocate(2*WIDTH*4)
                self.output=self.device.allocate(2*VOCAB*4)
                workspace=self.device.allocate(GATE_BLAS_WORKSPACE)
                self.blas.workspace(workspace,GATE_BLAS_WORKSPACE)
                for start,rows in self.blocks:
                    self.budget.acquire(0)
                    original=promoted=None
                    try:
                        original=self.row_reader(start,rows); validate_rows(original,rows)
                        promoted=bf16_values(original)
                        self.driver.check(self.driver.cuMemcpyHtoD_v2(self.weight_pointer+start*WIDTH*4,
                            promoted.ctypes.data,promoted.nbytes))
                        self.rows_read+=rows
                    finally:
                        original=promoted=None
                self.opened=True; return self
            except BaseException:
                self.stack.close(); raise

    @runtime_operation
    def project(self, values):
        validate_hidden(values)
        with self.lock:
            if not self.opened: raise RuntimeError('Head candidate is not open')
            self.budget.acquire(0); self.device.current()
            x=bf16_values(values); count=values.shape[0]
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
        head_bytes=device.allocations.get(getattr(self,'weight_pointer',None),0) if device else 0
        return {**self.target.describe(),'calls':self.calls,'rows_read':self.rows_read,
                'block_rows':self.block_rows,'full_vocabulary_rows':VOCAB,'head_compute_dtype':'float32',
                'retained_head':self.opened and head_bytes==VOCAB*WIDTH*4,
                'head_device_bytes':head_bytes,'device_cap_bytes':HEAD_DEVICE_CAP,
                'whole_host_head_materialized':False,'weight_reads_in_forward':False,
                'TF32_enabled':False,'host_allowance_bytes':self.budget.cap,'host_reserved_bytes':self.budget.reserved,
                'device_owned_bytes':sum(device.allocations.values()) if device else 0,
                'device_peak_owned_bytes':device.peak if device else 0,
                'reservation_is_RSS_limit':False}

    def __exit__(self,*_):
        with self.lock:
            self.opened=False
            if getattr(self,'device',None) is not None and self.device.context: self.device.current()
            self.stack.close()
