"""Publish token history only after both original Engram layers succeed."""

from contextlib import ExitStack
import threading
import numpy as np

from .memory import runtime_operation
from .engram import validate_input
from .ngram import LAYERS
from .resources import MemoryBudget,MiB


class EngramSession:
    def __init__(self,hash_state,operators):
        if set(operators)!=set(LAYERS) or any(operators[l].rows.layer!=l for l in LAYERS):
            raise ValueError("Expected both original Engram layer operators")
        self.hash=hash_state; self.operators=operators; self.budget=MemoryBudget(cap=32*MiB)
        self.lock=threading.RLock(); self.allocation_lock=threading.RLock(); self.stack=ExitStack(); self.opened=False; self.peak=0; self.primary_cap=384*MiB
        self.combined_cap_refusals=0

    def _owned(self): return sum(sum(o.device.allocations.values()) for o in self.operators.values())

    def __enter__(self):
        with self.lock:
            if self.opened or any(not o.opened for o in self.operators.values()): raise RuntimeError("Engram session requires open arithmetic operators")
            if len({id(o.device) for o in self.operators.values()})!=2 or len({o.ordinal for o in self.operators.values()})!=1:
                raise ValueError("Engram session requires distinct owned ledgers on one compute device")
            try:
                self.budget.acquire(self.budget.cap); self.stack.callback(self.budget.release,self.budget.cap)
                for op in self.operators.values():
                    original=op.device.allocate
                    def allocate(size,original=original):
                        with self.allocation_lock:
                            if self._owned()+size>self.primary_cap:
                                self.combined_cap_refusals+=1; raise MemoryError("Combined native Engram GPU cap")
                            pointer=original(size); self.peak=max(self.peak,self._owned()); return pointer
                    op.device.allocate=allocate; self.stack.callback(setattr,op.device,'allocate',original)
                self.peak=self._owned()
                if self.peak>self.primary_cap: raise MemoryError("Initial combined native Engram cap")
                self.opened=True; return self
            except BaseException: self.stack.close(); raise

    @runtime_operation
    def step(self,ids,hidden,mask=None,start_pos=None):
        with self.lock,self.hash.lock:
            if not self.opened: raise RuntimeError("Engram session is not open")
            self.budget.acquire(0)
            if set(hidden)!=set(LAYERS): raise ValueError("Expected each independent original Engram hidden input")
            candidate=self.hash.prepare(ids,start_pos,mask)
            indices={l:np.ascontiguousarray(candidate.hash_ids[:,:,i,:]) for i,l in enumerate(LAYERS)}
            for l in LAYERS: validate_input(hidden[l],indices[l],mask)
            result={l:self.operators[l].forward(hidden[l],indices[l],mask) for l in LAYERS}
            self.budget.acquire(0)
            if self._owned()>self.primary_cap: raise MemoryError("Combined native Engram publication cap")
            with self.budget.publication(): self.hash.commit(candidate)
            return {'hash_ids':candidate.hash_ids.copy(),'compressed':candidate.compressed.copy(),'lookbacks':candidate.lookbacks.copy(),
                'layers':result,'state_sha256':self.hash.state_sha256()}

    def stats(self):
        return {'host_allowance_bytes':self.budget.cap,'host_reserved_bytes':self.budget.reserved,
            'combined_primary_cap_bytes':self.primary_cap,'combined_primary_peak_owned_bytes':self.peak,
            'combined_cap_refusals_before_backend':self.combined_cap_refusals,
            'reservation_is_RSS_limit':False,'history_tokens_per_batch_row':3}

    def __exit__(self,*_):
        with self.lock: self.opened=False; self.stack.close()
