"""Exact original Engram addressing with three tokens of owned history."""

from dataclasses import dataclass
import math
import threading

import numpy as np

from .model_artifacts import canonical

LAYERS=(1,14)
TABLE_ROWS=(384006168,384016682)


def next_prime(start, seen):
    candidate=start+1
    while candidate in seen or candidate<2 or (candidate!=2 and (candidate%2==0 or any(candidate%d==0 for d in range(3,math.isqrt(candidate)+1,2)))):
        candidate+=1
    return candidate


def layout():
    seen,primes=set(),[]
    for _ in LAYERS:
        per_layer=[]
        for _ in range(3):
            group=[]; current=15999999
            for _ in range(8):
                current=next_prime(current,seen); seen.add(current); group.append(current)
            per_layer.append(group)
        primes.append(per_layer)
    primes=np.array(primes,np.int64)
    if tuple(primes.sum(axis=(1,2)))!=TABLE_ROWS: raise ValueError("Original Engram bucket totals differ")
    offsets=np.array([np.cumsum([0,*p.reshape(-1)[:-1]]) for p in primes],np.int64)
    bound=(np.iinfo(np.int64).max//99092)//2
    multipliers=np.array([np.random.default_rng(10007*l).integers(0,bound,size=4,dtype=np.int64)*2+1 for l in LAYERS])
    return primes,offsets,multipliers


@dataclass(frozen=True)
class HashStep:
    owner: object
    generation: object
    start: int
    compressed: np.ndarray
    lookbacks: np.ndarray
    hash_ids: np.ndarray
    history: np.ndarray


class HashSnapshot: pass


class NgramState:
    def __init__(self, token_map):
        if not isinstance(token_map,np.ndarray) or token_map.shape!=(129280,) or token_map.dtype!=np.int64 or token_map.min()!=0 or token_map.max()!=99091:
            raise ValueError("Expected the qualified compressed vocabulary")
        self.token_map=token_map.copy(); self.token_map.flags.writeable=False
        self.pad=int(token_map[2]); self.primes,self.offsets,self.multipliers=layout()
        self.lock=threading.RLock(); self.owner=object(); self.handles={}; self.reset()

    def reset(self):
        with self.lock:
            self.position=0; self.history=None; self.generation=object(); self.handles.clear()

    def inspect(self):
        with self.lock:
            return {"position":self.position,"history":None if self.history is None else self.history.copy()}

    def prepare(self, ids, start_pos=None, mask=None):
        with self.lock:
            pos=self.position if start_pos is None else start_pos
            if (not isinstance(ids,np.ndarray) or ids.dtype!=np.int64 or ids.ndim!=2 or not ids.flags.c_contiguous
                    or ids.shape[0] not in (1,2) or not 1<=ids.shape[1]<=3 or np.any(ids<0) or np.any(ids>=129280)
                    or type(pos) is not int or pos!=self.position or pos+ids.shape[1]>1048576
                    or (self.history is not None and ids.shape[0]!=self.history.shape[0])):
                raise ValueError("Expected the next bounded original token chunk")
            if mask is not None and (not isinstance(mask,np.ndarray) or mask.dtype!=np.bool_ or mask.shape!=ids.shape or not mask.flags.c_contiguous):
                raise ValueError("Expected a contiguous original text-token mask")
            compressed=self.token_map[ids].copy()
            if mask is not None: compressed[~mask]=-1
            prefix=np.empty((ids.shape[0],0),np.int64) if self.history is None else self.history
            joined=np.concatenate((prefix,compressed),axis=1); width=prefix.shape[1]
            tokens=[]; blocked=np.zeros_like(ids,np.bool_)
            for shift in range(4):
                columns=np.arange(width,width+ids.shape[1])-shift
                source=joined[:,np.maximum(columns,0)]
                blocked=blocked | (columns<0)[None,:] | (source==-1)
                tokens.append(np.where(blocked,self.pad,source))
            lookbacks=np.stack(tokens,axis=-1)
            products=lookbacks[:,:,None,:]*self.multipliers
            rolling=products[...,0].copy(); hashes=[]
            for i in range(1,4):
                rolling=np.bitwise_xor(rolling,products[...,i]); hashes.append(rolling[...,None]%self.primes[:,i-1])
            result=np.concatenate(hashes,axis=-1)+self.offsets
            history=joined[:,-3:].copy()
            for a in (compressed,lookbacks,result,history): a.flags.writeable=False
            return HashStep(self.owner,self.generation,pos,compressed,lookbacks,result,history)

    def commit(self, step):
        with self.lock:
            if not isinstance(step,HashStep) or step.owner is not self.owner or step.generation is not self.generation or step.start!=self.position:
                raise ValueError("Foreign or stale Engram hash candidate")
            history=step.history.copy(); position=self.position+step.compressed.shape[1]; generation=object()
            self.history,self.position,self.generation=history,position,generation

    def snapshot(self):
        with self.lock:
            if len(self.handles)>=16: raise MemoryError('Engram history snapshot handle cap')
            token=HashSnapshot(); self.handles[token]=self.inspect(); return token

    def rollback(self, token):
        with self.lock:
            if type(token) is not HashSnapshot or token not in self.handles: raise ValueError("Consumed, foreign or reset-expired Engram snapshot")
            saved=self.handles[token]
            history=None if saved["history"] is None else saved["history"].copy(); generation=object()
            self.position,self.history,self.generation=saved["position"],history,generation
            del self.handles[token]

    def discard(self, token):
        with self.lock:
            if type(token) is not HashSnapshot or token not in self.handles:
                raise ValueError("Consumed, foreign or reset-expired Engram snapshot")
            del self.handles[token]

    def state_sha256(self):
        state=self.inspect()
        return canonical({"position":state["position"],"history":None if state["history"] is None else state["history"].tolist()})
