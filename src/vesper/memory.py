"""Process-wide cooperative memory admission, owned buffers and pressure recovery.

The host ceiling covers explicit payloads plus conservative legacy component
admissions. Process RSS/private commit and driver/compiler phases are observed
separately: none of these counters is an operating-system memory limit.
"""

from collections import OrderedDict,deque
from contextlib import contextmanager
import ctypes as C
from dataclasses import dataclass,field
from functools import wraps
import os
from pathlib import Path
import threading
import time
import weakref
from types import MappingProxyType

import numpy as np

from .resources import GiB,MiB,physical_memory

CATEGORIES=('cache','weights','activations','kv','engram','metadata','pinned','integrity','scratch','diagnostics','admission')
_active=None
_active_lock=threading.RLock()


class _ProcessCounters(C.Structure):
    _fields_=[('cb',C.c_uint32),('faults',C.c_uint32)]+[(n,C.c_size_t) for n in
        ('peak_ws','ws','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile','private')]


_process_api=None


def current_policy():
    with _active_lock: return _active


def process_memory():
    """Read this process only; no PID, private path or other-process inventory."""
    global _process_api
    if os.name=='nt':
        if _process_api is None:
            k=C.WinDLL('kernel32',use_last_error=True); k.GetCurrentProcess.restype=C.c_void_p
            lib=C.WinDLL('psapi',use_last_error=True)
            lib.GetProcessMemoryInfo.argtypes=[C.c_void_p,C.POINTER(_ProcessCounters),C.c_uint32]
            _process_api=(k,lib)
        k,lib=_process_api
        data=_ProcessCounters(); data.cb=C.sizeof(data)
        if not lib.GetProcessMemoryInfo(k.GetCurrentProcess(),C.byref(data),data.cb): raise C.WinError(C.get_last_error())
        return {'resident_bytes':data.ws,'peak_resident_bytes':data.peak_ws,'private_commit_bytes':data.private,
            'virtual_bytes':None,'locked_bytes':None,'platform':'Windows','measurement':'GetProcessMemoryInfo'}
    values={}
    for line in Path('/proc/self/status').read_text().splitlines():
        key,_,value=line.partition(':')
        if key in ('VmRSS','VmHWM','VmSize','VmLck'): values[key]=int(value.split()[0])*1024
    private=None; rollup=Path('/proc/self/smaps_rollup'); rss_source='proc_status_approximate'; huge=None
    if rollup.exists():
        fields={}
        try:
            for line in rollup.read_text().splitlines():
                key,_,value=line.partition(':')
                if key in ('Rss','Private_Clean','Private_Dirty','Private_Hugetlb'): fields[key]=int(value.split()[0])*1024
            values['VmRSS']=fields['Rss']; private=fields['Private_Clean']+fields['Private_Dirty']; huge=fields.get('Private_Hugetlb',0)
            rss_source='smaps_rollup'
        except OSError: pass
    return {'resident_bytes':values['VmRSS'],'peak_resident_bytes':values['VmHWM'],'private_commit_bytes':None,
        'private_resident_bytes':private,'virtual_bytes':values['VmSize'],'locked_bytes':values['VmLck'],
        'private_hugetlb_bytes':huge,'resident_measurement':rss_source,'platform':'Linux','measurement':'proc status/smaps_rollup'}


class RuntimeCancelled(RuntimeError): pass


@dataclass(frozen=True)
class MemoryPolicy:
    host_cap_bytes:int=96*GiB
    available_floor_bytes:int=32*GiB
    pinned_cap_bytes:int=GiB
    device_caps:dict=field(default_factory=lambda:{0:20*GiB,1:10*GiB})
    device_free_floor_bytes:int=2*GiB
    monitor_interval_seconds:float=.25

    def __post_init__(self):
        if (type(self.host_cap_bytes) is not int or not 0<self.host_cap_bytes<=96*GiB
                or type(self.available_floor_bytes) is not int or self.available_floor_bytes<32*GiB
                or type(self.pinned_cap_bytes) is not int or not 0<self.pinned_cap_bytes<=self.host_cap_bytes
                or type(self.device_free_floor_bytes) is not int or self.device_free_floor_bytes<GiB
                or not .05<=self.monitor_interval_seconds<=5
                or not isinstance(self.device_caps,dict) or not self.device_caps
                or any(type(k) is not int or k<0 or type(v) is not int or v<=0 for k,v in self.device_caps.items())):
            raise ValueError('Invalid explicit runtime memory policy')
        object.__setattr__(self,'device_caps',MappingProxyType(dict(self.device_caps)))


class Lease:
    def __init__(self,policy,key,size,category,owner,kind):
        self.policy,self.key,self.size,self.category,self.owner,self.kind=policy,key,size,category,owner,kind
        self.released=False

    def release(self):
        with self.policy.lock:
            if self.released: return
            self.policy._release(self); self.released=True


class RuntimeMemory:
    """One shared policy per process, including worker threads and all components.

    Injected observers are test seams. Live pressure tests use min(real available,
    simulated available); the normal runtime always reads physical memory.
    """
    CONTROL_METADATA_BYTES=64*MiB
    MAX_LIVE_RECORDS=16384
    MAX_RECLAIMERS=512
    ENABLE_WATCHDOG=True

    def __init__(self,config=None,*,available=None,process=None):
        self.config=config or MemoryPolicy(); self.available=available or (lambda:physical_memory()['available'])
        self.process=process or process_memory; self.lock=threading.RLock(); self.opened=False; self.closing=False; self.cancelled=False
        self.leases={}; self.budgets={}; self.reclaimers=OrderedDict(); self.devices={}; self.sequence=0
        self.host=0; self.host_peak=0; self.explicit_host=0; self.explicit_peak=0; self.pinned=0; self.pinned_peak=0
        self.device_used={}; self.device_peaks={}; self.refusals=0; self.reclaims=0; self.reclaimed_bytes=0; self.inflight=0
        self.minimum_available=None; self.samples=deque(maxlen=128); self.external=deque(maxlen=64)
        self._reclaiming=False; self.last_process_sample=0; self.baseline=None
        self.control_lease=None
        self.process_peaks={}; self.external_counts={}
        self.stop_monitor=threading.Event(); self.monitor=None; self.monitor_checks=0; self.monitor_pressure_refusals=0
        self.monitor_error=None; self.local=threading.local()

    def __enter__(self):
        global _active
        with _active_lock:
            if _active is not None or self.opened: raise RuntimeError('A process-wide memory policy is already active')
            if not self._floor(0): raise MemoryError('Runtime entry would consume the system memory reserve')
            self.baseline=self.process(); self.opened=True; _active=self
        try:
            if self.CONTROL_METADATA_BYTES:
                self.control_lease=self.reserve(self.CONTROL_METADATA_BYTES,'metadata','memory_policy_control',kind='control_admission')
            self.sample('runtime_entry')
            if self.ENABLE_WATCHDOG:
                self.monitor=threading.Thread(target=self._watch,name='vesper-memory-watch',daemon=True); self.monitor.start()
            return self
        except BaseException:
            if self.control_lease is not None: self.control_lease.release()
            with _active_lock: _active=None; self.opened=False
            raise

    def _watch(self):
        while not self.stop_monitor.wait(self.config.monitor_interval_seconds):
            if not self.lock.acquire(blocking=False): continue
            try:
                if self.closing: return
                self.monitor_checks+=1
                try: self._recover(0,'admission')
                except MemoryError: self.monitor_pressure_refusals+=1
                if time.monotonic()-self.last_process_sample>=1: self.sample('memory_watch')
            except BaseException as error:
                self.monitor_error=type(error).__name__; self.cancelled=True; return
            finally: self.lock.release()

    def touch_checkpoint(self,size):
        """Check physical room before touching a previously admitted buffer."""
        with self.lock:
            self._ready()
            if self._floor(size): return
            for callback in tuple(self.reclaimers.values()):
                while not self._floor(size):
                    before=self.host; callback(max(size,self.config.available_floor_bytes+size-self.available()))
                    if self.host>=before: break
                    self.reclaims+=1; self.reclaimed_bytes+=before-self.host
                if self._floor(size): break
            if not self._floor(size): self.refusals+=1; raise MemoryError('Physical reserve cannot accommodate the next bounded buffer touch')

    def _floor(self,size):
        available=self.available()
        if type(available) is not int or available<0: raise ValueError('Invalid available-memory observation')
        self.minimum_available=available if self.minimum_available is None else min(self.minimum_available,available)
        return available-size>=self.config.available_floor_bytes

    def _ready(self):
        if not self.opened or self.closing: raise RuntimeError('Runtime memory policy is not accepting work')
        if self.monitor_error is not None: raise RuntimeError('Runtime memory observation failed')
        if self.cancelled and not getattr(self.local,'restoring',0): raise RuntimeCancelled('Runtime work was cancelled')

    def _fits(self,size,category):
        return (self.host+size<=self.config.host_cap_bytes
            and (category!='pinned' or self.pinned+size<=self.config.pinned_cap_bytes) and self._floor(size))

    def _recover(self,size,category):
        if self._fits(size,category): return
        if size>self.config.host_cap_bytes or category=='pinned' and self.pinned+size>self.config.pinned_cap_bytes:
            self.refusals+=1; raise MemoryError('Allocation cannot fit the runtime host or pinned ceiling')
        if not self._reclaiming:
            self._reclaiming=True
            try:
                # Callbacks must try their own lock without waiting and skip live
                # borrowed entries. A busy cache never deadlocks an allocator.
                for callback in tuple(self.reclaimers.values()):
                    while not self._fits(size,category):
                        target=max(1,size,self.host+size-self.config.host_cap_bytes,self.config.available_floor_bytes+size-self.available())
                        before=self.host; callback(target); freed=before-self.host
                        if freed<0: raise RuntimeError('Pressure callback allocated memory')
                        if not freed: break
                        self.reclaims+=1; self.reclaimed_bytes+=freed
                    if self._fits(size,category): break
            finally: self._reclaiming=False
        if not self._fits(size,category):
            self.refusals+=1; raise MemoryError('Runtime host/pinned cap or available-memory reserve cannot be restored')

    def reserve(self,size,category,owner,*,kind='explicit'):
        if (type(size) is not int or size<=0 or category not in CATEGORIES or type(owner) is not str or not 0<len(owner)<=160
                or kind not in ('explicit','component_admission','control_admission','boundary_admission_only')):
            raise ValueError('Invalid owned runtime allocation identity')
        with self.lock:
            self._ready(); self._recover(size,category)
            if self._reclaiming and size: raise RuntimeError('Pressure callbacks cannot allocate')
            if len(self.leases)>=self.MAX_LIVE_RECORDS: raise MemoryError('Runtime allocation-record limit')
            self.sequence+=1; lease=Lease(self,self.sequence,size,category,owner,kind); self.leases[lease.key]=lease
            self.host+=size; self.host_peak=max(self.host_peak,self.host)
            if kind=='explicit': self.explicit_host+=size; self.explicit_peak=max(self.explicit_peak,self.explicit_host)
            if category=='pinned': self.pinned+=size; self.pinned_peak=max(self.pinned_peak,self.pinned)
            return lease

    def _release(self,lease):
        if self.leases.get(lease.key) is not lease: raise ValueError('Foreign runtime allocation release')
        del self.leases[lease.key]; self.host-=lease.size
        if lease.kind=='explicit': self.explicit_host-=lease.size
        if lease.category=='pinned': self.pinned-=lease.size

    def checkpoint(self):
        with self.lock: self._ready(); self._recover(0,'admission')

    @contextmanager
    def publication(self):
        with self.lock:
            self.checkpoint(); yield

    @contextmanager
    def restoration(self):
        """Allow exact snapshot recovery after cancellation, under the same caps.

        Cancellation still blocks ordinary work. Recovery cannot discard model
        history or bypass allocation/physical-memory admission.
        """
        with self.lock:
            previous=getattr(self.local,'restoring',0); self.local.restoring=previous+1
            try: self.checkpoint(); yield
            finally: self.local.restoring=previous

    @contextmanager
    def operation(self):
        with self.lock: self.checkpoint(); self.inflight+=1
        try: yield
        finally:
            with self.lock: self.inflight-=1

    def cancel(self):
        with self.lock: self.cancelled=True

    def resume(self):
        with self.lock:
            if self.inflight or self.closing: raise RuntimeError('Cannot resume while operations are active or cleanup is pending')
            self.cancelled=False; self.checkpoint()

    def register_reclaimer(self,callback):
        with self.lock:
            self._ready()
            if len(self.reclaimers)>=self.MAX_RECLAIMERS: raise MemoryError('Runtime reclaimer-record limit')
            self.sequence+=1; token=self.sequence; self.reclaimers[token]=callback; return token

    def unregister_reclaimer(self,token):
        with self.lock: self.reclaimers.pop(token,None)

    def acquire_budget(self,budget,size,category):
        with self.lock:
            if size==0: self.checkpoint(); return
            lease=self.reserve(size,category,budget.owner,kind='component_admission')
            self.budgets.setdefault(budget._token,[]).append(lease)

    def release_budget(self,budget,size,category=None):
        with self.lock:
            rows=self.budgets.get(budget._token,[])
            eligible=[l for l in rows if category is None or l.category==category]
            if sum(l.size for l in eligible)<size: raise ValueError('Runtime component release differs from its typed admissions')
            remaining=size
            for lease in reversed(eligible):
                if not remaining: break
                take=min(remaining,lease.size)
                if take==lease.size: lease.release(); rows.remove(lease)
                else:
                    lease.size-=take; self.host-=take
                    if lease.category=='pinned': self.pinned-=take
                remaining-=take
            if not rows: self.budgets.pop(budget._token,None)

    def reserve_device(self,ordinal,size):
        with self.lock:
            self._ready(); self._recover(0,'admission')
            if ordinal not in self.config.device_caps or self.device_used.get(ordinal,0)+size>self.config.device_caps[ordinal]:
                self.refusals+=1; raise MemoryError('Combined runtime device allocation cap')
            self.device_used[ordinal]=self.device_used.get(ordinal,0)+size
            self.device_peaks[ordinal]=max(self.device_peaks.get(ordinal,0),self.device_used[ordinal])

    def release_device(self,ordinal,size):
        with self.lock:
            if not 0<=size<=self.device_used.get(ordinal,0): raise ValueError('Foreign runtime device release')
            self.device_used[ordinal]-=size

    def sample(self,label):
        if type(label) is not str or not 0<len(label)<=160: raise ValueError('Process observation labels must be bounded')
        with self.lock:
            data=self.process(); available=self.available()
            for key in ('resident_bytes','private_commit_bytes','private_resident_bytes'):
                if data.get(key) is not None: self.process_peaks[key]=max(self.process_peaks.get(key,0),data[key])
            row={'phase':label,'process':data,'available_bytes':available,'charged_host_bytes':self.host,
                'explicit_buffer_bytes':self.explicit_host,'pinned_bytes':self.pinned,'device_bytes':dict(self.device_used)}
            self.samples.append(row); self.last_process_sample=time.monotonic(); return row

    def snapshot(self):
        with self.lock:
            by_category={n:0 for n in CATEGORIES}; kinds={}; owners={}
            for l in self.leases.values():
                by_category[l.category]+=l.size; kinds[l.kind]=kinds.get(l.kind,0)+l.size; owners[l.owner]=owners.get(l.owner,0)+l.size
            return {'host_cap_bytes':self.config.host_cap_bytes,'available_floor_bytes':self.config.available_floor_bytes,
                'charged_host_bytes':self.host,'charged_host_peak_bytes':self.host_peak,'explicit_host_bytes':self.explicit_host,
                'explicit_host_peak_bytes':self.explicit_peak,'pinned_bytes':self.pinned,'pinned_peak_bytes':self.pinned_peak,
                'pinned_cap_bytes':self.config.pinned_cap_bytes,'categories':by_category,'accounting_kinds':kinds,'owners':owners,
                'device_caps':dict(self.config.device_caps),'device_used_bytes':dict(self.device_used),'device_peak_bytes':dict(self.device_peaks),
                'minimum_observed_available_bytes':self.minimum_available,'refusals':self.refusals,'reclaim_events':self.reclaims,
                'reclaimed_charge_bytes':self.reclaimed_bytes,'live_leases':len(self.leases),'live_reclaimers':len(self.reclaimers),
                'inflight_operations':self.inflight,'cancelled':self.cancelled,'process_baseline':self.baseline,'samples':list(self.samples),
                'external_phase_samples':list(self.external),'reservation_is_RSS_or_OS_limit':False,
                'sampled_process_peaks':dict(self.process_peaks),'external_phase_counts':dict(self.external_counts),
                'control_metadata_admission_bytes':self.CONTROL_METADATA_BYTES,'maximum_live_allocation_records':self.MAX_LIVE_RECORDS,
                'maximum_reclaimers':self.MAX_RECLAIMERS,
                'monitor_interval_seconds':self.config.monitor_interval_seconds,'monitor_checks':self.monitor_checks,
                'monitor_pressure_refusals':self.monitor_pressure_refusals,'monitor_thread_stopped':self.monitor is None or not self.monitor.is_alive(),
                'monitor_error':self.monitor_error,
                'accounting':'explicit buffers plus conservative existing component admissions; overlapping admissions may overestimate payload bytes'}

    def close(self):
        global _active
        with self.lock:
            self.closing=True
            self.stop_monitor.set()
            if self.monitor is not None and self.monitor is not threading.current_thread(): self.monitor.join(timeout=2)
            other_leases=[l for l in self.leases.values() if l is not self.control_lease]
            if other_leases or any(self.device_used.values()) or self.inflight or self.reclaimers:
                raise RuntimeError('Runtime memory still owns allocations, borrowers, operations or reclaimers')
            if self.control_lease is not None: self.control_lease.release()
            with _active_lock:
                if _active is self: _active=None
            self.opened=False

    def __exit__(self,*_): self.close()


@contextmanager
def external_phase(name):
    policy=current_policy()
    if policy is None: yield; return
    if type(name) is not str or not 0<len(name)<=140: raise ValueError('External observation labels must be bounded')
    with policy.lock:
        if name not in policy.external_counts and len(policy.external_counts)>=64: raise MemoryError('External phase name limit')
        policy.external_counts.setdefault(name,0)
    before=policy.sample(name+'_entry')
    try: yield
    finally:
        after=policy.sample(name+'_exit')
        with policy.lock:
            policy.external_counts[name]=policy.external_counts.get(name,0)+1
            policy.external.append({'phase':name,'before':before['process'],'after':after['process'],
                'owned_payload_charged_separately':True,'delta_is_exact_component_attribution':False})


def observe_external(name):
    def decorate(function):
        @wraps(function)
        def wrapped(*args,**kwargs):
            with external_phase(name): return function(*args,**kwargs)
        return wrapped
    return decorate


def runtime_operation(function):
    """Keep cancellation recovery explicit until an entire public call exits."""
    @wraps(function)
    def wrapped(*args,**kwargs):
        policy=current_policy()
        if policy is None: return function(*args,**kwargs)
        with policy.operation(): return function(*args,**kwargs)
    return wrapped


class OwnedBuffer:
    """Charge before malloc; escaped NumPy views keep their real charge alive.

    Closing drops ownership, but the final lease is released only when the base
    array and every borrowed view die. This avoids an uncounted retained view or
    a dangling native pointer after pressure reclamation.
    """
    def __init__(self,policy,size,category,owner,*,fill=None):
        self.policy=policy; self.lease=policy.reserve(size,category,owner); self.closed=False; self.borrowers=0
        try:
            self._array=np.empty(size,np.uint8); self.finalizer=weakref.finalize(self._array,self.lease.release)
            if fill is not None:
                for start in range(0,size,64*MiB): policy.touch_checkpoint(min(64*MiB,size-start)); self._array[start:start+64*MiB].fill(fill)
        except BaseException:
            self._array=None; self.lease.release(); raise

    @contextmanager
    def borrow(self):
        with self.policy.lock:
            self.policy.checkpoint()
            if self.closed: raise RuntimeError('Owned runtime buffer is closed')
            self.borrowers+=1; view=self._array.view()
        try: yield view
        finally:
            del view
            with self.policy.lock: self.borrowers-=1

    def close(self):
        with self.policy.lock:
            if self.borrowers: raise RuntimeError('Cannot release an actively borrowed runtime buffer')
            if not self.closed: self.closed=True; self._array=None

    def __enter__(self): return self
    def __exit__(self,*_): self.close()


class ManagedCache:
    """LRU eligible bytes only; logical KV/history never enter this cache."""
    def __init__(self,policy,capacity,category='cache',owner='runtime_cache'):
        if type(capacity) is not int or capacity<0 or category not in ('cache','engram'): raise ValueError('Invalid reclaimable runtime cache')
        self.policy,self.capacity,self.category,self.owner=policy,capacity,category,owner
        self.entries=OrderedDict(); self.bytes=0; self.hits=self.misses=self.evictions=0; self.closed=False
        self.token=policy.register_reclaimer(self.reclaim)

    def reclaim(self,requested):
        with self.policy.lock:
            freed=0
            for key,buffer in list(self.entries.items()):
                if buffer.borrowers: continue
                before=self.policy.host; buffer.close(); del self.entries[key]; self.bytes-=buffer.lease.size
                self.evictions+=1; freed+=before-self.policy.host
                if freed>=requested: break
            return freed

    @contextmanager
    def acquire(self,key,size,loader):
        parts=key if type(key) is tuple else (key,)
        if not 1<=len(parts)<=4 or any(not (type(p) is int and -(2**63)<=p<2**63 or type(p) is str and len(p)<=160) for p in parts):
            raise ValueError('Cache keys must have bounded immutable identity metadata')
        with self.policy.lock:
            self.policy.checkpoint()
            if self.closed or not 0<size<=self.capacity: raise ValueError('Cache is closed or entry exceeds its explicit capacity')
            if key in self.entries:
                buffer=self.entries.pop(key); self.entries[key]=buffer; self.hits+=1
                if buffer.lease.size!=size: raise ValueError('Cached immutable payload identity changed size')
            else:
                while self.bytes+size>self.capacity:
                    before=self.bytes; self.reclaim(size)
                    if self.bytes==before: raise MemoryError('All runtime cache victims are actively borrowed')
                self.misses+=1; buffer=OwnedBuffer(self.policy,size,self.category,self.owner)
                try:
                    with buffer.borrow() as view: loader(view)
                    del view
                    with self.policy.publication(): self.entries[key]=buffer; self.bytes+=size
                except BaseException: buffer.close(); raise
            context=buffer.borrow(); view=context.__enter__()
        try: yield view
        finally:
            del view; context.__exit__(None,None,None)

    def close(self):
        with self.policy.lock:
            if any(b.borrowers for b in self.entries.values()): raise RuntimeError('Cache has active borrowers')
            for b in self.entries.values(): b.close()
            self.entries.clear(); self.bytes=0; self.closed=True; self.policy.unregister_reclaimer(self.token)

    def __enter__(self): return self
    def __exit__(self,*_): self.close()
