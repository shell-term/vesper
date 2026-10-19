"""Full-backbone resource workload with touched synthetic bytes and a companion.

Uses released tensor identities/shapes, never synthetic-model numerical claims.
The small companion is owned by this probe; no existing workload is controlled.
"""

from collections import deque
from contextlib import ExitStack,contextmanager
import gc
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import threading
import time

import numpy as np

from .memory import RuntimeMemory,MemoryPolicy,OwnedBuffer,ManagedCache,RuntimeCancelled,process_memory
from .model_artifacts import canonical
from .model_memory_plan import full_backbone_plan
from .resources import MemoryBudget,GiB,MiB,physical_memory
from .cuda import Driver,Device,Pinned


def companion_worker():
    budget=MemoryBudget(cap=GiB); budget.acquire(512*MiB)
    payload=bytearray(512*MiB); stop=threading.Event(); counter=0
    def read_stop():
        sys.stdin.readline(); stop.set()
    reader=threading.Thread(target=read_stop,daemon=True); reader.start()
    try:
        while not stop.is_set():
            counter+=1; payload[(counter*4096)%len(payload)]=(counter%251)
            print(json.dumps({'counter':counter,'process':process_memory(),'done':False}),flush=True)
            stop.wait(.1)
    finally:
        del payload; budget.release(512*MiB)
        print(json.dumps({'counter':counter,'process':process_memory(),'done':True}),flush=True)


class Companion:
    def __enter__(self):
        self.records=deque(maxlen=32); self.error=None
        flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
        self.child=subprocess.Popen([sys.executable,'-u','-m','vesper.memory_scale','worker'],stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,creationflags=flags)
        def read():
            try:
                for line in self.child.stdout: self.records.append(json.loads(line))
            except BaseException as e: self.error=type(e).__name__
        self.reader=threading.Thread(target=read,daemon=True); self.reader.start()
        deadline=time.monotonic()+20
        while not self.records:
            if self.child.poll() is not None or time.monotonic()>deadline:
                self.__exit__(None,None,None); raise RuntimeError('Bounded companion failed to start')
            time.sleep(.02)
        self.first=dict(self.records[-1]); return self

    def counter(self): return self.records[-1]['counter'] if self.records else 0

    def __exit__(self,*_):
        if self.child.poll() is None:
            try: self.child.stdin.write('stop\n'); self.child.stdin.flush()
            except BrokenPipeError: pass
        try: self.child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            # This handle was created above for this bounded probe only.
            self.child.terminate(); self.child.wait(timeout=10); self.error='owned_companion_required_termination'
        self.reader.join(timeout=5)
        error=self.child.stderr.read(); self.child.stdin.close(); self.child.stdout.close(); self.child.stderr.close()
        self.result={'first':getattr(self,'first',None),'last':dict(self.records[-1]) if self.records else None,
            'exit_code':self.child.returncode,'exited':self.child.poll() is not None,'reader_error':self.error,
            'stderr_empty':not error,'owned_by_probe':True,'existing_workloads_stopped':False}


def payload_loader(identity):
    tag=struct.pack('<Q',identity)
    def load(array):
        array.fill(identity%251)
        array[:8]=np.frombuffer(tag,np.uint8); array[-8:]=np.frombuffer(tag,np.uint8)
    return load


def check_payload(array,identity):
    tag=struct.pack('<Q',identity)
    if array[:8].tobytes()!=tag or array[-8:].tobytes()!=tag or int(array[8192])!=identity%251:
        raise RuntimeError('Synthetic resource payload changed identity')


def state_digest(buffers):
    sha=hashlib.sha256()
    for buffer in buffers:
        with buffer.borrow() as view: sha.update(view)
        del view
    return sha.hexdigest()


def scale_probe(source,*,experts_per_layer=384,cache_bytes=6*GiB,host_cap=8*GiB,progress=None,native_check=None):
    if type(experts_per_layer) is not int or not 1<=experts_per_layer<=384 or not GiB<=cache_bytes<host_cap or not 4*GiB<=host_cap<=16*GiB:
        raise ValueError('Invalid explicit resource-probe shape/cap')
    rows=[]; faults=[]; phases=[]; trace=hashlib.sha256(); touched=0; physical_min=physical_memory()['available']
    def observe_available():
        nonlocal physical_min
        available=physical_memory()['available']; physical_min=min(physical_min,available); return available
    policy=RuntimeMemory(MemoryPolicy(host_cap_bytes=host_cap),available=observe_available); native=None
    with Companion() as companion,policy:
        metadata=MemoryBudget(cap=512*MiB,owner='complete_model_catalog',category='metadata'); metadata.acquire(metadata.cap)
        try:
            plan=full_backbone_plan(source)
            with ExitStack() as stack:
                states=[stack.enter_context(OwnedBuffer(policy,t['bytes'],'kv',t['owner'],fill=0)) for t in plan['KV_allocations']]
                originals=state_digest(states)
                rollback=stack.enter_context(OwnedBuffer(policy,plan['KV_bytes'],'kv','exact_rollback_copy',fill=0))
                # Every shape is physically touched, including snapshot capacity.
                scratch=stack.enter_context(OwnedBuffer(policy,64*MiB,'scratch','bounded_transfer_scratch',fill=0))
                activations=stack.enter_context(OwnedBuffer(policy,16*MiB,'activations','HC_activation_workspace',fill=1))
                engram=stack.enter_context(ManagedCache(policy,128*1024,'engram','engram_rows'))
                cache=stack.enter_context(ManagedCache(policy,cache_bytes,'cache','all_layer_experts'))
                integrity_bytes=(cache_bytes//plan['expert_bytes'])*((plan['expert_bytes']+4095)//4096)*32
                integrity=stack.enter_context(OwnedBuffer(policy,integrity_bytes,'integrity','cached_expert_block_digests',fill=0))
                driver=Driver(); stack.enter_context(Device(driver,0,cap=MiB,free_floor=2*GiB))
                dma=MemoryBudget(cap=64*MiB,owner='shared_expert_transfer_buffers')
                for _ in range(2):
                    pinned=stack.enter_context(Pinned(driver,plan['expert_bytes'],dma)); pinned.array().fill(0)
                # Same packed byte footprint as the largest original BF16 tensor.
                with OwnedBuffer(policy,plan['largest_nonexpert_tensor']['bytes'],'weights','embedding_or_head',fill=29): policy.sample('largest_nonexpert_weights')
                for layer in range(40):
                    t=plan['layer_nonexpert_working_weights'][str(layer)]
                    with OwnedBuffer(policy,t['bytes'],'weights',t['name'],fill=layer%251):
                        for expert in range(experts_per_layer):
                            identity=layer*384+expert
                            with cache.acquire((layer,expert),plan['expert_bytes'],payload_loader(identity)) as view:
                                check_payload(view,identity); trace.update(struct.pack('<QQ',identity,len(view)))
                            del view; touched+=plan['expert_bytes']
                        policy.sample('layer_'+str(layer))
                    row={'layer':layer,'expert_entries':experts_per_layer,'live_cache_entries':len(cache.entries),'live_cache_bytes':cache.bytes,'status':'pass'}
                    rows.append(row); physical_min=min(physical_min,physical_memory()['available'])
                    if progress: progress(row)
                phases.append(policy.sample('all_backbone_experts_visited'))
                with OwnedBuffer(policy,plan['largest_nonexpert_tensor']['bytes'],'weights','head_after_warm_cache',fill=31):
                    phases.append(policy.sample('global_capacity_reclaims_for_head'))
                # A few actual original row-cache footprints remain ordinary LRU.
                for key in range(48):
                    with engram.acquire(key,264,lambda a:a.fill(0)) as view: pass
                    del view
                if native_check is not None:
                    native=native_check(policy)
                    if state_digest(states)!=originals: raise RuntimeError('Native work changed model-scale retained KV')
                before=policy.host; boundary=companion.counter()
                try: OwnedBuffer(policy,host_cap+1,'scratch','impossible_request')
                except MemoryError: faults.append({'name':'over_global_cap','status':'pass','no_allocation_published':True})
                else: raise RuntimeError('Global host cap admitted an impossible request')
                if policy.host!=before: raise RuntimeError('Impossible request changed existing ownership')
                def broken(view): view[:MiB].fill(9); raise OSError('Injected source read failure')
                before=policy.host
                try:
                    with cache.acquire(('failure',0),plan['expert_bytes'],broken): pass
                except OSError: pass
                else: raise RuntimeError('Failed loader published a cache entry')
                gc.collect()
                if policy.host>before or ('failure',0) in cache.entries: raise RuntimeError('Failed loader leaked owned bytes')
                faults.append({'name':'partial_source_read','status':'pass','no_allocation_published':True})
                before=policy.host
                def cancel(view): view[:MiB].fill(9); policy.cancel()
                try:
                    with cache.acquire(('cancel',0),plan['expert_bytes'],cancel): pass
                except RuntimeCancelled: pass
                else: raise RuntimeError('Cancellation published a cache entry')
                gc.collect()
                if policy.host>before or ('cancel',0) in cache.entries: raise RuntimeError('Cancelled loader leaked owned bytes')
                policy.resume(); faults.append({'name':'cancel_before_publication','status':'pass','no_allocation_published':True})
                for key in range(6):
                    with cache.acquire(('retry',key),plan['expert_bytes'],payload_loader(key)) as view: check_payload(view,key)
                    del view
                faults.append({'name':'retry_after_failure_and_cancel','status':'pass','entries_verified':6})
                real_available=policy.available; anchor=policy.host
                policy.available=lambda:min(real_available(),policy.config.available_floor_bytes-128*MiB+(anchor-policy.host))
                try: policy.checkpoint()
                finally: policy.available=real_available
                faults.append({'name':'injected_pressure_restored','status':'pass','live_pressure_is_injected':True})
                policy.available=lambda:min(real_available(),policy.config.available_floor_bytes-1)
                try:
                    try: policy.checkpoint()
                    except MemoryError: pass
                    else: raise RuntimeError('Unrestorable reserve did not refuse work')
                finally: policy.available=real_available
                faults.append({'name':'injected_unrestorable_reserve','status':'pass','live_pressure_is_injected':True})
                if state_digest(states)!=originals: raise RuntimeError('Reclaim/failure/cancel changed retained KV')
                policy.checkpoint()
                with cache.acquire(('recovered',0),plan['expert_bytes'],payload_loader(17)) as view: check_payload(view,17)
                del view
                phases.append(policy.sample('recovered_with_KV_preserved'))
                active=policy.snapshot(); counts={'hits':cache.hits,'misses':cache.misses,'evictions':cache.evictions}
                other_work={'counter_before_faults':boundary,'counter_after_faults':companion.counter()}
                # RSS need not fall by the charge total: NumPy/driver allocators
                # and metadata are measured separately after payload release.
            policy.sample('after_owned_buffers_closed')
        finally: metadata.release(metadata.cap)
        policy.sample('before_policy_close')
    if (companion.result['exit_code']!=0 or companion.result['reader_error'] is not None or not companion.result['stderr_empty']
            or other_work['counter_after_faults']<=other_work['counter_before_faults']): raise RuntimeError('Companion did not keep making progress')
    result={'schema_version':1,'probe':'representative_model_memory_policy','status':'pass','plan':plan,
        'complete_backbone_expert_pass':experts_per_layer==384,'layers':rows,'expert_entries_visited':40*experts_per_layer,
        'synthetic_weight_bytes_touched':touched,'synthetic_trace_sha256':trace.hexdigest(),'cache_counters':counts,'faults':faults,
        'retained_KV_sha256':originals,'KV_unchanged_across_all_faults':True,'physical_available_min_sampled_bytes':physical_min,
        'active_governor':active,'cleanup_governor':policy.snapshot(),'companion':companion.result,'other_work_progress':other_work,
        'native_comparisons':native,'phases':phases,'actual_96GiB_allocation_attempted':False,'full_checkpoint_downloaded':False,
        'weights_fetched_during_probe':0,'GPU_resets':0,'pagefile_changes':False,'full_model_quality':'not_run','end_to_end_speed':'not_measured',
        'scope':'Touched synthetic payloads at original full-backbone allocation geometry, bounded cache and actual process observations; no model-wide numerical inference'}
    if policy.host or any(policy.device_used.values()) or policy.leases or policy.reclaimers: raise RuntimeError('Global resource cleanup is incomplete')
    return result


if __name__=='__main__':
    if sys.argv[1:]!=['worker']: raise SystemExit('Use the bounded memory probe example')
    companion_worker()
