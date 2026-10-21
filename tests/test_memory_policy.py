import gc
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import unittest
from unittest.mock import patch

import numpy as np

from vesper.memory import MemoryPolicy,RuntimeMemory,RuntimeCancelled,OwnedBuffer,ManagedCache,current_policy,process_memory,external_phase,CATEGORIES,runtime_operation
from vesper.resources import MemoryBudget,GiB,MiB


class MemoryPolicyTests(unittest.TestCase):
    def setUp(self):
        # Counter-boundary unit cases deliberately use byte-sized caps. Live
        # resource probes always retain the default 64 MiB metadata admission.
        self.metadata=patch.object(RuntimeMemory,'CONTROL_METADATA_BYTES',0)
        self.metadata.start(); self.addCleanup(self.metadata.stop)
        self.watchdog=patch.object(RuntimeMemory,'ENABLE_WATCHDOG',False)
        self.watchdog.start(); self.addCleanup(self.watchdog.stop)

    def make(self,cap=1024,*,available=None,process=None):
        return RuntimeMemory(MemoryPolicy(host_cap_bytes=cap,pinned_cap_bytes=min(cap,128),device_caps={0:1024}),
            available=available or (lambda:128*GiB),process=process)

    def tearDown(self):
        policy=current_policy()
        if policy is not None:
            # Failed tests must not contaminate unrelated runtime tests.
            for callback in tuple(policy.reclaimers.values()): callback(2**63)
            for lease in list(policy.leases.values()): lease.release()
            policy.reclaimers.clear(); policy.device_used.clear(); policy.inflight=0; policy.close()

    def test_default_headroom_and_immutable_configuration(self):
        p=MemoryPolicy(); self.assertEqual(p.host_cap_bytes,96*GiB); self.assertEqual(p.available_floor_bytes,32*GiB)
        with self.assertRaises(TypeError): p.device_caps[0]=2**63
        for kwargs in ({'host_cap_bytes':97*GiB},{'available_floor_bytes':31*GiB},{'pinned_cap_bytes':0},{'device_caps':{True:5}}):
            with self.assertRaises(ValueError): MemoryPolicy(**kwargs)
        with self.assertRaises(MemoryError): self.make(available=lambda:GiB).__enter__()
        self.assertIsNone(current_policy())

    def test_all_categories_share_one_cap_and_leases_release_once(self):
        with self.make(110) as policy:
            leases=[policy.reserve(10,c,c) for c in CATEGORIES]
            self.assertEqual(policy.host,110)
            with self.assertRaises(MemoryError): policy.reserve(1,'weights','one_more')
            self.assertEqual(policy.host,110)
            for lease in leases: lease.release(); lease.release()
            self.assertEqual(policy.host,0); self.assertEqual(policy.explicit_host,0)

    def test_host_refusal_samples_physical_memory_after_cap_short_circuit(self):
        observations=[]
        def available(): observations.append(1); return 128*GiB
        with self.make(100,available=available) as policy:
            budget=MemoryBudget(cap=100,available_floor=0,owner='component')
            budget.acquire(90,category='metadata'); before=len(observations)
            with self.assertRaisesRegex(MemoryError,'cannot be restored'):
                policy.reserve(11,'weights','pending')
            self.assertEqual(len(observations),before+1)
            refusal=policy.snapshot()['last_refusal']
            self.assertEqual(refusal['causes'],['host_cap'])
            self.assertEqual((refusal['requested_bytes'],refusal['category']),(11,'weights'))
            self.assertEqual((refusal['charged_host_bytes'],refusal['host_cap_bytes']),(90,100))
            self.assertEqual((refusal['component_admission_bytes'],refusal['explicit_host_bytes']),(90,0))
            self.assertEqual(refusal['physical_available_bytes'],128*GiB)
            self.assertEqual(refusal['available_floor_bytes'],32*GiB)
            self.assertEqual(refusal['physical_observation'],'after_admission_refusal')
            self.assertIsNone(refusal['observation_error'])
            self.assertEqual((budget.reserved,policy.host),(90,90))
            budget.release(90,category='metadata')

    def test_refusal_distinguishes_physical_and_simultaneous_constraints_and_survives_cleanup(self):
        for used,causes in ((0,['physical_reserve']),(90,['host_cap','physical_reserve'])):
            with self.subTest(used=used):
                state={'available':128*GiB}
                with self.make(100,available=lambda:state['available']) as policy:
                    lease=policy.reserve(used,'weights','existing') if used else None
                    state['available']=32*GiB+10
                    with self.assertRaises(MemoryError): policy.reserve(11,'scratch','pending')
                    refusal=policy.snapshot()['last_refusal']
                    self.assertEqual(refusal['causes'],causes)
                    self.assertEqual(refusal['physical_available_bytes'],32*GiB+10)
                    self.assertEqual(refusal['physical_pending_bytes'],11)
                    # The monitor or an unwinding checkpoint cannot replace the
                    # allocation evidence with a subsequent zero-byte refusal.
                    state['available']=32*GiB-1
                    with self.assertRaises(MemoryError): policy.checkpoint()
                    self.assertEqual(policy.snapshot()['last_refusal'],refusal)
                    if lease is not None: lease.release()
                    state['available']=128*GiB; policy.checkpoint()
                closed=policy.snapshot()
                self.assertEqual(closed['charged_host_bytes'],0)
                self.assertEqual(closed['live_leases'],0)
                self.assertEqual(closed['last_refusal'],refusal)
                closed['last_refusal']['causes'].append('caller_mutation')
                self.assertEqual(policy.snapshot()['last_refusal']['causes'],causes)

    def test_early_host_and_pinned_refusals_retain_requested_and_existing_bytes(self):
        with self.make(100) as policy:
            with self.assertRaisesRegex(MemoryError,'host or pinned ceiling'):
                policy.reserve(101,'scratch','oversize')
            refusal=policy.snapshot()['last_refusal']
            self.assertEqual(refusal['causes'],['host_cap'])
            self.assertEqual((refusal['requested_bytes'],refusal['charged_host_bytes']),(101,0))
        with self.make(1024) as policy:
            lease=policy.reserve(100,'pinned','existing')
            with self.assertRaises(MemoryError): policy.reserve(29,'pinned','pending')
            refusal=policy.snapshot()['last_refusal']
            self.assertEqual(refusal['causes'],['pinned_cap'])
            self.assertEqual((refusal['pinned_bytes'],refusal['pinned_cap_bytes']),(100,128))
            self.assertEqual(refusal['physical_available_bytes'],128*GiB)
            self.assertEqual(policy.host,100); lease.release()

    def test_device_refusal_records_device_limit_and_shared_host_usage_without_spending(self):
        config=MemoryPolicy(host_cap_bytes=2048,pinned_cap_bytes=128,device_caps={0:1024},unified_device_memory=True)
        with RuntimeMemory(config,available=lambda:128*GiB) as policy:
            policy.reserve_device(0,600)
            with self.assertRaisesRegex(MemoryError,'device allocation cap'): policy.reserve_device(0,425)
            refusal=policy.snapshot()['last_refusal']
            self.assertEqual(refusal['causes'],['device_cap'])
            self.assertEqual((refusal['device_ordinal'],refusal['device_used_bytes'],refusal['device_cap_bytes']),(0,600,1024))
            self.assertEqual((refusal['charged_host_bytes'],refusal['explicit_host_bytes'],refusal['unified_device_bytes']),(600,600,600))
            self.assertEqual(refusal['physical_available_bytes'],128*GiB)
            self.assertEqual(policy.device_used,{0:600}); self.assertEqual(policy.host,600)
            policy.release_device(0,600)
            for ordinal in (1,None):
                with self.assertRaises(MemoryError): policy.reserve_device(ordinal,1)
                missing=policy.snapshot()['last_refusal']
                self.assertEqual(missing['stage'],'device_admission')
                self.assertEqual(missing['causes'],['device_not_configured'])
                self.assertIsNone(missing['device_cap_bytes'])
            self.assertEqual((policy.host,policy.device_used[0]),(0,0))

    def test_refusal_observation_failure_does_not_replace_the_original_memory_error(self):
        state={'failed':False}
        def available():
            if state['failed']: raise OSError('Physical observation unavailable')
            return 128*GiB
        with self.make(100,available=available) as policy:
            lease=policy.reserve(90,'weights','existing'); state['failed']=True
            with self.assertRaisesRegex(MemoryError,'cannot be restored'):
                policy.reserve(11,'scratch','pending')
            refusal=policy.snapshot()['last_refusal']
            self.assertEqual(refusal['causes'],['host_cap'])
            self.assertIsNone(refusal['physical_available_bytes'])
            self.assertEqual(refusal['observation_error'],'OSError')
            self.assertEqual(policy.host,90); lease.release()

    def test_device_refusal_reports_shared_pool_constraints_only_for_unified_memory(self):
        for unified in (False,True):
            with self.subTest(unified=unified):
                state={'available':128*GiB}
                config=MemoryPolicy(host_cap_bytes=1024,pinned_cap_bytes=128,device_caps={0:1024},unified_device_memory=unified)
                with RuntimeMemory(config,available=lambda:state['available']) as policy:
                    policy.reserve_device(0,600); state['available']=32*GiB+400
                    with self.assertRaises(MemoryError): policy.reserve_device(0,425)
                    refusal=policy.snapshot()['last_refusal']
                    expected=['host_cap','device_cap','physical_reserve'] if unified else ['device_cap']
                    self.assertEqual(refusal['causes'],expected)
                    self.assertEqual(refusal['physical_pending_bytes'],425 if unified else 0)
                    self.assertEqual(policy.host,600 if unified else 0)
                    self.assertEqual(policy.device_used,{0:600}); policy.release_device(0,600)

    def test_existing_component_budgets_cannot_each_spend_the_full_cap(self):
        with self.make(100) as policy:
            a,b=MemoryBudget(cap=100,available_floor=0,owner='a'),MemoryBudget(cap=100,available_floor=0,owner='b')
            a.acquire(60); b.acquire(30,category='metadata')
            with self.assertRaises(MemoryError): b.acquire(11)
            self.assertEqual((a.reserved,b.reserved,policy.host),(60,30,90))
            self.assertEqual(policy.snapshot()['accounting_kinds'],{'component_admission':90})
            with self.assertRaises(ValueError): b.release(1,category='pinned')
            b.release(30,category='metadata'); a.release(60)
            self.assertFalse(policy.budgets)

    def test_typed_admissions_cannot_be_released_as_unrelated_scratch(self):
        with self.make(100) as policy:
            b=MemoryBudget(cap=100,available_floor=0)
            b.acquire(20,category='scratch'); b.acquire(30,category='integrity')
            with self.assertRaises(ValueError): b.release(20)
            b.release(20,category='scratch')
            self.assertEqual(policy.snapshot()['categories']['integrity'],30)
            b.release(30,category='integrity'); self.assertFalse(policy.budgets)

    def test_control_metadata_is_admitted_and_record_counts_are_bounded(self):
        with patch.object(RuntimeMemory,'CONTROL_METADATA_BYTES',64*MiB):
            with self.make(128*MiB) as policy:
                self.assertEqual(policy.host,64*MiB); self.assertEqual(policy.snapshot()['categories']['metadata'],64*MiB)
            self.assertEqual(policy.host,0)
        with self.make() as policy,patch.object(policy,'MAX_LIVE_RECORDS',2):
            a=policy.reserve(1,'metadata','a'); b=policy.reserve(1,'metadata','b')
            with self.assertRaises(MemoryError): policy.reserve(1,'metadata','c')
            a.release(); b.release()

    def test_observation_metadata_cannot_grow_without_a_bound(self):
        with self.make() as policy:
            with self.assertRaises(ValueError): policy.sample('x'*161)
            with self.assertRaises(ValueError): policy.reserve(1,'metadata','owner',kind='x'*1000)
            for i in range(64):
                with external_phase('component_'+str(i)): pass
            with self.assertRaises(MemoryError):
                with external_phase('one_more_component'): pass
            self.assertEqual(len(policy.samples),128); self.assertEqual(len(policy.external),64)

    def test_public_nested_operations_cannot_resume_inflight_cancelled_work(self):
        with self.make() as policy:
            @runtime_operation
            def inner():
                self.assertEqual(policy.inflight,2); policy.cancel()
                with self.assertRaises(RuntimeError): policy.resume()
                with policy.publication(): pass
            @runtime_operation
            def outer(): return inner()
            with self.assertRaises(RuntimeCancelled): outer()
            self.assertEqual(policy.inflight,0); policy.resume()

    def test_process_policy_applies_to_worker_threads(self):
        with self.make(100) as policy:
            ready=threading.Barrier(3); release=threading.Event()
            def worker():
                b=MemoryBudget(cap=100,available_floor=0); b.acquire(40)
                ready.wait(); release.wait(5); b.release(40)
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures=[pool.submit(worker) for _ in range(2)]; ready.wait(timeout=5)
                self.assertEqual(policy.host,80)
                with self.assertRaises(MemoryError): policy.reserve(21,'scratch','main')
                release.set()
                for f in futures: f.result(timeout=5)
            self.assertEqual(policy.host,0)

    def test_reclaim_skips_a_live_borrow_and_restores_global_capacity(self):
        with self.make(100) as policy:
            with ManagedCache(policy,80) as cache:
                with cache.acquire('a',40,lambda a:a.fill(17)) as a:
                    with cache.acquire('b',40,lambda b:b.fill(19)) as b: self.assertTrue((b==19).all())
                    del b
                    with OwnedBuffer(policy,30,'scratch','work'):
                        self.assertIn('a',cache.entries); self.assertNotIn('b',cache.entries)
                        self.assertTrue((a==17).all())
                del a
            self.assertGreater(policy.reclaimed_bytes,0); self.assertEqual(policy.host,0)

    def test_available_floor_requires_observed_recovery_not_just_counter_release(self):
        state={'available':128*GiB}
        with self.make(100,available=lambda:state['available']) as policy:
            with ManagedCache(policy,80) as cache:
                with cache.acquire('a',40,lambda a:a.fill(1)) as a: pass
                del a
                state['available']=31*GiB
                with self.assertRaises(MemoryError): policy.checkpoint()
                self.assertEqual(policy.host,0); self.assertFalse(cache.entries)
                state['available']=128*GiB; policy.checkpoint()

    def test_escaped_views_remain_counted_until_the_last_view_dies(self):
        with self.make() as policy:
            buffer=OwnedBuffer(policy,64,'kv','non_reclaimable_history',fill=7)
            with buffer.borrow() as a: escaped=a[1:]
            del a; buffer.close(); self.assertEqual(policy.host,64)
            with self.assertRaises(RuntimeError): policy.close()
            np.testing.assert_array_equal(escaped,np.full(63,7,np.uint8))
            del escaped; gc.collect(); self.assertEqual(policy.host,0)
            # A failed close blocks new allocations; cleanup and final close remain possible.
            with self.assertRaises(RuntimeError): policy.reserve(1,'scratch','late')

    def test_failed_loader_and_cancelled_publication_do_not_publish_partial_entries(self):
        with self.make() as policy:
            with ManagedCache(policy,100) as cache:
                def broken(a): a.fill(4); raise OSError('I/O failure')
                with self.assertRaises(OSError):
                    with cache.acquire('broken',64,broken): pass
                gc.collect(); self.assertFalse(cache.entries)
                self.assertEqual(policy.host,0)
                def cancel(a): a.fill(5); policy.cancel()
                with self.assertRaises(RuntimeCancelled):
                    with cache.acquire('cancelled',64,cancel): pass
                gc.collect(); self.assertFalse(cache.entries); self.assertEqual(policy.host,0)
                policy.resume()
                with cache.acquire('retry',64,lambda a:a.fill(5)) as a: self.assertTrue((a==5).all())
                del a

    def test_cancellation_and_publication_are_serialized_and_retry_is_explicit(self):
        with self.make() as policy:
            state=[]
            with policy.operation():
                policy.cancel()
                with self.assertRaises(RuntimeError): policy.resume()
                with self.assertRaises(RuntimeCancelled):
                    with policy.publication(): state.append('partial')
            self.assertFalse(state); policy.resume()
            with policy.operation(),policy.publication(): state.append('committed')
            self.assertEqual(state,['committed'])

    def test_another_thread_cannot_cancel_halfway_through_publication(self):
        with self.make() as policy:
            entered=threading.Event(); commit=threading.Event(); cancelling=threading.Event(); state=[]
            def publish():
                with policy.publication(): entered.set(); commit.wait(5); state.append('complete')
            def cancel(): cancelling.set(); policy.cancel()
            with ThreadPoolExecutor(max_workers=2) as pool:
                first=pool.submit(publish)
                try:
                    self.assertTrue(entered.wait(5)); second=pool.submit(cancel); self.assertTrue(cancelling.wait(5))
                    self.assertFalse(policy.cancelled)
                finally: commit.set()
                first.result(5); second.result(5)
            self.assertEqual(state,['complete']); self.assertTrue(policy.cancelled)

    def test_exact_recovery_is_allowed_after_cancel_but_new_work_is_not(self):
        with self.make() as policy:
            b=MemoryBudget(cap=100,available_floor=0); b.acquire(16); state=[]; policy.cancel()
            with self.assertRaises(RuntimeCancelled): b.acquire(0)
            with b.restoration(),b.publication(): state.append('restored')
            with self.assertRaises(RuntimeCancelled): b.acquire(0)
            self.assertEqual(state,['restored']); b.release(16)

    def test_idle_pressure_monitor_reclaims_and_stops_on_cleanup(self):
        pressure={'enabled':False,'anchor':0}; holder={}
        def available():
            return 128*GiB if not pressure['enabled'] else 32*GiB-1+pressure['anchor']-holder['policy'].host
        with patch.object(RuntimeMemory,'ENABLE_WATCHDOG',True):
            with self.make(available=available) as policy:
                holder['policy']=policy
                with ManagedCache(policy,100) as cache:
                    with cache.acquire('cached',40,lambda a:a.fill(7)) as a: pass
                    del a; pressure.update(enabled=True,anchor=policy.host)
                    deadline=time.monotonic()+3
                    while cache.entries and time.monotonic()<deadline: time.sleep(.01)
                    self.assertFalse(cache.entries); self.assertGreater(policy.monitor_checks,0)
                pressure['enabled']=False
            self.assertTrue(policy.snapshot()['monitor_thread_stopped'])

    def test_failed_monitor_refuses_future_work_and_preserves_cleanup(self):
        failed={'now':False}
        def available():
            if failed['now']: raise OSError('Observation unavailable')
            return 128*GiB
        with patch.object(RuntimeMemory,'ENABLE_WATCHDOG',True):
            with self.make(available=available) as policy:
                failed['now']=True; deadline=time.monotonic()+3
                while policy.monitor_error is None and time.monotonic()<deadline: time.sleep(.01)
                self.assertEqual(policy.monitor_error,'OSError')
                with self.assertRaises(RuntimeError): policy.reserve(1,'scratch','new_work')
            self.assertTrue(policy.snapshot()['monitor_thread_stopped'])

    def test_device_and_pinned_limits_are_shared_independently_of_host(self):
        with self.make(1024) as policy:
            a=policy.reserve(100,'pinned','dma')
            with self.assertRaises(MemoryError): policy.reserve(29,'pinned','dma2')
            policy.reserve_device(0,600)
            with self.assertRaises(MemoryError): policy.reserve_device(0,425)
            self.assertEqual(policy.device_used,{0:600})
            policy.release_device(0,600); a.release()

    def test_existing_unmanaged_allocations_cannot_be_silently_adopted(self):
        b=MemoryBudget(cap=100,available_floor=0); b.acquire(10)
        with self.make():
            with self.assertRaises(RuntimeError): b.acquire(0)
        b.release(10)

    def test_failed_observation_does_not_leave_an_active_policy(self):
        calls=[]
        def observer():
            calls.append(1)
            if len(calls)>1: raise OSError('Process measurement failed')
            return {'resident_bytes':0}
        with self.assertRaises(OSError): self.make(process=observer).__enter__()
        self.assertIsNone(current_policy())

    def test_process_metrics_and_external_phase_are_separate_from_ownership(self):
        data=process_memory(); self.assertGreater(data['resident_bytes'],0)
        self.assertIs(data['private_commit_bytes'] is None, data['platform']=='Linux')
        with self.make() as policy:
            with external_phase('reference_observation'): policy.sample('inside')
            report=policy.snapshot(); self.assertEqual(report['charged_host_bytes'],0)
            self.assertFalse(report['reservation_is_RSS_or_OS_limit'])
            self.assertEqual(report['external_phase_samples'][0]['phase'],'reference_observation')
            self.assertFalse(report['external_phase_samples'][0]['delta_is_exact_component_attribution'])


if __name__=='__main__': unittest.main()
