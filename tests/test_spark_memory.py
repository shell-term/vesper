"""UMA CPU/GPU admission shares one ceiling and preserves rollback/retry."""
import unittest

from vesper.memory import MemoryPolicy, RuntimeMemory
from vesper.resources import GiB, MiB, MemoryBudget
from vesper.spark_backbone_runtime import spark_memory_policy, SPARK_ASSET_CAP


class SparkMemoryTests(unittest.TestCase):
    def runtime(self, **overrides):
        config=dict(host_cap_bytes=GiB, pinned_cap_bytes=128*MiB,
                    device_caps={0:GiB}, unified_device_memory=True)
        config.update(overrides)
        runtime=RuntimeMemory(MemoryPolicy(**config), available=lambda:48*GiB,
            process=lambda:{'resident_bytes':0})
        runtime.ENABLE_WATCHDOG=False
        return runtime

    def test_host_device_and_pinned_share_cap(self):
        with self.runtime() as memory:
            host=memory.reserve(400*MiB,'weights','host')
            pinned=memory.reserve(64*MiB,'pinned','staging')
            memory.reserve_device(0,400*MiB)
            before=memory.snapshot()
            self.assertEqual(before['charged_host_bytes'],928*MiB)
            self.assertEqual(before['explicit_host_bytes'],864*MiB)
            with self.assertRaises(MemoryError): memory.reserve_device(0,128*MiB)
            self.assertEqual(memory.snapshot()['device_used_bytes'],{0:400*MiB})
            memory.release_device(0,400*MiB)
            memory.reserve_device(0,128*MiB)
            memory.release_device(0,128*MiB)
            pinned.release(); host.release()
            self.assertEqual(memory.snapshot()['charged_host_bytes'],64*MiB)

    def test_available_floor_checks_pending_device_bytes(self):
        with self.runtime() as memory:
            memory.available=lambda:32*GiB+32*MiB
            with self.assertRaises(MemoryError): memory.reserve_device(0,64*MiB)
            self.assertEqual(memory.device_used,{})
            self.assertEqual(memory.unified_device_leases,{})
            memory.available=lambda:48*GiB
            memory.reserve_device(0,64*MiB)
            with self.assertRaises(ValueError): memory.release_device(0,32*MiB)
            self.assertEqual(memory.device_used,{0:64*MiB})
            memory.release_device(0,64*MiB)

    def test_discrete_default_does_not_charge_device_to_host(self):
        with self.runtime(unified_device_memory=False) as memory:
            before=memory.host
            memory.reserve_device(0,512*MiB)
            self.assertEqual(memory.host,before)
            memory.release_device(0,512*MiB)

    def test_invalid_unified_policy(self):
        with self.assertRaises(ValueError): MemoryPolicy(unified_device_memory=1)

    def test_first_moe_catalog_admission_after_recorded_uma_charge(self):
        # Reproduce the measured first-step ledger, without allocating GiB of
        # real memory. The prior 8 GiB cap refuses a 512 MiB catalog admission.
        recorded_charge = 8327851284
        for cap, expected_refusal in ((8 * GiB, True), (12 * GiB, False)):
            with self.subTest(cap=cap), self.runtime(host_cap_bytes=cap) as memory:
                prior = memory.reserve(recorded_charge - memory.CONTROL_METADATA_BYTES,
                    'admission', 'recorded_first_moe')
                catalog = MemoryBudget(cap=512 * MiB, owner='original_catalog')
                if expected_refusal:
                    with self.assertRaises(MemoryError): catalog.acquire(512 * MiB)
                    self.assertEqual(catalog.reserved, 0)
                else:
                    catalog.acquire(512 * MiB)
                    self.assertEqual(memory.host, recorded_charge + 512 * MiB)
                    catalog.release(512 * MiB)
                prior.release()

    def test_s1_composition_headroom_keeps_shared_ceiling_and_physical_floor(self):
        policy = spark_memory_policy()
        runtime = RuntimeMemory(policy, available=lambda:48 * GiB,
            process=lambda:{'resident_bytes':0})
        runtime.ENABLE_WATCHDOG = False
        with runtime as memory:
            # Entire conservative CPU allowance, the separately bounded
            # assets and full device cap must share one 12 GiB ledger.
            components = memory.reserve(6 * GiB - memory.CONTROL_METADATA_BYTES,
                'admission', 's1_conservative_components')
            assets = MemoryBudget(cap=SPARK_ASSET_CAP, owner='spark_model_assets')
            assets.acquire(SPARK_ASSET_CAP, category='weights')
            memory.reserve_device(0, 5 * GiB)
            self.assertEqual(memory.host, 12 * GiB)
            with self.assertRaises(MemoryError): memory.reserve(1, 'scratch', 'over_cap')
            memory.release_device(0, 5 * GiB)
            assets.release(SPARK_ASSET_CAP, category='weights')
            components.release()
            memory.available = lambda:32 * GiB + 256 * MiB
            with self.assertRaises(MemoryError): memory.reserve_device(0, 512 * MiB)
            self.assertEqual(memory.device_used, {0:0})
            memory.available = lambda:48 * GiB
        self.assertEqual(memory.host, 0)


if __name__=='__main__': unittest.main()
