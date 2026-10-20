"""UMA CPU/GPU admission shares one ceiling and preserves rollback/retry."""
import unittest

from vesper.memory import MemoryPolicy, RuntimeMemory
from vesper.resources import GiB, MiB


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


if __name__=='__main__': unittest.main()
