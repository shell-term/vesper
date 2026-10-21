"""CPU ownership/byte-overwrite checks; no CUDA or arithmetic qualification."""
from contextlib import ExitStack
import ctypes as C
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from vesper import spark_moe_auxiliaries as aux
from vesper import moe
from vesper.memory import RuntimeMemory, MemoryPolicy, RuntimeCancelled, current_policy
from vesper.resources import MiB, GiB
from tests.test_execution_target import BINARY, compiler_metadata


class Device:
    next_pointer = 1
    blocks = {}
    fail_load = False
    fail_alloc_after = None
    failed_frees_remaining = 0

    def __init__(self, driver, ordinal, *, cap, **_):
        self.driver, self.ordinal, self.cap = driver, ordinal, cap
        self.allocations, self.modules = {}, []
        self.context, self.peak = True, 0

    def current(self):
        if not self.context:
            raise RuntimeError('Closed test context')

    def allocate(self, size):
        if self.fail_alloc_after is not None and len(self.allocations) >= self.fail_alloc_after:
            raise MemoryError('Injected allocation refusal')
        if sum(self.allocations.values()) + size > self.cap:
            raise MemoryError('Native cap exceeded')
        current_policy().reserve_device(self.ordinal, size)
        pointer = Device.next_pointer
        Device.next_pointer += 1
        self.allocations[pointer] = size
        Device.blocks[pointer] = bytearray(size)
        self.peak = max(self.peak, sum(self.allocations.values()))
        return pointer

    def upload(self, array):
        pointer = self.allocate(array.nbytes)
        self.driver.check(self.driver.cuMemcpyHtoD_v2(pointer, array.ctypes.data, array.nbytes))
        return pointer

    def free(self, pointer):
        if Device.failed_frees_remaining:
            Device.failed_frees_remaining -= 1
            raise RuntimeError('Injected failed free')
        current_policy().release_device(self.ordinal, self.allocations.pop(pointer))
        del Device.blocks[pointer]

    def load(self, _):
        if self.fail_load:
            raise ValueError('Injected module load failure')
        module = object()
        self.modules.append(module)
        return module

    def launch(self, _module, name, args, _items):
        if name == 'moe_zero':
            Device.blocks[args[0].value][:] = bytes(args[1].value * 4)

    def close(self):
        if self.allocations or self.modules:
            raise AssertionError('Free/unload before context release')
        self.context = False


class SparkMoEAuxiliaryTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(RuntimeMemory, 'ENABLE_WATCHDOG', False))
        self.stack.enter_context(patch.object(RuntimeMemory, 'CONTROL_METADATA_BYTES', 0))
        for name in ('vesper.resources.physical_memory', 'vesper.memory.physical_memory'):
            self.stack.enter_context(patch(name, return_value={'available': 128 * GiB}))
        self.stack.enter_context(patch('vesper.execution_target.platform.system', return_value='Linux'))
        self.stack.enter_context(patch('vesper.execution_target.platform.machine', return_value='aarch64'))
        self.context = {'platform': 'dgx-spark', 'profile': 'native', 'model': aux.MODEL, 'revision': aux.REVISION,
            'repository': 'shell-term/vesper', 'git_head': 'a' * 40, 'dirty_diff_sha256': 'b' * 64,
            'registry_sha256': 'c' * 64, 'binding_id': 'runtime-one'}
        self.workspace = SimpleNamespace(check=Mock(), check_runtime_module=Mock(), context=lambda: dict(self.context))
        self.inventory = {'driver_api_version': 13000, 'devices': [{'ordinal': 0, 'name': 'NVIDIA GB10',
            'compute_capability': '12.1', 'total_bytes': 128 * GiB}]}
        self.driver = Mock()
        self.driver.inventory.return_value = self.inventory
        self.driver.check.side_effect = lambda code: None
        def copy(pointer, source, size):
            Device.blocks[pointer][:] = C.string_at(source, size)
            return 0
        self.driver.cuMemcpyHtoD_v2.side_effect = copy
        def compile(architecture, *, compiler_info=None, **flags):
            if compiler_info is not None:
                compiler_info.update(compiler_metadata(architecture, **flags))
            return BINARY
        for module in (aux, moe):
            for name, value in (('Driver', Mock(return_value=self.driver)), ('Device', Device),
                                ('compile_probe', Mock(side_effect=compile))):
                self.stack.enter_context(patch.object(module, name, value))
        self.stack.enter_context(patch.object(aux, 'load_toolkit_library', return_value=object()))
        self.stack.enter_context(patch.object(aux, '_compiler_library_identity', return_value=(1, 'private-compiler', 2)))
        Device.blocks = {}
        Device.fail_load = False
        Device.fail_alloc_after = None
        Device.failed_frees_remaining = 0

    def runtime(self, cap=32 * MiB):
        return RuntimeMemory(MemoryPolicy(host_cap_bytes=cap, device_caps={0: 8 * MiB},
            pinned_cap_bytes=MiB, unified_device_memory=True),
            available=lambda: 128 * GiB, process=lambda: {'resident_bytes': 0})

    def biases(self, value=0.):
        return {name: np.full(384, value + offset, np.float32) for offset, name in enumerate(('bias', 'bias_vl'))}

    def test_two_entries_reuse_complete_bias_uploads_and_native_zero_without_extra_residency(self):
        with self.runtime() as memory:
            with aux.SparkMoEAuxiliaries(self.workspace) as pool:
                identities = {}
                for value in (0., 3.):
                    arrays = self.biases(value)
                    with pool.router(arrays, 0) as router, pool.accumulator(0) as accumulator:
                        for kind, operator in (('router', router), ('accumulator', accumulator)):
                            identities.setdefault(kind, operator)
                            self.assertIs(operator, identities[kind])
                        for name, array in arrays.items():
                            self.assertEqual(bytes(Device.blocks[router.biases[name]]), array.tobytes())
                        self.assertIsNone(router.arrays)
                        Device.blocks[accumulator.work['sum']][:] = b'\xff' * (5120 * 4)
                        accumulator.begin(np.arange(6, dtype=np.int32).reshape(1, 6))
                        self.assertEqual(Device.blocks[accumulator.work['sum']], bytes(5120 * 4))
                stats = pool.stats()
                self.assertEqual((stats['opens'], stats['reuse_hits'], stats['borrows']), (2, 2, 4))
                self.assertEqual(stats['trained_bias_full_uploads'], 2)
                self.assertEqual(stats['device_owned_bytes'], 48724)
                self.assertEqual(stats['maximum_explicit_device_bytes'], 48724)
                self.assertEqual(memory.host, 5 * MiB + 48724)
                self.assertEqual(stats['accumulator_host_reserved_bytes'], 4 * MiB)
                self.assertNotIn('private-compiler', str(stats))
            self.assertEqual(memory.host, 0)
            self.assertFalse(memory.reclaimers)
            self.assertFalse(Device.blocks)

    def test_nested_borrow_refuses_before_overwriting_live_bias_and_bad_input_opens_nothing(self):
        with self.runtime():
            with aux.SparkMoEAuxiliaries(self.workspace) as pool:
                with self.assertRaises(ValueError):
                    with pool.router(self.biases(float('nan')), 0): pass
                self.assertEqual(pool.stats()['opens'], 0)
                with pool.router(self.biases(1), 0) as router:
                    before = bytes(Device.blocks[router.biases['bias']])
                    with self.assertRaisesRegex(RuntimeError, 'active borrower'):
                        with pool.router(self.biases(2), 0): pass
                    self.assertEqual(bytes(Device.blocks[router.biases['bias']]), before)
                with self.assertRaises(ValueError):
                    with pool.accumulator(True): pass

    def test_changed_source_compiler_bias_pointer_or_workspace_refuses_reuse(self):
        for change in ('workspace', 'source', 'compiler', 'bias', 'work', 'binary', 'device'):
            with self.subTest(change=change), self.runtime():
                with aux.SparkMoEAuxiliaries(self.workspace) as pool:
                    with pool.router(self.biases(), 0) as router: pass
                    with ExitStack() as edits:
                        if change == 'workspace':
                            edits.enter_context(patch.dict(self.context, git_head='f' * 40))
                        elif change == 'source':
                            edits.enter_context(patch.object(aux, '_sources', return_value={}))
                        elif change == 'compiler':
                            edits.enter_context(patch.object(aux, '_compiler_library_identity', return_value=('changed',)))
                        elif change == 'bias':
                            edits.enter_context(patch.dict(router.biases, bias=123456))
                        elif change == 'work':
                            edits.enter_context(patch.dict(router.work, logits=123456))
                        elif change == 'binary':
                            edits.enter_context(patch.object(router, 'cubin_sha256', 'f' * 64))
                        else:
                            edits.enter_context(patch.object(self.driver, 'inventory', return_value={}))
                        with self.assertRaises((ValueError, RuntimeError)):
                            with pool.router(self.biases(), 0): pass
                    self.assertTrue(pool.poisoned)

    def test_pressure_drops_idle_auxiliaries_but_never_active_workspace(self):
        with self.runtime() as memory:
            with aux.SparkMoEAuxiliaries(self.workspace) as pool:
                with pool.router(self.biases(), 0), pool.accumulator(0):
                    self.assertEqual(pool.reclaim(32 * MiB), 0)
                    with self.assertRaises(MemoryError): memory.reserve(30 * MiB, 'scratch', 'pressure')
                lease = memory.reserve(30 * MiB, 'scratch', 'pressure')
                self.assertEqual(pool.stats()['entries'], 0)
                lease.release()
                with pool.router(self.biases(4), 0), pool.accumulator(0): pass
                self.assertEqual(pool.stats()['opens'], 4)
            self.assertEqual(memory.host, 0)

    def test_initial_module_failure_cleans_and_healthy_retry_opens_fresh(self):
        with self.runtime() as memory:
            with aux.SparkMoEAuxiliaries(self.workspace) as pool:
                Device.fail_load = True
                with self.assertRaisesRegex(ValueError, 'module load'):
                    with pool.accumulator(0): pass
                self.assertEqual(memory.host, MiB)
                self.assertFalse(pool.poisoned)
                Device.fail_load = False
                with pool.accumulator(0): pass
            self.assertEqual(memory.host, 0)

    def test_failed_partial_entry_cleanup_retains_owner_until_explicit_close_retry(self):
        with self.runtime() as memory:
            pool = aux.SparkMoEAuxiliaries(self.workspace).__enter__()
            Device.fail_alloc_after = 1
            Device.failed_frees_remaining = 2
            with self.assertRaisesRegex(RuntimeError, 'failed free'):
                with pool.router(self.biases(), 0): pass
            self.assertTrue(pool.poisoned)
            self.assertEqual(pool.stats()['failed_cleanup_owners'], 1)
            self.assertGreater(pool.stats()['device_owned_bytes'], 0)
            self.assertTrue(pool.failed[0].operator.device.context)
            Device.fail_alloc_after = None
            pool.close()
            self.assertEqual(memory.host, 0)
            self.assertFalse(Device.blocks)

    def test_failed_bias_upload_poison_cannot_publish_or_reuse_and_close_still_releases(self):
        with self.runtime() as memory:
            with aux.SparkMoEAuxiliaries(self.workspace) as pool:
                with pool.router(self.biases(), 0): pass
                self.driver.cuMemcpyHtoD_v2.side_effect = ValueError('Injected trained bias upload failure')
                with self.assertRaisesRegex(ValueError, 'bias upload'):
                    with pool.router(self.biases(7), 0): self.fail('Partial bias was published')
                self.assertTrue(pool.poisoned)
                self.assertEqual(pool.stats()['borrowers'], 0)
                with self.assertRaisesRegex(RuntimeError, 'requires cleanup'):
                    with pool.accumulator(0): pass
            self.assertEqual(memory.host, 0)

    def test_failed_free_retains_context_and_runtime_cancel_cannot_start_new_borrow(self):
        with self.runtime() as memory:
            pool = aux.SparkMoEAuxiliaries(self.workspace).__enter__()
            with pool.accumulator(0): pass
            Device.failed_frees_remaining = 1
            with self.assertRaisesRegex(RuntimeError, 'cleanup requires'):
                pool.close()
            self.assertEqual(pool.stats()['entries'], 1)
            self.assertTrue(pool.entries['accumulator'].operator.device.context)
            self.assertGreater(memory.host, MiB)
            pool.close()
            self.assertEqual(memory.host, 0)
        with self.runtime() as memory:
            with aux.SparkMoEAuxiliaries(self.workspace) as pool:
                memory.cancel()
                with self.assertRaises(RuntimeCancelled):
                    with pool.accumulator(0): pass
                self.assertEqual(pool.stats()['opens'], 0)


if __name__ == '__main__':
    unittest.main()
