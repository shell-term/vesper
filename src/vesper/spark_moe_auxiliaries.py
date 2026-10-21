"""Spark-only reuse of one router and one single-token accumulation backend.

No expert, gate or shared-expert weights are retained here. Every router borrow
fully uploads both freshly authenticated trained biases. The existing native
accumulator begin() still zeros its sum before any expert contribution.
"""

from contextlib import contextmanager
from copy import deepcopy
import hashlib
from pathlib import Path
import threading

import numpy as np

from .checkpoint import MODEL, REVISION
from .cuda import Device, Driver, compile_probe, load_toolkit_library, _compiler_library_identity
from .execution_target import SPARK_TARGET, kernel_source_sha256, resolve_execution_target, _validate_spark_compiler
from .memory import current_policy
from .moe import MoEAccumulator
from .resources import MemoryBudget, MiB
from .router import RouterProbe, validate_router_arrays

METADATA_BYTES = MiB
ACCUMULATOR_BYTES = 5120 * (4 + 2 + 2) + 4
ROUTER_BYTES = 2 * 384 * 4 + 3 * 384 * 4 + 3 * 6 * 4 + 2 * 4
SOURCE_FILES = ('spark_moe_auxiliaries.py', 'connected.py', 'router.py', 'moe.py', 'cuda.py', 'execution_target.py')
CONTEXT_KEYS = ('platform', 'profile', 'repository', 'model', 'revision', 'git_head',
                'dirty_diff_sha256', 'registry_sha256', 'binding_id')


def _sources():
    root = Path(__file__).parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def _close_operator(operator):
    """Free before releasing the context; retain ownership on cleanup failure."""
    device = operator.device
    if device is not None:
        device.current()
        for pointer in list(device.allocations):
            device.free(pointer)
        while device.modules:
            device.driver.check(device.driver.cuModuleUnload(device.modules[-1]))
            device.modules.pop()
        device.close()
        operator.device = None
    if isinstance(operator, MoEAccumulator):
        if operator.budget.reserved:
            operator.budget.release(operator.budget.reserved)
        operator.state = 'closed'


class _ReusableAccumulator(MoEAccumulator):
    # MoEAccumulator.__enter__ invokes close on failed initialization. Keep the
    # same failure cleanup rule as normal pool cleanup, including partial entry.
    def close(self):
        _close_operator(self)


class _ReusableRouter(RouterProbe):
    def __enter__(self):
        # The same native layout and compiler path, with retryable cleanup of
        # partial allocation failures before the primary context is released.
        if self.device is not None:
            raise RuntimeError('Router probe is already open')
        driver = Driver()
        self.info = next(d for d in driver.inventory()['devices'] if d['ordinal'] == self.ordinal)
        self.device = Device(driver, self.ordinal, cap=MiB)
        self.device_object = self.device
        try:
            cubin = compile_probe(self.info['compute_capability'].replace('.', ''), routing=True)
            self.cubin_sha256 = hashlib.sha256(cubin).hexdigest()
            self.module = self.device.load(cubin)
            self.biases = {name: self.device.upload(array) for name, array in self.arrays.items()}
            self.layout = {'logits': ((384,), np.float32), 'scores': ((384,), np.float32),
                'selection': ((384,), np.float32), 'raw': ((6,), np.float32),
                'denominator': ((1,), np.float32), 'weights': ((6,), np.float32),
                'indices': ((6,), np.int32), 'tied': ((1,), np.int32)}
            self.work = {name: self.device.allocate(int(np.prod(shape)) * np.dtype(dtype).itemsize)
                for name, (shape, dtype) in self.layout.items()}
            return self
        except BaseException:
            _close_operator(self)
            raise


class _Entry:
    def __init__(self, kind, ordinal, arrays=None):
        self.kind, self.ordinal = kind, ordinal
        self.operator = _ReusableRouter(arrays, ordinal) if kind == 'router' else _ReusableAccumulator(ordinal, max_tokens=1)
        self.borrowed = False
        self.compiler, self.bias_sha256 = {}, {}

    @property
    def flags(self):
        return {'routing' if self.kind == 'router' else 'aggregation': True}

    def open(self):
        # The surrounding Spark compilation cache reproduces the same binary
        # used by the unchanged native operator; never substitute a new kernel.
        self.nvrtc = load_toolkit_library('nvrtc')
        self.compiler_identity = _compiler_library_identity(self.nvrtc)
        if self.compiler_identity is None:
            raise ValueError('Reusable MoE auxiliaries require an identified compiler')
        self.source_sha256 = kernel_source_sha256(execution_target=SPARK_TARGET, **self.flags)
        binary = compile_probe('121', compiler_info=self.compiler, **self.flags)
        _validate_spark_compiler(self.compiler, self.source_sha256, self.flags)
        self.binary_sha256 = hashlib.sha256(binary).hexdigest()
        self.operator.__enter__()
        op = self.operator
        if op.cubin_sha256 != self.binary_sha256:
            raise ValueError('Native auxiliary binary differs from its reproduced compiler identity')
        self.device, self.module = op.device, op.module
        self.inventory = self.device.driver.inventory()
        target = resolve_execution_target(SPARK_TARGET)
        target.validate_device(op.info, self.inventory)
        self.allocations, self.modules, self.work = dict(self.device.allocations), tuple(self.device.modules), dict(op.work)
        expected = ROUTER_BYTES if self.kind == 'router' else ACCUMULATOR_BYTES
        if sum(self.allocations.values()) != expected:
            raise ValueError('Native auxiliary workspace byte count differs from its fixed layout')
        if self.kind == 'router':
            self.biases, self.layout = dict(op.biases), deepcopy(op.layout)
            # The owner retains device buffers, not host fixture arrays.
            op.arrays = None
        self.validate()
        return self

    def validate(self):
        op, device = self.operator, self.operator.device
        if (device is None or device is not self.device or not device.context
                or op.ordinal != self.ordinal or op.module != self.module or tuple(device.modules) != self.modules
                or op.cubin_sha256 != self.binary_sha256 or device.allocations != self.allocations
                or op.work != self.work
                or (self.kind == 'router' and (op.biases != self.biases or op.layout != self.layout))
                or (self.kind == 'accumulator' and (op.max_tokens != 1 or op.budget.reserved != 4 * MiB))):
            raise RuntimeError('Reusable auxiliary device/binary/bias/workspace ownership changed')
        if (kernel_source_sha256(execution_target=SPARK_TARGET, **self.flags) != self.source_sha256
                or _compiler_library_identity(self.nvrtc) != self.compiler_identity
                or device.driver.inventory() != self.inventory):
            raise ValueError('Reusable auxiliary source/compiler/device identity changed')
        _validate_spark_compiler(self.compiler, self.source_sha256, self.flags)
        resolve_execution_target(SPARK_TARGET).validate_device(op.info, self.inventory)
        device.current()

    def upload_biases(self, arrays):
        op, device = self.operator, self.operator.device
        expected = {name: hashlib.sha256(memoryview(arrays[name])).hexdigest() for name in ('bias', 'bias_vl')}
        device.current()
        for name in ('bias', 'bias_vl'):
            array = arrays[name]
            if device.allocations.get(op.biases[name]) != array.nbytes:
                raise ValueError('Trained bias upload does not match its owned destination')
            device.driver.check(device.driver.cuMemcpyHtoD_v2(op.biases[name], array.ctypes.data, array.nbytes))
        device.driver.check(device.driver.cuCtxSynchronize())
        if any(hashlib.sha256(memoryview(arrays[name])).hexdigest() != expected[name] for name in expected):
            raise ValueError('Fresh trained bias changed during upload')
        self.bias_sha256 = expected

    def close(self):
        if self.borrowed:
            raise RuntimeError('A native auxiliary still has an active borrower')
        _close_operator(self.operator)


class SparkMoEAuxiliaries:
    """Two small runtime-owned auxiliaries; no simultaneous same-kind borrow.

    Explicit buffer ceilings stay unchanged. One metadata MiB and the native
    accumulator's four-MiB host admission cover this owner's host retention;
    the router/accumulator device allocations are separately charged once by
    their original Device owners in the unified policy. Context/module overhead
    remains externally observed, not an invented buffer-size measurement.
    """
    def __init__(self, workspace, ordinal=0):
        if type(ordinal) is not int or ordinal != 0:
            raise ValueError('Spark auxiliary reuse requires sole device ordinal zero')
        self.workspace, self.ordinal = workspace, ordinal
        self.lock = threading.RLock()
        self.entries, self.failed = {}, []
        self.opened = self.closed = self.poisoned = False
        self.opens = self.hits = self.borrows = self.evictions = self.bias_uploads = 0
        self.policy = self.reclaimer = None
        self.budget = MemoryBudget(cap=METADATA_BYTES, owner='spark_moe_auxiliary_metadata')

    def __enter__(self):
        with self.lock:
            if self.opened or self.closed:
                raise RuntimeError('Spark auxiliary owners open once')
            self.workspace.check()
            for name in SOURCE_FILES:
                self.workspace.check_runtime_module(Path(__file__).parent / name, 'src/vesper/' + name)
            self.context = self.workspace.context()
            if tuple(self.context[key] for key in ('platform', 'profile', 'model', 'revision')) != ('dgx-spark', 'native', MODEL, REVISION):
                raise ValueError('Auxiliary reuse requires the pinned Spark native workspace')
            self.policy = current_policy()
            if self.policy is None or not self.policy.config.unified_device_memory:
                raise ValueError('Auxiliary reuse requires an active unified runtime')
            self.sources = _sources()
            self.budget.acquire(METADATA_BYTES, category='metadata')
            try:
                self.reclaimer = self.policy.register_reclaimer(self.reclaim)
                self.opened = True
                return self
            except BaseException:
                self.budget.release(METADATA_BYTES, category='metadata')
                raise

    def _ready(self):
        if not self.opened or self.closed or self.poisoned or current_policy() is not self.policy:
            raise RuntimeError('Spark auxiliary owner is closed or requires cleanup')
        self.policy.checkpoint()
        context = self.workspace.context()
        if (any(context.get(key) != self.context.get(key) for key in CONTEXT_KEYS) or _sources() != self.sources):
            self.poisoned = True
            raise ValueError('Reusable auxiliary workspace/source identity changed')

    def invalidate(self):
        with self.lock:
            self.poisoned = True

    @contextmanager
    def router(self, arrays, ordinal):
        validate_router_arrays(arrays)
        if any(not arrays[name].flags.c_contiguous for name in ('bias', 'bias_vl')):
            raise ValueError('Expected contiguous authenticated trained biases')
        with self._acquire('router', ordinal, arrays) as operator:
            yield operator

    @contextmanager
    def accumulator(self, ordinal):
        with self._acquire('accumulator', ordinal) as operator:
            yield operator

    @contextmanager
    def _acquire(self, kind, ordinal, arrays=None):
        if type(ordinal) is not int or ordinal != self.ordinal or self.policy is None:
            raise ValueError('Expected this open Spark owner and sole device')
        with self.policy.lock, self.lock:
            self._ready()
            entry = self.entries.get(kind)
            if entry is not None and entry.borrowed:
                raise RuntimeError('Auxiliary workspace has an active borrower')
            if entry is None:
                entry = _Entry(kind, ordinal, arrays)
                try:
                    entry.open()
                except BaseException:
                    try:
                        entry.close()
                    except BaseException:
                        self.failed.append(entry)
                        self.poisoned = True
                    raise
                self.entries[kind] = entry
                self.opens += 1
            else:
                try:
                    entry.validate()
                except BaseException:
                    self.poisoned = True
                    raise
                self.hits += 1
            entry.borrowed = True
            try:
                if kind == 'router':
                    entry.upload_biases(arrays)
                    self.bias_uploads += 1
                self.borrows += 1
            except BaseException:
                entry.borrowed = False
                self.poisoned = True
                raise
        try:
            yield entry.operator
        except BaseException:
            self.invalidate()
            raise
        finally:
            with self.policy.lock, self.lock:
                entry.borrowed = False
                if entry.operator.device is not None and entry.operator.device.allocations != entry.allocations:
                    self.poisoned = True
                    raise RuntimeError('Borrower changed reusable auxiliary allocations')

    def reclaim(self, requested):
        with self.policy.lock, self.lock:
            freed = 0
            for kind, entry in list(self.entries.items()):
                if entry.borrowed or self.poisoned:
                    continue
                before = self.policy.host
                try:
                    entry.close()
                except BaseException:
                    self.poisoned = True
                    raise
                del self.entries[kind]
                self.evictions += 1
                freed += before - self.policy.host
                if freed >= requested:
                    break
            return freed

    def stats(self):
        with self.lock:
            entries = [*self.entries.values(), *self.failed]
            return {'opened': self.opened, 'closed': self.closed, 'poisoned': self.poisoned,
                'entries': len(self.entries), 'maximum_entries': 2, 'failed_cleanup_owners': len(self.failed),
                'opens': self.opens, 'reuse_hits': self.hits, 'borrows': self.borrows, 'evictions': self.evictions,
                'borrowers': sum(entry.borrowed for entry in entries), 'trained_bias_full_uploads': self.bias_uploads,
                'host_metadata_reserved_bytes': self.budget.reserved,
                'accumulator_host_reserved_bytes': sum(entry.operator.budget.reserved for entry in entries if entry.kind == 'accumulator'),
                'device_owned_bytes': sum(sum(entry.operator.device.allocations.values()) for entry in entries if entry.operator.device is not None),
                'maximum_explicit_device_bytes': ROUTER_BYTES + ACCUMULATOR_BYTES,
                'retained_expert_gate_shared_weights': False, 'reservation_is_RSS_limit': False,
                'source_sha256': dict(getattr(self, 'sources', {})), 'qualification_receipt_issued': False,
                'backends': {kind: {'compiler': deepcopy(entry.compiler), 'cubin_sha256': entry.binary_sha256,
                    'kernel_source_sha256': entry.source_sha256,
                    'compiler_library_identity_sha256': hashlib.sha256(repr(entry.compiler_identity).encode()).hexdigest(),
                    'last_trained_bias_sha256': dict(entry.bias_sha256)} for kind, entry in self.entries.items()}}

    def close(self):
        if self.policy is None:
            return
        with self.policy.lock, self.lock:
            errors = []
            for kind, entry in list(self.entries.items()):
                try:
                    entry.close()
                except BaseException as error:
                    errors.append(error)
                else:
                    del self.entries[kind]
            for entry in list(self.failed):
                try:
                    entry.close()
                except BaseException as error:
                    errors.append(error)
                else:
                    self.failed.remove(entry)
            if errors:
                self.poisoned = True
                raise RuntimeError('Auxiliary cleanup requires explicit retry') from errors[0]
            if self.reclaimer is not None:
                self.policy.unregister_reclaimer(self.reclaimer)
                self.reclaimer = None
            if self.budget.reserved:
                self.budget.release(self.budget.reserved, category='metadata')
            self.opened, self.closed = False, True

    def __exit__(self, *_):
        self.close()
