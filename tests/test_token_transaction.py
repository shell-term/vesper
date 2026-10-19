import copy
from collections import OrderedDict
import unittest
from unittest.mock import patch

import numpy as np

from vesper.kv import NativeWindowKV, WindowCheckpoint
from vesper.memory import MemoryPolicy, RuntimeMemory, RuntimeCancelled
from vesper.ngram import NgramState
from vesper.resources import GiB, MiB
from vesper.token_transaction import TokenTransaction
from vesper.prefix_profile import validate_layer


class State:
    """Protocol test double, not numerical model evidence."""
    def __init__(self):
        self.position, self.payload, self.handle = 0, b"original", None
        self.fail_restore, self.fail_discard = False, False

    def snapshot(self):
        if self.handle is not None:
            raise RuntimeError("occupied")
        self.handle = object()
        self.saved = self.position, self.payload
        return self.handle

    def rollback(self, handle):
        if self.fail_restore:
            raise OSError("restore transfer")
        if handle is not self.handle:
            raise ValueError("foreign")
        self.position, self.payload = self.saved
        self.handle = None

    def discard(self, handle):
        if self.fail_discard:
            raise OSError("discard observation")
        if handle is not self.handle:
            raise ValueError("foreign")
        self.handle = None


class Sampler(State):
    def __init__(self):
        super().__init__()
        self.position = 2026  # Emulated RNG offset, not a model sequence position.


class TokenTransactionTests(unittest.TestCase):
    def setUp(self):
        self.memory = patch("vesper.resources.physical_memory", return_value={"available": 64 * GiB})
        self.memory.start(); self.addCleanup(self.memory.stop)
        self.mapping = np.arange(129280, dtype=np.int64) % 99092
        self.ids = np.array([[0]], np.int64)

    def components(self):
        states = {"window0": State(), "window1": State(), "sharing": State()}
        history, rng = NgramState(self.mapping), Sampler()
        return states, history, rng

    @staticmethod
    def advance(states, rng):
        for state in states.values():
            state.position += 1; state.payload = b"new"
        rng.position += 4; rng.payload = b"new draw"
        return {"output": 42}

    def test_late_failure_restores_all_owners_and_retry_commits_once(self):
        states, history, rng = self.components()
        with TokenTransaction(states, history, rng) as transaction:
            def failing(candidate):
                self.advance(states, rng)
                raise OSError("after output and RNG")
            with self.assertRaisesRegex(OSError, "after output"):
                transaction.step(self.ids, failing)
            self.assertEqual((transaction.position, history.position, rng.position), (0, 0, 2026))
            self.assertTrue(all(s.position == 0 and s.payload == b"original" for s in states.values()))
            self.assertEqual(transaction.step(self.ids, lambda c: self.advance(states, rng)), {"output": 42})
            self.assertEqual((transaction.position, history.position, rng.position), (1, 1, 2030))
            self.assertFalse(history.handles)
            self.assertTrue(all(s.handle is None for s in (*states.values(), rng)))

    def test_partial_snapshot_failure_releases_earlier_slots(self):
        states, history, rng = self.components()
        with TokenTransaction(states, history, rng) as transaction, patch.object(states["window1"], "snapshot", side_effect=OSError("snapshot")):
            with self.assertRaises(OSError):
                transaction.step(self.ids, lambda c: self.fail("compute reached"))
            self.assertIsNone(states["window0"].handle)
            self.assertFalse(transaction.pending)

    def test_snapshot_handles_are_recorded_without_allocating_new_dictionary_slots(self):
        class GuardedSlots(OrderedDict):
            refuse_new = False
            def __setitem__(self, key, value):
                if self.refuse_new and key not in self:
                    raise MemoryError("metadata insertion after snapshot publication")
                return super().__setitem__(key, value)
        states, history, rng = self.components()
        original = states["window0"].snapshot
        def snapshot():
            handle = original(); GuardedSlots.refuse_new = True
            return handle
        with patch("vesper.token_transaction.OrderedDict", GuardedSlots), patch.object(states["window0"], "snapshot", snapshot):
            with TokenTransaction(states, history, rng) as transaction:
                transaction.step(self.ids, lambda c: self.advance(states, rng))
                self.assertTrue(all(s.handle is None for s in (*states.values(), rng)))
                self.assertFalse(transaction.pending)

    def test_blocked_restore_retains_only_failed_handle_and_recover_then_retry(self):
        states, history, rng = self.components()
        with TokenTransaction(states, history, rng) as transaction:
            def failing(c):
                self.advance(states, rng); states["window1"].fail_restore = True
                raise OSError("late layer")
            with self.assertRaisesRegex(RuntimeError, "recovery remains pending"):
                transaction.step(self.ids, failing)
            self.assertEqual(list(transaction.pending), ["window1"])
            self.assertEqual((states["window0"].position, states["sharing"].position, history.position, rng.position), (0, 0, 0, 2026))
            with self.assertRaisesRegex(RuntimeError, "recover before"):
                transaction.step(self.ids, lambda c: self.fail("blocked computation reached"))
            states["window1"].fail_restore = False
            self.assertEqual(transaction.recover()["position"], 0)
            transaction.step(self.ids, lambda c: self.advance(states, rng))
            self.assertEqual(transaction.position, 1)

    def test_history_commit_failure_rolls_back_already_published_history(self):
        states, history, rng = self.components()
        with TokenTransaction(states, history, rng) as transaction:
            original = history.commit
            def fail_after_commit(c):
                original(c)
                raise MemoryError("commit allocation")
            with patch.object(history, "commit", fail_after_commit), self.assertRaises(MemoryError):
                transaction.step(self.ids, lambda c: self.advance(states, rng))
            self.assertEqual((history.position, transaction.position, rng.position), (0, 0, 2026))
            self.assertFalse(transaction.pending)

    def test_cleanup_failure_does_not_report_a_committed_token_as_failed(self):
        states, history, rng = self.components()
        with TokenTransaction(states, history, rng) as transaction:
            states["window1"].fail_discard = True
            self.assertEqual(transaction.step(self.ids, lambda c: self.advance(states, rng)), {"output": 42})
            self.assertEqual((transaction.position, history.position), (1, 1))
            self.assertEqual(list(transaction.cleanup), ["window1", "sharing", "__history__", "__sampler__"])
            with self.assertRaisesRegex(OSError, "discard"):
                transaction.step(self.ids, lambda c: self.fail("cleanup must finish first"))
            states["window1"].fail_discard = False
            transaction.step(self.ids, lambda c: self.advance(states, rng))
            self.assertEqual(transaction.position, 2)

    def test_cancel_restores_then_explicit_resume_required(self):
        states, history, rng = self.components()
        config = MemoryPolicy(host_cap_bytes=512 * MiB, pinned_cap_bytes=128 * MiB)
        with RuntimeMemory(config, available=lambda: 64 * GiB) as policy:
            with TokenTransaction(states, history, rng) as transaction:
                def cancel(c):
                    self.advance(states, rng); policy.cancel()
                    return {"output": 42}
                with self.assertRaises(RuntimeCancelled):
                    transaction.step(self.ids, cancel)
                self.assertTrue(policy.cancelled)
                self.assertEqual((transaction.position, history.position, rng.position), (0, 0, 2026))
                with self.assertRaises(RuntimeCancelled):
                    transaction.step(self.ids, lambda c: self.fail("cancelled computation reached"))
                policy.resume()
                transaction.step(self.ids, lambda c: self.advance(states, rng))
                self.assertEqual(transaction.position, 1)

    def test_incomplete_state_advance_is_a_failed_token(self):
        states, history, rng = self.components()
        with TokenTransaction(states, history, rng) as transaction:
            def incomplete(c):
                states["window0"].position += 1
            with self.assertRaisesRegex(RuntimeError, "advance every state"):
                transaction.step(self.ids, incomplete)
            self.assertTrue(all(s.position == 0 for s in states.values()))

    def test_invalid_binding_input_and_expired_history_handle(self):
        state = State(); history = NgramState(self.mapping); rng = Sampler()
        for states in ({}, {"x": state, "y": state}, {"__history__": state}):
            with self.assertRaises(ValueError):
                TokenTransaction(states, history, rng)
        states, history, rng = self.components()
        with TokenTransaction(states, history, rng) as transaction:
            for ids in (np.array([[0.]], np.float32), np.array([[0, 2]], np.int64), np.array([[-1]], np.int64)):
                with self.assertRaises(ValueError): transaction.step(ids, lambda c: None)
            with self.assertRaises(ValueError): transaction.step(self.ids, lambda c: None, start_pos=True)
        token = history.snapshot(); history.discard(token)
        with self.assertRaises(ValueError): history.rollback(token)
        with self.assertRaises(ValueError): history.discard(token)

    def test_prefix_profile_does_not_admit_unqualified_backbone_layers(self):
        from vesper.token_prefix_runtime import control_folder
        for layer in (4, 8, 39, -1, True, 1.):
            with self.assertRaises(ValueError): validate_layer(layer, (0, 2, 3), "token_prefix")
            with self.assertRaises(ValueError): control_folder("artifacts", layer)
        with self.assertRaises(ValueError): validate_layer(1, (0, 2, 3))
        self.assertEqual(validate_layer(1, (0, 2, 3), "token_prefix"), 1)

    def test_history_restore_allocation_failure_keeps_live_snapshot_and_current_state(self):
        history = NgramState(self.mapping)
        history.commit(history.prepare(self.ids))
        handle = history.snapshot()
        saved = history.handles[handle]["history"]
        history.commit(history.prepare(np.array([[2]], np.int64)))
        before = history.state_sha256()
        class FailedCopy:
            def copy(self): raise MemoryError("history restore allocation")
        history.handles[handle]["history"] = FailedCopy()
        with self.assertRaises(MemoryError): history.rollback(handle)
        self.assertIn(handle, history.handles)
        self.assertEqual(history.state_sha256(), before)
        history.handles[handle]["history"] = saved
        history.rollback(handle)
        self.assertEqual(history.position, 1)
        np.testing.assert_array_equal(history.history, saved)


class WindowCheckpointTests(unittest.TestCase):
    def test_outer_host_copy_survives_inner_snapshot_and_failed_restore(self):
        native = NativeWindowKV()
        native.budget.available_floor = 0
        native.active, native.position, native.buffers = 0, 129, [1, 2, 3]
        original = np.arange(128 * 512, dtype=np.uint16).reshape(1, 128, 512)
        native.driver = type("Driver", (), {"check": lambda *args: None, "cuMemcpyHtoD_v2": lambda *args: 0})()
        with patch.object(native, "_require_open"), patch.object(native, "_copy"), patch.object(native, "_cache", return_value=original.copy()):
            outer = native.checkpoint()
            with self.assertRaises(RuntimeError): native.checkpoint()
            self.assertFalse(native.checkpoint_cache.flags.writeable)
            inner = native.snapshot()
            native.position, native.active = 131, 1
            native.rollback(inner)
            self.assertIs(native.checkpoint_handle, outer)
            for handle in (None, WindowCheckpoint(129), copy.copy(outer)):
                with self.assertRaises(ValueError): native.restore(handle)
            with patch.object(native, "_cache", return_value=np.zeros_like(original)), self.assertRaisesRegex(RuntimeError, "storage bits"):
                native.restore(outer)
            self.assertIs(native.checkpoint_handle, outer)
            self.assertEqual(native.restore(outer)["position"], 129)
            self.assertIsNone(native.checkpoint_cache)
            with self.assertRaises(ValueError): native.restore(outer)


if __name__ == "__main__": unittest.main()
