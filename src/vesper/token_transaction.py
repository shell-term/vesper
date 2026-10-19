"""Whole-token publication across independently transactional native components.

This coordinator does not execute, qualify or approximate a Transformer layer.
The caller supplies the connected computation and exclusively owns the bound
state objects for the coordinator's lifetime. Cache contents may change on a
failed attempt; logical KV, token history and sampling state must be restored.
"""

from collections import OrderedDict
import threading

from .memory import runtime_operation
from .resources import MemoryBudget, MiB


class TokenTransaction:
    def __init__(self, states, history, sampler):
        if (not isinstance(states, dict) or not 1 <= len(states) <= 40
                or any(type(n) is not str or not n or len(n) > 64 for n in states)
                or len({id(s) for s in (*states.values(), history, sampler)}) != len(states) + 2):
            raise ValueError("Expected distinct bounded token-state owners")
        self.states, self.history, self.sampler = dict(states), history, sampler
        self.participants = {**states, "__history__": history, "__sampler__": sampler}
        if len(self.participants) != len(states) + 2:
            raise ValueError("Reserved token-state owner name")
        self.budget = MemoryBudget(cap=32 * MiB, owner="token_transaction")
        self.lock, self.opened = threading.RLock(), False
        self.pending, self.cleanup = OrderedDict(), OrderedDict()
        self.position = history.position
        self.committed, self.failed, self.recovered = 0, 0, 0

    def _aligned(self):
        if self.history.position != self.position or any(s.position != self.position for s in self.states.values()):
            raise RuntimeError("Token-state positions differ from the last model publication")

    def __enter__(self):
        with self.lock:
            if self.opened:
                raise RuntimeError("Token transaction is already open")
            self.budget.acquire(self.budget.cap)
            try:
                self._aligned()
                self.opened = True
                return self
            except BaseException:
                self.budget.release(self.budget.cap)
                raise

    def _ready(self):
        if not self.opened:
            raise RuntimeError("Token transaction is not open")
        if self.pending:
            raise RuntimeError("Whole-token recovery remains pending; recover before another call")
        self.budget.acquire(0)
        self._cleanup()
        self._aligned()

    def _cleanup(self):
        # Publication already succeeded. Failure to discard an unneeded handle
        # must never turn a committed token into a reported failed token.
        for name, handle in list(self.cleanup.items()):
            self.budget.acquire(0)
            self.participants[name].discard(handle)
            del self.cleanup[name]

    def _restore(self):
        errors = []
        with self.budget.restoration():
            # Try every independent owner, retaining only unconsumed handles.
            # A physical floor can block recovery even after cancellation.
            for name, handle in reversed(list(self.pending.items())):
                if handle is None:
                    del self.pending[name]
                    continue
                try:
                    self.budget.acquire(0)
                    self.participants[name].rollback(handle)
                    del self.pending[name]
                except BaseException as error:
                    errors.append(error)
        if errors:
            raise RuntimeError("Whole-token recovery remains pending; further calls are blocked") from errors[0]
        self._aligned()

    def recover(self):
        with self.lock, self.history.lock:
            if not self.opened or not self.pending:
                raise RuntimeError("No whole-token recovery is pending")
            self._restore()
            self.recovered += 1
            return {"position": self.position, "recovery_pending": False}

    @runtime_operation
    def step(self, ids, compute, *, start_pos=None):
        """Compute from one prepared hash candidate, then commit one position.

        ``compute(candidate)`` must finish all layers, final norm/head and the
        sampler, and build its returned result before returning. No checkpoint
        downloads belong in that callback. Serial one-token text calls only.
        """
        with self.lock, self.history.lock:
            self._ready()
            if (not callable(compute) or getattr(ids, "shape", None) != (1, 1)
                    or (start_pos is not None and (type(start_pos) is not int or start_pos != self.position))):
                raise ValueError("Expected the next single text token and a connected computation")
            candidate = self.history.prepare(ids, self.position)
            position, committed = self.position, self.committed
            next_position, next_committed, empty = position + 1, committed + 1, OrderedDict()
            # Reserve every dictionary slot before any component publishes an
            # opaque snapshot. A failed metadata insertion must not orphan a
            # handle that the outer transaction never got to record.
            self.pending = OrderedDict.fromkeys(self.participants)
            try:
                for name, participant in self.participants.items():
                    self.pending[name] = participant.snapshot()
                result = compute(candidate)
                if self.history.position != self.position or any(s.position != self.position + 1 for s in self.states.values()):
                    raise RuntimeError("Connected computation must advance every state exactly once without committing history")
                with self.budget.publication():
                    self.history.commit(candidate)
                    self.position, self.committed = next_position, next_committed
                    self.cleanup, self.pending = self.pending, empty
            except BaseException:
                self.position, self.committed = position, committed
                self.failed += 1
                self._restore()
                raise
            try:
                self._cleanup()
            except BaseException:
                # Retain cleanup ownership for the next call or close. Logical
                # state and the returned result have already been committed.
                pass
            return result

    def stats(self):
        with self.lock:
            return {"position": self.position, "committed_tokens": self.committed,
                "failed_attempts": self.failed, "explicit_recoveries": self.recovered,
                "pending_recovery_owners": list(self.pending), "pending_cleanup_owners": list(self.cleanup),
                "host_reserved_bytes": self.budget.reserved, "host_allowance_bytes": self.budget.cap,
                "reservation_is_RSS_limit": False}

    def __exit__(self, exc_type, *_):
        with self.lock, self.history.lock:
            if not self.opened:
                return
            try:
                if self.pending:
                    self._restore()
                with self.budget.restoration():
                    self._cleanup()
            except BaseException:
                if exc_type is None:
                    raise
            finally:
                self.opened = False
                self.budget.release(self.budget.cap)
