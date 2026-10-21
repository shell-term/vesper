"""Opt-in bounded MoE context reuse; model arithmetic and cache audits unchanged."""

from dataclasses import asdict
import math
from pathlib import Path
import threading
import time

from .backbone_prepare import PreparedMoE
from .backbone_runtime import validate_moe_manifests
from .connected import ConnectedConfig, ConnectedMoE
from .indexed_store import IndexedStore
from .model_artifacts import digest
from .resources import MiB


class SessionMoE(PreparedMoE):
    """One serial layer owner with explicit ephemeral/retained lifetimes.

    Retention preserves the existing integrity-checked cache across tokens. It
    neither changes routes nor skips payload/upload validation. The historical
    verification providers remain unchanged.
    """

    def __init__(self, root, bindings, layer, prepared, expected_provenance, *,
                 retain=True, config=None, index_directory=None, deadline=None):
        super().__init__(root, bindings, layer, prepared, expected_provenance)
        if type(retain) is not bool:
            raise ValueError("Expected an explicit context retention mode")
        if deadline is not None and (type(deadline) not in (int, float) or not math.isfinite(deadline)):
            raise ValueError("Expected a finite cooperative deadline")
        if index_directory is not None:
            self.store = Path(index_directory)
        self.store = self.store.resolve(strict=True)
        if not self.store.is_relative_to(self.root.resolve(strict=True)):
            raise ValueError("MoE session metadata escaped its artifact root")
        self.indexed = index_directory is not None
        self.retain, self.config, self.deadline = retain, config or ConnectedConfig(), deadline
        self.component, self.opened, self.failed = None, False, False
        self.lock = threading.RLock()
        self.calls = self.opens = self.closes = self.reads = self.bytes_read = 0
        self.open_seconds = self.forward_seconds = 0.0
        self.tiers, self.stage_timings, self.last_cleanup = {}, {}, None
        self.manifest_sha256 = self._manifest_digest()

    def _manifest_digest(self):
        path = self.store / 'manifest.json'
        if path.is_symlink() or path.resolve(strict=True).parent != self.store:
            raise ValueError("Session metadata escaped its source directory")
        if not 1 <= path.stat().st_size <= 4 * MiB:
            raise ValueError("Unbounded session source metadata")
        with path.open('rb') as stream:
            raw = stream.read(4 * MiB + 1)
        if len(raw) > 4 * MiB:
            raise ValueError("Session source metadata grew beyond its bound")
        return digest(raw)

    def _check(self):
        if self.deadline is not None and time.perf_counter() >= self.deadline:
            raise TimeoutError("Session benchmark reached its cooperative time budget")
        if self._manifest_digest() != self.manifest_sha256:
            raise ValueError("Immutable session source metadata changed")

    def _open(self):
        self._check()
        factory = (lambda directory, budget: IndexedStore(directory, self.root, budget, layer=self.layer)) if self.indexed else None
        component = ConnectedMoE(self.store, *self.files, self.config, layer=self.layer,
                                 profile='backbone_probe', source_factory=factory)
        start = time.perf_counter()
        component.__enter__()
        self.opens += 1
        try:
            validate_moe_manifests(component, self.bindings, self.layer)
            source = component.experts.store
            if (set(source.entries) != {f'{self.layer}:{e}' for e in self.available}
                    or any(e['provenance'] != self.expected_provenance[k] for k, e in source.entries.items())):
                raise ValueError("Session source differs from the original expert provenance")
            self._check()
        except BaseException:
            self._close(component)
            raise
        finally:
            self.open_seconds += time.perf_counter() - start
        return component

    def _close(self, component):
        try:
            component.__exit__(None, None, None)
        finally:
            self.closes += 1
            self.last_cleanup = getattr(component, 'cleanup', None)

    def __enter__(self):
        with self.lock:
            if self.opened:
                raise RuntimeError("MoE session is already open")
            self._check()
            self.failed = False
            if self.retain:
                self.component = self._open()
            self.opened = True
            return self

    def _forward(self, component, values, image_span):
        source = component.experts.store
        before = source.read_calls, source.bytes_read
        timings = component.experts.timings.snapshot()
        start = time.perf_counter()
        try:
            result = component.forward(values, image_span)
            for event in result['expert_events']:
                tier = event['source_tier']
                self.tiers[tier] = self.tiers.get(tier, 0) + 1
            self.calls += 1
            return result
        except BaseException:
            self.failed = True
            raise
        finally:
            self.forward_seconds += time.perf_counter() - start
            self.reads += source.read_calls - before[0]
            self.bytes_read += source.bytes_read - before[1]
            for name, entry in component.experts.timings.since(timings).items():
                total = self.stage_timings.setdefault(name, {'calls': 0, 'total_ms': 0.0})
                total['calls'] += entry['calls']
                total['total_ms'] += entry['total_ms']

    def forward(self, values, image_span=False):
        with self.lock:
            if not self.opened:
                raise RuntimeError("Use the MoE session as a context manager")
            if self.failed:
                raise RuntimeError("Close the failed MoE session before reuse")
            self._check()
            if self.retain:
                return self._forward(self.component, values, image_span)
            component = self._open()
            try:
                return self._forward(component, values, image_span)
            finally:
                self._close(component)

    def stats(self):
        return {'layer': self.layer, 'retained': self.retain, 'calls': self.calls,
                'context_opens': self.opens, 'context_closes': self.closes,
                'direct_read_calls': self.reads, 'direct_read_bytes': self.bytes_read,
                'source_tiers': dict(self.tiers), 'open_seconds': self.open_seconds,
                'forward_seconds': self.forward_seconds, 'stage_timings': self.stage_timings,
                'manifest_sha256': self.manifest_sha256, 'opened': self.opened,
                'failed': self.failed,
                'cache_config': asdict(self.config.cache), 'last_cleanup': self.last_cleanup}

    def __exit__(self, *_):
        with self.lock:
            try:
                if self.component is not None:
                    self._close(self.component)
            finally:
                self.component = None
                self.opened = False
