"""Optional, non-overlapping host wall-time measurements for serialized probes."""

from contextlib import contextmanager
import time


class Timings:
    def __init__(self, enabled=False):
        self.enabled, self.stages = enabled, {}

    @contextmanager
    def section(self, name):
        if not self.enabled:
            yield
            return
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            entry = self.stages.setdefault(name, [0, 0])
            entry[0] += 1
            entry[1] += time.perf_counter_ns() - start

    def snapshot(self):
        return {name: {"calls": value[0], "total_ms": value[1] / 1e6} for name, value in self.stages.items()}

    def since(self, previous):
        return {name: {"calls": value["calls"] - previous.get(name, {}).get("calls", 0),
                       "total_ms": value["total_ms"] - previous.get(name, {}).get("total_ms", 0)}
                for name, value in self.snapshot().items()
                if value["calls"] != previous.get(name, {}).get("calls", 0)}
