"""Ordered, single-rank MoE accumulation; no routing or post-projection weights."""

from __future__ import annotations

import ctypes as C
import hashlib

import numpy as np

from .cuda import Device, Driver, compile_probe
from .resources import MemoryBudget, MiB

HIDDEN = 5120
TOP_K = 6
EXPERTS = 384


def participation(indices, max_tokens=32):
    indices = np.asarray(indices)
    if (indices.dtype not in (np.dtype(np.int32), np.dtype(np.int64)) or indices.ndim != 2
            or indices.shape[1] != TOP_K or indices.shape[0] > max_tokens):
        raise ValueError("Expected at most 32 tokens with six integer expert IDs each")
    if np.any(indices < 0) or np.any(indices >= EXPERTS):
        raise ValueError("Expert ID is outside the pinned model")
    if np.any(np.diff(np.sort(indices, axis=1), axis=1) == 0):
        # Gate.topk returns unique IDs. Repeated IDs would make the original
        # y[idx] += ... perform overlapping advanced-index writes on CUDA.
        raise ValueError("Repeated expert IDs within a token are outside the unique-topk scope")
    return {int(expert): np.flatnonzero(np.any(indices == expert, axis=1)).astype(np.int32)
            for expert in np.unique(indices)}


def _bf16_output(array, tokens):
    array = np.asarray(array)
    if array.dtype != np.uint16 or array.shape != (tokens, HIDDEN):
        raise ValueError("Expected BF16 storage bits with shape (participating_tokens, 5120)")
    if np.any((array & np.uint16(0x7f80)) == np.uint16(0x7f80)):
        raise ValueError("Expert output must be finite in this probe")
    return np.ascontiguousarray(array)


class MoEAccumulator:
    """Add weighted expert BF16 outputs in ID order and the shared output last.

    Routing weights belong inside Expert.forward, before w2. This class accepts
    only completed BF16 expert outputs and never multiplies them by a weight.
    Batch support tests participation/aggregation only, not batched GEMM parity.
    """

    def __init__(self, ordinal=0, max_tokens=32):
        if type(max_tokens) is not int or not 1 <= max_tokens <= 32:
            raise ValueError("Aggregation probe supports one to 32 workspace tokens")
        self.ordinal, self.max_tokens = ordinal, max_tokens
        self.device, self.state = None, "closed"
        self.budget = MemoryBudget(cap=4 * MiB)

    def __enter__(self):
        if self.device is not None:
            raise RuntimeError("Accumulator is already open")
        self.budget.acquire(4 * MiB)
        try:
            self.driver = Driver()
            self.info = next(d for d in self.driver.inventory()["devices"] if d["ordinal"] == self.ordinal)
            cubin = compile_probe(self.info["compute_capability"].replace(".", ""), aggregation=True)
            self.cubin_sha256 = hashlib.sha256(cubin).hexdigest()
            self.device = Device(self.driver, self.ordinal, cap=4 * MiB)
            self.module = self.device.load(cubin)
            count = self.max_tokens * HIDDEN
            self.work = {name: self.device.allocate(size) for name, size in
                         {"sum": count * 4, "input": count * 2,
                          "rows": self.max_tokens * 4, "output": count * 2}.items()}
            self.state = "ready"
            return self
        except Exception:
            self.close()
            raise

    def _open(self):
        if self.device is None:
            raise RuntimeError("Use the accumulator as a context manager")

    def _launch(self, name, pointers, trailing, items):
        if items:
            self.device.launch(self.module, name, [C.c_uint64(p) for p in pointers] + trailing, items)

    def _upload(self, name, array):
        if array.nbytes:
            self.device.current()
            self.driver.check(self.driver.cuMemcpyHtoD_v2(self.work[name], array.ctypes.data, array.nbytes))

    def _snapshot(self):
        if not self.tokens:
            return np.empty((0, HIDDEN), np.float32)
        return self.device.download(self.work["sum"], (self.tokens, HIDDEN), np.float32)

    def begin(self, indices):
        """Start/reset a diagnostic call; zero all active FP32 rows."""
        self._open()
        selected = participation(indices, self.max_tokens)
        self.tokens = np.asarray(indices).shape[0]
        self.selected = selected
        self.order = list(selected)
        self.position = 0
        self._launch("moe_zero", [self.work["sum"]], [C.c_int(self.tokens * HIDDEN)], self.tokens * HIDDEN)
        self.state = "routed"

    def add(self, expert_id, output):
        self._open()
        if (self.state != "routed" or self.position >= len(self.order)
                or type(expert_id) is not int or expert_id != self.order[self.position]):
            raise ValueError("Contributions must follow the selected ascending expert-ID order")
        rows = self.selected[expert_id]
        output = _bf16_output(output, len(rows))
        self._upload("input", output)
        self._upload("rows", rows)
        self._launch("moe_add_expert", [self.work["sum"], self.work["input"], self.work["rows"]],
                     [C.c_int(len(rows)), C.c_int(HIDDEN)], len(rows) * HIDDEN)
        self.position += 1
        return self._snapshot()

    def finish(self, shared):
        self._open()
        if self.state != "routed" or self.position != len(self.order):
            raise ValueError("All selected expert contributions must precede the shared expert")
        shared = _bf16_output(shared, self.tokens)
        before = self._snapshot()
        self._upload("input", shared)
        count = self.tokens * HIDDEN
        self._launch("moe_add_shared", [self.work["sum"], self.work["input"]], [C.c_int(count)], count)
        after = self._snapshot()
        self._launch("moe_finish", [self.work["sum"], self.work["output"]], [C.c_int(count)], count)
        output = (self.device.download(self.work["output"], (self.tokens, HIDDEN), np.uint16)
                  if self.tokens else np.empty((0, HIDDEN), np.uint16))
        self.state = "finished"
        return {"before_shared_fp32": before, "after_shared_fp32": after, "output_bf16": output}

    def run(self, indices, contributions, shared):
        selected = participation(indices, self.max_tokens)
        if set(contributions) != set(selected) or any(type(k) is not int for k in contributions):
            raise ValueError("Supply exactly the selected expert contributions")
        # Validate the complete call before changing device state.
        shared = _bf16_output(shared, np.asarray(indices).shape[0])
        for expert, rows in selected.items():
            _bf16_output(contributions[expert], len(rows))
        self.begin(indices)
        stages = {f"after_expert_{expert}_fp32": self.add(expert, contributions[expert]) for expert in self.order}
        stages.update(self.finish(shared))
        return stages

    def close(self):
        try:
            if self.device is not None:
                self.device.close()
        finally:
            self.device, self.state = None, "closed"
            if self.budget.reserved:
                self.budget.release(self.budget.reserved)

    def __exit__(self, *_):
        self.close()
