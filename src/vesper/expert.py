"""One resident native expert, one input token; a diagnostic runtime, not a model."""

from __future__ import annotations

import ctypes as C
import hashlib

import numpy as np

from .checkpoint import EXPERT_SHAPES, validate_expert_arrays
from .cuda import Device, Driver, compile_probe
from .formats import bf16_bits, bf16_values
from .resources import MiB
from .timing import Timings


class ExpertExecutor:
    """Qualified native arithmetic and reusable workspace, independent of residency."""

    def __init__(self, ordinal=0, swiglu_limit=10.0, device_cap=64 * MiB):
        if not np.isfinite(swiglu_limit) or swiglu_limit < 0:
            raise ValueError("Invalid SwiGLU limit")
        self.ordinal, self.limit, self.device_cap = ordinal, float(swiglu_limit), device_cap
        self.device = None
        self.timings = Timings()

    def __enter__(self):
        self.driver = Driver()
        self.info = next(d for d in self.driver.inventory()["devices"] if d["ordinal"] == self.ordinal)
        if tuple(map(int, self.info["compute_capability"].split("."))) < (8, 9):
            raise ValueError("Resident native expert requires FP8 MMA; the Ampere candidate is unqualified")
        self.device = Device(self.driver, self.ordinal, cap=self.device_cap)
        try:
            cubin = compile_probe(self.info["compute_capability"].replace(".", ""), expert=True)
            self.cubin_sha256 = hashlib.sha256(cubin).hexdigest()
            self.module = self.device.load(cubin)
            sizes = {"x": 5120 * 4, "q": 5120, "s": 160, "w1": 2304 * 2, "w3": 2304 * 2,
                     "down_input": 2304 * 4, "trace": 2304 * 5 * 4,
                     "down_q": 2304, "down_s": 72, "w2": 5120 * 2}
            self.work = {name: self.device.allocate(size) for name, size in sizes.items()}
            return self
        except Exception:
            self.close()
            raise

    def _launch(self, name, pointers, trailing, items):
        args = [C.c_uint64(p) for p in pointers] + trailing
        with self.timings.section("compute.kernels"):
            self.device.launch(self.module, name, args, items)

    def execute(self, x: np.ndarray, weights: dict[str, int], routing_weight: float | None = None) -> dict[str, np.ndarray]:
        if self.device is None:
            raise RuntimeError("Use the expert executor as a context manager")
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.shape != (5120,) or not np.isfinite(x).all():
            raise ValueError("This expert probe accepts one finite 5,120-element input")
        if routing_weight is not None and not np.isfinite(routing_weight):
            raise ValueError("Routing weight must be finite")
        x = np.ascontiguousarray(bf16_values(bf16_bits(x)))
        if not np.isfinite(x).all():
            raise ValueError("Input overflows this probe's finite BF16 scope")
        with self.timings.section("transfer.input_h2d"):
            self.device.current()
            self.driver.check(self.driver.cuMemcpyHtoD_v2(self.work["x"], x.ctypes.data, x.nbytes))
        p = self.work
        self._launch("activation32", [p["x"], p["q"], p["s"]], [C.c_int(160)], 160)
        for projection in ("w1", "w3"):
            self._projection(projection, p["q"], p["s"], p[projection], weights)
        self._launch("expert_swiglu", [p["w1"], p["w3"], p["down_input"], p["trace"]],
                     [C.c_int(2304), C.c_float(self.limit), C.c_float(1.0 if routing_weight is None else routing_weight),
                      C.c_int(routing_weight is not None)], 2304)
        self._launch("activation32", [p["down_input"], p["down_q"], p["down_s"]], [C.c_int(72)], 72)
        self._projection("w2", p["down_q"], p["down_s"], p["w2"], weights)
        layout = {"q": ((5120,), np.uint8), "s": ((160,), np.uint8),
                  "w1": ((2304,), np.uint16), "w3": ((2304,), np.uint16),
                  "down_input": ((2304,), np.float32), "trace": ((2304, 5), np.float32),
                  "down_q": ((2304,), np.uint8), "down_s": ((72,), np.uint8), "w2": ((5120,), np.uint16)}
        with self.timings.section("transfer.diagnostics_d2h"):
            return {name: self.device.download(p[name], shape, dtype) for name, (shape, dtype) in layout.items()}

    def _projection(self, name, q, scales, output, weights):
        rows, width = EXPERT_SHAPES[name]
        self._launch("gemv_fp4_mma", [q, scales, weights[f"{name}_weight"], weights[f"{name}_scale"], output],
                     [C.c_int(rows), C.c_int(width)], rows * 4)

    def close(self):
        if self.device is not None:
            self.device.close()
            self.device = None

    def __exit__(self, *_):
        self.close()


class ResidentExpert(ExpertExecutor):
    """Convenience wrapper preserving the original one-resident-expert API."""

    def __init__(self, arrays: dict[str, np.ndarray], ordinal=0, swiglu_limit=10.0):
        validate_expert_arrays(arrays)
        super().__init__(ordinal, swiglu_limit)
        self.arrays = arrays
        self.weights = {}

    def __enter__(self):
        super().__enter__()
        try:
            self.weights = {name: self.device.upload(array) for name, array in self.arrays.items()}
            return self
        except Exception:
            self.close()
            raise

    def forward(self, x: np.ndarray, routing_weight: float | None = None) -> dict[str, np.ndarray]:
        return self.execute(x, self.weights, routing_weight)
