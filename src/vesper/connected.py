"""Original single-token gate -> audited byte cache -> exact MoE accumulation."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
import hashlib
from pathlib import Path
import threading
import time

import numpy as np

from .bandwidth import EXPERT_BYTES
from .formats import bf16_bits, bf16_values
from .gate import GateProjection, load_gate_fixture
from .moe import MoEAccumulator, participation
from .resources import MemoryBudget, MiB, physical_memory
from .router import RouterProbe, load_router_fixture
from .shared import ResidentSharedExpert, load_shared_fixture
from .storage import expert_key
from .tiered import CacheConfig, TieredExpertRuntime


def _cache_config():
    return CacheConfig(host_cap_bytes=128 * MiB, ram_cache_bytes=2 * EXPERT_BYTES,
                       primary_slots=2, helper_slots=2, integrity_mode="gpu-sha256", max_store_experts=48)


@dataclass(frozen=True)
class ConnectedConfig:
    cache: CacheConfig = field(default_factory=_cache_config)
    host_cap_bytes: int = 320 * MiB
    primary_explicit_cap_bytes: int = 128 * MiB

    def __post_init__(self):
        # 96 MiB: original fixtures, upload preparation, per-call diagnostics.
        # Gate: 32 MiB; accumulator: 4 MiB; cache: its entire independent cap.
        if self.host_cap_bytes < self.cache.host_cap_bytes + 132 * MiB:
            raise ValueError("Connected host cap must cover all component reservation allowances")
        if self.primary_explicit_cap_bytes <= 0:
            raise ValueError("Invalid connected GPU allocation cap")


def bf16_activation(activation):
    x = np.asarray(activation)
    if x.shape != (5120,) or x.dtype not in (np.dtype(np.uint16), np.dtype(np.float32)):
        raise ValueError("Connected MoE accepts 5,120 BF16 storage bits or FP32 activation values")
    bits = np.ascontiguousarray(x if x.dtype == np.uint16 else bf16_bits(x), dtype=np.uint16)
    values = bf16_values(bits)
    if not np.isfinite(values).all():
        raise ValueError("Connected activation leaves the finite BF16 scope")
    return np.ascontiguousarray(values), bits


class ConnectedMoE:
    """Serial registered-layer BF16 calls. Missing experts fail before expert I/O.

    Gate decisions depend only on the original input, weights and trained bias.
    Failed attempts return no output; a retry starts a new zeroed accumulator.
    No checkpoint/network fetch is permitted in forward().
    """

    def __init__(self, store: Path, gate_fixture: Path, router_fixture: Path, shared_fixture: Path,
                 config: ConnectedConfig | None = None, *, cublas_library=None, layer=0):
        if type(layer) is not int or layer not in (0, 2):
            raise ValueError("Unregistered connected MoE layer")
        self.layer = layer
        self.store_path, self.gate_fixture = Path(store), Path(gate_fixture)
        self.router_fixture, self.shared_fixture = Path(router_fixture), Path(shared_fixture)
        self.config, self.cublas_library = config or ConnectedConfig(), cublas_library
        self.stack, self.active = None, False
        self.lock = threading.RLock()
        self.stats = {"calls_started": 0, "calls_completed": 0, "calls_failed": 0}
        self.last_attempt = None

    def __enter__(self):
        if self.stack is not None:
            raise RuntimeError("Connected MoE is already open")
        self.stack = ExitStack()
        c = self.config
        self.budget = MemoryBudget(c.host_cap_bytes, c.cache.available_floor)
        try:
            allowance = c.cache.host_cap_bytes + 100 * MiB  # own 96 + accumulator 4
            self.budget.acquire(allowance)
            self.stack.callback(self.budget.release, allowance)
            weight, self.gate_manifest = load_gate_fixture(self.gate_fixture)
            biases, self.router_manifest = load_router_fixture(self.router_fixture)
            shared, self.shared_manifest = load_shared_fixture(self.shared_fixture)
            if any(m["layer"] != self.layer for m in (self.gate_manifest, self.router_manifest, self.shared_manifest)):
                raise ValueError("Connected gate, router and shared artifacts must match the selected layer")
            ordinal = c.cache.primary_device
            self.gate = self.stack.enter_context(GateProjection(weight, ordinal, self.cublas_library, host_budget=self.budget))
            self.router = self.stack.enter_context(RouterProbe(biases, ordinal))
            self.shared = self.stack.enter_context(ResidentSharedExpert(shared, ordinal))
            self.accumulator = self.stack.enter_context(MoEAccumulator(ordinal, max_tokens=1))
            self.experts = self.stack.enter_context(TieredExpertRuntime(self.store_path, c.cache))
            if any(entry["provenance"]["layer"] != self.layer for entry in self.experts.store.entries.values()):
                raise ValueError("Connected store contains artifacts from another layer")
            self.devices = {"gate": self.gate.device, "router": self.router.device,
                            "shared": self.shared.device, "accumulator": self.accumulator.device,
                            "cache_primary": self.experts.primary_device_object}
            self.helper_device = self.experts.helper_device_object
            self.minimum_available = physical_memory()["available"]
            self._headroom()
            return self
        except BaseException:
            self.stack.close()
            self.stack = None
            raise

    def _headroom(self):
        available = physical_memory()["available"]
        self.minimum_available = min(self.minimum_available, available)
        self.budget.acquire(0, available)
        total = sum(sum(device.allocations.values()) for device in self.devices.values())
        if total > self.config.primary_explicit_cap_bytes:
            raise MemoryError("Combined native GPU buffers exceed the connected explicit cap")
        for device in (*self.devices.values(), self.helper_device):
            if device is not None and device.memory()["free"] < device.free_floor:
                raise MemoryError("Connected runtime refuses work below the GPU reserve")

    def forward(self, activation, image_span=False):
        if self.stack is None:
            raise RuntimeError("Use connected MoE as a context manager")
        if type(image_span) is not bool:
            raise ValueError("Connected image-span flag must be boolean")
        x, bits = bf16_activation(activation)
        with self.lock:
            if self.active:
                raise RuntimeError("Nested MoE calls would overwrite live workspaces")
            before = self.experts.snapshot()
            self.active = True
            self.stats["calls_started"] += 1
            attempt = {"status": "running", "input_bf16_sha256": hashlib.sha256(bits.tobytes()).hexdigest(),
                       "image_span": image_span, "completed_expert_ids": [], "stage": "headroom"}
            self.last_attempt = attempt
            started = time.perf_counter()
            try:
                self.experts.cache.enforce_pressure()
                self._headroom()
                attempt["stage"] = "gate"
                logits = self.gate.project(x)
                routing = self.router.inspect(logits, image_span)
                weights, indices = routing["weights"], routing["indices"]
                selected = participation(indices.reshape(1, 6), max_tokens=1)
                attempt.update(indices=indices.tolist(), weight_bits=weights.view(np.uint32).tolist())
                missing = [expert for expert in selected if expert_key(self.layer, expert) not in self.experts.store.entries]
                if missing:
                    raise KeyError(f"Selected original experts absent from this bounded store: {missing}")
                self.accumulator.begin(indices.reshape(1, 6))
                outputs, stages, events = {}, {}, []
                for expert in selected:
                    attempt["stage"] = f"expert:{expert}"
                    rank = int(np.flatnonzero(indices == expert)[0])
                    result = self.experts.forward(expert_key(self.layer, expert), x, float(weights[rank]))
                    outputs[expert] = result
                    events.append(dict(self.experts.last_event))
                    stages[f"after_expert_{expert}_fp32"] = self.accumulator.add(expert, result["w2"].reshape(1, 5120))
                    attempt["completed_expert_ids"].append(expert)
                attempt["stage"] = "shared"
                shared = self.shared.forward(x)
                attempt["stage"] = "final_accumulation"
                stages.update(self.accumulator.finish(shared["w2"].reshape(1, 5120)))
                self.experts.cache.enforce_pressure()
                self._headroom()
                attempt.update(status="pass", stage="complete")
                self.stats["calls_completed"] += 1
                return {"input_bf16": bits, "logits": logits, "routing": routing, "experts": outputs,
                        "expert_events": events, "shared": shared, "aggregation": stages,
                        "output_bf16": stages["output_bf16"].reshape(5120)}
            except BaseException as error:
                self.stats["calls_failed"] += 1
                attempt.update(status="failed", error_type=type(error).__name__)
                raise
            finally:
                try:
                    after = self.experts.snapshot()
                    attempt.update(wall_ms=(time.perf_counter() - started) * 1000,
                                   store_direct_read_calls=after["store_direct_read_calls"] - before["store_direct_read_calls"],
                                   store_direct_read_bytes=after["store_direct_read_bytes"] - before["store_direct_read_bytes"],
                                   cache_counter_delta={n: after["counters"][n] - v for n, v in before["counters"].items()})
                finally:
                    self.active = False

    def snapshot(self):
        if self.stack is None:
            raise RuntimeError("Connected MoE is closed")
        return {"calls": dict(self.stats), "cache": self.experts.snapshot(),
                "combined_host_reservation_peak_bytes": self.budget.peak,
                "combined_host_cap_bytes": self.budget.cap,
                "minimum_observed_host_available_bytes": self.minimum_available,
                "host_available_floor_bytes": self.budget.available_floor,
                "component_peak_explicit_device_bytes": {name: device.peak for name, device in self.devices.items()},
                "primary_explicit_cap_bytes": self.config.primary_explicit_cap_bytes,
                "cache_config": asdict(self.config.cache), "cublas_version": self.gate.blas.version,
                "compiled_cubin_sha256": {"router": self.router.cubin_sha256, "expert": self.experts.executor.cubin_sha256,
                                          "shared": self.shared.fp8_cubin_sha256, "aggregation": self.accumulator.cubin_sha256}}

    def __exit__(self, *_):
        with self.lock:
            if self.active:
                raise RuntimeError("Cannot close connected MoE during a live call")
            if self.stack is not None:
                self.stack.close()
                self.stack = None
            self.cleanup = {"parent_reservation_bytes": self.budget.reserved,
                            "accumulator_reservation_bytes": self.accumulator.budget.reserved,
                            "cache_reservation_bytes": self.experts.budget.reserved,
                            "primary_explicit_live_bytes": sum(sum(d.allocations.values()) for d in self.devices.values()),
                            "helper_explicit_live_bytes": 0 if self.helper_device is None else sum(self.helper_device.allocations.values())}
