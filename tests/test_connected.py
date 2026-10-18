import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from vesper.bandwidth import EXPERT_BYTES
from vesper.checkpoint import MODEL, REVISION
from vesper.connected import ConnectedConfig, ConnectedMoE, bf16_activation
from vesper.connected_fixture import load_connected_plan
from vesper.formats import bf16_bits
from vesper.resources import MemoryBudget, MiB
from vesper.storage import NativeStore, build_store, tensor_layout
from vesper.tiered import CacheConfig


class ConnectedTests(unittest.TestCase):
    def runtime(self):
        runtime = ConnectedMoE(*[Path("unused")] * 4)
        runtime.stack = object()
        runtime._headroom = MagicMock()
        runtime.gate = SimpleNamespace(project=MagicMock(return_value=np.zeros(384, np.float32)))
        runtime.router = SimpleNamespace(inspect=MagicMock(return_value={
            "indices": np.array([5, 2, 3, 1, 0, 4], np.int32),
            "weights": np.array([.1, .2, .3, .4, .25, .25], np.float32)}))
        class Experts:
            def __init__(self):
                self.cache = SimpleNamespace(enforce_pressure=lambda: None)
                self.store = SimpleNamespace(entries={f"0:{e}": {} for e in range(6)})
                self.calls, self.fail_fourth = [], False
                self.last_event = {"source_tier": "fake"}
            def snapshot(self):
                return {"store_direct_read_calls": 0, "store_direct_read_bytes": 0, "counters": {"test": 0}}
            def forward(self, key, x, route):
                self.calls.append((key, route, x.copy()))
                if self.fail_fourth and len(self.calls) == 4:
                    raise OSError("Injected fourth expert failure")
                return {"w2": np.zeros(5120, np.uint16)}
        class Accumulator:
            begins = 0
            def begin(self, ids):
                self.begins += 1
                self.order = []
            def add(self, expert, output):
                self.order.append(expert)
                return np.zeros((1, 5120), np.float32)
            def finish(self, shared):
                if self.order != list(range(6)):
                    raise AssertionError("Incomplete/reordered expert set")
                return {"output_bf16": np.zeros((1, 5120), np.uint16)}
        runtime.experts, runtime.accumulator = Experts(), Accumulator()
        runtime.shared = SimpleNamespace(forward=MagicMock(return_value={"w2": np.zeros(5120, np.uint16)}))
        return runtime

    def test_gate_and_experts_receive_the_same_bf16_rounded_input(self):
        x = np.full(5120, 1.00390625, np.float32)
        x[1] = -0.0
        runtime = self.runtime()
        runtime.forward(x)
        promoted, bits = bf16_activation(x)
        self.assertTrue(np.array_equal(bits, bf16_bits(x)))
        self.assertTrue(np.array_equal(runtime.gate.project.call_args.args[0].view(np.uint32), promoted.view(np.uint32)))
        self.assertEqual(bits[1], 0x8000)
        self.assertEqual([key for key, _, _ in runtime.experts.calls], [f"0:{e}" for e in range(6)])
        ids = runtime.router.inspect.return_value["indices"]
        weights = runtime.router.inspect.return_value["weights"]
        for expert, (_, weight, actual_input) in enumerate(runtime.experts.calls):
            self.assertEqual(np.float32(weight).view(np.uint32), weights[np.flatnonzero(ids == expert)[0]].view(np.uint32))
            self.assertTrue(np.array_equal(actual_input.view(np.uint32), promoted.view(np.uint32)))

    def test_missing_selected_expert_refuses_before_any_expert_or_accumulator_call(self):
        runtime = self.runtime()
        del runtime.experts.store.entries["0:5"]
        with self.assertRaisesRegex(KeyError, "absent"):
            runtime.forward(np.zeros(5120, np.float32))
        self.assertFalse(runtime.experts.calls)
        self.assertEqual(runtime.accumulator.begins, 0)
        runtime.shared.forward.assert_not_called()
        self.assertFalse(runtime.active)

    def test_partial_failure_returns_no_output_and_retry_restarts_all_six_experts(self):
        runtime = self.runtime()
        runtime.experts.fail_fourth = True
        x = np.ones(5120, np.float32)
        with self.assertRaises(OSError):
            runtime.forward(x)
        failed = copy.deepcopy(runtime.last_attempt)
        self.assertEqual(failed["completed_expert_ids"], [0, 1, 2])
        runtime.shared.forward.assert_not_called()
        result = runtime.forward(x)
        self.assertEqual(result["routing"]["indices"].tolist(), failed["indices"])
        self.assertEqual(runtime.last_attempt["weight_bits"], failed["weight_bits"])
        self.assertEqual(runtime.accumulator.begins, 2)
        self.assertEqual([c[0] for c in runtime.experts.calls[4:]], [f"0:{e}" for e in range(6)])
        self.assertEqual(runtime.stats, {"calls_started": 2, "calls_completed": 1, "calls_failed": 1})
        runtime.shared.forward.assert_called_once()

    def test_invalid_input_mask_and_reentry_fail_before_device_or_cache_work(self):
        runtime = self.runtime()
        for x in (np.zeros(5119, np.float32), np.full(5120, np.inf, np.float32), np.full(5120, 0x7fc0, np.uint16),
                  np.zeros(5120, np.float64), np.zeros(5120, np.complex64)):
            with self.assertRaises(ValueError):
                runtime.forward(x)
        with self.assertRaises(ValueError):
            runtime.forward(np.zeros(5120, np.float32), 1)
        runtime.gate.project.assert_not_called()
        runtime.active = True
        with self.assertRaisesRegex(RuntimeError, "Nested"):
            runtime.forward(np.zeros(5120, np.float32))
        with self.assertRaisesRegex(RuntimeError, "already open"):
            runtime.__enter__()

    def test_failed_entry_releases_the_parent_allowance(self):
        runtime = ConnectedMoE(*[Path("unused")] * 4, ConnectedConfig(cache=CacheConfig(
            host_cap_bytes=96 * MiB, available_floor=0)))
        with patch("vesper.connected.load_gate_fixture", side_effect=ValueError("fixture failure")):
            with self.assertRaisesRegex(ValueError, "fixture failure"):
                runtime.__enter__()
        self.assertEqual(runtime.budget.reserved, 0)
        self.assertIsNone(runtime.stack)

    def test_parent_cap_covers_all_component_caps(self):
        with self.assertRaisesRegex(ValueError, "all component"):
            ConnectedConfig(host_cap_bytes=128 * MiB)

    def test_import_scope_is_the_unique_gate_selected_union(self):
        plan = {"probe": "connected_moe_plan", "status": "pass", "model": MODEL, "revision": REVISION, "layer": 0,
                "max_experts": 6, "expert_ids": list(range(6)), "selected_weight_bytes": 6 * EXPERT_BYTES,
                "cases": [{"image_span": False, "indices": list(range(6))}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps(plan))
            self.assertEqual(load_connected_plan(path)["expert_ids"], list(range(6)))
            for mutate in (lambda p: p.update(expert_ids=list(range(7))),
                           lambda p: p["cases"][0].update(indices=[0, 0, 2, 3, 4, 5]),
                           lambda p: p.update(selected_weight_bytes=EXPERT_BYTES),
                           lambda p: p.update(max_experts=49)):
                bad = copy.deepcopy(plan)
                mutate(bad)
                path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    load_connected_plan(path)

    def test_larger_store_requires_explicit_bounded_opt_in(self):
        payload = np.zeros(EXPERT_BYTES, np.uint8)
        layout, entries = tensor_layout(), []
        for expert in range(5):
            tensors = [{"name": name, "tensor": f"layers.0.ffn.experts.{expert}." + name.replace("_", "."),
                        "dtype": "I8" if name.endswith("weight") else "F8_E8M0", "shape": region["shape"],
                        "bytes": region["bytes"], "sha256": hashlib.sha256(memoryview(payload)[region["offset"]:region["offset"]+region["bytes"]]).hexdigest()}
                       for name, region in layout.items()]
            entries.append({"key": f"0:{expert}", "offset": expert * EXPERT_BYTES, "bytes": EXPERT_BYTES,
                            "sha256": hashlib.sha256(payload).hexdigest(),
                            "provenance": {"model": MODEL, "revision": REVISION, "kind": "complete_expert",
                                           "layer": 0, "expert": expert, "swiglu_limit": 10.0, "tensors": tensors}})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {"schema_version": 1, "kind": "vesper_native_expert_store", "model": MODEL, "revision": REVISION,
                        "expert_bytes": EXPERT_BYTES, "file": "experts.bin", "file_bytes": 5 * EXPERT_BYTES,
                        "layout": layout, "entries": entries}
            (root / "manifest.json").write_text(json.dumps(manifest))
            with (root / "experts.bin").open("wb") as stream:
                stream.truncate(5 * EXPERT_BYTES)
            with self.assertRaisesRegex(ValueError, "probe size"):
                NativeStore(root, MemoryBudget(cap=MiB, available_floor=0))
            store = NativeStore(root, MemoryBudget(cap=MiB, available_floor=0), max_experts=5)
            self.assertEqual(len(store.entries), 5)
            for limit in (0, 49, True):
                with self.assertRaises(ValueError):
                    build_store([], root, max_experts=limit)


if __name__ == "__main__":
    unittest.main()
