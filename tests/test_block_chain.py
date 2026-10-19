from contextlib import ExitStack
import copy
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from vesper.attention_topology import REFRESH_CHAIN, validate_block_chain
from vesper.block_chain import NativeBlockChain
from vesper.block_chain_fixture import prepare_store, PREPARATION_HOST_MIB
from vesper.resources import GiB, MiB
from vesper.shared_attention import SharedAttentionFrame
from vesper.topology_validation import state_digest
from test_block_pair import Projection, Norm, Compression, Attention, Device, Indexer


class RefreshIndexer(Indexer):
    layers = (20, 24)
    def select(self, score, lengths, offset, **kwargs):
        column = int(self.query_layers[-1] == 24 and score.shape[-1] > 1)
        result = {"indices_i32": np.full((1, 1, 1), offset + column, np.int32)}
        if kwargs["candidate_source"]: result["candidates_bool"] = np.ones((1, 1, 1), np.bool_)
        return result


def chain_frame():
    arrays = {f"layers.{l}.attn.{name}.weight": np.zeros(width, np.uint16) for l in REFRESH_CHAIN for name, width in (("q_norm", 1280), ("kv_norm", 512))}
    arrays.update({f"layers.{l}.attn.attn_sink": np.zeros(64, np.float32) for l in REFRESH_CHAIN})
    return SharedAttentionFrame(arrays, {l: Projection(l) for l in REFRESH_CHAIN}, Norm(), Compression(20), RefreshIndexer(), Attention(), REFRESH_CHAIN)


def ledger_chain():
    def op(layer=20): return SimpleNamespace(opened=True, ordinal=0, layer=layer, device=Device())
    hc, ps = {l: op(l) for l in REFRESH_CHAIN}, {l: op(l) for l in REFRESH_CHAIN}
    norm, comp, index, sparse = [op() for _ in range(4)]
    comp.layers, index.layers, sparse.profile = {20: 1}, (20, 24), "long_context"
    comp.norm, index.norm = op(), op()
    moe = {l: SimpleNamespace(layer=l, stack=ExitStack(), config=SimpleNamespace(cache=SimpleNamespace(primary_device=0)),
        devices={"moe": Device()}, helper_device=Device(1)) for l in REFRESH_CHAIN}
    return NativeBlockChain({}, hc, ps, norm, comp, index, sparse, moe)


class BlockChainTests(unittest.TestCase):
    def test_actual_pipeline_executes_all_intervening_callbacks_and_refreshes_at24(self):
        carry, order = [None], []
        def prepare(l):
            order.append(("prepare", l))
            if l == 20: return np.zeros((1, 1, 5120), np.uint16)
            self.assertEqual(int(carry[0][0, 0, 0]), l - 1)
            return carry[0]
        def finish(l, stages):
            order.append(("finish", l)); carry[:] = [stages["wo_b_output_bf16"]]
        with patch("vesper.resources.physical_memory", return_value={"available": 64 * GiB}), chain_frame() as frame:
            frame.step_connected(prepare, finish)
            self.assertEqual(order, [(stage, l) for l in REFRESH_CHAIN for stage in ("prepare", "finish")])
            result = frame.step_connected(prepare, finish)
            self.assertEqual(frame.indexer.query_layers, [20, 24, 20, 24])
            self.assertEqual(frame.state["sharing"]["kv_owner"], 20)
            self.assertEqual(frame.state["sharing"]["index_owner"], 24)
            self.assertTrue(np.array_equal(result["layer25_selected_indices_i32"], result["layer24_selected_indices_i32"]))
            self.assertFalse(np.array_equal(result["layer25_selected_indices_i32"], result["layer20_selected_indices_i32"]))

    def test_failure_after_refreshed_index_and_late_publication_preserves_all_six_states(self):
        x = lambda l: np.zeros((1, 1, 5120), np.uint16)
        with patch("vesper.resources.physical_memory", return_value={"available": 64 * GiB}), chain_frame() as frame:
            frame.step_connected(x, lambda *a: None)
            before = state_digest(frame.inspect()); handle = frame.snapshot()
            def fail(l, stages):
                if l == 24: raise OSError("after refreshed index/Attention, before last consumer")
            with self.assertRaises(OSError): frame.step_connected(x, fail)
            self.assertEqual(state_digest(frame.inspect()), before)
            def late(stages): raise MemoryError("publication after six complete callbacks")
            with self.assertRaises(MemoryError): frame.step_connected(x, lambda *a: None, finalize=late)
            self.assertEqual(state_digest(frame.inspect()), before)
            frame.step_connected(x, lambda *a: None); frame.rollback(handle)
            self.assertEqual(state_digest(frame.inspect()), before)

    def test_chain_caps_guard_nested_devices_and_reject_missing_or_wrong_owners(self):
        chain = ledger_chain()
        original = chain.index.norm.device.allocate
        with patch("vesper.resources.physical_memory", return_value={"available": 64 * GiB}), chain:
            self.assertEqual(len(chain.devices), 24)
            self.assertEqual(Device.selected_ordinal, 0)
            for device, cap in ((chain.index.norm.device, chain.primary_cap), (chain.moe[25].helper_device, chain.helper_cap)):
                calls = device.calls
                with self.assertRaises(MemoryError): device.allocate(cap)
                self.assertEqual(device.calls, calls)
        self.assertEqual(chain.index.norm.device.allocate, original)
        self.assertEqual(chain.budget.reserved, 0)
        for change in (lambda c: setattr(c.index, "layers", (20,)), lambda c: setattr(c.hc[22], "layer", 21),
                       lambda c: setattr(c.moe[24], "layer", 20), lambda c: setattr(c.index.norm, "device", c.comp.norm.device)):
            bad = ledger_chain(); change(bad)
            with patch("vesper.resources.physical_memory", return_value={"available": 64 * GiB}):
                with self.assertRaises(ValueError): bad.__enter__()
            self.assertEqual(bad.budget.reserved, 0)
        for bad in ((20, 21, 24, 25), (20, 21, 22, 24, 25), (20, 21), (20.0, 21, 22, 23, 24, 25)):
            with self.assertRaises(ValueError): validate_block_chain(bad)

    def test_expert_import_has_at_most_two_bounded_workers_and_no_store_after_error(self):
        plan = {"target_layer": 22, "expert_ids": [0, 1, 2, 3], "max_experts": 4}
        lock, active, peak = threading.Lock(), [0], [0]
        def fetch(*a, **k):
            with lock:
                active[0] += 1; peak[0] = max(peak[0], active[0])
            time.sleep(0.02)
            with lock: active[0] -= 1
        with tempfile.TemporaryDirectory() as tmp, patch("vesper.block_chain_fixture.load_plan", return_value=plan), \
                patch("vesper.resources.physical_memory", return_value={"available": 64 * GiB}), \
                patch("vesper.block_chain_fixture.fetch_expert_fixture", side_effect=fetch), \
                patch("vesper.block_chain_fixture.build_store", return_value={}) as build:
            result = prepare_store("unused-plan", tmp)
            self.assertEqual(peak[0], 2)
            self.assertEqual(result["host_allowance_bytes"], PREPARATION_HOST_MIB * MiB)
            self.assertEqual(len(build.call_args.args[0]), 4)
            build.reset_mock()
            with patch("vesper.block_chain_fixture.fetch_expert_fixture", side_effect=OSError("range failure")):
                with self.assertRaises(OSError): prepare_store("unused-plan", tmp)
            build.assert_not_called()

    def test_published_chain_rejects_missing_handoffs_refresh_controls_and_partial_cleanup(self):
        from vesper.block_chain_validation import validate_goldens, expected_metadata
        root = Path(__file__).resolve().parents[1]
        report = json.loads((root / "results/2026-10-19-refresh-block-chain/chain.json").read_bytes())
        self.assertEqual(len(validate_goldens(report)), 40)
        self.assertEqual(len(list(expected_metadata())), 40)
        for change in (lambda r: r["chain_cases"].pop(),
                lambda r: r["chain_cases"][0]["handoffs"].pop("23"),
                lambda r: r["chain_cases"][0].update(index_query_layers=[20]),
                lambda r: r["chain_cases"][1]["reused_index_owners"].update({"25": 20}),
                lambda r: r["chain_cases"][0]["boundaries"].pop(),
                lambda r: r["unchanged_Block_forward_controls"][0].pop(),
                lambda r: r["unchanged_Attention_forward_controls"][0][0].update(mismatches=1),
                lambda r: r["whole_chain_state_controls"][0].update(actual_sha256="0" * 64),
                lambda r: r["faults"][4]["completed_experts_in_failed_attempt"].update({"24": 2}),
                lambda r: r["snapshot_events"][0].update(all_six_states_restored=False),
                lambda r: r["memory"].update(minimum_sampled_host_available_bytes=16 * GiB),
                lambda r: r["memory"]["moe_cleanup"]["23"].update(primary_explicit_live_bytes=1),
                lambda r: r["memory"]["chain_active_stats"].update(peak_owned_primary_bytes=2 * GiB + 1),
                lambda r: r["native_profiles"]["projections"]["24"].update(converted_wo_a_bf16_sha256="0" * 64),
                lambda r: r["negative_controls"].pop()):
            bad = copy.deepcopy(report); change(bad)
            with self.assertRaises(ValueError): validate_goldens(bad)

    def test_staged_expert_plan_refuses_missing_intervening_blocks_before_any_fetch(self):
        from vesper.block_chain_fixture import validate_plan
        root = Path(__file__).resolve().parents[1]
        report = json.loads((root / "results/2026-10-19-refresh-block-chain/chain.json").read_bytes())
        for plan in report["expert_plans"]: validate_plan(plan)
        plan = report["plan"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.json"
            for change in (lambda p: p["layers"].remove(22),
                    lambda p: p["cases"][0]["handoffs"].pop("23"),
                    lambda p: p["fixtures"]["attention24"].update(layer=20),
                    lambda p: p["gate"].update(layer=24),
                    lambda p: p["expert_ids"].pop()):
                bad = copy.deepcopy(plan); change(bad); path.write_bytes(json.dumps(bad).encode())
                with patch("vesper.block_chain_fixture.fetch_expert_fixture", side_effect=AssertionError("No expert I/O allowed")):
                    with self.assertRaises(ValueError): prepare_store(path, tmp)


if __name__ == "__main__": unittest.main()
