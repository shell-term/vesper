from contextlib import ExitStack
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from vesper.block_fixture import block_layout, LAYOUT
from vesper.block_pair import NativeBlockPair, PRIMARY_CAP, HELPER_CAP
from vesper.block_pair_validation import expected_metadata, validate_goldens
from vesper.attention_topology import role, validate_block_pair
from vesper.hc import NativeHC
from vesper.resources import GiB
from vesper.shared_attention import SharedAttentionFrame
from vesper.topology_validation import state_digest


class Projection:
    opened, ordinal = True, 0
    def __init__(self, layer): self.layer = layer
    def fp8(self, name, x):
        width = {"wq_a": 1280, "wq_b": 32768, "wkv": 512, "wo_b": 5120}[name]
        return {"output_bf16": np.full((1, 1, width), self.layer, np.uint16)}
    def grouped(self, x): return {"output_bf16": np.zeros((1, 1, 8, 1024), np.uint16)}
    def round_window(self, x): return {"rounded_bf16": x.copy()}


class Norm:
    opened, ordinal = True, 0
    def norm(self, x, weight): return {"output_bf16": x.copy()}
    def rope(self, x, *a, **k): return {"output_bf16": x.copy()}


class Compression:
    opened, ordinal = True, 0
    def __init__(self, layer=2):
        self.layers, self.norm = {layer: role(layer).ratio}, Norm()
        self.arrays = {f"layers.{layer}.attn.compressor.norm.weight": np.zeros(512, np.uint16)}
    def project(self, x, layer):
        if self.layers[layer] == 1: return {"wkv_bf16": np.full((1, 1, 512), layer, np.uint16)}
        return {"wkv_fp32": np.ones((1, 1, 512), np.float32), "wgate_fp32": np.zeros((1, 1, 512), np.float32)}
    def round_kv(self, x): return {"rounded_bf16": x.copy()}


class Indexer:
    opened, ordinal, layers = True, 0, (20,)
    def __init__(self): self.query_layers = []
    def key(self, latent, layer, positions): return {"quant_rounded_bf16": latent[..., :128].copy()}
    def queries(self, x, qr, layer, pos):
        self.query_layers.append(layer)
        return {"quant_rounded_bf16": np.zeros((1, 1, 32, 128), np.uint16), "weights_scaled_bf16": np.zeros((1, 1, 32), np.uint16)}
    def score(self, q, keys, weights): return {"score_bf16": np.ones((1, 1, keys.shape[1]), np.uint16)}
    def select(self, score, lengths, offset, **kwargs):
        return {"indices_i32": np.full((1, 1, 1), offset, np.int32), "candidates_bool": np.ones((1, 1, 1), np.bool_)}


class Attention:
    opened, ordinal, profile = True, 0, "long_context"
    def __call__(self, *a, **k): return {"output_bf16": np.zeros((1, 1, 64, 512), np.uint16)}


def frame(layers=(2, 3)):
    arrays = {f"layers.{l}.attn.{name}.weight": np.zeros(width, np.uint16) for l in layers for name, width in (("q_norm", 1280), ("kv_norm", 512))}
    arrays.update({f"layers.{l}.attn.attn_sink": np.zeros(64, np.float32) for l in layers})
    index = Indexer() if layers == (20, 21) else SimpleNamespace(opened=True, ordinal=0, layers=(2,))
    return SharedAttentionFrame(arrays, {l: Projection(l) for l in layers}, Norm(), Compression(layers[0]), index, Attention(), layers)


class Device:
    free_floor = 0
    selected_ordinal = None
    def __init__(self, ordinal=0): self.allocations, self.calls, self.ordinal = {1: 4096}, 0, ordinal
    def allocate(self, size):
        self.calls += 1; ptr = max(self.allocations) + 1; self.allocations[ptr] = size
        return ptr
    def memory(self):
        Device.selected_ordinal = self.ordinal
        return {"free": 8 * GiB}


def ledger_pair(layers=(2, 3)):
    def op(layer=2): return SimpleNamespace(opened=True, ordinal=0, layer=layer, device=Device())
    hc, ps = {l: op(l) for l in layers}, {l: op(l) for l in layers}
    norm, comp, index, attn = op(), op(), op(), op()
    comp.layers, index.layers, attn.profile = {layers[0]: role(layers[0]).ratio}, (layers[0],), "long_context"
    comp.norm, index.norm = op(), op()
    moe = {l: SimpleNamespace(layer=l, stack=ExitStack(), config=SimpleNamespace(cache=SimpleNamespace(primary_device=0)),
        devices={"moe": Device()}, helper_device=Device(1)) for l in layers}
    return NativeBlockPair({}, hc, ps, norm, comp, index, attn, moe, layers=layers)


class BlockPairTests(unittest.TestCase):
    def test_ratio1_real_pipeline_handoff_and_candidate_snapshot_are_atomic(self):
        def prepare(l):
            if l == 21: self.assertEqual(carry[0][0, 0, 0], 20)
            return np.zeros((1, 1, 5120), np.uint16) if l == 20 else carry[0]
        def finish(l, stages):
            carry[:] = [stages["wo_b_output_bf16"]]
        carry = [None]
        with patch("vesper.resources.physical_memory", return_value={"available": 64 * GiB}), frame((20, 21)) as f:
            f.step_connected(prepare, finish)
            self.assertEqual(f.indexer.query_layers, [20])
            self.assertTrue(f.state["sharing"]["candidates"].all())
            self.assertEqual(f.state["sharing"]["candidate_owner"], 20)
            self.assertEqual(f.state["compressors"][20][0].shape, (1, 0, 512))
            saved = state_digest(f.inspect()); handle = f.snapshot()
            def fail(l, stages):
                finish(l, stages)
                if l == 21: raise OSError("consumer after source candidates")
            with self.assertRaises(OSError): f.step_connected(prepare, fail)
            self.assertEqual(state_digest(f.inspect()), saved)
            def late_failure(stages): raise MemoryError("publication after candidates")
            with self.assertRaises(MemoryError): f.step_connected(prepare, finish, finalize=late_failure)
            self.assertEqual(state_digest(f.inspect()), saved)
            f.step_connected(prepare, finish)
            f.rollback(handle)
            self.assertEqual(state_digest(f.inspect()), saved)
            copy = f.inspect(); copy["state"]["sharing"]["candidates"].fill(False)
            self.assertEqual(state_digest(f.inspect()), saved)

    def test_ratio1_operator_caps_and_original_layer_identity(self):
        pair = ledger_pair((20, 21))
        with patch("vesper.resources.physical_memory", return_value={"available": 64 * GiB}), pair:
            for device, cap in ((pair.comp.norm.device, PRIMARY_CAP), (pair.moe[21].helper_device, HELPER_CAP)):
                before = device.calls
                with self.assertRaises(MemoryError): device.allocate(cap)
                self.assertEqual(device.calls, before)
        self.assertEqual(pair.budget.reserved, 0)
        bad = ledger_pair((20, 21)); bad.hc[21].layer = 3
        with patch("vesper.resources.physical_memory", return_value={"available": 64 * GiB}):
            with self.assertRaises(ValueError): bad.__enter__()
        for l in (20, 21):
            specs = block_layout(l)
            arrays = {n: np.zeros(s, np.float32) for n, (d, s) in specs.items() if d == "F32"}
            self.assertEqual(NativeHC(arrays, layer=l).layer, l)
        for bad in ((20.0, 21), (2, 21), (21, 20), (24, 25), (20, 21, 24), [20, 21]):
            with self.assertRaises(ValueError): validate_block_pair(bad)

    def test_real_pipeline_requests_consumer_only_after_source_callback(self):
        events, carry = [], [None]
        def prepare(l):
            events.append(("prepare", l))
            if l == 3:
                self.assertEqual(int(carry[0][0, 0, 0]), 2)
                return carry[0]
            return np.zeros((1, 1, 5120), np.uint16)
        def finish(l, stages):
            events.append(("finish", l))
            carry[0] = stages["wo_b_output_bf16"]
        with patch("vesper.resources.physical_memory", return_value={"available": 64 * GiB}), frame() as f:
            f.step_connected(prepare, finish)
            self.assertEqual(events, [("prepare", 2), ("finish", 2), ("prepare", 3), ("finish", 3)])
            self.assertEqual(f.position, 1)
            self.assertEqual(f.state["sharing"]["index_owner"], 2)
            self.assertEqual(set(f.state["compressors"]), {2})
        two = frame(); two.batch = 2
        with patch("vesper.resources.physical_memory", return_value={"available": 64 * GiB}), two:
            with self.assertRaises(ValueError): two.step_connected(prepare, finish)

    def test_actual_source_candidate_is_not_published_after_consumer_failure(self):
        def prepare(l): return np.zeros((1, 1, 5120), np.uint16)
        def fail(l, stages):
            if l == 3: raise OSError("consumer failure after source partial/cache work")
        with patch("vesper.resources.physical_memory", return_value={"available": 64 * GiB}), frame() as f:
            before = state_digest(f.inspect())
            with self.assertRaises(OSError): f.step_connected(prepare, fail)
            self.assertEqual(state_digest(f.inspect()), before)
            f.step_connected(prepare, lambda *a: None)
            self.assertEqual(f.position, 1)
            self.assertTrue(np.all(f.state["compressors"][2][0][:, 0] == 1))

    def test_late_reserve_refusal_leaves_both_windows_and_partial_state_unchanged(self):
        with patch("vesper.resources.physical_memory", return_value={"available": 64 * GiB}), frame() as f:
            before, acquire = state_digest(f.inspect()), f.budget.acquire
            def refuse(*a, **k): raise MemoryError("publication")
            def after(l, stages):
                if l == 3: f.budget.acquire = refuse
            try:
                with self.assertRaises(MemoryError): f.step_connected(lambda l: np.zeros((1, 1, 5120), np.uint16), after)
            finally: f.budget.acquire = acquire
            self.assertEqual(state_digest(f.inspect()), before)

    def test_pair_primary_and_helper_caps_guard_all_nested_ledgers(self):
        p = ledger_pair()
        originals = p.index.norm.device.allocate, p.moe[3].helper_device.allocate
        with patch("vesper.resources.physical_memory", return_value={"available": 64 * GiB}), p:
            self.assertEqual(len(p.devices), 12)
            self.assertEqual(Device.selected_ordinal, 0)
            p._headroom()
            self.assertEqual(Device.selected_ordinal, 0)
            for d, limit in ((p.index.norm.device, PRIMARY_CAP), (p.moe[3].helper_device, HELPER_CAP)):
                calls = d.calls
                with self.assertRaises(MemoryError): d.allocate(limit)
                self.assertEqual(d.calls, calls)
        self.assertEqual((p.index.norm.device.allocate, p.moe[3].helper_device.allocate), originals)
        self.assertEqual(p.budget.reserved, 0)
        self.assertEqual(p.attention.budget.reserved, 0)

    def test_wrong_layer_or_aliased_device_ownership_refuses_pair_admission(self):
        for mutate in (lambda p: setattr(p.hc[3], "layer", 2), lambda p: setattr(p.index.norm, "device", p.comp.norm.device),
                       lambda p: setattr(p.moe[3], "layer", 2)):
            p = ledger_pair(); mutate(p)
            with patch("vesper.resources.physical_memory", return_value={"available": 64 * GiB}):
                with self.assertRaises(ValueError): p.__enter__()
            self.assertEqual(p.budget.reserved, 0)
            self.assertEqual(p.attention.budget.reserved, 0)

    def test_consumer_HC_uses_its_own_original_layer_and_fixture_layout(self):
        specs = block_layout(3)
        self.assertEqual(len(specs), 8)
        self.assertEqual(block_layout(2), LAYOUT)
        arrays = {n: np.zeros(s, np.float32) for n, (d, s) in specs.items() if d == "F32"}
        self.assertEqual(NativeHC(arrays, layer=3).layer, 3)
        with self.assertRaises(ValueError): NativeHC(arrays, layer=2)
        for bad in (True, 3.0, 4):
            with self.assertRaises(ValueError): block_layout(bad)

    def test_complete_published_pair_goldens_and_provenance(self):
        from vesper.block_pair_fixture import validate_plan
        root = Path(__file__).resolve().parents[1]
        report = json.loads((root / "results/2026-10-19-block-pair/pair.json").read_bytes())
        self.assertEqual(len(validate_goldens(report)), 36)
        self.assertEqual(len(list(expected_metadata())), 36)
        validate_plan(report["plan"])
        for change in (lambda r: r["pair_cases"].pop(), lambda r: r["pair_cases"][0]["boundaries"].pop(),
                lambda r: r["pair_cases"][1]["boundaries"][0].__setitem__(3, "0" * 64),
                lambda r: r["pair_cases"][0]["decisions"]["3"]["indices"].reverse(),
                lambda r: r["pair_cases"][0]["handoff"].update(pre_sha256="0" * 64),
                lambda r: r["pair_cases"][0].update(consumer_reused_source_KV=False),
                lambda r: r["pair_cases"][1].update(index_query_layers=[2, 3]),
                lambda r: r["unchanged_Block_forward_controls"][0].pop(),
                lambda r: r["unchanged_Attention_forward_controls"][0][0].update(mismatches=1),
                lambda r: r["faults"][3].update(source_block_completed_before_failure=False),
                lambda r: r["snapshot_events"][0].update(both_states_restored=False),
                lambda r: r["memory"].update(minimum_sampled_host_available_bytes=16 * GiB),
                lambda r: r["memory"]["pair_active_stats"].update(peak_owned_helper_bytes=HELPER_CAP + 1),
                lambda r: r["memory"]["moe_cleanup"]["3"].update(primary_explicit_live_bytes=1),
                lambda r: r["native_profiles"]["projections"]["3"].update(converted_wo_a_bf16_sha256="0" * 64),
                lambda r: r["negative_controls"].pop()):
            bad = copy.deepcopy(report); change(bad)
            with self.assertRaises(ValueError): validate_goldens(bad)

    def test_complete_ratio1_goldens_include_initial_query_and_candidate_state(self):
        root = Path(__file__).resolve().parents[1]
        report = json.loads((root / "results/2026-10-19-ratio1-block-pair/pair.json").read_bytes())
        self.assertEqual(len(validate_goldens(report)), 36)
        self.assertEqual([r["meta"] for r in report["pair_cases"]], list(expected_metadata((20, 21))))
        for row in report["pair_cases"]:
            self.assertEqual(row["index_query_layers"], [20])
            names = {report["boundary_names"][b[0]] for b in row["boundaries"]}
            self.assertIn("state/sharing/candidates/", names)
            self.assertIn("attention/layer20_index_selection_candidates_bool/", names)
        for change in (lambda r: r["pair_cases"].pop(),
                lambda r: r["pair_cases"][0].update(index_query_layers=[]),
                lambda r: r["pair_cases"][1].update(index_query_layers=[20, 21]),
                lambda r: r["pair_cases"][0]["boundaries"].pop(),
                lambda r: r["pair_cases"][0]["handoff"].update(pre_sha256="0" * 64),
                lambda r: r["plan"]["consumer_gate"].update(layer=3),
                lambda r: r["plan"]["fixtures"]["attention20"].update(layer=2),
                lambda r: r["whole_pair_state_controls"][0].update(actual_sha256="0" * 64),
                lambda r: r["unchanged_Attention_forward_controls"][0].pop(),
                lambda r: r["faults"][2].update(both_block_states_and_position_unchanged=False),
                lambda r: r["snapshot_events"][0].update(both_states_restored=False),
                lambda r: r["negative_controls"].pop(),
                lambda r: r["memory"]["operator_cleanup"]["projection_21"].update(owned_context_active=True),
                lambda r: r["memory"].update(minimum_sampled_host_available_bytes=16 * GiB),
                lambda r: r["memory"]["pair_active_stats"].update(peak_owned_primary_bytes=PRIMARY_CAP + 1)):
            bad = copy.deepcopy(report); change(bad)
            with self.assertRaises(ValueError): validate_goldens(bad)

    def test_ratio1_source_and_consumer_plans_refuse_cross_layer_imports(self):
        from vesper.block_pair_fixture import prepare_consumer_store
        from vesper.compressed_block_fixture import validate_plan as validate_source_plan
        root = Path(__file__).resolve().parents[1] / "results/2026-10-19-ratio1-block-pair"
        source = json.loads((root / "source-expert-plan.json").read_bytes())
        pair = json.loads((root / "pair.json").read_bytes())["plan"]
        validate_source_plan(source)
        self.assertEqual(source["expert_ids"], pair["expert_ids"]["20"])
        for change in (lambda p: p.update(layer=2), lambda p: p["expert_ids"].pop(),
                       lambda p: p["gate_fixture"].update(layer=21), lambda p: p["fixtures"]["hc_norm"].update(layer=2)):
            bad = copy.deepcopy(source); change(bad)
            with self.assertRaises(ValueError): validate_source_plan(bad)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.json"
            for change in (lambda p: p["consumer_gate"].update(layer=3),
                    lambda p: p["fixtures"]["attention20"].update(layer=2),
                    lambda p: p["expert_ids"]["21"].pop()):
                bad = copy.deepcopy(pair); change(bad)
                path.write_bytes(json.dumps(bad).encode())
                with patch("vesper.block_pair_fixture.fetch_expert_fixture", side_effect=AssertionError("No fetching allowed")):
                    with self.assertRaises(ValueError): prepare_consumer_store(path, Path(tmp) / "experts", Path(tmp) / "store")


if __name__ == "__main__": unittest.main()
