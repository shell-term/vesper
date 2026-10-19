from contextlib import ExitStack
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from vesper.block import NativeBlock
from vesper.block_fixture import LAYOUT, TENSOR_BYTES, load_block_fixture
from vesper.checkpoint import MODEL, REVISION, LIMIT
from vesper.compressed_block_fixture import validate_plan
from vesper.compressed_block_validation import block_inputs, expected_metadata, expected_names, validate_goldens
from vesper.connected import ConnectedMoE
from vesper.hc import NativeHC
from vesper.prefix import STORAGE_DTYPES
from vesper.resources import GiB, MiB
from vesper.shared_attention import validate_frame_layers


class Device:
    free_floor = 0
    def __init__(self): self.allocations = {1: 4096}; self.calls = 0
    def allocate(self, size):
        self.calls += 1
        ptr = max(self.allocations) + 1
        self.allocations[ptr] = size
        return ptr
    def memory(self): return {"free": 8 * GiB}


class Frame:
    def __init__(self, *a): self.position = 0
    def __enter__(self): return self
    def __exit__(self, *a): pass


def block_ledger():
    def op(): return SimpleNamespace(opened=True, ordinal=0, layer=2, device=Device())
    hc, norm, proj, sparse, comp, index = [op() for _ in range(6)]
    comp.norm, index.norm = op(), op()
    moe = SimpleNamespace(layer=2, stack=ExitStack(), devices={"moe": Device()},
        config=SimpleNamespace(cache=SimpleNamespace(primary_device=0)))
    with patch("vesper.block.SharedAttentionFrame", Frame):
        return NativeBlock({}, hc, norm, proj, sparse, moe, layer=2, compressor=comp, indexer=index)


class CompressedBlockTests(unittest.TestCase):
    def test_input_is_explicit_independent_HC_lanes_and_incoming_mixture(self):
        inputs = [block_inputs(i) for i in range(4)]
        for hidden, pre in inputs:
            self.assertEqual(hidden.dtype, np.uint16)
            self.assertFalse(np.array_equal(hidden[..., 0, :], hidden[..., 1, :]))
            self.assertEqual(pre.dtype, np.float32)
            self.assertTrue(np.all((pre > 0) & (pre < 1)))
        self.assertEqual(len(list(expected_metadata())), 32)
        self.assertNotEqual(inputs[0][0].tobytes(), inputs[1][0].tobytes())
        for value in (-1, 4, True):
            with self.assertRaises(ValueError): block_inputs(value)

    def test_source_only_frame_does_not_admit_arbitrary_groups_or_consumers(self):
        self.assertEqual(validate_frame_layers((2,)), (2,))
        for layers in ((3,), (20,), (2, 4), (2, 2), [2], (True,), (2.0,), (np.int32(2),)):
            with self.assertRaises(ValueError): validate_frame_layers(layers)

    def test_layer2_HC_requires_its_own_original_arrays_before_CUDA(self):
        arrays = {n: np.zeros(s, np.float32) for n, (d, s) in LAYOUT.items() if d == "F32"}
        with patch("vesper.hc.Driver", side_effect=AssertionError("No CUDA access expected")):
            self.assertEqual(NativeHC(arrays, layer=2).layer, 2)
            with self.assertRaises(ValueError): NativeHC(arrays)
            for layer in (True, 1, 3, 20):
                with self.assertRaises(ValueError): NativeHC(arrays, layer=layer)

    def test_block_owned_cap_covers_nested_compression_and_index_norm_devices(self):
        block = block_ledger()
        original = block.indexer.norm.device.allocate
        with patch("vesper.resources.physical_memory", return_value={"available": 64 * GiB}), block:
            self.assertEqual(len(block.devices), 9)
            calls = block.indexer.norm.device.calls
            with self.assertRaises(MemoryError): block.indexer.norm.device.allocate(512 * MiB)
            self.assertEqual(block.indexer.norm.device.calls, calls)
        self.assertEqual(block.indexer.norm.device.allocate, original)
        self.assertEqual(block.budget.reserved, 0)

    def test_block_mismatched_HC_layer_refuses_before_frame_or_reservation(self):
        block = block_ledger()
        block.hc.layer = 0
        with patch.object(block.attention, "__enter__", side_effect=AssertionError("No frame admission expected")):
            with self.assertRaisesRegex(ValueError, "layer identities"): block.__enter__()
        self.assertEqual(block.budget.reserved, 0)

    def test_connected_rejects_cross_layer_artifacts_before_opening_GPU(self):
        runtime = ConnectedMoE(*[Path("unused")] * 4, layer=2)
        with patch("vesper.resources.physical_memory", return_value={"available": 64 * GiB}), \
             patch("vesper.connected.load_gate_fixture", return_value=(None, {"layer": 2})), \
             patch("vesper.connected.load_router_fixture", return_value=(None, {"layer": 0})), \
             patch("vesper.connected.load_shared_fixture", return_value=(None, {"layer": 2})), \
             patch("vesper.connected.GateProjection", side_effect=AssertionError("No GPU access expected")):
            with self.assertRaisesRegex(ValueError, "selected layer"): runtime.__enter__()
        self.assertIsNone(runtime.stack)
        self.assertEqual(runtime.budget.reserved, 0)

    def test_fixture_original_hash_layer_layout_and_archive_bounds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arrays = {n: np.zeros(s, STORAGE_DTYPES[d]) for n, (d, s) in LAYOUT.items()}
            manifest = {"kind": "original_layer2_block_fixture", "model": MODEL, "revision": REVISION, "layer": 2,
                "tensor_bytes": TENSOR_BYTES, "weights_fetched_bytes": TENSOR_BYTES, "maximum_range_bytes": LIMIT,
                "tensors": [{"name": n, "dtype": d, "shape": list(s), "bytes": arrays[n].nbytes,
                    "sha256": hashlib.sha256(arrays[n].tobytes()).hexdigest()} for n, (d, s) in LAYOUT.items()]}
            np.savez(root / "weights.npz", **arrays)
            def save(value): (root / "manifest.json").write_bytes(json.dumps(value).encode())
            save(manifest)
            loaded, _ = load_block_fixture(root)
            self.assertEqual(sum(a.nbytes for a in loaded.values()), TENSOR_BYTES)
            for mutate in (lambda m: m.update(layer=0), lambda m: m["tensors"][0].update(sha256="0" * 64),
                           lambda m: m["tensors"].pop(), lambda m: m["tensors"][0].update(shape=[10**9])):
                bad = copy.deepcopy(manifest); mutate(bad); save(bad)
                with self.assertRaises(ValueError): load_block_fixture(root)
            save(manifest)
            arrays[next(iter(arrays))] = np.zeros((10**6,), np.float32)
            np.savez(root / "weights.npz", **arrays)
            with self.assertRaises(ValueError): load_block_fixture(root)

    def test_complete_published_goldens_reject_missing_or_changed_evidence(self):
        root = Path(__file__).resolve().parents[1]
        report = json.loads((root / "results/2026-10-19-compressed-block/block.json").read_bytes())
        self.assertEqual(len(validate_goldens(report)), 32)
        validate_plan(report["plan"])
        for change in (lambda r: r["block_cases"].pop(), lambda r: r["block_cases"][0]["boundaries"].pop(),
                lambda r: r["block_cases"][0]["boundaries"][0].__setitem__(2, 1),
                lambda r: r["block_cases"][0]["selected_expert_ids"].reverse(),
                lambda r: r["block_cases"][1].update(committed_state_sha256="0" * 64),
                lambda r: r["unchanged_Block_forward_controls"].pop(),
                lambda r: r["unchanged_Attention_forward_controls"][0][0].update(mismatches=1),
                lambda r: r["faults"][0].update(all_committed_state_unchanged=False),
                lambda r: r["negative_controls"].pop(),
                lambda r: r["snapshot_events"][0].update(consumed_handle_refusals=0),
                lambda r: r["memory"]["HC_cleanup"].update(host_reserved_bytes=1),
                lambda r: r["faults"][0].update(completed_experts_before_failure=6),
                lambda r: r["native_profiles"]["indexer"].update(cubin_sha256="0" * 64)):
            bad = copy.deepcopy(report); change(bad)
            with self.assertRaises(ValueError): validate_goldens(bad)
        observed = copy.deepcopy(report)
        observed["attention_reference_qualification"]["memory"]["minimum_sampled_host_available_bytes"] = 64 * GiB
        self.assertEqual(len(validate_goldens(observed)), 32)
        for change in (lambda p: p["cases"].pop(), lambda p: p["cases"][0].update(hidden_sha256="0" * 64),
                       lambda p: p.update(layer=0), lambda p: p["expert_ids"].pop()):
            bad = copy.deepcopy(report["plan"]); change(bad)
            with self.assertRaises(ValueError): validate_plan(bad)


if __name__ == "__main__": unittest.main()
