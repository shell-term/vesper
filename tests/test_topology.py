import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from vesper.attention_topology import GROUPS, INDEX_SOURCES, KV_SOURCES, compression_layout, indexer_layout, role, validate_config, validate_group
from vesper.checkpoint import MODEL, REVISION, LIMIT
from vesper.compression import NativeCompression
from vesper.compression_fixture import COMPRESSION_LAYOUT
from vesper.indexer import NativeIndexer, validate_query
from vesper.indexer_fixture import INDEXER_LAYOUT
from vesper.prefix import STORAGE_DTYPES
from vesper.resources import GiB
from vesper.shared_attention import SharedAttentionFrame, initial_state, validate_frame, validate_seed
from vesper.topology_fixture import fetch_layer, layout, load_layer
from vesper.topology_validation import state_digest


def frame_args(layers):
    ps = {l: SimpleNamespace(opened=True, ordinal=0, layer=l) for l in layers}
    norm = SimpleNamespace(opened=True, ordinal=0)
    comp = SimpleNamespace(opened=True, ordinal=0, layers={l: role(l).ratio for l in layers if role(l).owns_kv})
    index = SimpleNamespace(opened=True, ordinal=0, layers=tuple(l for l in layers if role(l).owns_index))
    attn = SimpleNamespace(opened=True, ordinal=0, profile="long_context")
    return {}, ps, norm, comp, index, attn, layers


def xs(layers, s=1): return {l: np.zeros((1, s, 5120), np.uint16) for l in layers}


class TopologyTests(unittest.TestCase):
    def test_backbone_owners_match_explicit_pinned_intervals(self):
        kv = [None] * 2 + [2] * 6 + [8] * 6 + [14] * 6 + [20] * 20
        index = [None] * 2 + [2] * 6 + [8] * 6 + [14] * 6 + [20] * 4 + [24] * 4 + [28] * 4 + [32] * 4 + [36] * 4
        for l in range(40):
            r = role(l)
            self.assertEqual((r.kv_owner, r.index_owner), (kv[l], index[l]))
            self.assertEqual(r.owns_kv, l in (2, 8, 14, 20))
            self.assertEqual(r.owns_index, l in (2, 8, 14, 20, 24, 28, 32, 36))
        for l in (-1, 40, True, 2.0):
            with self.assertRaises(ValueError): role(l)
        for group in ((3, 2), (8, 3), (20, 25), [2, 3]):
            with self.assertRaises(ValueError): validate_group(group)

    def test_config_and_default_layouts_remain_exact(self):
        self.assertEqual(compression_layout((2, 20)), COMPRESSION_LAYOUT)
        self.assertEqual(indexer_layout((2, 20, 24)), INDEXER_LAYOUT)
        for layers in ((), (8, 8), (24,), [2]):
            with self.assertRaises(ValueError): compression_layout(layers)
        for layers in ((3,), (28, 28), (True,), [2]):
            with self.assertRaises(ValueError): indexer_layout(layers)
        config = {"n_layers": 40, "dim": 5120, "n_heads": 64, "head_dim": 512, "q_lora_rank": 1280, "o_groups": 8,
            "o_lora_rank": 1024, "window_size": 128, "kv_source_layers": list(KV_SOURCES), "index_source_layers": list(INDEX_SOURCES),
            "candidate_source_layer": 20, "candidate_topk_blocks": 2048, "candidate_block_size": 8, "compress_ratios": [0, 0] + [2] * 18 + [1] * 20}
        self.assertEqual(len(validate_config(config)), 40)
        for key, value in (("kv_source_layers", [2, 20]), ("candidate_source_layer", 24), ("compress_ratios", [1] * 40)):
            with self.assertRaises(ValueError): validate_config({**config, key: value})

    def test_new_operator_weights_and_plain_consumer_refusal_before_cuda(self):
        cl = compression_layout((8, 14))
        arrays = {n: np.zeros(s, np.uint16) for n, s in cl.items()}
        self.assertEqual(NativeCompression(arrays, layers=(8, 14)).layers, {8: 2, 14: 2})
        with self.assertRaises(ValueError): NativeCompression(arrays)
        il = indexer_layout((28, 32, 36))
        arrays = {n: np.zeros(s, STORAGE_DTYPES[d]) for n, (d, s) in il.items()}
        self.assertEqual(NativeIndexer(arrays, layers=(28, 32, 36)).layers, (28, 32, 36))
        x, qr = np.zeros((1, 1, 5120), np.uint16), np.zeros((1, 1, 1280), np.uint16)
        validate_query(x, qr, 28, 0, layers=(28, 32, 36))
        for l in (25, 29, 33, 37):
            with self.assertRaises(ValueError): validate_query(x, qr, l, 0, layers=(28, 32, 36))

    def test_group_identity_pressure_and_input_admission(self):
        for layers in GROUPS:
            self.assertEqual(validate_frame(xs(layers, 3), layers, 1, 0), 3)
            self.assertEqual(validate_frame(xs(layers), layers, 1, 1151), 1)
            with self.assertRaises(ValueError): validate_frame(xs(layers, 2), layers, 1, 1)
            with self.assertRaises(ValueError): validate_frame(xs(layers), layers, 1, 1152)
            args = frame_args(layers)
            args[1][layers[-1]].layer = layers[0]
            with patch("vesper.shared_attention.initial_state", side_effect=AssertionError("no allocation")):
                with self.assertRaises(ValueError): SharedAttentionFrame(*args).__enter__()
        with patch("vesper.resources.physical_memory", return_value={"available": 16 * GiB}), patch("vesper.shared_attention.initial_state", side_effect=AssertionError("no allocation")):
            with self.assertRaises(MemoryError): SharedAttentionFrame(*frame_args((8, 9))).__enter__()

    def test_seed_layout_future_owner_and_nonfinite_refusal(self):
        state = initial_state((8, 9), 1)
        validate_seed(state, (8, 9), 1, 1024)
        for change in (lambda s: s["windows"].update({3: s["windows"].pop(9)}),
                       lambda s: s["index_keys"][8].__setitem__((0, 512, 0), 1),
                       lambda s: s["compressors"][8][2].fill(0x4529),
                       lambda s: s["compressors"][8][0].fill(np.nan),
                       lambda s: s["compressors"][8][1].fill(np.inf)):
            candidate = copy.deepcopy(state)
            change(candidate)
            with self.assertRaises(ValueError): validate_seed(candidate, (8, 9), 1, 1024)
        for pos in (0, 1023, 1152, True):
            with self.assertRaises(ValueError): validate_seed(state, (8, 9), 1, pos)

    def test_seed_publication_and_late_failure_preserve_all_state(self):
        layers = (14, 15)
        state = initial_state(layers, 1)
        state["windows"][14].fill(0x3f80)
        with patch("vesper.resources.physical_memory", return_value={"available": 64 * GiB}):
            with SharedAttentionFrame(*frame_args(layers)) as f:
                before = state_digest(f.inspect())
                with patch("vesper.shared_attention.copy.deepcopy", side_effect=MemoryError("copy")):
                    with self.assertRaises(MemoryError): f.prime_diagnostic(state, 1024)
                self.assertEqual(state_digest(f.inspect()), before)
                f.prime_diagnostic(state, 1024)
                state["windows"][14].fill(0)
                self.assertTrue((f.state["windows"][14] == 0x3f80).all())
                before = state_digest(f.inspect())
                candidate = initial_state(layers, 1)
                def fail(_): raise OSError("after compute")
                with patch("vesper.shared_attention.stage_group", return_value=({}, candidate)):
                    with self.assertRaises(OSError): f.step(xs(layers), finalize=fail)
                self.assertEqual(state_digest(f.inspect()), before)
                handle = f.snapshot()
                f.position = 1025
                f.state["windows"][15].fill(0x3f80)
                self.assertEqual(state_digest(f.rollback(handle)), before)
                with self.assertRaises(ValueError): f.rollback(handle)
            self.assertEqual(f.budget.reserved, 0)

    def test_fixture_destination_and_forged_header_refuse_before_network_or_allocation(self):
        with tempfile.TemporaryDirectory() as tmp, patch("vesper.topology_fixture.read_small", side_effect=AssertionError("no network")):
            with self.assertRaises(FileExistsError): fetch_layer(tmp, 8)
        with tempfile.TemporaryDirectory() as tmp:
            root, specs = Path(tmp), layout(9)
            entries = [{"name": n, "dtype": d, "shape": list(s), "bytes": int(np.prod(s)) * STORAGE_DTYPES[d].itemsize, "sha256": "0" * 64} for n, (d, s) in specs.items()]
            size = sum(e["bytes"] for e in entries)
            (root / "manifest.json").write_text(json.dumps({"kind": "original_topology_layer_fixture", "model": MODEL, "revision": REVISION,
                "layer": 9, "tensors": entries, "tensor_bytes": size, "weights_fetched_bytes": size, "weights_reused_bytes": 0, "maximum_range_bytes": LIMIT}))
            with zipfile.ZipFile(root / "weights.npz", "w") as archive:
                for n in specs:
                    with archive.open(n + ".npy", "w") as stream:
                        np.lib.format.write_array_header_1_0(stream, {"shape": (2**50,), "fortran_order": False, "descr": "|u1"})
            with patch("vesper.topology_fixture.np.load", side_effect=AssertionError("no array allocation")):
                with self.assertRaises(ValueError): load_layer(root, 9)

    def test_published_group_goldens_refuse_role_coverage_and_identity_tampering(self):
        from vesper.topology_comparison import group_name
        from vesper.topology_validation import validate_goldens
        root = Path(__file__).resolve().parents[1] / "results/2026-10-19-attention-topology"
        for group in GROUPS:
            report = json.loads((root / (group_name(group) + ".json")).read_bytes())
            self.assertEqual(len(validate_goldens(report)), 556)
        for mutate in (lambda r: r["meta"].update(position=1),
                       lambda r: r["boundaries"][0].__setitem__(1, r["boundaries"][0][1] + 1),
                       lambda r: r.update(actual_layout_sha256="0" * 64),
                       lambda r: r["observations"]["reused_index_owners"].update({"25": 20})):
            row = copy.deepcopy(report["frame_cases"][0])
            mutate(row)
            with self.assertRaises(ValueError): validate_goldens({**report, "frame_cases": [row, *report["frame_cases"][1:]]})
        with self.assertRaises(ValueError): validate_goldens({**report, "frame_cases": report["frame_cases"][:-1]})
        for key, mutate in (("native_profiles", lambda v: v["projections"]["28"].update(converted_wo_a_bf16_sha256="0" * 64)),
                ("fixtures", lambda v: v[-1]["tensors"][0].update(sha256="0" * 64)),
                ("original_methods_ast_sha256", lambda v: v.update({"Attention.forward": "0" * 64})),
                ("original_seed_controls", lambda v: v[0]["checks"].pop()),
                ("faults", lambda v: v[0].update(state_after_sha256="0" * 64)),
                ("snapshot_events", lambda v: v[0].update(diagnostic_copy_isolated=False)),
                ("instrumentation_counts", lambda v: v.update(index_sum=0)),
                ("negative_controls", lambda v: v.pop()),
                ("memory", lambda v: v.update(minimum_sampled_host_available_bytes=16 * GiB))):
            value = copy.deepcopy(report[key])
            mutate(value)
            with self.assertRaises(ValueError): validate_goldens({**report, key: value})

    def test_published_summary_binds_all_group_and_native_replay_bytes(self):
        root = Path(__file__).resolve().parents[1] / "results/2026-10-19-attention-topology"
        summary = json.loads((root / "summary.json").read_bytes())
        for name, digest in summary["artifacts_sha256"].items():
            self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), digest)
        frames, calls, comparisons, elements = 0, 0, 0, 0
        for g in summary["groups"]:
            p = json.loads((root / g["paired_report"]).read_bytes())
            n = json.loads((root / g["native_only_report"]).read_bytes())
            self.assertEqual(n["reference_report_sha256"], summary["artifacts_sha256"][g["paired_report"]])
            self.assertEqual(n["status"], "pass")
            self.assertEqual(n["diagnostic_seeds"], p["diagnostic_seeds"])
            self.assertFalse(any(n[f + suffix] for f in ("torch", "tilelang") for suffix in ("_installed", "_imported")))
            self.assertEqual(len(n["frame_cases"]), 556)
            for left, right in zip(p["frame_cases"], n["frame_cases"]):
                self.assertEqual(left["meta"], right["meta"])
                self.assertEqual(left["observations"], right["observations"])
                self.assertEqual(right["status"], "pass")
                self.assertEqual(right["actual_boundary_root_sha256"], left["expected_boundary_root_sha256"])
                self.assertEqual(right["actual_layout_sha256"], left["expected_layout_sha256"])
                self.assertEqual(right["boundary_comparisons"], len(left["boundaries"]))
                self.assertEqual(right["compared_storage_elements"], sum(b[1] for b in left["boundaries"]))
            frames += len(n["frame_cases"])
            calls += len(n["frame_cases"]) * len(g["layers"])
            comparisons += sum(r["boundary_comparisons"] for r in n["frame_cases"])
            elements += sum(r["compared_storage_elements"] for r in n["frame_cases"])
        self.assertEqual((frames, calls, comparisons, elements), (2224, 8896, 684616, 13174870167))
        self.assertEqual(summary["boundary_comparisons"], comparisons)
        self.assertEqual(summary["compared_storage_elements"], elements)


if __name__ == "__main__": unittest.main()
