"""Synthetic provider/lease tests; no official model or CUDA qualification."""

import gc
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from vesper import spark_tensor_views as views
from vesper.engram_assets import CONTROL_LAYOUT
from vesper.gate import GATE_SHAPE, load_gate_fixture
from vesper.router import load_router_fixture
from vesper.shared import SHARED_LAYOUT, load_shared_fixture
from vesper.resources import MemoryBudget, MiB, GiB


class Workspace:
    def __init__(self, root):
        self.base = Path(root) / "artifacts/dgx-spark"
        self.check = Mock()
        self.check_runtime_module = Mock()

    def context(self):
        return {"platform": "dgx-spark", "profile": "native", "model": views.MODEL, "revision": views.REVISION}

    def artifact_path(self, name):
        path = self.base / name
        if not path.absolute().is_relative_to(self.base.absolute()):
            raise ValueError("outside namespace")
        return path


class FakeVerifiedReader:
    """Inject the already-tested verified span primitive, never raw networking."""
    def __init__(self, metadata, originals, integrity, budget, *, tensor_names, workspace):
        self.budget, self.names = budget, tensor_names
        self.tensors = {name: CATALOG[name] for name in tensor_names}
        self.reads = []
        self.failure = None
        self.nonfinite = False

    def __enter__(self):
        self.budget.acquire(16 * MiB, category="metadata")
        return self

    def read_into(self, name, offset, target):
        self.budget.acquire(0)
        raw = memoryview(target).cast("B")
        if not 0 < raw.nbytes <= 8 * MiB or offset + raw.nbytes > self.tensors[name]["bytes"]:
            raise AssertionError("unbounded original span")
        self.reads.append((name, offset, raw.nbytes))
        if self.failure:
            raise self.failure
        array = np.frombuffer(raw, np.uint8)
        array.fill(0)
        if self.nonfinite and self.tensors[name]["dtype"] == "BF16":
            array[:2] = (128, 127)

    def __exit__(self, *_):
        self.budget.release(16 * MiB, category="metadata")


CATALOG = {}


class TensorViewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.addCleanup(self.temp.cleanup)
        self.workspace = Workspace(self.temp.name)
        self.metadata, self.originals, self.integrity = [self.workspace.base / n for n in ("metadata", "originals", "integrity")]
        for path in (self.metadata, self.originals, self.integrity):
            path.mkdir(parents=True)
        catalog = {}
        def add(name, dtype, shape):
            catalog[name] = {"name": name, "dtype": dtype, "shape": list(shape),
                             "bytes": int(np.prod(shape)) * views.STORAGE[dtype].itemsize}
        add("layers.0.attn_norm.weight", "BF16", (5120,))
        add("layers.39.ffn_norm.weight", "BF16", (5120,))
        add("layers.0.large.weight", "U8", (17 * MiB,))
        for layer in (1, 14):
            for short, (dtype, shape) in CONTROL_LAYOUT.items():
                add(f"layers.{layer}.engram.{short}", dtype, shape)
        add("layers.0.ffn.gate.weight", "BF16", GATE_SHAPE)
        for short in ("bias", "bias_vl"):
            add("layers.0.ffn.gate." + short, "F32", (384,))
        for short, shape in SHARED_LAYOUT.items():
            p, s = short.split("_")
            add(f"layers.0.ffn.shared_experts.{p}.{s}", "F8_E4M3" if s == "weight" else "F8_E8M0", shape)
        for name, shape in (("head.weight", (views.VOCAB, views.WIDTH)), ("embed.weight", (views.VOCAB, views.WIDTH)),
                            ("norm.weight", (views.WIDTH,))):
            add(name, "BF16", shape)
        header, end = {}, 0
        for name, t in catalog.items():
            header[name] = {"dtype": t["dtype"], "shape": t["shape"], "data_offsets": [end, end + t["bytes"]]}
            end += t["bytes"]
        raw = json.dumps(header, separators=(",", ":")).encode()
        shard = "model-00003-of-00048.safetensors"
        for name, t in catalog.items():
            t.update(shard=shard, shard_byte_offset=8 + len(raw) + header[name]["data_offsets"][0])
        global CATALOG
        CATALOG = self.catalog = catalog
        (self.metadata / "headers").mkdir()
        (self.metadata / "headers" / (shard + ".json")).write_bytes(raw)
        self.identity = {"shards": [{"shard": shard, "header_bytes": len(raw), "header_sha256": hashlib.sha256(raw).hexdigest()}]}
        selected = views.controls(catalog)
        config = {"text_config": {"hidden_size": 5120, "n_routed_experts": 384, "num_experts_per_tok": 6,
            "scoring_func": "sqrtsoftplus", "norm_topk_prob": True, "routed_scaling_factor": 1.5,
            "moe_intermediate_size": 2304, "swiglu_limit": 10.0}}
        inference = {"n_routed_experts": 384, "n_activated_experts": 6, "score_func": "sqrtsoftplus", "route_scale": 1.5}
        self.small = {"config.json": json.dumps(config).encode(), "inference/config.json": json.dumps(inference).encode(),
                      "model.safetensors.index.json": json.dumps({"weight_map": {n: shard for n in catalog}}).encode()}
        patches = [patch.object(views, "load_catalog", return_value=(catalog, self.identity)),
                   patch.object(views, "OriginalTensorReader", FakeVerifiedReader),
                   patch.object(views, "_metadata_paths"),
                   patch.object(views, "CONTROL_COUNT", len(selected)),
                   patch.object(views, "CONTROL_BYTES", sum(t["bytes"] for t in selected.values())),
                   patch.object(views, "read_checked", side_effect=lambda root, name: self.small[name]),
                   patch("vesper.router.ROUTER_INFERENCE_CONFIG_HASH", hashlib.sha256(self.small["inference/config.json"]).hexdigest()),
                   patch("vesper.resources.physical_memory", return_value={"available": 128 * GiB}),
                   patch.object(views.shutil, "disk_usage", return_value=Mock(free=2**40)),
                   patch("urllib.request.urlopen", side_effect=AssertionError("network forbidden in provider"))]
        for value in patches:
            value.start(); self.addCleanup(value.stop)
        self.budget = MemoryBudget(cap=8 * GiB, owner="synthetic_shared_parent")

    def assets(self):
        return views.SparkModelAssets(self.metadata, self.originals, self.integrity,
                                      workspace=self.workspace, budget=self.budget)

    def test_open_and_mapping_enumeration_are_metadata_only(self):
        with self.assets() as a:
            self.assertEqual(len(a.arrays), views.CONTROL_COUNT + 1)
            self.assertNotIn("head.weight", a.arrays)
            self.assertNotIn("embed.weight", a.arrays)
            self.assertIn("norm.weight", a.arrays)
            self.assertIs(a.compression_arrays(), a.arrays)
            self.assertIs(a.indexer_arrays(), a.arrays)
            self.assertEqual(set(a.layer_arrays(39)), {"layers.39.ffn_norm.weight"})
            self.assertEqual(a.reader.reads, [])
        self.assertEqual(self.budget.reserved, 0)

    def test_explicit_authenticated_catalog_avoids_second_unscoped_loader(self):
        catalog = SimpleNamespace(select=Mock(return_value={"tensors": self.catalog, "identity": self.identity}))
        received = []
        def reader(*args, catalog=None, **kwargs):
            received.append(catalog)
            return FakeVerifiedReader(*args, **kwargs)
        with patch.object(views, "load_catalog", side_effect=AssertionError("Use the explicit catalog owner")), \
                patch.object(views, "OriginalTensorReader", side_effect=reader):
            with views.SparkModelAssets(self.metadata, self.originals, self.integrity,
                    workspace=self.workspace, budget=self.budget, catalog=catalog) as assets:
                self.assertIs(assets.catalog, catalog)
                self.assertEqual(assets.reader.reads, [])
        self.assertEqual(received, [catalog])
        self.assertEqual(catalog.select.call_count, 1)
        predicate = catalog.select.call_args.args[0]
        self.assertTrue(predicate("head.weight"))
        self.assertFalse(predicate("layers.0.ffn.experts.0.w1.weight"))
        self.assertFalse(predicate("layers.1.engram.embed.weight"))
        self.assertEqual(self.budget.reserved, 0)

    def test_large_control_splits_source_reads_without_a_resident_cache(self):
        with self.assets() as a:
            first = a.arrays_for_layer(0)["layers.0.large.weight"]
            self.assertEqual([r[2] for r in a.reader.reads], [8 * MiB, 8 * MiB, MiB])
            second = a.arrays["layers.0.large.weight"]
            self.assertIsNot(first, second)
            self.assertEqual(a.stats()["live_array_bytes"], 34 * MiB)
            self.assertFalse(first.flags.writeable)
            self.assertEqual(first.dtype, np.uint8)
            del first, second
        gc.collect()
        self.assertEqual(self.budget.reserved, 0)

    def test_lease_survives_provider_close_and_numpy_borrowed_view(self):
        a = self.assets().__enter__()
        owner = a.norm_weight()
        borrowed = owner[7:12]
        size = owner.nbytes
        del owner
        a.__exit__()
        gc.collect()
        self.assertEqual(self.budget.reserved, size)
        self.assertEqual(a.stats()["live_arrays"], 1)
        self.assertTrue(np.all(borrowed == 0))
        del borrowed
        gc.collect()
        self.assertEqual(self.budget.reserved, 0)
        self.assertEqual(a.stats()["live_arrays"], 0)

    def test_temporary_upload_reads_once_then_releases_cache_but_not_borrowers(self):
        with self.assets() as a:
            baseline = self.budget.reserved
            name = "layers.0.large.weight"
            with a.upload_arrays(0) as arrays:
                first = arrays[name]
                read_count = len(a.reader.reads)
                self.assertIs(arrays[name], first)
                self.assertEqual(len(a.reader.reads), read_count)
                self.assertFalse(first.flags.writeable)
                borrowed = first[1:5]
                del first
            self.assertEqual(arrays.values, {})
            with self.assertRaises(RuntimeError): arrays[name]
            self.assertEqual(self.budget.reserved, baseline + 17 * MiB)
            del borrowed
            gc.collect()
            self.assertEqual(self.budget.reserved, baseline)
            with a.upload_arrays(0) as next_view:
                next_view[name]
                self.assertGreater(len(a.reader.reads), read_count)
        self.assertEqual(self.budget.reserved, 0)

    def test_temporary_upload_cap_and_exception_do_not_leak(self):
        with self.assets() as a:
            baseline = self.budget.reserved
            with patch.object(views, "UPLOAD_ARRAY_CAP", MiB), a.upload_arrays(0) as arrays:
                with self.assertRaises(MemoryError): arrays["layers.0.large.weight"]
                self.assertEqual(a.reader.reads, [])
            with self.assertRaisesRegex(ValueError, "injected upload"):
                with a.upload_arrays(0) as arrays:
                    arrays["layers.0.large.weight"]
                    raise ValueError("injected upload")
            gc.collect()
            self.assertEqual(self.budget.reserved, baseline)

    def test_read_error_and_nonfinite_values_never_publish_or_leak_a_lease(self):
        with self.assets() as a:
            baseline = self.budget.reserved
            a.reader.failure = ValueError("synthetic verified-block mismatch")
            with self.assertRaisesRegex(ValueError, "verified-block mismatch"):
                a.norm_weight()
            gc.collect()
            self.assertEqual(self.budget.reserved, baseline)
            a.reader.failure = None; a.reader.nonfinite = True
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                a.embedding(1)
            gc.collect()
            self.assertEqual(self.budget.reserved, baseline)

    def test_memory_floor_and_array_cap_reject_before_source_io(self):
        with self.assets() as a:
            with patch("vesper.resources.physical_memory", return_value={"available": 32 * GiB + 3}):
                with self.assertRaises(MemoryError):
                    a.norm_weight()
            self.assertEqual(a.reader.reads, [])
            with patch.object(views, "MAX_ARRAY_BYTES", 1024):
                with self.assertRaisesRegex(ValueError, "256 MiB"):
                    a.norm_weight()
            self.assertEqual(a.reader.reads, [])
        with self.assertRaisesRegex(ValueError, "32 GiB"):
            views.SparkModelAssets(self.metadata, self.originals, self.integrity,
                                  workspace=self.workspace, budget=MemoryBudget(cap=GiB, available_floor=0))

    def test_embedding_contract_is_per_lookup_and_rejects_nonoriginal_ids(self):
        with self.assets() as a:
            rows = a.embedding_rows
            self.assertFalse(hasattr(rows, "items"))
            self.assertIn(views.VOCAB - 1, rows)
            self.assertNotIn(True, rows)
            self.assertEqual(a.reader.reads, [])
            result = rows[views.VOCAB - 1]
            self.assertEqual(result.shape, (5120,))
            self.assertEqual(result.dtype, np.uint16)
            self.assertEqual(a.reader.reads, [("embed.weight", (views.VOCAB - 1) * 10240, 10240)])
            for token in (-1, views.VOCAB, True, 1.0):
                with self.assertRaises(ValueError):
                    a.embedding(token)
            del result

    def test_head_rows_keep_full_addressing_and_split_at_eight_mib(self):
        with self.assets() as a:
            values = a.head_rows(5, 1640)
            self.assertEqual(values.shape, (1640, 5120))
            self.assertTrue(values.flags.c_contiguous)
            self.assertEqual(a.reader.reads[0], ("head.weight", 5 * 10240, 8 * MiB))
            self.assertEqual(sum(r[2] for r in a.reader.reads), 1640 * 10240)
            self.assertLessEqual(max(r[2] for r in a.reader.reads), 8 * MiB)
            for start, count in ((-1, 1), (views.VOCAB, 1), (0, views.VOCAB), (0, True)):
                with self.assertRaises(ValueError):
                    a.head_rows(start, count)
            del values

    def test_engram_aliases_match_existing_control_layout(self):
        with self.assets() as a:
            local = a.engram_controls(14)
            self.assertEqual(set(local), set(CONTROL_LAYOUT))
            self.assertEqual(a.reader.reads, [])
            q = local["q_weight"]
            self.assertEqual(q.shape, (4, 5120))
            self.assertEqual(q.dtype, np.uint16)
            self.assertEqual(a.reader.reads[0][0], "layers.14.engram.q_weight")
            with self.assertRaises(ValueError):
                a.engram_controls(0)
            del q

    def test_missing_moe_paths_never_prepare_or_write_during_forward(self):
        with self.assets() as a:
            with self.assertRaises(FileNotFoundError):
                a.moe_paths(0)
            self.assertFalse(a.fixtures.exists())
            self.assertEqual(a.reader.reads, [])

    def test_local_moe_prepare_preserves_existing_validators_and_is_reusable(self):
        with self.assets() as a:
            result = a.prepare_moe_fixtures((0,))
            self.assertEqual(result["network_payload_bytes"], 0)
            self.assertFalse(result["second_expert_pack_created"])
            paths = a.moe_paths(0)
            self.assertTrue(all(isinstance(p, Path) for p in paths))
            for path, loader in zip(paths, (load_gate_fixture, load_router_fixture, load_shared_fixture)):
                arrays, manifest = loader(path)
                self.assertEqual(manifest["layer"], 0)
                del arrays
            reads = len(a.reader.reads)
            a.prepare_moe_fixtures((0,))
            self.assertEqual(len(a.reader.reads), reads)
            manifest = paths[0].parent / "manifest.json"
            manifest.write_bytes(manifest.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "manifest changed"):
                a.moe_paths(0)
        gc.collect()
        self.assertEqual(self.budget.reserved, 0)

    def test_moe_preparation_refuses_space_pressure_without_creating_layer(self):
        with self.assets() as a:
            with patch.object(views.shutil, "disk_usage", return_value=Mock(free=64 * GiB)):
                with self.assertRaises(OSError):
                    a.prepare_moe_fixtures((0,))
            self.assertFalse((a.fixtures / "layer0").exists())
            self.assertFalse((a.fixtures / ".layer0.part").exists())
            self.assertEqual(a.reader.reads, [])

    def test_self_consistent_changed_fixture_still_refuses_original_identity(self):
        with self.assets() as a:
            a.prepare_moe_fixtures((0,))
            gate = a.moe_paths(0)[0]
        weight, manifest = load_gate_fixture(gate)
        weight[0, 0] = 0x3f80
        np.savez(gate, weight=weight)
        manifest["tensor"]["sha256"] = hashlib.sha256(memoryview(weight)).hexdigest()
        (gate.parent / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        # The existing fixture validator accepts this self-consistent altered
        # payload; provider admission must additionally bind it to the source.
        changed, _ = load_gate_fixture(gate)
        self.assertEqual(changed[0, 0], 0x3f80)
        del weight, changed
        with self.assets() as a:
            with self.assertRaisesRegex(ValueError, "registered original tensor bytes"):
                a.moe_paths(0)
            self.assertEqual(a.moe_ready, {})
            self.assertLessEqual(max(r[2] for r in a.reader.reads), 8 * MiB)
        gc.collect()
        self.assertEqual(self.budget.reserved, 0)

    def test_borrowed_lock_inode_refuses_before_any_prepare_write(self):
        with self.assets() as a:
            a.fixtures.mkdir(parents=True)
            guard = a.fixtures / ".download.lock"
            guard.write_bytes(b"preserve existing borrowed inode")
            original_stat = Path.stat
            def observed(path, *args, **kwargs):
                result = original_stat(path, *args, **kwargs)
                if path == guard:
                    fields = {name: getattr(result, name) for name in dir(result) if name.startswith("st_")}
                    fields["st_nlink"] = 2
                    return SimpleNamespace(**fields)
                return result
            with patch.object(Path, "stat", new=observed), self.assertRaisesRegex(ValueError, "one owned link"):
                a.prepare_moe_fixtures((0,))
            self.assertEqual(guard.read_bytes(), b"preserve existing borrowed inode")
            self.assertEqual(a.reader.reads, [])

    def test_reader_cleanup_error_still_releases_owned_small_metadata(self):
        a = self.assets().__enter__()
        a._small("config.json")
        before = self.budget.reserved
        original = a.reader.__exit__
        def fail_after_release(*args):
            original(*args)
            raise OSError("synthetic cleanup failure")
        a.reader.__exit__ = fail_after_release
        with self.assertRaisesRegex(OSError, "cleanup failure"):
            a.__exit__()
        self.assertGreater(before, 16 * MiB)
        self.assertIsNone(a.reader)
        self.assertEqual(self.budget.reserved, 0)

    def test_wrong_control_census_and_runtime_source_refuse_before_reader(self):
        with patch.object(views, "CONTROL_COUNT", 1251), patch.object(views, "OriginalTensorReader") as reader:
            with self.assertRaisesRegex(ValueError, "census"):
                self.assets().__enter__()
            reader.assert_not_called()
        self.workspace.check_runtime_module.side_effect = ValueError("wrong imported provider")
        with self.assertRaisesRegex(ValueError, "wrong imported provider"):
            self.assets()


if __name__ == "__main__":
    unittest.main()
