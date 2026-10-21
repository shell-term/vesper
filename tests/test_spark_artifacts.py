"""Synthetic CPU-only acquisition/integrity tests, not model-quality evidence."""

import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from vesper import spark_artifacts as assets
from vesper.artifact_repository import register_blob
from vesper.resources import MemoryBudget, MiB, GiB


class Workspace:
    def __init__(self, root):
        self.root = Path(root)
        self.base = self.root / "artifacts/dgx-spark"
        self.check = Mock()
        self.check_runtime_module = Mock()

    def context(self):
        return {"platform": "dgx-spark", "profile": "native", "model": assets.MODEL, "revision": assets.REVISION}

    def stamp_result(self, payload, *, initial_context=None):
        return {**payload, "workspace_context": initial_context or self.context()}

    def artifact_path(self, name):
        path = self.base / name
        if not path.absolute().is_relative_to(self.base.absolute()):
            raise ValueError("outside bound artifacts")
        return path


class Response:
    def __init__(self, raw, start=0, *, status=206, fail_after=None, wrong_range=False, encoding="identity"):
        self.status = status
        self.headers = {"Content-Range": f"bytes {start}-{len(raw)-1}/{len(raw)}",
                        "Content-Length": str(len(raw) - start), "Content-Encoding": encoding}
        if wrong_range:
            self.headers["Content-Range"] = "bytes 0-5/6"
        self.stream = io.BytesIO(raw[start:])
        self.fail_after, self.calls = fail_after, 0

    def read(self, size):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise OSError("synthetic interrupted transfer")
        return self.stream.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.stream.close()


def expected(raw, ordinal=3):
    return {"file": f"model-{ordinal:05d}-of-00048.safetensors", "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ws = Workspace(self.root)
        self.metadata = self.ws.base / "metadata"
        self.originals = self.ws.base / "originals"
        self.metadata.mkdir(parents=True)
        (self.metadata / "tree.json").write_bytes(b"synthetic inventory seam")
        self.originals.mkdir()
        self.raw = bytes(range(256)) * (2 * MiB // 256) + b"last original bytes"
        self.row = expected(self.raw)
        self.inventory = patch.object(assets, "shard_inventory", return_value=[self.row])
        self.inventory.start(); self.addCleanup(self.inventory.stop)
        physical = patch("vesper.resources.physical_memory", return_value={"available": 128 * GiB})
        physical.start(); self.addCleanup(physical.stop)
        disk = patch.object(assets.shutil, "disk_usage", return_value=Mock(free=2**40))
        disk.start(); self.addCleanup(disk.stop)

    def one(self, opener, row=None, checkpoint=lambda: None):
        return assets._download_one(self.originals, row or self.row, 2 * MiB,
                                    opener=opener, checkpoint=checkpoint)

    def test_full_download_requires_explicit_opt_in_before_network(self):
        opener = Mock()
        with self.assertRaises(ValueError):
            assets.download_originals(self.metadata, self.originals, quota_bytes=GiB, opener=opener, workspace=self.ws)
        opener.assert_not_called()

    def test_foreign_model_or_revision_refuses_before_inventory_or_network(self):
        for change in ({"model": "another/public-model"}, {"revision": "0" * 40}):
            with self.subTest(change=change):
                context = {**self.ws.context(), **change}
                opener = Mock()
                with patch.object(self.ws, "context", return_value=context), patch.object(assets, "shard_inventory") as inventory:
                    with self.assertRaisesRegex(ValueError, "pinned original model and revision"):
                        assets.download_originals(self.metadata, self.originals, quota_bytes=GiB,
                            execute=True, workspace=self.ws, opener=opener)
                    inventory.assert_not_called()
                    opener.assert_not_called()

    def test_readonly_plan_checks_quota_namespace_and_disk(self):
        with self.assertRaises(ValueError):
            assets.plan_originals(self.metadata, self.originals, quota_bytes=len(self.raw), workspace=self.ws)
        with self.assertRaises(ValueError):
            assets.plan_originals(self.metadata, self.root / "pc-other", quota_bytes=GiB, workspace=self.ws)
        plan = assets.plan_originals(self.metadata, self.originals, quota_bytes=GiB, workspace=self.ws)
        self.assertFalse(plan["network_started"])
        self.assertEqual(plan["original_bytes"], len(self.raw))
        with patch.object(assets.shutil, "disk_usage", return_value=Mock(free=GiB)):
            with self.assertRaises(OSError):
                assets.plan_originals(self.metadata, self.originals, quota_bytes=GiB, workspace=self.ws)
        self.assertEqual(list(self.originals.iterdir()), [])

    def test_whole_hash_verified_then_exclusive_publication_and_reuse(self):
        count = self.one(lambda req, **kw: Response(self.raw))
        self.assertEqual(count, len(self.raw))
        final = self.originals / self.row["file"]
        self.assertEqual(final.read_bytes(), self.raw)
        self.assertFalse(final.with_name(final.name + ".part").exists())
        opener = Mock(side_effect=AssertionError("no network for complete original"))
        self.assertEqual(self.one(opener), 0)
        final.write_bytes(b"x" * len(self.raw))
        with self.assertRaises(ValueError):
            self.one(opener)
        self.assertEqual(final.read_bytes(), b"x" * len(self.raw))

    def test_interruption_resumes_only_verified_prefix_with_exact_range(self):
        with self.assertRaises(OSError):
            self.one(lambda req, **kw: Response(self.raw, fail_after=1))
        part = self.originals / (self.row["file"] + ".part")
        self.assertEqual(part.stat().st_size, 2 * MiB)
        starts = []
        def resume(req, **kw):
            start = int(req.get_header("Range").split("=")[1].split("-")[0])
            starts.append(start)
            return Response(self.raw, start)
        self.assertEqual(self.one(resume), len(self.raw) - 2 * MiB)
        self.assertEqual(starts, [2 * MiB])
        self.assertEqual((self.originals / self.row["file"]).read_bytes(), self.raw)

    def test_changed_acknowledged_partial_refuses_before_network(self):
        with self.assertRaises(OSError):
            self.one(lambda req, **kw: Response(self.raw, fail_after=1))
        part = self.originals / (self.row["file"] + ".part")
        with part.open("r+b") as file:
            file.write(b"tamper")
        opener = Mock()
        with self.assertRaises(ValueError):
            self.one(opener)
        opener.assert_not_called()

    def test_owned_unacknowledged_tail_is_removed_before_resume(self):
        with self.assertRaises(OSError):
            self.one(lambda req, **kw: Response(self.raw, fail_after=1))
        part = self.originals / (self.row["file"] + ".part")
        with part.open("ab") as file:
            file.write(b"unacknowledged")
        self.one(lambda req, **kw: Response(self.raw, 2 * MiB))
        self.assertEqual((self.originals / self.row["file"]).read_bytes(), self.raw)

    def test_wrong_range_or_encoding_is_rejected_before_body_read(self):
        for bad in (dict(wrong_range=True), dict(encoding="gzip")):
            with self.subTest(bad=bad):
                response = Response(self.raw, **bad)
                with self.assertRaises(ValueError):
                    self.one(lambda req, **kw: response)
                self.assertEqual(response.calls, 0)

    def test_wrong_whole_sha_preserves_partial_without_publishing(self):
        row = {**self.row, "sha256": "0" * 64}
        with self.assertRaises(ValueError):
            self.one(lambda req, **kw: Response(self.raw), row)
        self.assertFalse((self.originals / row["file"]).exists())
        self.assertEqual((self.originals / (row["file"] + ".part")).stat().st_size, len(self.raw))

    def test_resumed_200_response_is_rejected_before_read(self):
        with self.assertRaises(OSError):
            self.one(lambda req, **kw: Response(self.raw, fail_after=1))
        response = Response(self.raw, 2 * MiB, status=200)
        with self.assertRaises(ValueError):
            self.one(lambda req, **kw: response)
        self.assertEqual(response.calls, 0)

    def test_public_acquisition_uses_pinned_model_and_bounded_receipts(self):
        urls = []
        def opener(request, **kwargs):
            urls.append(request.full_url)
            return Response(self.raw)
        result = assets.download_originals(self.metadata, self.originals, quota_bytes=GiB,
            chunk_bytes=2 * MiB, execute=True, workspace=self.ws, opener=opener)
        self.assertTrue(result["whole_shard_SHA256_verified"])
        self.assertFalse(result["whole_checkpoint_downloaded"])
        self.assertEqual(urls, [f"{assets.BASE}/resolve/{assets.REVISION}/{self.row['file']}"])
        self.assertLess((self.originals / (self.row["file"] + ".chunks.jsonl")).stat().st_size, 4096)

    def test_publication_race_never_replaces_a_completed_destination(self):
        original_publish = assets._publish
        raced = self.originals / self.row["file"]
        def race(part, final):
            final.write_bytes(b"another writer")
            return original_publish(part, final)
        with patch.object(assets, "_publish", side_effect=race):
            with self.assertRaises(OSError):
                self.one(lambda req, **kw: Response(self.raw))
        self.assertEqual(raced.read_bytes(), b"another writer")
        self.assertEqual((self.originals / (self.row["file"] + ".part")).read_bytes(), self.raw)

    def test_unknown_directory_contents_and_source_guard_failure_prevent_network(self):
        (self.originals / "someone-elses-model.bin").write_bytes(b"preserve")
        opener = Mock()
        with self.assertRaises(ValueError):
            assets.download_originals(self.metadata, self.originals, quota_bytes=GiB,
                execute=True, workspace=self.ws, opener=opener)
        opener.assert_not_called()
        self.assertEqual((self.originals / "someone-elses-model.bin").read_bytes(), b"preserve")
        self.ws.check_runtime_module.side_effect = ValueError("wrong runtime source")
        with self.assertRaises(ValueError):
            assets.download_originals(self.metadata, self.originals, quota_bytes=GiB,
                execute=True, workspace=self.ws, opener=opener)
        opener.assert_not_called()

    def test_hardlinked_journal_refuses_before_network_or_truncation(self):
        journal = self.originals / (self.row["file"] + ".chunks.jsonl")
        journal.write_bytes(b"preserve this unrelated inode\n")
        original_stat = Path.stat
        def linked_stat(path, *args, **kwargs):
            result = original_stat(path, *args, **kwargs)
            if path == journal:
                values = list(result); values[3] = 2
                return os.stat_result(values)
            return result
        opener = Mock()
        # Link creation is unavailable in some Windows sandboxes. Inject only
        # the OS-observed link count; all namespace and file I/O stays real.
        with patch.object(Path, "stat", new=linked_stat), self.assertRaisesRegex(ValueError, "one owned link"):
            assets.download_originals(self.metadata, self.originals, quota_bytes=GiB,
                execute=True, workspace=self.ws, opener=opener)
        opener.assert_not_called()
        self.assertEqual(journal.read_bytes(), b"preserve this unrelated inode\n")

    def test_interrupted_chunk_receipt_tail_resumes_but_changed_source_does_not(self):
        with self.assertRaises(OSError):
            self.one(lambda req, **kw: Response(self.raw, fail_after=1))
        journal = self.originals / (self.row["file"] + ".chunks.jsonl")
        with journal.open("ab") as file:
            file.write(b'{"offset":')
        self.one(lambda req, **kw: Response(self.raw, 2 * MiB))
        self.assertEqual((self.originals / self.row["file"]).read_bytes(), self.raw)

    def test_memory_or_space_pressure_check_runs_between_transfer_chunks(self):
        calls = 0
        def checkpoint():
            nonlocal calls
            calls += 1
            if calls == 3:
                raise MemoryError("synthetic floor pressure")
        # Resume performs the first checkpoint; the third is before chunk two.
        with self.assertRaises(MemoryError):
            self.one(lambda req, **kw: Response(self.raw), checkpoint=checkpoint)
        self.assertFalse((self.originals / self.row["file"]).exists())
        self.assertEqual((self.originals / (self.row["file"] + ".part")).stat().st_size, 2 * MiB)


class OriginalReadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        cls.root = Path(cls.temp.name); cls.ws = Workspace(cls.root)
        cls.metadata, cls.originals, cls.integrity = [cls.ws.base / n for n in ("metadata", "originals", "integrity")]
        for p in (cls.metadata, cls.originals, cls.integrity):
            p.mkdir(parents=True)
        for filename in [*(value[0] for value in assets.FILES.values()), "tree.json"]:
            (cls.metadata / filename).write_bytes(b"synthetic catalog qualification seam")
        (cls.metadata / "headers").mkdir()
        for i in range(1, 49):
            (cls.metadata / "headers" / f"model-{i:05d}-of-00048.safetensors.json").write_bytes(b"{}")
        cls.inventory, receipts = [], []
        cls.first, cls.last = b"\x11" * assets.EXPERT_BYTES, b"\x2e" * assets.EXPERT_BYTES
        memory = patch("vesper.resources.physical_memory", return_value={"available": 128 * GiB})
        with memory:
            for i in range(1, 49):
                raw = cls.first + cls.last if i == 3 else b"small synthetic source bytes"
                row = expected(raw, i); cls.inventory.append(row)
                path = cls.originals / row["file"]; path.write_bytes(raw)
                receipts.append(register_blob(path, row, cls.integrity / (row["file"] + ".sha256"),
                                                MemoryBudget(cap=MiB, available_floor=0)))
        cls.registration = {"kind": "registered_original_artifacts", "model": assets.MODEL, "revision": assets.REVISION,
            "catalog_sha256": assets.CATALOG_SHA256, "shard_inventory_sha256": assets.SHARD_INVENTORY_SHA256,
            "entries": receipts}
        (cls.integrity / "manifest.json").write_bytes(assets._json(cls.registration))
        # A synthetic authenticated-catalog seam: tests exercise source reading
        # and integrity, not official metadata qualification or model arithmetic.
        cls.catalog = {}
        for expert in range(384):
            start = assets.EXPERT_BYTES if expert == 383 else 0
            for short, region in assets.tensor_layout().items():
                projection, suffix = short.split("_")
                name = f"layers.39.ffn.experts.{expert}.{projection}.{suffix}"
                cls.catalog[name] = {"name": name, "dtype": "I8" if suffix == "weight" else "F8_E8M0",
                    "shape": region["shape"], "bytes": region["bytes"], "shard": cls.inventory[2]["file"],
                    "shard_byte_offset": start + region["offset"]}
        for name, ordinal, offset, size, dtype in (("head.weight", 3, 0, 8 * MiB, "BF16"),
            ("embed.weight", 1, 0, 16, "BF16"), ("control.weight", 2, 1, 8, "U8")):
            cls.catalog[name] = {"name": name, "dtype": dtype, "shape": [size // (2 if dtype == "BF16" else 1)],
                "bytes": size, "shard": cls.inventory[ordinal - 1]["file"], "shard_byte_offset": offset}

    @classmethod
    def tearDownClass(cls):
        del cls.first, cls.last
        cls.temp.cleanup()

    def setUp(self):
        for target, value in (("vesper.resources.physical_memory", {"available": 128 * GiB}),
            ("vesper.spark_artifacts.shard_inventory", self.inventory),
            ("vesper.spark_artifacts.load_catalog", (self.catalog, {"catalog_sha256": assets.CATALOG_SHA256}))):
            p = patch(target, return_value=value); p.start(); self.addCleanup(p.stop)
        self.budget = MemoryBudget(cap=128 * MiB, available_floor=0)

    def test_arbitrary_last_expert_has_six_original_tensors_without_network_or_repack(self):
        target = np.empty(assets.EXPERT_BYTES, np.uint8)
        with patch.object(assets.urllib.request, "urlopen", side_effect=AssertionError("forward must be offline")):
            with assets.OriginalShardExpertStore(self.metadata, self.originals, self.integrity,
                    self.budget, layer=39, workspace=self.ws) as store:
                self.assertEqual(set(store.entries), {f"39:{i}" for i in range(384)})
                with patch.object(assets.VerifiedBlob, "read", side_effect=AssertionError("Spark spans must use bulk I/O")):
                    store.read_into("39:383", target)
                self.assertEqual(hashlib.sha256(target).hexdigest(), hashlib.sha256(self.last).hexdigest())
                store.validate("39:383", target, tensors=True)
                self.assertEqual(store.stats()["network_payload_bytes_in_forward"], 0)
                self.assertEqual(store.stats()["payload_bytes_repacked"], 0)
                self.assertEqual(store.stats()["maximum_verified_span_read_bytes"], 8 * MiB)
                self.assertEqual(store.stats()["verification_block_bytes"], 65536)
                self.assertLessEqual(len(store.blobs), 2)
        self.assertEqual(self.budget.reserved, 0)
        self.assertFalse((self.originals / "experts.bin").exists())

    def test_expert_invalid_key_or_target_fails_before_payload_access(self):
        with assets.OriginalShardExpertStore(self.metadata, self.originals, self.integrity,
                self.budget, layer=39, workspace=self.ws) as store:
            with patch.object(store, "_blob", side_effect=AssertionError("no I/O")):
                with self.assertRaises(ValueError):
                    store.read_into("39:0", np.empty(3, np.uint8))
                with self.assertRaises(KeyError):
                    store.read_into("38:0", np.empty(assets.EXPERT_BYTES, np.uint8))
        self.assertEqual(self.budget.reserved, 0)

    def test_dense_rows_are_caller_owned_bounded_and_no_more_than_two_shards_open(self):
        with assets.OriginalTensorReader(self.metadata, self.originals, self.integrity,
                self.budget, tensor_names=("head.weight", "embed.weight", "control.weight"), workspace=self.ws) as reader:
            target = np.empty((512, 5120), np.uint16)
            with patch.object(assets.VerifiedBlob, "read", side_effect=AssertionError("Spark spans must use bulk I/O")):
                reader.read_into("head.weight", 10240, target)
            self.assertTrue(np.all(target == 0x1111))
            tiny = bytearray(8)
            reader.read_into("embed.weight", 0, tiny)
            self.assertEqual(tiny, b"small sy")
            reader.read_into("control.weight", 0, tiny)
            self.assertEqual(tiny, b"mall syn")
            self.assertEqual(reader.stats()["open_shard_readers"], 2)
            self.assertGreater(reader.stats()["physical_data_bytes_read"], target.nbytes)
            self.assertEqual(reader.stats()["maximum_direct_read_bytes"], 8 * MiB)
        self.assertEqual(self.budget.reserved, 0)

    def test_dense_extent_dtype_alignment_and_size_fail_before_payload_io(self):
        with assets.OriginalTensorReader(self.metadata, self.originals, self.integrity,
                self.budget, workspace=self.ws) as reader:
            with patch.object(reader, "_blob", side_effect=AssertionError("no payload")):
                for offset, target in ((1, bytearray(2)), (0, bytearray(3)), (0, b"read only"),
                                       (8 * MiB, bytearray(2)), (0, bytearray(8 * MiB + 2))):
                    with self.subTest(offset=offset), self.assertRaises(ValueError):
                        reader.read_into("head.weight", offset, target)

    def test_source_block_tamper_prevents_expert_publication_and_releases_budget(self):
        path = self.originals / self.inventory[2]["file"]
        with path.open("r+b", buffering=0) as file:
            file.write(b"X"); file.flush()
        try:
            with assets.OriginalShardExpertStore(self.metadata, self.originals, self.integrity,
                    self.budget, layer=39, workspace=self.ws) as store:
                with self.assertRaisesRegex(ValueError, "block SHA"):
                    store.read_into("39:0", np.empty(assets.EXPERT_BYTES, np.uint8))
                self.assertEqual(store.read_calls, 0)
                self.assertIsNone(store.entries["39:0"]["sha256"])
        finally:
            with path.open("r+b", buffering=0) as file:
                file.write(b"\x11"); file.flush()
        self.assertEqual(self.budget.reserved, 0)

    def test_incomplete_registration_refuses_before_catalog_or_payload(self):
        path = self.integrity / "manifest.json"
        original = path.read_bytes()
        changed = {**self.registration, "entries": self.registration["entries"][:-1]}
        path.write_bytes(assets._json(changed))
        try:
            with patch.object(assets, "load_catalog", side_effect=AssertionError("must stop before catalog")):
                with self.assertRaisesRegex(ValueError, "All original shards"):
                    with assets.OriginalShardExpertStore(self.metadata, self.originals, self.integrity,
                            self.budget, layer=39, workspace=self.ws):
                        self.fail("Incomplete registration cannot qualify arbitrary routes")
        finally:
            path.write_bytes(original)
        self.assertEqual(self.budget.reserved, 0)


if __name__ == "__main__":
    unittest.main()
