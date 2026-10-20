from copy import deepcopy
import hashlib
import json
import os
import shutil
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from vesper.bandwidth import EXPERT_BYTES
from vesper.disk import DirectReader
from vesper.indexed_store import build_index, IndexedStore, INDEX_METADATA_BYTES
from vesper.resources import MemoryBudget, MiB, GiB
import test_storage


class IndexedStoreTests(unittest.TestCase):
    def setUp(self):
        memory = patch("vesper.resources.physical_memory", return_value={"available":64*GiB})
        memory.start(); self.addCleanup(memory.stop)

    def sources(self, root, count=49):
        # Synthetic full-sized bytes/provenance for I/O capacity, not model
        # weights. Hard links avoid writing the same 18.8 MB forty-nine times.
        payload = np.zeros(EXPERT_BYTES, np.uint8)
        template = test_storage.StorageTests().manifest(payload)
        backing = root / "synthetic.bin"; backing.write_bytes(payload.tobytes())
        sources = []
        for expert in range(count):
            path = root / f"source-{expert:03d}"; path.mkdir()
            try: os.link(backing, path / "experts.bin")
            except OSError: shutil.copyfile(backing, path / "experts.bin")
            m = deepcopy(template); e = m["entries"][0]; e["key"] = f"0:{expert}"; e["provenance"]["expert"] = expert
            for t in e["provenance"]["tensors"]: t["tensor"] = t["tensor"].replace("experts.0.", f"experts.{expert}.")
            (path / "manifest.json").write_text(json.dumps(m), encoding="utf-8"); sources.append(path)
        return sources

    def test_49_experts_reuse_one_buffer_and_validate_actual_direct_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); sources = self.sources(root)
            with patch("vesper.indexed_store.DirectReader", side_effect=AssertionError("Index construction must not read payloads")):
                receipt = build_index(sources, root / "index", root, layer=0)
            self.assertEqual((receipt["experts"], receipt["payload_bytes_copied"]), (49, 0))
            budget = MemoryBudget(cap=96*MiB, available_floor=0); budget.acquire(EXPERT_BYTES)
            target = np.empty(EXPERT_BYTES, np.uint8)
            with IndexedStore(root / "index", root, budget, layer=0) as store:
                self.assertIsNone(store.reader)
                self.assertEqual(budget.reserved, INDEX_METADATA_BYTES + EXPERT_BYTES)
                buffer = None
                for expert in (0, 24, 48, 0):
                    store.read_into(f"0:{expert}", target)
                    self.assertFalse(target.any())
                    if buffer is None: buffer = store.reader.buffer
                    else: self.assertIs(store.reader.buffer, buffer)
                self.assertEqual((store.read_calls, store.file_switches), (4, 4))
                self.assertEqual(budget.reserved, INDEX_METADATA_BYTES + 2*EXPERT_BYTES)
                with self.assertRaises(KeyError): store.read_into("0:99", target)
            self.assertEqual(budget.reserved, EXPERT_BYTES)
            budget.release(EXPERT_BYTES); self.assertEqual(budget.reserved, 0)

    def test_failed_switch_keeps_old_source_readable_and_never_changes_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); sources=self.sources(root,2); build_index(sources,root/"index",root,layer=0)
            budget=MemoryBudget(cap=96*MiB,available_floor=0); target=np.empty(EXPERT_BYTES,np.uint8)
            with IndexedStore(root/"index",root,budget,layer=0) as store:
                store.read_into("0:0",target); reader=store.reader; handle=reader.handle
                with self.assertRaises(OSError): reader.switch_file(root/"missing.bin")
                self.assertEqual(reader.handle,handle)
                target.fill(7)
                with patch.object(reader,"switch_file",side_effect=OSError("owned test failure")):
                    with self.assertRaises(OSError): store.read_into("0:1",target)
                self.assertTrue(np.all(target==7)); self.assertEqual(store.current_source,0)
                store.read_into("0:0",target); self.assertFalse(target.any())
            self.assertEqual(budget.reserved,0)

    def test_corrupt_bytes_and_changed_source_manifest_are_not_published(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); sources=self.sources(root,1); build_index(sources,root/"index",root,layer=0)
            budget=MemoryBudget(cap=96*MiB,available_floor=0)
            with IndexedStore(root/"index",root,budget,layer=0) as store:
                with (sources[0]/"experts.bin").open("r+b") as f: f.seek(EXPERT_BYTES-1); f.write(b'\x01')
                with self.assertRaisesRegex(ValueError,"store hash"): store.read_into("0:0",np.empty(EXPERT_BYTES,np.uint8))
            self.assertEqual(budget.reserved,0)
            path=sources[0]/"manifest.json"; path.write_bytes(path.read_bytes()+b' ')
            with self.assertRaisesRegex(ValueError,"manifest/extent"):
                with IndexedStore(root/"index",root,budget,layer=0): pass
            self.assertEqual(budget.reserved,0)

    def test_invalid_index_paths_extents_and_layer_fail_before_direct_io(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); sources=self.sources(root,1); build_index(sources,root/"index",root,layer=0)
            path=root/"index/manifest.json"; original=json.loads(path.read_bytes()); budget=MemoryBudget(cap=64*MiB,available_floor=0)
            for mutate in (lambda m:m["sources"][0].update(directory="../escape"),
                    lambda m:m["entries"][0].update(offset=4096),lambda m:m["entries"][0].update(source=True),
                    lambda m:m.update(layer=1),lambda m:m["entries"].append(deepcopy(m["entries"][0])),
                    lambda m:m.update(entries=m['entries']*385)):
                m=deepcopy(original); mutate(m); path.write_text(json.dumps(m),encoding="utf-8")
                with self.subTest(change=m), patch("vesper.indexed_store.DirectReader",side_effect=AssertionError("No payload I/O on invalid metadata")):
                    with self.assertRaises(ValueError):
                        with IndexedStore(root/"index",root,budget,layer=0): pass
                self.assertEqual(budget.reserved,0)

    def test_existing_index_and_unavailable_selection_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); sources=self.sources(root,1)
            with self.assertRaises(ValueError): build_index(sources,root/"absent",root,layer=0,experts=[0,1])
            self.assertFalse((root/"absent").exists())
            build_index(sources,root/"index",root,layer=0)
            with self.assertRaises(FileExistsError):build_index(sources,root/"index",root,layer=0)
