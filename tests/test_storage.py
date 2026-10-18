import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from vesper.bandwidth import EXPERT_BYTES
from vesper.checkpoint import MODEL, REVISION
from vesper.resources import MemoryBudget, MiB
from vesper.storage import NativeStore, build_store, tensor_layout
from vesper.tiered import CacheConfig, CudaCacheBackend, TieredExpertRuntime


class StorageTests(unittest.TestCase):
    def manifest(self, payload):
        layout = tensor_layout()
        tensors = []
        for name, region in layout.items():
            start, size = region["offset"], region["bytes"]
            tensors.append({"name": name, "tensor": "layers.0.ffn.experts.0." + name.replace("_", "."),
                            "dtype": "I8" if name.endswith("weight") else "F8_E8M0",
                            "shape": region["shape"], "bytes": size,
                            "sha256": hashlib.sha256(payload[start:start + size]).hexdigest()})
        provenance = {"model": MODEL, "revision": REVISION, "kind": "complete_expert",
                      "layer": 0, "expert": 0, "swiglu_limit": 10.0, "tensors": tensors}
        return {"schema_version": 1, "kind": "vesper_native_expert_store", "model": MODEL, "revision": REVISION,
                "expert_bytes": EXPERT_BYTES, "file": "experts.bin", "file_bytes": EXPERT_BYTES,
                "layout": layout, "entries": [{"key": "0:0", "offset": 0, "bytes": EXPERT_BYTES,
                                               "sha256": hashlib.sha256(payload).hexdigest(), "provenance": provenance}]}

    def test_layout_is_contiguous_and_every_expert_region_is_direct_io_aligned(self):
        cursor = 0
        for region in tensor_layout().values():
            self.assertEqual(region["offset"], cursor)
            self.assertEqual(region["offset"] % 4096, 0)
            self.assertEqual(region["bytes"] % 4096, 0)
            cursor += region["bytes"]
        self.assertEqual(cursor, EXPERT_BYTES)

    def test_metadata_and_payload_identity_are_both_checked(self):
        payload = np.zeros(EXPERT_BYTES, np.uint8)
        manifest = self.manifest(payload)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (root / "experts.bin").open("wb") as stream:
                stream.truncate(EXPERT_BYTES)
            def opened(changed):
                (root / "manifest.json").write_text(json.dumps(changed), encoding="utf-8")
                return NativeStore(root, MemoryBudget(cap=EXPERT_BYTES, available_floor=0))
            store = opened(manifest)
            store.validate("0:0", payload, tensors=True)
            payload[-1] ^= 1
            with self.assertRaisesRegex(ValueError, "store hash"):
                store.validate("0:0", payload, tensors=True)
            payload[-1] ^= 1
            for modify in (
                lambda m: m.update(file="../experts.bin"),
                lambda m: m.update(schema_version=2),
                lambda m: m["entries"][0].update(offset=4096),
                lambda m: m["entries"][0]["provenance"].update(expert=1),
                lambda m: m["entries"].append(copy.deepcopy(m["entries"][0])),
            ):
                changed = copy.deepcopy(manifest)
                modify(changed)
                with self.assertRaises(ValueError):
                    opened(changed)
            changed = copy.deepcopy(manifest)
            changed["entries"][0]["provenance"]["tensors"][0]["sha256"] = "0" * 64
            store = opened(changed)
            with self.assertRaisesRegex(ValueError, "original source hash"):
                store.validate("0:0", payload, tensors=True)

    def test_builder_never_overwrites_existing_store_or_fetches_on_invalid_count(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            marker = target / "keep"
            marker.write_text("existing", encoding="utf-8")
            with patch("vesper.storage.load_expert_fixture") as loader:
                with self.assertRaises(FileExistsError):
                    build_store([Path("unused")], target)
                with self.assertRaises(ValueError):
                    build_store([], target)
                loader.assert_not_called()
            self.assertEqual(marker.read_text(encoding="utf-8"), "existing")

    def test_failed_runtime_entry_releases_store_and_metadata_before_any_gpu(self):
        class Store:
            size = 4
            def __init__(self, path, budget):
                self.budget = budget
            def __enter__(self):
                self.budget.acquire(4)
                return self
            def __exit__(self, *_):
                self.budget.release(4)
        runtime = TieredExpertRuntime(Path("unused"), CacheConfig(host_cap_bytes=MiB + 4, available_floor=0))
        with patch("vesper.tiered.NativeStore", Store), patch("vesper.tiered.ExpertExecutor") as executor:
            with self.assertRaisesRegex(MemoryError, "Host cap"):
                runtime.__enter__()
            executor.assert_not_called()
        self.assertEqual(runtime.budget.reserved, 0)
        self.assertIsNone(runtime.stack)

    def test_reentry_does_not_replace_a_live_resource_stack(self):
        marker = object()
        runtime = TieredExpertRuntime(Path("unused"))
        runtime.stack = marker
        backend = CudaCacheBackend(None, 4, CacheConfig(), None)
        backend.stack = marker
        for resource in (runtime, backend):
            with self.assertRaisesRegex(RuntimeError, "already open"):
                resource.__enter__()
            self.assertIs(resource.stack, marker)


if __name__ == "__main__":
    unittest.main()
