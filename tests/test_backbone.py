import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

from vesper.attention_topology import BACKBONE_COMPRESSED_LAYERS
from vesper.catalog_reader import CatalogReader
from vesper.model_artifacts import digest
from vesper.prefix_profile import validate_layer
from vesper.resources import GiB
from vesper.shared_attention import initial_state, validate_frame_layers


class BackboneBindingTests(unittest.TestCase):
    def setUp(self):
        memory = patch("vesper.resources.physical_memory", return_value={"available": 64 * GiB})
        memory.start(); self.addCleanup(memory.stop)

    def test_all_layer_opt_in_preserves_qualified_group_and_layer_guards(self):
        self.assertEqual(validate_frame_layers(BACKBONE_COMPRESSED_LAYERS, binding_profile="backbone_probe"), tuple(range(2, 40)))
        for layers in (tuple(range(2, 39)), tuple(range(3, 40)), tuple(reversed(range(2, 40))), (2, 8, 14, 20)):
            with self.assertRaises(ValueError): validate_frame_layers(layers, binding_profile="backbone_probe")
        with self.assertRaises(ValueError): validate_frame_layers(BACKBONE_COMPRESSED_LAYERS)
        for layer in (4, 14, 39):
            validate_layer(layer, (0, 2), "backbone_probe")
            with self.assertRaises(ValueError): validate_layer(layer, (0, 2))
        for layer in (True, -1, 40):
            with self.assertRaises(ValueError): validate_layer(layer, (0, 2), "backbone_probe")
        state = initial_state(BACKBONE_COMPRESSED_LAYERS, 1, binding_profile="backbone_probe")
        self.assertEqual(set(state["windows"]), set(range(2, 40)))
        self.assertEqual(set(state["compressors"]), {2, 8, 14, 20})
        self.assertEqual(set(state["index_keys"]), {2, 8, 14, 20})

    def catalog(self, root):
        header = json.dumps({"weight": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}}).encode()
        (root / "headers").mkdir()
        (root / "headers/model.safetensors.json").write_bytes(header)
        return header, ({"weight": {}}, {"shards": [{"shard": "model.safetensors", "header_bytes": len(header),
            "header_sha256": digest(header), "file_bytes": len(header) + 12}]})

    def test_checked_detached_metadata_is_reused_and_payload_remains_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); header, loaded = self.catalog(root)
            with patch("vesper.catalog_reader.load_catalog", return_value=loaded), patch("vesper.catalog_reader.read_checked", return_value=b"checked") as small:
                with CatalogReader(root) as reader:
                    with patch("vesper.catalog_reader.read_range", return_value=b"data") as fetch:
                        self.assertEqual(reader.small("config.json"), reader.small("config.json")); small.assert_called_once()
                        self.assertEqual(reader.range("model.safetensors", 0, 8), struct.pack("<Q", len(header)))
                        self.assertEqual(reader.range("model.safetensors", 8, len(header)), header); fetch.assert_not_called()
                        self.assertEqual(reader.range("model.safetensors", 8 + len(header), 4), b"data"); fetch.assert_called_once()
                        for shard, start, size in (("other", 0, 8), ("model.safetensors", 9, 1), ("model.safetensors", 8 + len(header), 5),
                                ("model.safetensors", True, 4), ("model.safetensors", 0, 8 * 1024**2 + 1)):
                            with self.assertRaises(ValueError): reader.range(shard, start, size)
                    self.assertEqual(reader.stats()["payload_bytes_fetched"], 4)
                self.assertEqual(reader.budget.reserved, 0)
                with self.assertRaises(RuntimeError): reader.small("config.json")

    def test_changed_header_and_permanent_http_errors_are_not_retried_or_admitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); header, loaded = self.catalog(root)
            with patch("vesper.catalog_reader.load_catalog", return_value=loaded), CatalogReader(root) as reader:
                (root / "headers/model.safetensors.json").write_bytes(b"changed")
                with patch("vesper.catalog_reader.read_range") as fetch:
                    with self.assertRaises(ValueError): reader.range("model.safetensors", 0, 8)
                    fetch.assert_not_called()
                error = urllib.error.HTTPError("https://example.invalid", 503, "temporary", {}, None)
                with patch("vesper.catalog_reader.read_range", side_effect=[error, b"data"]) as fetch, patch("vesper.catalog_reader.time.sleep") as sleep:
                    self.assertEqual(reader.range("model.safetensors", 8 + len(header), 4), b"data")
                    self.assertEqual(fetch.call_count, 2); sleep.assert_called_once_with(1)
                error = urllib.error.HTTPError("https://example.invalid", 404, "missing", {}, None)
                with patch("vesper.catalog_reader.read_range", side_effect=error) as fetch, patch("vesper.catalog_reader.time.sleep") as sleep:
                    with self.assertRaises(urllib.error.HTTPError): reader.range("model.safetensors", 8 + len(header), 4)
                    fetch.assert_called_once(); sleep.assert_not_called()
                self.assertEqual(reader.stats()["payload_bytes_fetched"], 4)


if __name__ == "__main__": unittest.main()
