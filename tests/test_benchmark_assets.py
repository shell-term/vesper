import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from vesper.benchmark_assets import BenchmarkReader, MissingBenchmarkAsset
from vesper.checkpoint import MODEL, REVISION
from vesper.engram_assets import expected_row_address
from vesper.model_artifacts import digest
from vesper.resources import GiB


class BenchmarkAssetTests(unittest.TestCase):
    def setUp(self):
        memory = patch('vesper.resources.physical_memory', return_value={'available': 64 * GiB})
        memory.start(); self.addCleanup(memory.stop)
        catalog = patch('vesper.catalog_reader.load_catalog', return_value=({}, {'shards': []}))
        catalog.start(); self.addCleanup(catalog.stop)

    def rows(self, root):
        directory = root / 'engram-fixture/rows/layer1'; directory.mkdir(parents=True)
        raw = b'\x38' * 256 + b'\x7f' * 8
        (directory / 'rows.bin').write_bytes(raw + bytes(4096 - 264))
        source = expected_row_address(1, 0)
        m = {'kind': 'original_engram_selected_rows', 'model': MODEL, 'revision': REVISION,
             'layer': 1, 'file_bytes': 4096, 'rows': [{'row': 0, 'offset': 0,
             'source': source, 'sha256': digest(raw)}]}
        (directory / 'manifest.json').write_text(json.dumps(m))
        return directory, source, raw

    def test_verified_row_ranges_reused_without_network_and_corruption_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); directory, source, raw = self.rows(root)
            with BenchmarkReader(root, time.perf_counter() + 60) as reader:
                with patch('vesper.catalog_reader.read_range', side_effect=AssertionError('Network called')):
                    chunks = [reader.range(s['shard'], s['shard_byte_offset'], s['bytes']) for s in source]
                self.assertEqual(b''.join(chunks), raw)
                self.assertEqual(reader.reused_bytes, 264)
                (directory / 'rows.bin').write_bytes(bytes(4096))
                with self.assertRaisesRegex(ValueError, 'payload changed'):
                    s = source[0]; reader.range(s['shard'], s['shard_byte_offset'], s['bytes'])

    def test_missing_asset_and_expired_deadline_cannot_start_a_fetch(self):
        with tempfile.TemporaryDirectory() as tmp:
            with BenchmarkReader(Path(tmp), time.perf_counter() + 60) as reader:
                with patch('vesper.catalog_reader.read_range', side_effect=AssertionError('Network called')):
                    with self.assertRaises(MissingBenchmarkAsset): reader.range('missing', 100, 1)
                    reader.deadline = 0
                    with self.assertRaises(TimeoutError): reader.range('missing', 100, 1)

    def test_wrong_original_row_source_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); directory, _, _ = self.rows(root)
            path = directory / 'manifest.json'; m = json.loads(path.read_text())
            m['rows'][0]['source'][0]['shard_byte_offset'] += 1
            path.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError, 'address differs'):
                with BenchmarkReader(root, time.perf_counter() + 60): pass

    def test_fetch_allowance_must_be_explicit_and_bounded(self):
        for value in (-1, True, 1024**3 + 1):
            with self.assertRaises(ValueError): BenchmarkReader(Path('.'), 1., maximum_fetch_bytes=value)
