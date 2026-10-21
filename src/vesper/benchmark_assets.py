"""Bounded original-range reader with verified reuse of existing input rows."""

import json
from pathlib import Path
import threading
import time

from .catalog_reader import CatalogReader
from .checkpoint import MODEL, REVISION
from .engram_assets import expected_row_address
from .model_artifacts import digest
from .resources import MiB


class MissingBenchmarkAsset(RuntimeError): pass


class BenchmarkReader(CatalogReader):
    def __init__(self, root, deadline, *, maximum_fetch_bytes=0):
        super().__init__(Path(root) / 'import-source')
        if type(maximum_fetch_bytes) is not int or not 0 <= maximum_fetch_bytes <= 1024 * MiB:
            raise ValueError('Expected a bounded explicit original-range import allowance')
        self.root, self.deadline, self.maximum_fetch_bytes = Path(root), deadline, maximum_fetch_bytes
        self.cached, self.pending, self.reused_bytes = {}, 0, 0
        self.import_lock = threading.RLock()

    def check(self):
        if time.perf_counter() >= self.deadline:
            raise TimeoutError('Benchmark source deadline reached')

    def __enter__(self):
        super().__enter__()
        try:
            directories = [self.root / 'engram-fixture']
            directories += sorted((self.root / 'pc-heterogeneous/trajectories').glob('*'))
            directories += [self.root / 'generation/story-v1']
            for directory in directories:
                self.check()
                for metadata in sorted((directory / 'embeddings').glob('*.json')):
                    if metadata.is_symlink() or metadata.stat().st_size > 4096:
                        raise ValueError('Invalid cached embedding metadata')
                    entry = json.loads(metadata.read_bytes()); source = entry['source']
                    if (not isinstance(source, dict) or source.get('tensor') != 'embed.weight'
                            or source.get('bytes') != 10240):
                        raise ValueError('Different cached embedding source')
                    data = metadata.with_suffix('.bin')
                    if data.is_symlink() or data.stat().st_size != 10240:
                        raise ValueError('Cached embedding extent differs')
                    key = source['shard'], source['shard_byte_offset'], source['bytes']
                    self.cached.setdefault(key, (data, 0, entry['sha256'], None))
                for layer in (1, 14):
                    path = directory / f'rows/layer{layer}/manifest.json'
                    if not path.is_file(): continue
                    if path.is_symlink() or path.stat().st_size > 12 * MiB:
                        raise ValueError('Invalid cached Engram manifest')
                    m = json.loads(path.read_bytes())
                    if (m.get('kind'), m.get('model'), m.get('revision'), m.get('layer')) != (
                            'original_engram_selected_rows', MODEL, REVISION, layer):
                        raise ValueError('Different cached original row identity')
                    if not 0 < len(m['rows']) <= 12288:
                        raise ValueError('Unbounded cached row inventory')
                    data = path.parent / 'rows.bin'
                    if data.is_symlink() or data.stat().st_size != m['file_bytes']:
                        raise ValueError('Cached Engram extent differs')
                    for row in m['rows']:
                        if (row['source'] != expected_row_address(layer, row['row'])
                                or type(row['offset']) is not int or row['offset'] < 0
                                or row['offset'] + 264 > m['file_bytes']):
                            raise ValueError('Cached original row address differs')
                        for index, source in enumerate(row['source']):
                            key = source['shard'], source['shard_byte_offset'], source['bytes']
                            self.cached.setdefault(key, (data, row['offset'], row['sha256'], index))
            return self
        except BaseException:
            super().__exit__(); raise

    def range(self, shard, start, size):
        self.check()
        key = shard, start, size
        if key in self.cached:
            path, offset, sha, part = self.cached[key]
            extent = 10240 if part is None else 264
            with path.open('rb') as stream:
                stream.seek(offset); raw = stream.read(extent)
            if len(raw) != extent or digest(raw) != sha:
                raise ValueError('Cached original row payload changed')
            with self.import_lock: self.reused_bytes += size
            return raw if part is None else raw[:256] if part == 0 else raw[256:]
        meta = self.shards.get(shard)
        if meta is not None and (start, size) in ((0, 8), (8, meta['header_bytes'])):
            return super().range(shard, start, size)
        with self.import_lock:
            if self.payload_bytes + self.pending + size > self.maximum_fetch_bytes:
                raise MissingBenchmarkAsset('Original range unavailable within the explicit import allowance')
            self.pending += size
        try:
            # CatalogReader retains pinned address/header/transport validation.
            # Its bounded transport retries and in-flight requests may overrun
            # the cooperative deadline; this is not an OS timeout guarantee.
            return super().range(shard, start, size)
        finally:
            with self.import_lock: self.pending -= size

    def stats(self):
        return {**super().stats(), 'verified_input_row_reuse_bytes': self.reused_bytes,
                'maximum_successful_payload_fetch_bytes': self.maximum_fetch_bytes,
                'pending_requested_bytes': self.pending, 'cached_original_range_count': len(self.cached)}

    def __exit__(self, *_):
        self.cached.clear()
        super().__exit__()
