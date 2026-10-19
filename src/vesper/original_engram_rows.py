"""All original Engram addresses backed by registered local checkpoint shards."""

import json
from pathlib import Path

from .artifact_repository import VerifiedBlob
from .checkpoint import MODEL,REVISION
from .engram_assets import TABLE_BASES,expected_row_address
from .engram_rows import EngramRows,METADATA_ALLOWANCE
from .model_artifacts import CATALOG_SHA256,SHARD_INVENTORY_SHA256,shard_inventory,digest
from .ngram import TABLE_ROWS,LAYERS


class OriginalShardRows(EngramRows):
    def __init__(self,metadata,originals,integrity,layer,*,cache_rows=32):
        super().__init__(originals,layer,cache_rows=cache_rows)
        self.metadata=Path(metadata); self.integrity=Path(integrity)

    def __enter__(self):
        if self.reader is not None: raise RuntimeError('Original shard rows already open')
        self.budget.acquire(METADATA_ALLOWANCE)
        try:
            path=self.integrity/'manifest.json'
            if path.stat().st_size>65536: raise ValueError('Excessive original registration receipt')
            raw=path.read_bytes(); receipt=json.loads(raw)
            if (receipt.get('kind'),receipt.get('model'),receipt.get('revision'),receipt.get('catalog_sha256'),receipt.get('shard_inventory_sha256'))!=(
                    'registered_original_artifacts',MODEL,REVISION,CATALOG_SHA256,SHARD_INVENTORY_SHA256): raise ValueError('Original registration identity differs')
            inventory={r['file']:r for r in shard_inventory(self.metadata)}
            entries=receipt.get('entries',[])
            if len(entries)!=48 or {r['file'] for r in entries}!=set(inventory): raise ValueError('Incomplete original registration')
            for r in entries:
                if any(r.get(k)!=v for k,v in inventory[r['file']].items()): raise ValueError('Registered original shard identity differs')
            filename=TABLE_BASES[self.layer][0]; entry=next(r for r in entries if r['file']==filename)
            self.reader=VerifiedBlob(self.root/filename,self.integrity,entry,inventory[filename],self.budget)
            self.manifest={'kind':'registered_original_Engram_rows','layer':self.layer,'receipt_sha256':digest(raw),
                'original_shard':inventory[filename],'catalog_sha256':CATALOG_SHA256,'all_original_row_addresses_available':True}
            return self
        except BaseException:
            if self.reader: self.reader.close(); self.reader=None
            self.budget.release(METADATA_ALLOWANCE); raise

    def _contains(self,row): return 0<=row<TABLE_ROWS[LAYERS.index(self.layer)]

    def _load_row(self,row):
        before=self.reader.data.aligned_bytes_requested+self.reader.index.aligned_bytes_requested
        addresses=expected_row_address(self.layer,row)
        raw=b''.join(self.reader.read(e['shard_byte_offset'],e['bytes']) for e in addresses)
        self.physical_bytes+=self.reader.data.aligned_bytes_requested+self.reader.index.aligned_bytes_requested-before
        self.logical_bytes+=len(raw)
        if any((b&127)==127 for b in raw[:256]) or 255 in raw[256:]: raise ValueError('Original full-shard Engram row is nonfinite')
        return raw

    def clear(self):
        super().clear()
        if self.reader is not None: self.reader.clear()

    def stats(self):
        result=super().stats(); result.update(backend='registered_original_shard_blocks',all_original_row_addresses_available=True,
            integrity_block_bytes=65536,index_initialization_bytes=self.reader.index_initialization_bytes if self.reader else 0)
        return result
