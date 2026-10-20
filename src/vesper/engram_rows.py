"""Bounded, integrity-checked Engram row LRU over uncached aligned disk reads."""

from collections import OrderedDict
import json
from pathlib import Path
import threading

import numpy as np

from .checkpoint import MODEL, REVISION
from .disk import DirectReader
from .engram_assets import MAX_ROWS,expected_row_address
from .model_artifacts import CATALOG_SHA256, digest
from .ngram import LAYERS, TABLE_ROWS
from .resources import MemoryBudget, MiB

ENTRY_ALLOWANCE=520  # 264 original bytes plus per-entry Python/key/LRU allowance
METADATA_ALLOWANCE=6*MiB


class EngramRows:
    def __init__(self,directory,layer,*,cache_rows=32,max_rows=MAX_ROWS):
        from .sequence import MAX_ROWS as sequence_rows
        if type(max_rows) is not int or max_rows not in (MAX_ROWS,sequence_rows): raise ValueError("Unknown explicit Engram row profile")
        if type(layer) is not int or layer not in LAYERS or type(cache_rows) is not int or not 1<=cache_rows<=128:
            raise ValueError("Invalid bounded Engram row-cache profile")
        self.root=Path(directory); self.layer=layer; self.cache_rows=cache_rows
        self.max_rows=max_rows; self.metadata_allowance=METADATA_ALLOWANCE if max_rows==MAX_ROWS else 64*MiB
        self.metadata_limit=MiB if max_rows==MAX_ROWS else 12*MiB
        self.budget=MemoryBudget(cap=self.metadata_allowance+2*MiB); self.cache=OrderedDict(); self.lock=threading.RLock(); self.reader=None
        self.entries={}; self.manifest=None
        self._memory_reclaimer=None
        self.hits=self.misses=self.physical_bytes=self.logical_bytes=0

    def __enter__(self):
        with self.lock:
            if self.reader is not None: raise RuntimeError("Engram rows are already open")
            self.budget.acquire(self.metadata_allowance,category='metadata')
            try:
                path=self.root/'manifest.json'
                if path.stat().st_size>self.metadata_limit: raise ValueError("Excessive selected-row metadata")
                with path.open('rb') as f: raw=f.read(self.metadata_limit+1)
                if len(raw)>self.metadata_limit: raise ValueError("Selected-row metadata grew beyond its bound")
                m=json.loads(raw); rows=m.get('rows',[])
                if (m.get('kind'),m.get('model'),m.get('revision'),m.get('catalog_sha256'),m.get('layer'),m.get('record_bytes'),
                        m.get('maximum_rows'),m.get('file_bytes'))!=('original_engram_selected_rows',MODEL,REVISION,CATALOG_SHA256,self.layer,512,self.max_rows,
                            ((len(rows)*512+4095)//4096)*4096) or not 1<=len(rows)<=self.max_rows:
                    raise ValueError("Original Engram selected-row identity differs")
                if [r['row'] for r in rows]!=sorted(set(r['row'] for r in rows)) or any(
                        type(r['row']) is not int or not 0<=r['row']<TABLE_ROWS[LAYERS.index(self.layer)] or r['offset']!=i*512
                        or any(type(r.get(k)) is not str or len(r[k])!=64 or any(c not in '0123456789abcdef' for c in r[k])
                            for k in ('sha256','weight_sha256','scale_sha256')) for i,r in enumerate(rows)):
                    raise ValueError("Original row addresses, extents or hashes differ")
                if any(r.get('source')!=expected_row_address(self.layer,r['row']) for r in rows):
                    raise ValueError('Original Engram row source address differs')
                self.entries={r['row']:r for r in rows}; self.manifest=m
                if (self.root/'rows.bin').stat().st_size!=m['file_bytes']: raise ValueError("Selected-row store byte count differs")
                self.reader=DirectReader(self.root/'rows.bin',4096,self.budget)
                self._register_pressure()
                return self
            except BaseException:
                if self.reader: self.reader.close(); self.reader=None
                self.entries={}; self.manifest=None
                self.budget.release(self.metadata_allowance,category='metadata'); raise

    def _register_pressure(self):
        self._memory_reclaimer=self.budget.register_reclaimer(self.reclaim)

    def reclaim(self,requested):
        if not self.lock.acquire(blocking=False): return 0
        try:
            before=self.budget.reserved; self.clear(); return before-self.budget.reserved
        finally: self.lock.release()

    def clear(self):
        with self.lock:
            self.budget.release(len(self.cache)*ENTRY_ALLOWANCE,category='engram'); self.cache.clear()

    def _headroom(self):
        try: self.budget.acquire(0)
        except MemoryError:
            self.clear(); self.budget.acquire(0)

    def _contains(self,row): return row in self.entries

    def _load_row(self,row):
        entry=self.entries[row]; offset=entry['offset']
        page=self.reader.read((offset//4096)*4096,4096); self.physical_bytes+=4096
        raw=page[offset%4096:offset%4096+264]; self.logical_bytes+=264
        if (digest(raw)!=entry['sha256'] or digest(raw[:256])!=entry['weight_sha256'] or digest(raw[256:])!=entry['scale_sha256']):
            raise ValueError("Original Engram row SHA-256 differs; refusing cache publication")
        return raw

    def get(self,ids):
        with self.lock:
            if self.reader is None: raise RuntimeError("Engram rows are not open")
            if (not isinstance(ids,np.ndarray) or ids.dtype!=np.int64 or ids.ndim!=3 or ids.shape[-1]!=24
                    or ids.shape[0] not in (1,2) or not 1<=ids.shape[1]<=3 or not ids.flags.c_contiguous
                    or any(not self._contains(int(i)) for i in ids.reshape(-1))):
                raise ValueError("Expected available bounded original Engram row IDs")
            self._headroom()
            codes=np.empty((*ids.shape,256),np.uint8); scales=np.empty((*ids.shape,8),np.uint8)
            for i,row in enumerate(map(int,ids.reshape(-1))):
                if row in self.cache:
                    raw=self.cache.pop(row); self.cache[row]=raw; self.hits+=1
                else:
                    self._headroom(); raw=self._load_row(row)
                    self.misses+=1
                    if len(self.cache)==self.cache_rows:
                        self.cache.popitem(last=False); self.budget.release(ENTRY_ALLOWANCE,category='engram')
                    self.budget.acquire(ENTRY_ALLOWANCE,category='engram'); self.cache[row]=raw
                codes.reshape(-1,256)[i]=np.frombuffer(raw[:256],np.uint8)
                scales.reshape(-1,8)[i]=np.frombuffer(raw[256:],np.uint8)
            return codes,scales

    def stats(self):
        return {'host_allowance_bytes':self.budget.cap,'host_reserved_bytes':self.budget.reserved,'host_peak_bytes':self.budget.peak,
            'cache_capacity_rows':self.cache_rows,'cached_rows':len(self.cache),'hits':self.hits,'misses':self.misses,
            'logical_original_payload_bytes_read':self.logical_bytes,'aligned_physical_bytes_read':self.physical_bytes,
            'physical_alignment_bytes':4096,'model_wide_mapping':False,'buffered_row_read_fallback':False,
            'reservation_is_RSS_limit':False}

    def __exit__(self,*_):
        with self.lock:
            self.budget.unregister_reclaimer(self._memory_reclaimer); self._memory_reclaimer=None
            self.clear()
            if self.reader is not None:
                self.reader.close(); self.reader=None; self.budget.release(self.metadata_allowance,category='metadata')
                self.entries={}; self.manifest=None
