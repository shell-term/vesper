"""Bounded preparation-only imports into immutable original expert objects.

Previously packed experts are referenced in place. New experts are downloaded
once, checked by the existing importer, packed once and atomically published.
The temporary NPZ is removed by its owning TemporaryDirectory. Offline readers
have no importer or HTTP fallback.
"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import json
import os
from pathlib import Path
import tempfile
import threading

import numpy as np

from .bandwidth import EXPERT_BYTES
from .checkpoint import fetch_expert_fixture, validate_expert_identity
from .generation_assets import prefetch_expert, reserve_disk
from .indexed_store import build_index
from .model_artifacts import digest
from .resources import GiB, MiB, MemoryBudget
from .storage import NativeStore, build_store, tensor_layout


class ExpertBank:
    """One anchored source inventory, with a 64 GiB new packed-payload cap."""
    def __init__(self, root, directory, sources=(), *, new_payload_cap=64*GiB):
        self.root = Path(root).resolve(strict=True); self.directory = Path(directory).resolve()
        if not self.directory.is_relative_to(self.root) or type(new_payload_cap) is not int or not EXPERT_BYTES <= new_payload_cap <= 64*GiB:
            raise ValueError('Expected an anchored expert bank and bounded import cap')
        self.cap = new_payload_cap; self.sources = tuple(sources); self.entries = {}; self.receipts = {}
        self.new_bytes = self.pending_bytes = self.fetched_bytes = self.imports = 0
        self.lock = threading.RLock(); self.budget = None

    def __enter__(self):
        if self.budget is not None: raise RuntimeError('Expert bank is already open')
        self.budget = MemoryBudget(cap=256*MiB, owner='original_expert_bank_inventory'); self.budget.acquire(self.budget.cap, category='metadata')
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            paths = list(self.sources) + sorted(self.directory.glob('layer*/expert-*'))
            if len(paths)>15360+80: raise ValueError('Excessive expert-bank sources')
            for path in paths: self._register(path, owned=Path(path).resolve().is_relative_to(self.directory))
            if self.new_bytes>self.cap: raise MemoryError('Existing new expert objects exceed the explicit bank cap')
            return self
        except BaseException:
            self.__exit__(); raise

    def _register(self, directory, *, owned):
        path = Path(directory).resolve(strict=True)
        if not path.is_relative_to(self.root): raise ValueError('Expert source escapes the declared root')
        if not (path/'experts.bin').resolve(strict=True).is_relative_to(self.root):raise ValueError('Expert payload escapes the declared root')
        metadata=path/'manifest.json'
        if not 1<=metadata.stat().st_size<=MiB: raise ValueError('Excessive expert bank source metadata')
        with metadata.open('rb') as stream:raw=stream.read(MiB+1)
        if len(raw)>MiB:raise ValueError('Expert metadata grew beyond its bound')
        source = NativeStore(path, self.budget, max_experts=48)
        if source.manifest!=json.loads(raw):raise ValueError('Expert source changed during validation')
        receipt = {'manifest_sha256':digest(raw),'file_bytes':source.manifest['file_bytes']}
        if path in self.receipts:
            if self.receipts[path]!=receipt: raise ValueError('An immutable expert source changed')
            return
        for key, entry in source.entries.items():
            if key in self.entries:
                old = self.entries[key][1]
                if any(old[n]!=entry[n] for n in ('bytes','sha256','provenance')): raise ValueError('Conflicting original expert identity')
            else: self.entries[key]=(path,entry)
        self.receipts[path]=receipt
        if owned: self.new_bytes+=source.manifest['file_bytes']

    def has(self, layer, expert):
        validate_expert_identity(layer,expert)
        return f'{layer}:{expert}' in self.entries

    def prepare(self, layer, experts, reader, transfer_pool):
        if self.budget is None: raise RuntimeError('Expert bank is closed')
        ids=sorted(set(experts))
        if not 1<=len(ids)<=6: raise ValueError('Prepare exactly the actual bounded top-six route')
        for expert in ids: validate_expert_identity(layer,expert)
        missing=[e for e in ids if not self.has(layer,e)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda e:self._import(layer,e,reader,transfer_pool),missing))

    def _import(self, layer, expert, reader, transfer_pool):
        with self.lock:
            if self.has(layer,expert): return
            if self.new_bytes+self.pending_bytes+EXPERT_BYTES>self.cap: raise MemoryError('New original expert payload cap reached')
            reserve_disk(self.directory, 8*EXPERT_BYTES)
            self.pending_bytes+=EXPERT_BYTES
        try:
            with ExitStack() as stack:
                allowance=MemoryBudget(cap=128*MiB, owner='bank_single_expert_import');allowance.acquire(allowance.cap)
                stack.callback(allowance.release,allowance.cap)
                staging=self.directory/'staging';staging.mkdir(exist_ok=True)
                temporary=Path(stack.enter_context(tempfile.TemporaryDirectory(prefix='expert-',dir=staging)))
                with prefetch_expert(reader,layer,expert,transfer_pool) as prepared:
                    provenance=fetch_expert_fixture(temporary/'fixture',layer,expert,reader=prepared)
                build_store([temporary/'fixture/native-expert.npz'],temporary/'packed',max_experts=1)
                target=self.directory/f'layer{layer}/expert-{expert:03d}';target.parent.mkdir(exist_ok=True)
                if not target.parent.resolve().is_relative_to(self.directory) or not (temporary/'packed').resolve().is_relative_to(staging.resolve()):
                    raise ValueError('Expert publication escaped its owning directories')
                with self.lock:
                    if target.exists(): raise FileExistsError('Refusing to replace an original expert object')
                    os.rename(temporary/'packed',target)
                    self._register(target,owned=True)
                    self.fetched_bytes+=provenance['weights_fetched_bytes'];self.imports+=1
        finally:
            with self.lock:self.pending_bytes-=EXPERT_BYTES

    def index(self, layer, experts, directory):
        ids=sorted(set(experts))
        if not ids or any(not self.has(layer,e) for e in ids): raise ValueError('Unprepared actual expert route')
        paths=sorted({self.entries[f'{layer}:{e}'][0] for e in ids})
        return build_index(paths,directory,self.root,layer=layer,experts=ids)

    def arrays(self, layer, expert):
        """Offline exact source for original math; no fixture conversion or fetch."""
        if self.budget is None: raise RuntimeError('Expert bank is closed')
        path, entry=self.entries[f'{layer}:{expert}']
        if not (path/'experts.bin').resolve(strict=True).is_relative_to(self.root):raise ValueError('Expert payload escaped before read')
        if digest((path/'manifest.json').read_bytes())!=self.receipts[path]['manifest_sha256']:
            raise ValueError('Immutable original expert source changed before read')
        budget=MemoryBudget(cap=96*MiB,owner='original_expert_bank_read');budget.acquire(EXPERT_BYTES,category='scratch')
        try:
            data=np.empty(EXPERT_BYTES,np.uint8)
            with NativeStore(path,budget,max_experts=48) as source:
                source.read_into(f'{layer}:{expert}',data)
                source._validate(f'{layer}:{expert}',data,tensors=True)
            # The reference caller's fixture admission owns the returned arrays.
            arrays={n:data[v['offset']:v['offset']+v['bytes']].reshape(v['shape']) for n,v in tensor_layout().items()}
            return arrays,entry['provenance']
        finally:budget.release(EXPERT_BYTES,category='scratch')

    def stats(self):
        return {'kind':'immutable_original_expert_bank','available_experts':len(self.entries),'source_files':len(self.receipts),
            'new_packed_payload_bytes':self.new_bytes,'new_packed_payload_cap_bytes':self.cap,'pending_payload_bytes':self.pending_bytes,
            'expert_payload_bytes_fetched_this_process':self.fetched_bytes,'expert_imports_this_process':self.imports,
            'host_reserved_bytes':self.budget.reserved if self.budget else 0,'persistent_NPZ_copies':False}

    def __exit__(self,*_):
        if self.pending_bytes: raise RuntimeError('Expert imports are still in flight')
        self.entries.clear();self.receipts.clear()
        if self.budget is not None:self.budget.release(self.budget.cap,category='metadata');self.budget=None


def existing_sources(root):
    root=Path(root);paths=set()
    for name in ('backbone-plan.json','generation/story-v1/plan.json'):
        for e in json.loads((root/name).read_bytes())['experts'].values():
            path=(root/e['store']).resolve(strict=True)
            if not path.is_relative_to(root.resolve()):raise ValueError('Historical store escapes its source root')
            paths.add(path)
    return sorted(paths)
