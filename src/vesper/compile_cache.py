"""Opt-in process-local reuse of unchanged NVRTC output, with bounded ownership.

This caches immutable cubin bytes, not CUDA modules, handles or model weights.
The source/toolchain is sealed for the context's lifetime. A change refuses the
run; no persistent cache or cross-toolchain binary reuse is implemented here.
"""

from contextvars import ContextVar
from copy import deepcopy
from functools import wraps
import hashlib
import inspect
import os
from pathlib import Path
import threading

from .model_artifacts import canonical
from .resources import MemoryBudget, MiB

_ACTIVE = ContextVar('vesper_compilation_cache', default=None)


def toolchain_files():
    if os.name == 'nt':
        roots = [Path(v) for k in ('CUDA_PATH', 'CUDA_HOME') if (v := os.environ.get(k))]
        roots += sorted(Path('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA').glob('v*'), reverse=True)
        root = next((p for p in roots if list((p/'bin').glob('nvrtc64_*.dll'))), None)
        if root is None: raise RuntimeError('NVRTC toolkit was not found')
        binaries = sorted((root/'bin').glob('nvrtc*.dll'))
    else:
        root = Path(os.environ.get('CUDA_HOME', '/usr/local/cuda'))
        # The loader may select a system library. Include its resolved filename
        # as well as the toolkit headers; reject an unidentifiable library.
        import ctypes.util
        selected = ctypes.util.find_library('nvrtc')
        candidates = [root/'lib64'/selected] if selected else []
        candidates += list((root/'lib64').glob('libnvrtc*.so*'))
        candidates += list(Path('/usr/lib/x86_64-linux-gnu').glob('libnvrtc*.so*'))
        binaries = [p for p in candidates if p.is_file()]
    if not binaries or not (root/'include').is_dir(): raise RuntimeError('Cannot seal the NVRTC binary/header inputs')
    files = [p for p in (root/'include').rglob('*') if p.is_file()]
    files += binaries
    files += list((Path(__file__).parent/'kernels').glob('*.cu'))
    files += [Path(__file__).with_name('cuda.py'), Path(__file__)]
    return sorted(set(p.resolve(strict=True) for p in files))


def _sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(MiB), b''): h.update(chunk)
    return h.hexdigest()


def cached_compilation(function):
    signature = inspect.signature(function)
    @wraps(function)
    def call(*args, **kwargs):
        cache = _ACTIVE.get()
        if cache is None: return function(*args, **kwargs)
        bound = signature.bind(*args, **kwargs); bound.apply_defaults()
        return cache.compile(function, bound.arguments)
    return call


class CompilationCache:
    """One explicit compile scope, at most 32 entries / 32 MiB cubin payload."""
    def __init__(self):
        self.entries = {}; self.images_bytes = self.hits = self.misses = 0
        self.token = None; self.lock = threading.RLock(); self.budget = None

    def __enter__(self):
        if self.token is not None or _ACTIVE.get() is not None: raise RuntimeError('Nested compilation caches are not supported')
        # Cover the 32 MiB retained images, the compiler's bounded image copy
        # during a miss, and source/entry metadata with explicit headroom.
        self.budget = MemoryBudget(cap=96*MiB, owner='process_local_compilation_cache')
        self.budget.acquire(self.budget.cap, category='metadata')
        try:
            files = toolchain_files()
            if not 1 <= len(files) <= 8192: raise ValueError('Unbounded compiler input inventory')
            self.inputs = {p: (p.stat().st_size, p.stat().st_mtime_ns, _sha(p)) for p in files}
            self.environment = tuple(os.environ.get(k) for k in ('CUDA_PATH','CUDA_HOME','LD_LIBRARY_PATH'))
            self.input_sha256 = canonical([{'ordinal':i,'bytes':v[0],'sha256':v[2]} for i,v in enumerate(self.inputs.values())])
            self.token = _ACTIVE.set(self)
            return self
        except BaseException:
            self.budget.release(self.budget.cap, category='metadata'); self.budget = None
            raise

    def _check(self, *, content=False):
        if self.environment != tuple(os.environ.get(k) for k in ('CUDA_PATH','CUDA_HOME','LD_LIBRARY_PATH')):
            raise ValueError('Compiler selection changed inside the sealed cache scope')
        if content and toolchain_files() != list(self.inputs): raise ValueError('Compiler input inventory changed')
        for p, (size, stamp, sha) in self.inputs.items():
            # A cache hit executes the already sealed immutable binary and
            # never invokes the compiler/headers. Check all inputs by content
            # on every miss and at close, plus source/binary stamps on hits.
            # No successful capture is published before the final full seal.
            if not content and p.suffix not in ('.cu','.py','.dll') and 'libnvrtc' not in p.name: continue
            actual = p.stat()
            if (actual.st_size, actual.st_mtime_ns) != (size, stamp) or (content and _sha(p) != sha):
                raise ValueError('Compiler source, binary or header changed inside the sealed scope')

    def compile(self, function, arguments):
        with self.lock:
            if self.token is None: raise RuntimeError('Compilation cache is closed')
            self._check()
            values = dict(arguments); info = values.pop('compiler_info', None)
            key = canonical({'inputs':self.input_sha256,'function':function.__module__+'.'+function.__qualname__,'request':values})
            if key in self.entries:
                image, metadata, sha = self.entries[key]
                if hashlib.sha256(image).hexdigest() != sha: raise ValueError('Cached cubin integrity differs')
                self.hits += 1
            else:
                if len(self.entries) >= 32: raise MemoryError('Compilation cache entry bound exceeded')
                metadata = {}; image = function(**values, compiler_info=metadata)
                if type(image) is not bytes or not image or self.images_bytes + len(image) > 32*MiB:
                    raise MemoryError('Compilation cache cubin payload bound exceeded')
                self._check(content=True)
                sha = hashlib.sha256(image).hexdigest()
                self.entries[key] = (image, deepcopy(metadata), sha)
                self.images_bytes += len(image); self.misses += 1
            if info is not None: info.update(deepcopy(metadata))
            return image

    def stats(self):
        return {'kind':'process_local_sealed_NVRTC_cubins','hits':self.hits,'misses':self.misses,
            'entries':len(self.entries),'cubin_bytes':self.images_bytes,'input_identity_sha256':getattr(self,'input_sha256',None),
            'host_reserved_bytes':self.budget.reserved if self.budget else 0,'persistent_binary_cache':False,
            'live_CUDA_modules_cached':False}

    def manifest(self):
        return [{'request_sha256':key,'bytes':len(image),'sha256':sha,'compiler':deepcopy(metadata)}
            for key,(image,metadata,sha) in sorted(self.entries.items())]

    def __exit__(self, *_):
        try: self._check(content=True)
        finally:
            _ACTIVE.reset(self.token); self.token = None; self.entries.clear(); self.images_bytes = 0
            self.budget.release(self.budget.cap, category='metadata'); self.budget = None
