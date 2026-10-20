"""Explicit PC expert-bank mounts; indexes retain their logical relative names.

Only two named PC banks can be mounted. Paths are private local configuration;
public receipts contain marker/config digests, never physical directories.
No links, implicit fallback, payload conversion or automatic bank rollover.
"""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import stat
import uuid

from .checkpoint import MODEL, REVISION
from .model_artifacts import digest
from .resources import GiB

CONFIG = 'pc-heterogeneous/source-roots.json'
MARKER = '.vesper-expert-bank.json'
BANKS = ('pc-heterogeneous/expert-bank-v1', 'pc-heterogeneous/expert-bank-v2')
MAX_CONFIG = 64 * 1024
_active = None


def _plain(path):
    path = Path(os.path.abspath(path))
    for node in (*reversed(path.parents), path):
        try: info = node.lstat()
        except FileNotFoundError: continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('Expert mounts cannot contain symlinks or reparse points')
    return path


def _read(path, limit):
    path = _plain(path)
    if not path.is_file() or not 1 <= path.stat().st_size <= limit:
        raise ValueError('Missing or excessive expert-root metadata')
    raw = path.read_bytes()
    if len(raw) > limit: raise ValueError('Expert-root metadata grew beyond its bound')
    def unique(pairs):
        result = {}
        for k, v in pairs:
            if k in result: raise ValueError('Duplicate expert-root metadata key')
            result[k] = v
        return result
    return json.loads(raw, object_pairs_hook=unique), raw


def marker(prefix, store_id=None):
    if prefix not in BANKS: raise ValueError('Unregistered logical PC expert bank')
    return {'schema_version': 1, 'kind': 'vesper_owned_pc_expert_bank',
        'platform': 'pc-heterogeneous', 'profile': 'native', 'model': MODEL, 'revision': REVISION,
        'logical_prefix': prefix, 'store_id': store_id or uuid.uuid4().hex,
        'packed_payload_cap_bytes': 64 * GiB}


def _valid_marker(value, prefix):
    if (type(value) is not dict or type(value.get('schema_version')) is not int
            or type(value.get('store_id')) is not str or type(value.get('packed_payload_cap_bytes')) is not int
            or not re.fullmatch(r'[0-9a-f]{32}', value['store_id'])
            or value != marker(prefix, value['store_id'])):
        raise ValueError('Expert-root ownership/model/profile marker differs')


def _name(name):
    if (type(name) is not str or not 1 <= len(name) <= 512 or '\\' in name or ':' in name
            or name.startswith('/') or any(p in ('', '.', '..') for p in name.split('/'))):
        raise ValueError('Expected a bounded logical relative expert path')
    return name


class SourceRoots:
    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)
        self.mounts = {}; self.config_sha256 = None; self.active_bank = BANKS[0]
        path = self.root / CONFIG
        if path.exists() or path.is_symlink():
            c, raw = _read(path, MAX_CONFIG)
            if (set(c) != {'schema_version', 'kind', 'platform', 'profile', 'model', 'revision', 'active_bank', 'mounts'}
                    or type(c['schema_version']) is not int or c['schema_version'] != 1
                    or (c['kind'], c['platform'], c['profile'], c['model'], c['revision']) != (
                        'vesper_pc_source_roots', 'pc-heterogeneous', 'native', MODEL, REVISION)
                    or type(c['mounts']) is not list or not 1 <= len(c['mounts']) <= 2):
                raise ValueError('Different or unbounded PC source-root configuration')
            for m in c['mounts']:
                if type(m) is not dict or set(m) != {'logical_prefix', 'physical_directory', 'marker_sha256'}:
                    raise ValueError('Invalid explicit expert-root descriptor')
                prefix = m['logical_prefix']
                if prefix not in BANKS or prefix in self.mounts: raise ValueError('Duplicate or unregistered PC bank mount')
                if (type(m['physical_directory']) is not str or not 1 <= len(m['physical_directory']) <= 4096
                        or not Path(m['physical_directory']).is_absolute()):
                    raise ValueError('Expected a private absolute expert-bank directory')
                physical = _plain(m['physical_directory']).resolve(strict=True)
                if not physical.is_dir() or physical.is_relative_to(self.root) or self.root.is_relative_to(physical):
                    raise ValueError('An external expert root must be separate from the artifact root')
                for old in self.mounts.values():
                    if physical.is_relative_to(old['path']) or old['path'].is_relative_to(physical):
                        raise ValueError('Expert-bank roots overlap')
                value, data = _read(physical / MARKER, 4096); _valid_marker(value, prefix)
                if digest(data) != m['marker_sha256']: raise ValueError('Expert-root marker digest changed')
                if any(value['store_id'] == old['marker']['store_id'] for old in self.mounts.values()):
                    raise ValueError('Duplicate expert-root ownership identity')
                self.mounts[prefix] = {'path': physical, 'marker': value, 'marker_sha256': digest(data)}
            if c['active_bank'] not in self.mounts: raise ValueError('Active bank is not explicitly mounted')
            self.active_bank = c['active_bank']; self.config_sha256 = digest(raw)

    def path(self, name, *, strict=True):
        name = _name(name)
        for prefix, m in self.mounts.items():
            if name == prefix or name.startswith(prefix + '/'):
                suffix = name[len(prefix):].lstrip('/')
                result = _plain(m['path'] / suffix).resolve(strict=strict)
                if not result.is_relative_to(m['path']): raise ValueError('Expert path escapes its mounted root')
                return result
        result = (self.root / name).resolve(strict=strict)
        if not result.is_relative_to(self.root): raise ValueError('Expert source escapes the declared source root')
        return result

    def relative(self, path, *, strict=True):
        path = Path(path).resolve(strict=strict)
        for prefix, m in self.mounts.items():
            if path.is_relative_to(m['path']):
                suffix = path.relative_to(m['path']).as_posix()
                name = prefix if suffix == '.' else prefix + '/' + suffix
                if self.path(name, strict=strict) != path: raise ValueError('Noncanonical mounted expert path')
                return _name(name)
        if not path.is_relative_to(self.root): raise ValueError('Expert source escapes the declared source root')
        name = _name(path.relative_to(self.root).as_posix())
        if self.path(name, strict=False) != path: raise ValueError('An external mount shadows this old physical path')
        return name

    def check_file(self, path):
        # Resolve and round-trip the payload itself, not only its directory.
        self.relative(path)

    def public(self):
        return {'schema_version': 1, 'kind': 'vesper_pc_source_root_identity',
            'platform': 'pc-heterogeneous', 'profile': 'native', 'model': MODEL, 'revision': REVISION,
            'config_sha256': self.config_sha256, 'active_bank': self.active_bank,
            'maximum_mounted_banks': 2, 'per_bank_packed_payload_cap_bytes': 64 * GiB,
            'disk_available_floor_bytes': 64 * GiB,
            'mounts': [{'logical_prefix': p, 'marker': m['marker'], 'marker_sha256': m['marker_sha256']}
                for p, m in sorted(self.mounts.items())]}

    def check_unchanged(self):
        if SourceRoots(self.root).public() != self.public(): raise ValueError('Source-root configuration changed during execution')


def roots(root):
    path = Path(root).resolve(strict=True)
    return _active if _active is not None and _active.root == path else SourceRoots(path)


@contextmanager
def source_lease(root):
    """One PC capture or migration; cooperating processes cannot race a move."""
    path = _plain(Path(root).resolve(strict=True) / 'pc-heterogeneous/source-roots.lock')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        stream.seek(0, 2)
        if not stream.tell(): stream.write(b'0'); stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            try: msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error: raise RuntimeError('Another PC capture or storage migration is active') from error
        else:
            import fcntl
            try: fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error: raise RuntimeError('Another PC capture or storage migration is active') from error
        try: yield
        finally:
            stream.seek(0)
            if os.name == 'nt': msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


@contextmanager
def source_scope(root):
    global _active
    if _active is not None: raise RuntimeError('A source-root capture scope is already active')
    with source_lease(root):
        selected = SourceRoots(root); _active = selected
        try:
            yield selected
            selected.check_unchanged()
        finally: _active = None
