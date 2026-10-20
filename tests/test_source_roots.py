from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from vesper.bandwidth import EXPERT_BYTES
from vesper.expert_bank import ExpertBank
from vesper.indexed_store import build_index, IndexedStore
from vesper.model_artifacts import digest
from vesper.resources import GiB, MiB, MemoryBudget
from vesper.source_roots import SourceRoots, CONFIG, MARKER, BANKS, marker, roots, source_scope, source_lease
import test_indexed_store


def mounted(root, external):
    mounts = []
    for i, name in enumerate(BANKS):
        path = external / ('bank' + str(i + 1)); path.mkdir(parents=True)
        raw = (json.dumps(marker(name), indent=2) + '\n').encode()
        (path / MARKER).write_bytes(raw)
        mounts.append({'logical_prefix': name, 'physical_directory': str(path.resolve()), 'marker_sha256': digest(raw)})
    c = {'schema_version': 1, 'kind': 'vesper_pc_source_roots', 'platform': 'pc-heterogeneous', 'profile': 'native',
        'model': marker(BANKS[0])['model'], 'revision': marker(BANKS[0])['revision'], 'active_bank': BANKS[1], 'mounts': mounts}
    path = root / CONFIG; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(c), encoding='utf-8')
    return c


class SourceRootTests(unittest.TestCase):
    def setUp(self):
        p = patch('vesper.resources.physical_memory', return_value={'available': 64 * GiB}); p.start(); self.addCleanup(p.stop)

    def test_default_indexes_have_the_same_logical_schema_and_zero_payload_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = test_indexed_store.IndexedStoreTests().sources(root, 1)[0]
            receipt = build_index([source], root / 'index', root, layer=0)
            self.assertEqual(receipt['payload_bytes_copied'], 0)
            self.assertEqual(json.loads((root/'index/manifest.json').read_bytes())['sources'][0]['directory'], 'source-000')
            r = SourceRoots(root); self.assertEqual(r.path('source-000'), source.resolve())
            self.assertEqual(r.public()['mounts'], []); self.assertIsNone(r.public()['config_sha256'])

    def test_old_index_still_reads_exact_bytes_after_a_declared_external_move(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp); root=base/'artifacts'; root.mkdir(); external=base/'ssd'
            bank=root/BANKS[0]; bank.mkdir(parents=True)
            source=test_indexed_store.IndexedStoreTests().sources(bank,1)[0]
            build_index([source],root/'index',root,layer=0); before=(root/'index/manifest.json').read_bytes()
            c=mounted(root,external); target=Path(c['mounts'][0]['physical_directory'])/'source-000'
            source.rename(target)
            r=SourceRoots(root); self.assertEqual(r.path(BANKS[0]+'/source-000'),target)
            self.assertEqual(r.relative(target),BANKS[0]+'/source-000')
            budget=MemoryBudget(cap=96*MiB,available_floor=0); output=np.empty(EXPERT_BYTES,np.uint8)
            with source_scope(root) as selected:
                self.assertIs(roots(root),selected)
                with IndexedStore(root/'index',root,budget,layer=0) as store:
                    store.read_into('0:0',output); self.assertFalse(output.any())
                with ExpertBank(root,r.path(BANKS[1]),[target]) as inventory:
                    inventory.prepare(0,[0],None,None)
                    self.assertFalse(any(a.any() for a in inventory.arrays(0,0)[0].values()))
                    self.assertEqual(inventory.new_bytes,0)
            self.assertEqual(budget.reserved,0); self.assertEqual((root/'index/manifest.json').read_bytes(),before)
            public=json.dumps(r.public()); self.assertNotIn(str(external),public); self.assertNotIn('physical_directory',public)

    def test_scope_rejects_changed_configuration_and_releases_lease_after_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp); root=base/'artifacts'; root.mkdir(); c=mounted(root,base/'ssd')
            with self.assertRaisesRegex(ValueError,'changed during'):
                with source_scope(root): (root/CONFIG).write_text(json.dumps(c)+' ',encoding='utf-8')
            with source_scope(root):
                with self.assertRaisesRegex(RuntimeError,'already active'):
                    with source_scope(root): pass
                with self.assertRaisesRegex(RuntimeError,'Another PC'):
                    with source_lease(root): pass
            with source_lease(root): pass

    def test_foreign_duplicate_overlapping_unbounded_and_changed_marker_mounts_refuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp); root=base/'artifacts'; root.mkdir(); c=mounted(root,base/'ssd'); path=root/CONFIG
            for mutate in (lambda d:d.update(profile='other'),lambda d:d.update(active_bank='undeclared'),
                    lambda d:d['mounts'].append(deepcopy(d['mounts'][0])),
                    lambda d:d['mounts'][1].update(logical_prefix=BANKS[0]),
                    lambda d:d['mounts'][0].update(logical_prefix='../escape'),
                    lambda d:d['mounts'][0].update(physical_directory='relative'),
                    lambda d:d['mounts'][1].update(physical_directory=d['mounts'][0]['physical_directory']),
                    lambda d:d['mounts'][0].update(marker_sha256='0'*64)):
                d=deepcopy(c); mutate(d); path.write_text(json.dumps(d),encoding='utf-8')
                with self.assertRaises(ValueError):SourceRoots(root)
            path.write_text(json.dumps(c),encoding='utf-8')
            m=Path(c['mounts'][0]['physical_directory'])/MARKER; m.write_bytes(m.read_bytes()+b' ')
            with self.assertRaisesRegex(ValueError,'digest'):SourceRoots(root)

    def test_path_escape_shadow_and_stale_marker_refuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp); root=base/'artifacts'; root.mkdir(); c=mounted(root,base/'ssd'); r=SourceRoots(root)
            for name in ('../escape','/absolute','a//b','a/./b','C:/escape','a\\b','a:stream'):
                with self.assertRaises(ValueError):r.path(name,strict=False)
            old=root/BANKS[0]/'stale'; old.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError,'shadows'):r.relative(old)
            with self.assertRaises(ValueError):r.relative(base/'outside',strict=False)
            marker_path=Path(c['mounts'][0]['physical_directory'])/MARKER
            marker_path.write_bytes(marker_path.read_bytes()+b' ')
            with self.assertRaises(ValueError):r.check_unchanged()

    def test_linked_external_directory_is_not_an_implicit_mount(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp); root=base/'artifacts'; root.mkdir(); c=mounted(root,base/'ssd')
            physical=Path(c['mounts'][0]['physical_directory']); link=physical/'escape'
            try:link.symlink_to(root,target_is_directory=True)
            except (OSError,NotImplementedError):self.skipTest('Directory symlink privilege unavailable')
            with self.assertRaisesRegex(ValueError,'symlinks|reparse'):SourceRoots(root).path(BANKS[0]+'/escape')
