from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from vesper.moe_session import SessionMoE
from vesper.timing import Timings


class FakeConnected:
    instances = []
    fail = False
    bad_provenance = False

    def __init__(self, *args, **kwargs):
        self.closed = False
        self.hits = 0
        provenance = {'source': 'changed' if self.bad_provenance else 'original'}
        source = SimpleNamespace(entries={'0:1': {'provenance': provenance}}, read_calls=0, bytes_read=0)
        self.experts = SimpleNamespace(store=source, timings=Timings(True))
        self.instances.append(self)

    def __enter__(self):
        return self

    def forward(self, values, image_span):
        if self.fail:
            raise OSError('injected device failure')
        self.hits += 1
        if self.hits == 1:
            self.experts.store.read_calls += 1
            self.experts.store.bytes_read += 16
        return {'output_bf16': values, 'expert_events': [{'source_tier': 'disk' if self.hits == 1 else 'primary'}]}

    def __exit__(self, *_):
        self.closed = True
        self.cleanup = {'parent_reservation_bytes': 0}


class SessionMoETests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = self.root / 'store'
        self.store.mkdir()
        (self.store / 'manifest.json').write_bytes(b'{}')
        self.bindings = {'layers': {'0': {k: {'path': k} for k in ('gate', 'router', 'shared')}}}
        FakeConnected.instances = []
        FakeConnected.fail = FakeConnected.bad_provenance = False
        self.addCleanup(patch.stopall)
        patch('vesper.moe_session.ConnectedMoE', FakeConnected).start()
        patch('vesper.moe_session.validate_moe_manifests').start()

    def session(self, **kwargs):
        return SessionMoE(self.root, self.bindings, 0, {'ids': [1], 'store': 'store'},
                          {'0:1': {'source': 'original'}}, **kwargs)

    def test_retained_context_keeps_cache_and_releases_once(self):
        session = self.session()
        with session:
            session.forward('first')
            session.forward('second')
            self.assertEqual(session.stats()['direct_read_calls'], 1)
        stats = session.stats()
        self.assertEqual((stats['calls'], stats['context_opens'], stats['context_closes']), (2, 1, 1))
        self.assertEqual(stats['source_tiers'], {'disk': 1, 'primary': 1})
        self.assertTrue(all(c.closed for c in FakeConnected.instances))
        self.assertFalse(stats['opened'])

    def test_ephemeral_comparator_opens_and_reads_for_each_call(self):
        session = self.session(retain=False)
        with session:
            session.forward('first')
            session.forward('second')
        self.assertEqual((session.opens, session.closes, session.reads), (2, 2, 2))
        self.assertTrue(all(c.closed for c in FakeConnected.instances))

    def test_failed_forward_cannot_be_silently_reused_and_closes(self):
        session = self.session()
        with session:
            FakeConnected.fail = True
            with self.assertRaises(OSError):
                session.forward('first')
            with self.assertRaisesRegex(RuntimeError, 'failed MoE session'):
                session.forward('retry')
        self.assertEqual(session.closes, 1)
        self.assertTrue(FakeConnected.instances[0].closed)
        FakeConnected.fail = False
        with session:
            self.assertEqual(session.forward('healthy')['output_bf16'], 'healthy')

    def test_mutated_manifest_is_refused_before_cached_execution(self):
        session = self.session()
        with session:
            (self.store / 'manifest.json').write_bytes(b'{"changed": true}')
            with self.assertRaisesRegex(ValueError, 'metadata changed'):
                session.forward('input')
        self.assertEqual(session.calls, 0)
        self.assertEqual(session.opens, session.closes)

    def test_provenance_failure_closes_partially_opened_context(self):
        session = self.session()
        FakeConnected.bad_provenance = True
        with self.assertRaisesRegex(ValueError, 'original expert provenance'):
            session.__enter__()
        self.assertFalse(session.opened)
        self.assertEqual((session.opens, session.closes), (1, 1))
        self.assertTrue(FakeConnected.instances[0].closed)

    def test_expired_budget_refuses_before_device_setup(self):
        session = self.session(deadline=0)
        with self.assertRaises(TimeoutError):
            session.__enter__()
        self.assertEqual(FakeConnected.instances, [])

    def test_context_usage_is_explicit(self):
        session = self.session()
        with self.assertRaises(RuntimeError):
            session.forward('unopened')
        with session:
            with self.assertRaises(RuntimeError):
                session.__enter__()
        with self.assertRaises(RuntimeError):
            session.forward('closed')

    def test_failed_ephemeral_session_can_be_reopened_explicitly(self):
        session = self.session(retain=False)
        with session:
            FakeConnected.fail = True
            with self.assertRaises(OSError):
                session.forward('failed')
        FakeConnected.fail = False
        with session:
            session.forward('healthy')
        self.assertFalse(session.failed)
        self.assertEqual((session.opens, session.closes), (2, 2))


if __name__ == '__main__':
    unittest.main()
