from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from vesper.benchmark_attempt import AttemptLedger, run_identity, run_subset
from vesper.gpqa_subset import FIELDS, make_cases
from vesper.messages import EOS


class Clock:
    def __init__(self): self.now = 0.
    def __call__(self): return self.now


class BenchmarkAttemptTests(unittest.TestCase):
    def setUp(self):
        rows = [dict(zip(FIELDS, (f'id{i}', 'Synthetic?', 'yes', 'no', 'other', 'none'))) for i in range(5)]
        cases = make_cases(rows, 3, seed=20261021, max_new_tokens=20)
        self.plan = {'cases': cases, 'case_count': 3, 'full_evaluation_run': False}
        self.candidate = {'workspace_context': {'git_head': 'example', 'created_at': 'first'},
                          'numerical_profile': 'synthetic-test-only', 'runtime_settings': {'temperature': 1.},
                          'resource_policy': {'host_cap_bytes': 96 * 1024**3}}
        self.clock = Clock(); self.opens = []; self.closes = []; self.mode = 'correct'

    @contextmanager
    def backend(self, case, deadline):
        self.opens.append(case['id'])
        self.clock.now += .2
        count = 0
        class Tokenizer:
            def encode(self, text): return [10, 11]
            def decode(inner, ids):
                answer = case['scoring']['expected'] if self.mode != 'wrong' else 'Z'
                return f'reason</think>({answer})' + (EOS if ids[-1:] == [1] else '')
        def step(ids, *, sample):
            nonlocal count
            self.clock.now += 1.; count += 1
            if self.mode == 'error': raise OSError('private source details must not escape')
            if self.mode == 'interrupt': raise KeyboardInterrupt()
            return {'sampling': {'ids_i64': np.array([1 if count >= 3 else 42], np.int64)}} if sample else {}
        try:
            yield SimpleNamespace(tokenizer=Tokenizer(), step=step, maximum_model_calls=1000)
        finally:
            self.clock.now += .1; self.closes.append(case['id'])

    def run_at(self, directory, budget=30):
        return run_subset(self.plan, self.candidate, directory, self.backend,
                          budget_seconds=budget, clock=self.clock)

    def test_real_eos_feedback_and_timing_and_completed_questions_are_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_at(directory)
            self.assertEqual(len(result['completed_case_ids']), 3)
            self.assertEqual(result['pending_case_ids'], [])
            for row in result['results']:
                self.assertEqual(row['selected_ids_including_eos'], [42, 1])
                self.assertTrue(row['score']['passed'])
                self.assertEqual(row['timing']['decode_tokens_per_second'], 1.)
                self.assertAlmostEqual(row['request_TTFT_seconds'], 2.)
                self.assertAlmostEqual(row['engine_TTFT_seconds'], 2.2)
            self.assertEqual(self.opens, self.closes)
            self.run_at(directory)
            self.assertEqual(len(self.opens), 3)

    def test_timeout_is_saved_without_score_and_restarts_only_unfinished_question(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_at(directory, budget=1.5)
            self.assertEqual(result['results'][0]['stop_reason'], 'budget_exceeded')
            self.assertIsNone(result['results'][0]['score']['passed'])
            self.assertEqual(result['completed_case_ids'], [])
            first = self.plan['cases'][0]['id']
            saved = (Path(directory) / (first + '-00001') / 'final.json').read_bytes()
            result = self.run_at(directory)
            self.assertEqual(len(result['completed_case_ids']), 3)
            self.assertEqual((Path(directory) / (first + '-00001') / 'final.json').read_bytes(), saved)
            journal = Path(directory) / (first + '-00002') / 'calls.jsonl'
            self.assertEqual(json.loads(journal.read_text().splitlines()[0])['position'], 0)

    def test_wrong_completed_answer_is_not_retried_until_correct(self):
        self.mode = 'wrong'
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_at(directory)
            self.assertTrue(all(r['score']['passed'] is False for r in result['results']))
            self.mode = 'correct'; self.run_at(directory)
            self.assertEqual(len(self.opens), 3)

    def test_error_and_user_interrupt_preserve_attempt_and_cleanup(self):
        for mode, stop in (('error', 'error'), ('interrupt', 'interrupted')):
            self.mode = mode
            with tempfile.TemporaryDirectory() as directory:
                result = self.run_at(directory)
                self.assertEqual(result['results'][0]['stop_reason'], stop)
                self.assertIsNone(result['results'][0]['score']['passed'])
                raw = next(Path(directory).glob('gpqa-*/final.json')).read_text()
                self.assertNotIn('private source details', raw)
        self.assertEqual(self.opens, self.closes)

    def test_changed_candidate_or_input_is_refused_before_model_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_at(directory); before = len(self.opens)
            self.candidate['numerical_profile'] = 'changed'
            with self.assertRaisesRegex(ValueError, 'Cannot resume'): self.run_at(directory)
            self.assertEqual(len(self.opens), before)
        self.plan['cases'][0]['max_new_tokens'] = 19
        with self.assertRaisesRegex(ValueError, 'question/settings changed'):
            run_identity(self.plan, self.candidate)

    def test_full_split_is_refused_and_fresh_timestamp_does_not_change_identity(self):
        first = run_identity(self.plan, self.candidate)
        self.candidate['workspace_context']['created_at'] = 'later'
        self.assertEqual(first, run_identity(self.plan, self.candidate))
        self.plan['full_evaluation_run'] = True
        with self.assertRaisesRegex(ValueError, 'development subset'):
            run_identity(self.plan, self.candidate)

    def test_backend_capacity_never_silently_clips_declared_budget(self):
        @contextmanager
        def limited(case, deadline):
            with self.backend(case, deadline) as backend:
                backend.maximum_model_calls = 2
                yield backend
        with tempfile.TemporaryDirectory() as directory:
            result = run_subset(self.plan, self.candidate, directory, limited, budget_seconds=20, clock=self.clock)
            self.assertEqual(result['results'][0]['stop_reason'], 'error')
            self.assertEqual(result['results'][0]['selected_ids_including_eos'], [])
            self.assertEqual(result['results'][0]['timing']['completed_model_calls'], 0)
            self.assertEqual(self.plan['cases'][0]['max_new_tokens'], 20)

    def test_crashed_attempt_with_only_journal_is_pending_and_not_overwritten(self):
        identity = run_identity(self.plan, self.candidate); first = self.plan['cases'][0]['id']
        with tempfile.TemporaryDirectory() as directory:
            with AttemptLedger(directory, identity) as ledger:
                with ledger.begin(first) as (path, journal):
                    journal.write('{"position": 0}\n'); journal.flush()
                self.assertIsNone(ledger.completed(first))
            self.run_at(directory)
            self.assertFalse((path / 'final.json').exists())
            self.assertTrue((Path(directory) / (first + '-00002') / 'final.json').exists())

    def test_large_declared_allowance_is_preserved_for_capable_backend(self):
        from vesper.model_artifacts import canonical
        self.plan['cases'] = self.plan['cases'][:1]; self.plan['case_count'] = 1
        case = self.plan['cases'][0]; case['max_new_tokens'] = 262144
        case['case_sha256'] = canonical({k: v for k, v in case.items() if k != 'case_sha256'})
        @contextmanager
        def capable(case, deadline):
            with self.backend(case, deadline) as backend:
                backend.maximum_model_calls = 262145
                yield backend
        with tempfile.TemporaryDirectory() as directory:
            result = run_subset(self.plan, self.candidate, directory, capable, budget_seconds=20, clock=self.clock)
            self.assertEqual(result['results'][0]['stop_reason'], 'eos')
            self.assertEqual(case['max_new_tokens'], 262144)

    def test_second_writer_and_changed_resource_policy_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            identity = run_identity(self.plan, self.candidate)
            with AttemptLedger(directory, identity):
                with self.assertRaises(OSError):
                    with AttemptLedger(directory, identity): self.fail('Second writer admitted')
            self.candidate['resource_policy']['host_cap_bytes'] -= 1
            with self.assertRaisesRegex(ValueError, 'Cannot resume'): self.run_at(directory)

    def test_opt_in_partial_capacity_preserves_allowance_and_has_no_quality_pass(self):
        @contextmanager
        def partial(case, deadline):
            with self.backend(case, deadline) as backend:
                backend.maximum_model_calls = 2; backend.allow_partial_capacity = True
                yield backend
        with tempfile.TemporaryDirectory() as directory:
            result = run_subset(self.plan, self.candidate, directory, partial, budget_seconds=20, clock=self.clock)
            row = result['results'][0]
            self.assertEqual(row['stop_reason'], 'backend_scope_exhausted')
            self.assertEqual(row['selected_ids_including_eos'], [42])
            self.assertEqual(row['declared_max_new_tokens'], 20)
            self.assertIsNone(row['score']['passed'])
            self.assertEqual(result['completed_case_ids'], [])


if __name__ == '__main__': unittest.main()
