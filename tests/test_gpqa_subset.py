import csv
import io
from pathlib import Path
import tempfile
import unittest

from vesper.gpqa_subset import FIELDS, _rows, load_archive, make_cases, preflight, score_response
from vesper.messages import EOS


class GPQASubsetTests(unittest.TestCase):
    def rows(self):
        return [{k: v for k, v in zip(FIELDS, (f'id{i}', f'Synthetic question {i}?',
                 f'correct{i}', f'wrong{i}x', f'wrong{i}y', f'wrong{i}z'))} for i in range(8)]

    def test_selection_is_reproducible_and_independent_of_csv_order(self):
        rows = self.rows()
        a = make_cases(rows, 3, seed=42, max_new_tokens=4096)
        b = make_cases(list(reversed(rows)), 3, seed=42, max_new_tokens=4096)
        self.assertEqual(a, b)
        self.assertEqual(len({c['id'] for c in a}), 3)
        for case in a:
            source = next(r for r in rows if r['Record ID'] == case['source_record_id'])
            self.assertIn(source['Question'], case['messages'][0]['content'])
            expected = case['scoring']['expected']
            self.assertIn(f"({expected}) {source['Correct Answer']}", case['messages'][0]['content'])
            self.assertEqual(case['max_new_tokens'], 4096)

    def test_full_suite_and_invalid_budgets_are_not_silently_accepted(self):
        for count in (0, True, 8, 9):
            with self.assertRaises(ValueError):
                make_cases(self.rows(), count, seed=1, max_new_tokens=4096)
        for budget in (0, True, 262145):
            with self.assertRaises(ValueError):
                make_cases(self.rows(), 3, seed=1, max_new_tokens=budget)

    def test_duplicate_id_and_missing_schema_fail(self):
        for rows in ([self.rows()[0]] * 2, [{'Question': 'missing fields'}]):
            buffer = io.StringIO()
            fields = list(rows[0]); writer = csv.DictWriter(buffer, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
            with self.assertRaises(ValueError):
                _rows(buffer.getvalue().encode())

    def test_preflight_keeps_full_budget_and_cannot_claim_inference(self):
        cases = make_cases(self.rows(), 3, seed=1, max_new_tokens=262144)
        class Tokenizer:
            def encode(self, text): return list(range(100))
        report = preflight(cases, Tokenizer())
        self.assertFalse(report['model_execution_run'])
        self.assertFalse(report['quality_scored'])
        self.assertEqual(report['native_attention_position_capacity'], 1152)
        self.assertEqual(report['native_maximum_decode_ids'], 4096)
        self.assertTrue(report['benchmark_attempt_controller_implemented'])
        self.assertFalse(report['native_dynamic_asset_resolution_implemented'])
        for case in report['cases']:
            self.assertEqual(case['maximum_model_calls'], 262243)
            self.assertFalse(case['ready_for_native_inference'])
            self.assertIn('prompt_and_generation_budget_exceed_native_512_call_scope', case['reasons'])
            self.assertIn('prompt_and_generation_budget_exceed_native_attention_capacity', case['reasons'])
            self.assertIn('completion_budget_exceeds_native_tokenizer_decode_capacity', case['reasons'])

    def test_oversized_question_is_preserved_and_refused_not_shortened(self):
        rows = self.rows()
        for row in rows: row['Question'] = 'Q' * 5000
        cases = make_cases(rows, 3, seed=1, max_new_tokens=4096)
        self.assertTrue(all(c['encoded_prompt'] is None for c in cases))
        self.assertTrue(all('Q' * 5000 in c['messages'][0]['content'] for c in cases))
        report = preflight(cases, None)
        self.assertTrue(all(c['prompt_tokens'] is None for c in report['cases']))

    def test_actual_eos_final_channel_and_strict_answer_format(self):
        case = make_cases(self.rows(), 1, seed=1, max_new_tokens=4096)[0]
        answer = case['scoring']['expected']
        self.assertTrue(score_response(case, f'analysis</think>({answer})' + EOS, stop_reason='eos')['passed'])
        self.assertTrue(score_response(case, 'x' * 20000 + f'</think>({answer})' + EOS, stop_reason='eos')['passed'])
        for text in (f'analysis</think>({answer}', f'analysis</think>({answer}' + EOS,
                     f'analysis ({answer})</think>no choice' + EOS):
            self.assertFalse(score_response(case, text, stop_reason='eos')['passed'])
        result = score_response(case, f'({answer})', stop_reason='budget_exceeded')
        self.assertIsNone(result['passed'])

    def test_duplicate_source_choices_are_preserved_and_flagged_not_dropped(self):
        rows = self.rows()
        for row in rows: row['Incorrect Answer 1'] = row['Correct Answer']
        cases = make_cases(rows, 3, seed=1, max_new_tokens=4096)
        self.assertEqual(len(cases), 3)
        self.assertTrue(all(c['source_has_duplicate_choice_text'] for c in cases))
        result = score_response(cases[0], 'reason</think>(A)' + EOS, stop_reason='eos')
        self.assertIsNone(result['passed'])

    def test_unpinned_archive_is_refused_before_decompression(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'dataset.zip'; path.write_bytes(b'untrusted archive')
            with self.assertRaisesRegex(ValueError, 'pinned upstream'):
                load_archive(path)
