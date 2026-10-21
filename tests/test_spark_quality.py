"""CPU-only evaluation-contract checks; synthetic questions, no model or judge."""

from copy import deepcopy
from contextlib import redirect_stdout
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from vesper import spark_quality as quality


def dataset():
    output = io.StringIO(newline="")
    fields = ["Record ID", "High-level domain", "Question", "Correct Answer",
              "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3", "Explanation"]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for n in range(198):
        domain = "Biology" if n < 19 else "Chemistry" if n < 112 else "Physics"
        writer.writerow({"Record ID": f"synthetic-{n}", "High-level domain": domain,
                         "Question": f"Synthetic question {n}?", "Correct Answer": f"yes-{n}",
                         "Incorrect Answer 1": "wrong1", "Incorrect Answer 2": "wrong2",
                         "Incorrect Answer 3": "wrong3", "Explanation": "SECRET EXPLANATION"})
    raw = output.getvalue().encode("utf-8")
    return raw, quality.read_dataset(raw, revision="a" * 40, expected_sha256=hashlib.sha256(raw).hexdigest())


def identity(commit="b" * 40):
    return {"model": quality.MODEL, "model_revision": quality.REVISION, "profile": "native",
            "code_commit": commit, "dirty_diff_sha256": None, "runtime": "direct",
            "hardware": "synthetic-cpu", "toolchain": "fixture", "diagnostic_mode": "full"}


def budget():
    return {"context_tokens": 1152, "max_generated_tokens": 768, "max_model_calls": 1152,
            "per_question_seconds": 600, "total_seconds": 3600}


def signed(value, field):
    value[field] = quality.digest({k: v for k, v in value.items() if k != field})
    return value


def records(manifest, plan, *, seconds=10):
    entries = {r["id"]: r for r in manifest["population"]}
    return [{"question_id": qid, "plan_sha256": plan["plan_sha256"], "status": "completed",
             "eos_complete": True, "final_text": "Answer: (" + entries[qid]["correct_label"] + ")",
             "input_tokens": 50, "generated_tokens": 10, "thinking_tokens": 7,
             "answer_tokens": 2, "special_tokens": 1, "model_calls": 59,
             "wall_seconds": seconds, "phase_seconds": {key: seconds if key == "answer" else 0 for key in quality.PHASES}}
            for qid in plan["question_ids"]]


class SparkQualityTests(unittest.TestCase):
    def setUp(self):
        self.raw, (self.pin, self.rows) = dataset()
        self.manifest = quality.prepare(self.pin, self.rows)
        self.plan = quality.plan(self.manifest, subset="development", identity=identity(), budget=budget())

    def test_dataset_file_and_complete_split_are_pinned(self):
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            quality.read_dataset(self.raw + b" ", revision="a" * 40, expected_sha256=self.pin["file_sha256"])
        short = self.raw.splitlines(keepends=True)
        raw = b"".join(short[:-1])
        with self.assertRaisesRegex(ValueError, "198"):
            quality.read_dataset(raw, revision="a" * 40, expected_sha256=hashlib.sha256(raw).hexdigest())
        with self.assertRaisesRegex(ValueError, "revision"):
            quality.read_dataset(self.raw, revision="main", expected_sha256=self.pin["file_sha256"])

    def test_deterministic_stratified_and_disjoint_order_independent(self):
        other = quality.prepare(self.pin, list(reversed(self.rows)))
        self.assertEqual(other, self.manifest)
        self.assertEqual(len(self.manifest["development"]), 12)
        self.assertEqual(len(self.manifest["held_out"]), 6)
        self.assertFalse(set(self.manifest["development"]) & set(self.manifest["held_out"]))
        population = {r["id"]: r for r in self.manifest["population"]}
        for subset in ("development", "held_out"):
            self.assertEqual({population[q]["domain"] for q in self.manifest[subset]}, set(quality.DOMAINS))
        self.assertNotEqual(quality.prepare(self.pin, self.rows, seed=1)["development"], self.manifest["development"])

    def test_selection_does_not_read_validation_or_explanation_fields(self):
        self.assertNotIn("SECRET", json.dumps(self.rows))
        r = quality.request(self.rows[0])
        self.assertNotIn("SECRET", r["prompt"])
        self.assertEqual(sorted(r["choice_order"]), [0, 1, 2, 3])
        position = "ABCD".index(r["correct_label"])
        self.assertEqual(r["choice_order"][position], 0)
        self.assertEqual(r["source_record_id"], "synthetic-0")

    def test_dataset_changes_and_invalid_subset_sizes_refuse(self):
        rows = deepcopy(self.rows)
        rows[0]["choices"][0] = "modified"
        with self.assertRaisesRegex(ValueError, "changed"):
            quality.prepare(self.pin, rows)
        for n in (0, 2, 199, True):
            with self.assertRaises(ValueError):
                quality.prepare(self.pin, self.rows, development=n)

    def test_plan_requires_explicit_finite_budgets_and_native_profile(self):
        for name in budget():
            b = budget()
            del b[name]
            with self.assertRaisesRegex(ValueError, "budget"):
                quality.plan(self.manifest, subset="development", identity=identity(), budget=b)
        for key, value in (("context_tokens", 1153), ("per_question_seconds", float("inf")), ("max_model_calls", True)):
            b = budget()
            b[key] = value
            with self.assertRaises(ValueError):
                quality.plan(self.manifest, subset="development", identity=identity(), budget=b)
        approximate = identity()
        approximate["profile"] = "lossy"
        with self.assertRaisesRegex(ValueError, "approximate"):
            quality.plan(self.manifest, subset="development", identity=approximate, budget=budget())
        with self.assertRaisesRegex(ValueError, "small"):
            quality.plan(self.manifest, subset="full", identity=identity(), budget=budget())

    def test_plan_settings_do_not_share_mutable_global_defaults(self):
        self.plan["settings"]["tools"].append("evil")
        self.assertEqual(quality.PROPOSED_SETTINGS["tools"], [])

    def test_parser_only_accepts_unambiguous_terminal_final_line(self):
        self.assertEqual(quality.parse_answer("Explanation mentions (B).\nAnswer: (A)"), "A")
        for value in ("(A)", "Answer: (E)", "Answer: (A) trailing", "Answer: (A)\nAnswer: (B)",
                      "<think>Answer: (A)</think>", "Answer: (A)\nmore", "answer: (A)",
                      "text<｜end▁of▁sentence｜>\nAnswer: (A)", "｜DSML｜\nAnswer: (A)"):
            self.assertIsNone(quality.parse_answer(value), value)

    def test_timeout_truncation_cancellation_error_and_missing_never_disappear(self):
        rows = records(self.manifest, self.plan)
        for row, status in zip(rows, ("timeout", "truncated", "cancelled", "error", "unsupported")):
            row["status"] = status
        rows.pop()
        report = quality.grade(self.manifest, self.plan, rows)
        self.assertEqual(report["summary"]["planned_questions"], 12)
        self.assertEqual(report["summary"]["correct"], 6)
        self.assertEqual(report["summary"]["completed_accuracy"], 1)
        self.assertEqual(report["summary"]["correct_over_planned"], .5)
        self.assertIsNone(report["summary"]["full_subset_score"])
        estimate = quality.estimate(self.manifest, [report], repetitions=1, fixed_overhead_seconds=0, excluded_costs=[])
        self.assertIsNone(estimate["total_seconds"]["stratified_mean"])
        self.assertFalse(estimate["configurations"][0]["protocol_invalid"])
        self.assertIsNone(report["records"][0]["score"])
        for status in ("timeout", "truncated", "cancelled", "error", "unsupported", "not_run"):
            self.assertEqual(report["summary"]["status_counts"][status], 1)
        self.assertFalse(report["full_run_authorized"])

    def test_incomplete_eos_and_budget_overrun_are_not_correct(self):
        rows = records(self.manifest, self.plan)
        rows[0]["eos_complete"] = False
        rows[1]["wall_seconds"] = 601
        rows[2]["final_text"] = "Cannot answer"
        report = quality.grade(self.manifest, self.plan, rows)
        self.assertEqual(report["summary"]["correct"], 9)
        self.assertEqual(report["records"][0]["status"], "invalid_completion")
        self.assertEqual(report["records"][1]["budget_violations"], ["per_question_time"])
        self.assertEqual(report["records"][2]["status"], "invalid_answer")
        self.assertFalse(report["summary"]["protocol_valid"])

    def test_duplicate_foreign_or_mismatched_results_refuse(self):
        rows = records(self.manifest, self.plan)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            quality.grade(self.manifest, self.plan, rows + rows[:1])
        rows[0]["plan_sha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "mismatch"):
            quality.grade(self.manifest, self.plan, rows)

    def test_token_and_phase_accounting_fail_closed(self):
        for key, value in (("thinking_tokens", 100), ("wall_seconds", float("nan")), ("input_tokens", -1),
                           ("model_calls", True), ("phase_seconds", {"answer": 20})):
            rows = records(self.manifest, self.plan)
            rows[0][key] = value
            with self.assertRaises(ValueError):
                quality.grade(self.manifest, self.plan, rows)

    def test_manifest_and_plan_tampering_or_resampling_refuse(self):
        changed = deepcopy(self.manifest)
        changed["development"][0] = changed["held_out"][0]
        with self.assertRaisesRegex(ValueError, "Manifest changed"):
            quality.grade(changed, self.plan, [])
        changed_plan = deepcopy(self.plan)
        changed_plan["question_ids"] = changed_plan["question_ids"][:-1]
        signed(changed_plan, "plan_sha256")
        with self.assertRaisesRegex(ValueError, "membership"):
            quality.grade(self.manifest, changed_plan, [])

    def test_native_pair_reports_regressions_without_adopting_tolerance(self):
        a = quality.grade(self.manifest, self.plan, records(self.manifest, self.plan))
        bplan = quality.plan(self.manifest, subset="development", identity=identity("c" * 40), budget=budget())
        braw = records(self.manifest, bplan)
        braw[0]["status"] = "timeout"
        b = quality.grade(self.manifest, bplan, braw)
        result = quality.compare(a, b)
        self.assertEqual(len(result["completion_loss"]), 1)
        self.assertEqual(len(result["correct_to_incorrect"]), 0)
        self.assertEqual(result["decision"], "review_required_no_quality_tolerance_adopted")
        self.assertIsNone(result["full_subset_score_delta"])
        b["run_plan"]["identity"]["diagnostic_mode"] = "serving"
        signed(b["run_plan"], "plan_sha256")
        signed(b, "report_sha256")
        with self.assertRaisesRegex(ValueError, "diagnostic_mode"):
            quality.compare(a, b)

    def test_estimate_uses_all_configurations_full_population_and_repetitions(self):
        a = quality.grade(self.manifest, self.plan, records(self.manifest, self.plan, seconds=10))
        bplan = quality.plan(self.manifest, subset="development", identity=identity("c" * 40), budget=budget())
        b = quality.grade(self.manifest, bplan, records(self.manifest, bplan, seconds=20))
        result = quality.estimate(self.manifest, [a, b], repetitions=3, fixed_overhead_seconds=100, excluded_costs=["scheduler queue"])
        self.assertEqual(result["total_seconds"]["stratified_mean"], 198 * 30 * 3 + 100)
        self.assertEqual(result["full_coverage_questions_per_repetition"], 198)
        self.assertFalse(result["full_run_authorized"])
        self.assertIn("explicit decision", result["required_user_decision"])
        self.assertIn("not confidence", result["uncertainty"])

    def test_censored_estimate_does_not_turn_timeout_limit_into_completion_cost(self):
        rows = records(self.manifest, self.plan)
        rows[0]["status"] = "truncated"
        report = quality.grade(self.manifest, self.plan, rows)
        result = quality.estimate(self.manifest, [report], repetitions=1, fixed_overhead_seconds=0, excluded_costs=[])
        self.assertIsNone(result["total_seconds"]["stratified_mean"])
        self.assertIsNone(result["total_seconds"]["observed_max"])
        self.assertEqual(len(result["configurations"][0]["censored_question_ids"]), 1)

    def test_missing_strata_cannot_produce_a_full_time_estimate(self):
        rows = records(self.manifest, self.plan)
        entries = {r["id"]: r for r in self.manifest["population"]}
        rows = [r for r in rows if entries[r["question_id"]]["domain"] != "biology"]
        report = quality.grade(self.manifest, self.plan, rows)
        result = quality.estimate(self.manifest, [report], repetitions=1, fixed_overhead_seconds=0, excluded_costs=[])
        self.assertEqual(result["configurations"][0]["missing_domains"], ["biology"])
        self.assertIsNone(result["total_seconds"]["stratified_mean"])

    def test_empty_no_measurements_is_not_a_zero_time_prediction(self):
        report = quality.grade(self.manifest, self.plan, [])
        self.assertEqual(set(report["summary"]["unmeasured_phases"]), set(quality.PHASES))
        result = quality.estimate(self.manifest, [report], repetitions=1, fixed_overhead_seconds=0, excluded_costs=[])
        self.assertIsNone(result["total_seconds"]["observed_min"])

    def test_completed_answer_cannot_claim_zero_work_or_inconsistent_calls(self):
        for key, value in (("input_tokens", 0), ("generated_tokens", 0), ("model_calls", 0), ("model_calls", 60), ("special_tokens", 0)):
            rows = records(self.manifest, self.plan)
            rows[0][key] = value
            with self.assertRaises(ValueError):
                quality.grade(self.manifest, self.plan, rows)

    def test_total_budget_overrun_keeps_descriptive_records_but_no_qualifying_score(self):
        rows = records(self.manifest, self.plan, seconds=301)
        report = quality.grade(self.manifest, self.plan, rows)
        self.assertEqual(report["summary"]["correct"], 12)
        self.assertTrue(report["summary"]["total_budget_exceeded"])
        self.assertFalse(report["summary"]["protocol_valid"])
        self.assertIsNone(report["summary"]["full_subset_score"])
        estimate = quality.estimate(self.manifest, [report], repetitions=1, fixed_overhead_seconds=0, excluded_costs=[])
        self.assertIsNone(estimate["total_seconds"]["stratified_mean"])
        self.assertTrue(estimate["configurations"][0]["protocol_invalid"])

    def test_duplicate_incorrect_choices_are_kept_as_the_author_released_them(self):
        raw = self.raw.replace(b'wrong1,wrong2,wrong3', b'wrong1,wrong1,wrong3')
        _, rows = quality.read_dataset(raw, revision='a' * 40, expected_sha256=hashlib.sha256(raw).hexdigest())
        self.assertEqual(rows[0]['choices'][1:3], ['wrong1', 'wrong1'])

    def test_all_four_cli_operations_work_offline_and_never_replace_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            class Workspace:
                def check(self): pass
                def context(self): return {'platform': 'dgx-spark'}
                def artifact_path(self, relative, create_parent=False):
                    path = base / relative
                    if create_parent: path.parent.mkdir(parents=True, exist_ok=True)
                    return path
            (base / 'dataset.csv').write_bytes(self.raw)
            (base / 'conditions.json').write_text(json.dumps({'identity': identity(), 'budget': budget()}), encoding='utf-8')
            calls = [
                ['prepare', '--dataset', 'dataset.csv', '--revision', 'a' * 40,
                 '--sha256', self.pin['file_sha256'], '--conditions', 'conditions.json', '--output', 'prepared.json'],
                ['grade', '--prepared', 'prepared.json', '--records', 'observations.json', '--output', 'graded.json'],
                ['compare', '--baseline', 'graded.json', '--candidate', 'graded.json', '--output', 'paired.json'],
                ['estimate', '--prepared', 'prepared.json', '--reports', 'graded.json', '--repetitions', '1',
                 '--fixed-overhead-seconds', '0', '--output', 'estimate.json']]
            with patch('vesper.workspace.Workspace.open', return_value=Workspace()), redirect_stdout(io.StringIO()):
                self.assertEqual(quality.main(calls[0]), 0)
                prepared = json.loads((base / 'prepared.json').read_text(encoding='utf-8'))
                self.assertEqual([r['id'] for r in prepared['requests']], prepared['run_plan']['question_ids'])
                self.assertFalse(any('correct_label' in r for r in prepared['requests']))
                (base / 'observations.json').write_text(json.dumps(records(prepared['manifest'], prepared['run_plan'])), encoding='utf-8')
                for call in calls[1:]:
                    self.assertEqual(quality.main(call), 0)
                original = (base / 'estimate.json').read_bytes()
                with self.assertRaises(FileExistsError):
                    quality.main(calls[-1])
                self.assertEqual((base / 'estimate.json').read_bytes(), original)
            estimate = json.loads(original)
            self.assertEqual(estimate['total_seconds']['stratified_mean'], 1980)
            self.assertFalse(estimate['full_run_authorized'])


if __name__ == "__main__":
    unittest.main()
