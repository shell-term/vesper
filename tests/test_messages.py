from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from vesper.evaluation import score_completion
from vesper.message_cases import encoding_cases, parsing_cases, rejection_cases, quality_cases
from vesper.message_validation import read_report, validate_reference, validate_native
from vesper.messages import BOS, EOS, USER, ASSISTANT, END_THINK, encode_messages, parse_completion

RESULTS = Path(__file__).resolve().parents[1] / "results/2026-10-20-message-format"


class MessageFormatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original, cls.raw = read_report(RESULTS / "original.json")
        cls.native, _ = read_report(RESULTS / "native.json")

    def test_all_original_prompts_without_mutation(self):
        for case, expected in zip(encoding_cases(), self.original["encoding"]):
            with self.subTest(case=case["id"]):
                before = deepcopy(case)
                self.assertEqual(encode_messages(case["messages"], **case["options"]), expected["prompt"])
                self.assertEqual(case, before)
                self.assertTrue(expected["prompt"].startswith(BOS))

    def test_all_original_parser_results(self):
        for case, expected in zip(parsing_cases(), self.original["parsing"]):
            with self.subTest(case=case["id"]):
                self.assertEqual(parse_completion(case["text"], thinking_mode=case["thinking_mode"]), expected["parsed"])

    def test_reused_message_object_keeps_exact_turn_count(self):
        msg = {"role": "user", "content": "u"}
        self.assertEqual(encode_messages([msg, msg, msg]), BOS + USER + "u\n\nu\n\nu" + ASSISTANT + END_THINK)
        self.assertEqual(msg, {"role": "user", "content": "u"})

    def test_both_reject_malformed_format_cases(self):
        for case in rejection_cases():
            with self.subTest(case=case["id"]), self.assertRaises(ValueError):
                if case["operation"] == "encode": encode_messages(case["messages"], **case["options"])
                else: parse_completion(case["text"], thinking_mode=case["thinking_mode"])

    def test_separate_complete_evidence_and_preflight(self):
        self.assertTrue(validate_reference(self.original))
        self.assertTrue(validate_native(self.native, self.original, self.raw))
        self.assertEqual(len(self.original["encoding"]), 30)
        self.assertEqual(len(self.original["parsing"]), 11)
        self.assertEqual(len(self.original["rejection"]), 13)
        for case in self.native["quality_preflight"]:
            self.assertFalse(case["fits_current_fixture"])
            self.assertFalse(case["model_inference_run"])
            self.assertIsNone(case["score"])

    def test_changed_goldens_cannot_self_qualify(self):
        from vesper.model_artifacts import canonical, digest
        changed = deepcopy(self.original)
        changed["encoding"][0]["prompt"] += "x"
        changed["encoding"][0]["prompt_utf8_sha256"] = digest(changed["encoding"][0]["prompt"].encode())
        changed["complete_goldens_sha256"] = canonical({n: changed[n] for n in ("encoding", "parsing", "rejection", "quality_preflight")})
        with self.assertRaises(ValueError): validate_reference(changed)
        validate_reference(changed, require_qualified=False)  # Explicit discovery mode is not qualification.

    def test_native_negative_provenance_and_false_quality_claim(self):
        for mutate in (lambda r: r["encoding"].pop(),
                lambda r: r["reference_frameworks"]["torch"].update(imported=True),
                lambda r: r.update(HTTP_calls_during_execution=1),
                lambda r: r.update(reference_sha256="0" * 64),
                lambda r: r["quality_preflight"][0].update(model_inference_run=True, score=1),
                lambda r: r["quality_preflight"][0].update(status="pass")):
            changed = deepcopy(self.native); mutate(changed)
            with self.assertRaises(ValueError): validate_native(changed, self.original, self.raw)

    def test_explicit_unsupported_fields_and_bounds(self):
        bad = [[{"role": "user", "content": "x", "task": "action"}],
            [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "unused"}}]}],
            [{"role": "user", "content": "x", "context": []}],
            [{"role": "unknown", "content": "x"}],
            [{"role": [], "content": "x"}],
            [{"role": "assistant", "content": "", "reasoning_content": []}],
            [{"role": "assistant", "content": "", "tool_calls": "invalid"}],
            [{"role": "assistant", "tool_calls": [{"id": [], "function": {"name": "f", "arguments": {}}}]}],
            [{"role": "tool", "tool_call_id": [], "content": "x"}],
            [{"role": "user", "content": None}],
            [{"role": "user", "content": [{"type": "text", "text": None}]}],
            [{"role": "tool", "content": None}],
            [{"role": "latest_reminder"}],
            [{"role": "user", "content": "x"}] * 33,
            [{"role": "user", "content": "x" * 4097}],
            [{"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": "broken"}}]}]]
        for messages in bad:
            with self.subTest(messages=str(messages)[:90]), self.assertRaises(ValueError): encode_messages(messages)
        with self.assertRaises(ValueError): encode_messages([{"role": "user", "content": "x"}], thinking_mode="invented")

    def test_read_bound_precedes_payload_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "oversized.json"
            with path.open("wb") as f: f.truncate(2 * 1024**2 + 1)
            with patch.object(Path, "open", side_effect=AssertionError("Must reject before opening payload")):
                with self.assertRaises(ValueError): read_report(path)


class QualityScorerTests(unittest.TestCase):
    def test_reasoning_correct_wrong_and_truncated(self):
        case = quality_cases()[0]
        self.assertTrue(score_completion(case, "3+2-2=3.</think>3" + EOS, stop_reason="eos")["passed"])
        self.assertFalse(score_completion(case, "wrong.</think>4" + EOS, stop_reason="eos")["passed"])
        self.assertEqual(score_completion(case, "3+2-2=3.</think>3", stop_reason="max_new_tokens")["status"], "incomplete")
        self.assertFalse(score_completion(case, "3+2-2=3.</think>3", stop_reason="eos")["passed"])

    def test_coding_semantics_and_invalid_outputs(self):
        case = quality_cases()[1]
        for expression in ("xs[-1]", "xs[len(xs)-1]"):
            self.assertTrue(score_completion(case, expression + EOS, stop_reason="eos")["passed"])
        for expression in ("xs[0]", "xs[999]", "```python\nxs[-1]\n```", "xs["):
            self.assertFalse(score_completion(case, expression + EOS, stop_reason="eos")["passed"])
        # An arbitrary call is never executed. Its absence from the supported
        # AST subset is reported separately from an incorrect model answer.
        self.assertEqual(score_completion(case, "__import__('os').system('invalid')" + EOS, stop_reason="eos")["status"], "unscorable")

    def test_tools_require_exact_name_and_typed_arguments(self):
        case = quality_cases()[2]
        text = '\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="add">\n<｜DSML｜ parameter name="a" string="false">2</｜DSML｜ parameter>\n<｜DSML｜ parameter name="b" string="false">3</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>' + EOS
        self.assertTrue(score_completion(case, text, stop_reason="eos")["passed"])
        for bad in (text.replace('name="add"', 'name="subtract"'), text.replace('>3</', '>4</'),
                text.replace('>2</', '>2.0</'), text.replace('string="false">2', 'string="true">2'),
                text.replace('name="b"', 'name="a"'), "The answer is 5." + EOS):
            self.assertFalse(score_completion(case, bad, stop_reason="eos")["passed"])
        self.assertEqual(score_completion(case, text[:-len(EOS)], stop_reason="max_new_tokens")["status"], "incomplete")

    def test_model_quality_is_not_inferred_from_scorer_examples(self):
        for case in quality_cases():
            self.assertIn(case["category"], ("reasoning", "coding", "tool_use"))
            self.assertNotIn("completion", case)
            self.assertNotIn("model_quality_passed", case)


if __name__ == "__main__": unittest.main()
