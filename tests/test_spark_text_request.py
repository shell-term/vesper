"""CPU request-admission checks; no model, dataset or hardware is opened."""

import unittest
from types import SimpleNamespace

from vesper.spark_text_request import (SparkTextLimits, SparkTextRequest,
                                       MAX_CONTEXT_TOKENS, MAX_COMPLETION_CHARS)
from vesper.sequence import MAX_CALLS, scope


class SparkTextRequestTests(unittest.TestCase):
    def tokenizer(self, ids):
        return SimpleNamespace(encode=lambda text: ids)

    def test_historical_default_scope_and_feedback_call_accounting(self):
        limits = SparkTextLimits()
        request = SparkTextRequest("original prompt", max_new_tokens=510, limits=limits)
        self.assertEqual(request.encode(self.tokenizer([10, 11, 12])), (10, 11, 12))
        self.assertEqual(limits.context_tokens, MAX_CALLS)
        for key, value in scope().items():
            self.assertEqual(limits.scope()[key], value)
        with self.assertRaisesRegex(ValueError, "Input and continuation"):
            SparkTextRequest("prompt", max_new_tokens=511).encode(self.tokenizer([10, 11, 12]))

    def test_actual_composition_capacity_and_no_claim_of_long_context_qualification(self):
        self.assertEqual(MAX_CONTEXT_TOKENS, 1152)
        limits = SparkTextLimits(1152, 1152, 1152, 60.)
        ids = [10] * 1152
        self.assertEqual(len(SparkTextRequest("prompt", max_new_tokens=1, limits=limits).encode(self.tokenizer(ids))), 1152)
        self.assertEqual(limits.scope()["maximum_completion_characters"], MAX_COMPLETION_CHARS)
        self.assertFalse(limits.scope()["extended_context_hardware_qualified"])
        for context in (1153, 4096, 131072, 1048576):
            with self.subTest(context=context), self.assertRaises(ValueError):
                SparkTextLimits(context, 1, 1, 1.)

    def test_independent_input_and_output_limits_refuse_without_truncation(self):
        limits = SparkTextLimits(1152, 3, 1, 60)
        self.assertEqual(SparkTextRequest("prompt", max_new_tokens=1000, limits=limits).encode(self.tokenizer([10, 11, 12])), (10, 11, 12))
        with self.assertRaisesRegex(ValueError, "Encoded input"):
            SparkTextRequest("prompt", max_new_tokens=1, limits=limits).encode(self.tokenizer([10] * 4))
        for ids in ([], [-1], [129280], [True], [1.0]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                SparkTextRequest("prompt", max_new_tokens=1, limits=limits).encode(self.tokenizer(ids))
        for tokens in (0, 1153, True, 2.0):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                SparkTextRequest("prompt", max_new_tokens=tokens, limits=limits).encode(self.tokenizer([10]))

    def test_explicit_duration_required_for_extended_scope_and_malformed_limits_rejected(self):
        with self.assertRaisesRegex(ValueError, "duration"):
            SparkTextLimits(513, 512, 512)
        for seconds in (0, -1, True, float("nan"), float("inf"), 86401, "10"):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                SparkTextLimits(duration_seconds=seconds)
        for args in ((True, 1, 1), (512, True, 1), (512, 1, True), (512, 513, 1),
                     (512, 1, 513), (0, 1, 1), (512, 0, 1), (512, 1, 0)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                SparkTextLimits(*args)

    def test_existing_serializer_character_and_eos_bounds_remain_explicit(self):
        for prompt in ("", "x" * 4097, None):
            with self.subTest(prompt_type=type(prompt)), self.assertRaises(ValueError):
                SparkTextRequest(prompt).encode(self.tokenizer([10]))
        for eos in (-1, 129280, True):
            with self.subTest(eos=eos), self.assertRaises(ValueError):
                SparkTextRequest("prompt", eos_id=eos).encode(self.tokenizer([10]))
        self.assertTrue(SparkTextLimits().admission(64)["one_inflight_model_call_may_overrun_duration"])
        self.assertFalse(SparkTextLimits().admission(64)["silent_truncation"])


if __name__ == "__main__":
    unittest.main()
