"""CPU transaction proofs for the Spark-only stateless prefill-tail omission.

Fake arithmetic is used solely to test call/state/feedback contracts. These
checks are not model numerical parity or Spark hardware qualification.
"""

from contextlib import contextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from vesper.formats import bf16_bits, bf16_values
from vesper.generation import GenerationRequest, TextGeneration
from vesper.ngram import NgramState
from vesper.resources import GiB
from vesper.spark_backbone_runtime import SparkBackboneCandidate
from vesper.token_transaction import TokenTransaction


class State:
    def __init__(self):
        self.position, self.payload, self.handle = 0, b"initial", None

    def snapshot(self):
        if self.handle is not None:
            raise RuntimeError("Snapshot already owned")
        self.handle = object()
        self.saved = self.position, self.payload
        return self.handle

    def rollback(self, handle):
        if handle is not self.handle:
            raise ValueError("Foreign snapshot")
        self.position, self.payload = self.saved
        self.handle = None

    def discard(self, handle):
        if handle is not self.handle:
            raise ValueError("Foreign snapshot")
        self.handle = None


class Sampler(State):
    def __init__(self):
        super().__init__()
        self.position = 2026
        self.calls = 0

    @property
    def offset(self):
        return self.position

    def sample(self, logits, temperature):
        before = self.offset
        selected = 4 if self.calls == 0 else 1
        self.calls += 1
        self.position += 4
        return {"ids_i64": np.array([selected], np.int64), "offset_before": before, "offset_after": self.offset}


class Tokenizer:
    def __init__(self, ids): self.ids = ids
    def encode(self, text): return list(self.ids)
    def decode(self, ids): return "".join(str(i) for i in ids)


class SparkPrefillTests(unittest.TestCase):
    def setUp(self):
        memory = patch("vesper.resources.physical_memory", return_value={"available": 64 * GiB})
        memory.start()
        self.addCleanup(memory.stop)

    @contextmanager
    def engine(self):
        # The real Spark method and inherited ordinary step/_finish are used;
        # only layer operators and output arithmetic are CPU test doubles.
        engine = object.__new__(SparkBackboneCandidate)
        states = {name: State() for name in ("window0", "window1", "sharing2_39")}
        history = NgramState(np.arange(129280, dtype=np.int64) % 99092)
        sampler = Sampler()
        engine.transaction = TokenTransaction(states, history, sampler)
        engine.temperature = .7
        engine.embeddings = {i: np.zeros(5120, np.uint16) for i in (1, 4, 10, 11, 12)}
        engine.norm_weight = np.ones(5120, np.uint16)
        collapse = Mock(side_effect=lambda hidden, pre: {"output_bf16": bf16_bits(np.sum(bf16_values(hidden) * pre[..., None], axis=2))})
        norm = Mock(side_effect=lambda hidden, weight: {"output_bf16": hidden.copy()})
        head = Mock(side_effect=lambda hidden: {"logits_fp32": np.sum(bf16_values(hidden), axis=-1, keepdims=True)})
        engine.hc = {39: SimpleNamespace(pre=collapse)}
        engine.norm = SimpleNamespace(norm=norm)
        engine.head = SimpleNamespace(project=head)
        engine.evaluate_calls = []
        def evaluate(candidate, ids):
            engine.evaluate_calls.append((candidate.start, int(ids[0, 0])))
            for state in states.values():
                state.position += 1
                state.payload = candidate.hash_ids.tobytes()
            hidden = bf16_bits(np.full((1, 1, 4, 5120), float(ids[0, 0]), np.float32))
            return {"output_bf16": hidden, "next_pre_mix_fp32": np.full((1, 1, 4), .25, np.float32),
                    "all_backbone_layers": list(range(40)), "both_engrams": [1, 14]}
        engine._evaluate = evaluate
        with engine.transaction:
            yield engine, states, history, sampler
        self.assertEqual(engine.transaction.budget.reserved, 0)
        self.assertTrue(all(state.handle is None for state in (*states.values(), sampler)))
        self.assertFalse(history.handles)

    def drive(self, *, optimized, ids=(10, 11, 12)):
        with self.engine() as (engine, states, history, sampler):
            session = None
            def step(token, *, sample):
                if optimized and not sample:
                    self.assertLess(session.position, len(session.prompt_ids) - 1)
                    return engine.prefill_step(token)
                return engine.step(token, sample=sample)
            session = TextGeneration(Tokenizer(ids), step, GenerationRequest("prompt", max_new_tokens=3))
            results = []
            while not session.done:
                meta, value = session.advance()
                results.append((meta, value))
            return {"summary": session.summary(), "results": results,
                    "calls": list(engine.evaluate_calls), "state": [(s.position, s.payload) for s in states.values()],
                    "history": history.state_sha256(), "sampler_offset": sampler.offset, "sampler_calls": sampler.calls,
                    "collapse_calls": engine.hc[39].pre.call_count, "norm_calls": engine.norm.norm.call_count,
                    "head_calls": engine.head.project.call_count}

    def test_only_first_n_minus_one_input_tails_are_omitted_and_feedback_is_equal(self):
        baseline = self.drive(optimized=False)
        optimized = self.drive(optimized=True)
        for key in ("summary", "calls", "state", "history", "sampler_offset", "sampler_calls"):
            self.assertEqual(baseline[key], optimized[key])
        self.assertEqual(optimized["calls"], [(0, 10), (1, 11), (2, 12), (3, 4)])
        self.assertEqual(baseline["head_calls"], 4)
        self.assertEqual(optimized["head_calls"], 2)
        self.assertEqual(optimized["collapse_calls"], 2)
        self.assertEqual(optimized["norm_calls"], 2)
        for index, ((left_meta, left), (right_meta, right)) in enumerate(zip(baseline["results"], optimized["results"])):
            self.assertEqual(left_meta, right_meta)
            for boundary in ("output_bf16", "next_pre_mix_fp32"):
                np.testing.assert_array_equal(left[boundary], right[boundary])
            self.assertEqual(right["all_backbone_layers"], list(range(40)))
            self.assertEqual(right["both_engrams"], [1, 14])
            if index < 2:
                self.assertNotIn("head", right)
                self.assertEqual(right["output_tail"], "skipped_nonfinal_input")
                self.assertEqual(right["sampling"], {"offset_before": 2026, "offset_after": 2026})
            else:
                np.testing.assert_array_equal(left["head"]["logits_fp32"], right["head"]["logits_fp32"])
                np.testing.assert_array_equal(left["sampling"]["ids_i64"], right["sampling"]["ids_i64"])

    def test_single_input_and_ordinary_sample_false_keep_complete_output_path(self):
        value = self.drive(optimized=True, ids=(10,))
        self.assertEqual(value["head_calls"], 2)
        self.assertFalse(any("output_tail" in item for _, item in value["results"]))
        with self.engine() as (engine, _, _, sampler):
            result = engine.step(np.array([[10]], np.int64), sample=False)
            self.assertIn("head", result)
            self.assertNotIn("output_tail", result)
            self.assertEqual(engine.head.project.call_count, 1)
            self.assertEqual(sampler.offset, 2026)
            self.assertEqual(sampler.calls, 0)

    def test_late_prefill_observer_failure_restores_state_and_retry_commits_once(self):
        with self.engine() as (engine, states, history, sampler):
            before = history.state_sha256()
            def fail_after_backbone(result):
                self.assertNotIn("head", result)
                sampler.position += 99
                raise OSError("injected late prefill observation")
            with self.assertRaisesRegex(OSError, "late prefill"):
                engine.prefill_step(np.array([[10]], np.int64), observe=fail_after_backbone)
            self.assertEqual(engine.position, 0)
            self.assertEqual(history.state_sha256(), before)
            self.assertTrue(all(s.position == 0 and s.payload == b"initial" for s in states.values()))
            self.assertEqual(sampler.offset, 2026)
            result = engine.prefill_step(np.array([[10]], np.int64), start_pos=0)
            self.assertEqual(result["output_tail"], "skipped_nonfinal_input")
            self.assertEqual(engine.position, 1)
            self.assertEqual(history.position, 1)
            self.assertEqual(engine.head.project.call_count, 0)
            self.assertEqual(engine.transaction.stats()["failed_attempts"], 1)
            self.assertEqual(engine.transaction.stats()["committed_tokens"], 1)

    def test_invalid_input_or_observer_refuses_before_computation(self):
        with self.engine() as (engine, _, _, _):
            for ids in (np.array([[10, 11]], np.int64), np.array([[10]], np.int32), np.array([[99]], np.int64)):
                with self.assertRaises(ValueError):
                    engine.prefill_step(ids)
            with self.assertRaises(ValueError):
                engine.prefill_step(np.array([[10]], np.int64), observe="invalid")
            self.assertEqual(engine.evaluate_calls, [])


if __name__ == "__main__":
    unittest.main()
