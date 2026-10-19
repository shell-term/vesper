import unittest
from contextlib import redirect_stderr
from concurrent.futures import ThreadPoolExecutor
import io
from pathlib import Path
from types import SimpleNamespace
import tempfile
from unittest.mock import Mock

import numpy as np

from vesper.generation import GenerationRequest, TextGeneration
from vesper.backbone import NativeBackbone
from vesper.generation_assets import checked_cache, embedding_address, prefetch_expert
from vesper.generation_probe import main, read_evidence
from vesper.generation_validation import plan_identity


class Tokenizer:
    def encode(self, text): return [10, 11, 12]
    def decode(self, ids): return "/".join(map(str, ids))


class GenerationTests(unittest.TestCase):
    def test_prefix_does_not_sample_and_continuation_is_fed_back(self):
        calls = []
        def step(ids, *, sample):
            token = int(ids[0, 0]); calls.append((token, sample))
            return {"sampling": {"ids_i64": np.array([token + 1], np.int64)}} if sample else {"sampling": {}}
        session = TextGeneration(Tokenizer(), step, GenerationRequest("a prompt", max_new_tokens=3))
        while not session.done: session.advance()
        self.assertEqual(calls, [(10, False), (11, False), (12, True), (13, True), (14, True)])
        self.assertEqual(session.summary()["completion_ids"], [13, 14, 15])
        self.assertEqual(session.summary()["completion_text"], "13/14/15")
        self.assertEqual(session.stop_reason, "max_new_tokens")
        with self.assertRaises(StopIteration): session.advance()

    def test_eos_stops_and_is_omitted_from_text(self):
        def step(ids, *, sample): return {"sampling": {"ids_i64": np.array([1], np.int64)}}
        session = TextGeneration(Tokenizer(), step, GenerationRequest("prompt"))
        while not session.done: session.advance()
        self.assertEqual(session.position, 3)
        self.assertEqual(session.generated, (1,))
        self.assertEqual(session.summary()["completion_text"], "")
        self.assertEqual(session.stop_reason, "eos")

    def test_failed_token_can_retry_without_publishing_a_continuation(self):
        fail = [True]
        def step(ids, *, sample):
            if sample and fail[0]: raise RuntimeError("transaction restored")
            return {"sampling": {"ids_i64": np.array([77], np.int64)}}
        session = TextGeneration(Tokenizer(), step, GenerationRequest("prompt", max_new_tokens=1))
        session.advance(); session.advance()
        with self.assertRaisesRegex(RuntimeError, "transaction restored"): session.advance()
        self.assertEqual((session.position, session.generated, session.done), (2, (), False))
        fail[0] = False; session.advance()
        self.assertEqual(session.generated, (77,))

    def test_invalid_and_excessive_requests_fail_before_model_execution(self):
        def forbidden(*a, **kw): self.fail("Model was called")
        for request in (GenerationRequest(""), GenerationRequest("a", 7), GenerationRequest("a", True), GenerationRequest("a", 0),
                GenerationRequest("a", eos_id=True), GenerationRequest("a", eos_id=129280)):
            with self.assertRaises(ValueError): TextGeneration(Tokenizer(), forbidden, request)

    def test_intermediate_prompt_head_preserves_sampler_state(self):
        model = NativeBackbone.__new__(NativeBackbone)
        hidden = np.zeros((1, 1, 4, 5120), np.uint16); pre = np.zeros((1, 1, 4), np.float32)
        collapsed = np.zeros((1, 1, 5120), np.uint16)
        model.hc = {39: SimpleNamespace(pre=Mock(return_value={"output_bf16": collapsed}))}
        model.norm = SimpleNamespace(norm=Mock(return_value={"output_bf16": collapsed}))
        logits = np.zeros((1, 129280), np.float32)
        model.head = SimpleNamespace(project=Mock(return_value={"logits_fp32": logits}))
        model.transaction = SimpleNamespace(sampler=SimpleNamespace(offset=0, sample=Mock(return_value={"ids_i64": np.array([88], np.int64)})))
        model.norm_weight, model.temperature = np.ones(5120, np.uint16), .7
        result = model._finish({"output_bf16": hidden, "next_pre_mix_fp32": pre}, sample=False)
        self.assertIs(result["head"]["logits_fp32"], logits)
        self.assertEqual(result["sampling"], {"offset_before": 0, "offset_after": 0})
        model.transaction.sampler.sample.assert_not_called()
        result = model._finish({"output_bf16": hidden, "next_pre_mix_fp32": pre})
        model.transaction.sampler.sample.assert_called_once_with(logits, .7)
        self.assertEqual(result["sampling"]["ids_i64"].tolist(), [88])

    def test_original_input_cache_refuses_wrong_extent_or_source(self):
        catalog = {"embed.weight": {"dtype": "BF16", "shape": [129280, 5120], "shard": "original", "shard_byte_offset": 123}}
        source = embedding_address(catalog, 5)
        self.assertEqual(source["shard_byte_offset"], 123 + 5 * 10240)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "5.bin"; fetch = Mock(return_value=bytes(10240))
            raw, _ = checked_cache(path, source, fetch)
            self.assertEqual(raw, bytes(10240))
            checked_cache(path, source, fetch); fetch.assert_called_once()
            with self.assertRaisesRegex(ValueError, "differs"): checked_cache(path, {**source, "shard_byte_offset": 124}, fetch)
            path.write_bytes(bytes(10241))
            with self.assertRaisesRegex(ValueError, "Incomplete"): checked_cache(path, source, fetch)
        for token in (True, -1, 129280):
            with self.assertRaises(ValueError): embedding_address(catalog, token)

    def test_probe_rejects_unbounded_run_path_before_opening_assets(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit): main(["native", "--run", "../other"])

    def test_oversized_evidence_is_rejected_before_reading_it(self):
        path = Mock(); path.stat.return_value.st_size = 64 * 1024**2 + 1
        with self.assertRaises(ValueError): read_evidence(path)
        path.open.assert_not_called()

    def test_cached_download_accounting_does_not_change_fixture_identity(self):
        plan = {k: k for k in ("model", "revision", "request", "layers", "temperature", "seed", "bindings_sha256",
            "output_fixture", "tokenizer", "calls", "completion", "assets")}
        plan["experts"] = {str(l): {"ids": [1, 2], "store": f"stores/layer{l}", "new_original_bytes": 42} for l in range(40)}
        original = plan_identity(plan)
        plan["reader"] = {"payload_bytes_fetched": 0}
        for entry in plan["experts"].values(): entry["new_original_bytes"] = 0
        self.assertEqual(plan_identity(plan), original)
        for l, entry in plan["experts"].items(): entry["store"] = f"generation/another-location/stores/layer{l}"
        self.assertEqual(plan_identity(plan), original)
        plan["experts"]["39"]["ids"] = [1, 3]
        self.assertNotEqual(plan_identity(plan), original)

    def test_parallel_import_preserves_exact_six_original_ranges_and_headers(self):
        catalog, expected = {}, {}
        offset = 1000
        for p in ("w1", "w2", "w3"):
            for suffix, size in (("weight", 5898240), ("scale", 368640)):
                catalog[f"layers.2.ffn.experts.7.{p}.{suffix}"] = {"shard": "shard", "shard_byte_offset": offset, "bytes": size}
                expected[("shard", offset, size)] = str(offset).encode(); offset += size
        calls = []
        def ranges(shard, start, size):
            calls.append((shard, start, size))
            return b"header" if start in (0, 8) else expected[(shard, start, size)]
        reader = SimpleNamespace(catalog=catalog, range=ranges, small=lambda path: b"metadata")
        with ThreadPoolExecutor(max_workers=2) as pool:
            with prefetch_expert(reader, 2, 7, pool) as prepared:
                self.assertEqual(prepared.small("index"), b"metadata")
                self.assertEqual(prepared.range("shard", 0, 8), b"header")
                for key, value in expected.items(): self.assertEqual(prepared.range(*key), value)
                with self.assertRaises(ValueError): prepared.range("shard", 123, 16)
        self.assertEqual(sorted(c for c in calls if c[1] != 0), sorted(expected))
        catalog["layers.2.ffn.experts.7.w3.scale"]["bytes"] += 1
        with ThreadPoolExecutor(max_workers=2) as pool, self.assertRaises(ValueError):
            with prefetch_expert(reader, 2, 7, pool): self.fail("Bad original extent was admitted")


if __name__ == "__main__": unittest.main()
