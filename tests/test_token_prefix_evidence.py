import copy
import json
from pathlib import Path
import unittest

from vesper.model_artifacts import digest
from vesper.token_prefix_validation import validate_original, validate_native


class TokenPrefixEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1] / "results/2026-10-19-token-prefix"
        cls.raw = (cls.root / "original.json").read_bytes()
        cls.original = json.loads(cls.raw)
        cls.native = json.loads((cls.root / "native.json").read_bytes())

    def test_complete_published_original_native_and_file_receipts(self):
        self.assertTrue(validate_native(self.native, self.original, self.raw))
        summary = json.loads((self.root / "summary.json").read_bytes())
        for name, receipt in summary["files"].items():
            raw = (self.root / name).read_bytes()
            self.assertEqual((len(raw), digest(raw)), (receipt["bytes"], receipt["sha256"]))
        self.assertEqual(summary["original_baseline"], {"token_calls": 2, "complete_block_calls": 8,
            "typed_shaped_boundaries": 2037, "storage_elements": 14697514})
        self.assertEqual(summary["native_replay"]["retry_boundaries"], 9414)
        self.assertFalse(summary["issue_complete"])
        self.assertFalse(summary["milestone_M2_complete"])

    def test_original_omission_wrong_shape_or_model_claim_is_rejected(self):
        for mutation in (lambda r: r["cases"][1]["boundaries"].pop(),
                         lambda r: r["cases"][0]["boundaries"][0].__setitem__("shape", [1]),
                         lambda r: r["scope"].__setitem__("head_attached_after_layer", 39),
                         lambda r: r["scope"].__setitem__("text_generation_available", True),
                         lambda r: r["expert_plan"]["1"].__setitem__("store", "../unregistered")):
            report = copy.deepcopy(self.original); mutation(report)
            with self.assertRaises(ValueError): validate_original(report)

    def test_native_incomplete_or_changed_retry_and_lost_recovery_owner_are_rejected(self):
        for mutation in (lambda r: r["faults"].pop(),
                         lambda r: r["faults"][0]["retry_boundaries"].pop(),
                         lambda r: r["faults"][1]["retry_boundaries"][0].__setitem__("actual_sha256", "0" * 64),
                         lambda r: r["faults"][-1]["recovery_pending_before_disarm"].pop(),
                         lambda r: r["cases"][1].__setitem__("state_sha256", "0" * 64)):
            report = copy.deepcopy(self.native); mutation(report)
            with self.assertRaises(ValueError): validate_native(report, self.original, self.raw)

    def test_native_framework_dependency_leaks_and_unbound_raw_file_are_rejected(self):
        for mutation in (lambda r: r["native_frameworks"]["torch"].__setitem__("installed", True),
                         lambda r: r["memory_after_policy_close"].__setitem__("charged_host_bytes", 1),
                         lambda r: r["outer_KV_checkpoint"]["cases"][0].__setitem__("restored_ring_sha256", "0" * 64),
                         lambda r: r.__setitem__("weight_payloads_fetched_in_recheck", 1)):
            report = copy.deepcopy(self.native); mutation(report)
            with self.assertRaises(ValueError): validate_native(report, self.original, self.raw)
        with self.assertRaises(ValueError): validate_native(self.native, self.original, self.raw + b"\n")


if __name__ == "__main__": unittest.main()
