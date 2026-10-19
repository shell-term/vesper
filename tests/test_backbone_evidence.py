import copy
import json
from pathlib import Path
import unittest

from vesper.backbone_assets import _entries
from vesper.backbone_validation import validate_original, validate_native
from vesper.model_artifacts import canonical, digest


class BackboneEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1] / "results/2026-10-19-backbone"
        cls.raw = (cls.root / "original.json").read_bytes()
        cls.original = json.loads(cls.raw)
        cls.native = json.loads((cls.root / "native.json").read_bytes())

    def test_complete_original_native_controls_and_file_receipts(self):
        self.assertTrue(validate_native(self.native, self.original, self.raw))
        controls = json.loads((self.root / "controls.json").read_bytes())
        self.assertEqual(canonical(controls["layers"]), self.original["bindings_sha256"])
        entries = [e for parts in controls["layers"].values() for p in parts.values() for e in _entries(p["manifest"])]
        self.assertEqual(len(entries), 1251)
        self.assertEqual(len({e.get("tensor", e["name"]) for e in entries}), 1251)
        self.assertEqual(sum(e["bytes"] for e in entries), 7199083968)
        summary = json.loads((self.root / "summary.json").read_bytes())
        for file, receipt in summary["files"].items():
            raw = (self.root / file).read_bytes()
            self.assertEqual((len(raw), digest(raw)), (receipt["bytes"], receipt["sha256"]))
        self.assertEqual(summary["original_baseline"]["complete_block_calls"], 80)
        self.assertEqual(summary["original_baseline"]["full_vocabulary_logits"], 258560)
        self.assertEqual(summary["original_baseline"]["ordered_router_decisions"], 80)
        self.assertFalse(summary["issue_complete"])
        self.assertFalse(summary["milestone_M2_complete"])
        self.assertFalse(summary["scope"]["text_generation_available"])

    def test_changed_original_shape_route_payload_scope_or_layer_is_rejected(self):
        for mutate in (lambda r: r["cases"][0]["boundaries"].pop(),
                       lambda r: r["cases"][0]["boundaries"][0].__setitem__("shape", [1]),
                       lambda r: r["cases"][1]["ordered_routes"].pop("39"),
                       lambda r: r["expert_provenance"].pop(next(iter(r["expert_provenance"]))),
                       lambda r: r["scope"].__setitem__("text_generation_available", True),
                       lambda r: r["scope"]["layers"].pop(),
                       lambda r: r["expert_plan"]["39"].__setitem__("store", "../unregistered")):
            report = copy.deepcopy(self.original); mutate(report)
            with self.assertRaises(ValueError): validate_original(report)

    def test_incomplete_native_comparison_or_wrong_state_is_rejected(self):
        for mutate in (lambda r: r["cases"][1]["boundaries"].pop(),
                       lambda r: r["cases"][0]["boundaries"][0].__setitem__("actual_sha256", "0" * 64),
                       lambda r: r["cases"][1].__setitem__("state_sha256", "0" * 64),
                       lambda r: r["cases"][1].__setitem__("sampled_ids", [-1]),
                       lambda r: r["native_profiles"].pop("39")):
            report = copy.deepcopy(self.native); mutate(report)
            with self.assertRaises(ValueError): validate_native(report, self.original, self.raw)

    def test_false_failure_cancellation_framework_or_cleanup_claim_is_rejected(self):
        for mutate in (lambda r: r["faults"].pop(),
                       lambda r: r["faults"][0].__setitem__("injection_reached", False),
                       lambda r: r["faults"][-1].__setitem__("caught", "TypeError"),
                       lambda r: r["faults"][-1].__setitem__("runtime_cancelled_before_resume", False),
                       lambda r: r["native_frameworks"]["torch"].__setitem__("installed", True),
                       lambda r: r["memory_after_policy_close"].__setitem__("charged_host_bytes", 1),
                       lambda r: r.__setitem__("python_http_disabled_during_native_execution", False)):
            report = copy.deepcopy(self.native); mutate(report)
            with self.assertRaises(ValueError): validate_native(report, self.original, self.raw)
        with self.assertRaises(ValueError): validate_native(self.native, self.original, self.raw + b"\n")


if __name__ == "__main__": unittest.main()
