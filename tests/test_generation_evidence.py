import copy
import json
from pathlib import Path
import unittest

from vesper.generation_validation import validate_original, validate_native
from vesper.model_artifacts import canonical, digest

ROOT = Path(__file__).resolve().parents[1] / "results/2026-10-19-generation"


class GenerationEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = (ROOT / "original.json").read_bytes()
        cls.original = json.loads(cls.raw)
        cls.native = json.loads((ROOT / "native.json").read_bytes())
        cls.summary = json.loads((ROOT / "summary.json").read_bytes())

    def test_complete_original_and_offline_native_match(self):
        self.assertTrue(validate_original(self.original))
        self.assertTrue(validate_native(self.native, self.original, self.raw))
        for name, receipt in self.summary["files"].items():
            raw = (ROOT / name).read_bytes()
            self.assertEqual((len(raw), digest(raw)), (receipt["bytes"], receipt["sha256"]))
        self.assertEqual(self.summary["typed_shaped_boundaries"], 26250)
        self.assertEqual(self.summary["complete_FP32_logits"], 646400)
        self.assertEqual(self.summary["RNG_offsets"], [0, 0, 4, 8, 12])
        self.assertEqual(self.summary["complete_original_block_calls"], 200)
        self.assertEqual(self.summary["completion"], self.original["completion"])

    def test_reject_teacher_forcing_missing_boundaries_or_wrong_rng(self):
        changed = copy.deepcopy(self.native)
        changed["cases"][3]["meta"]["token_id"] = 2
        with self.assertRaises(ValueError): validate_native(changed, self.original, self.raw)
        changed = copy.deepcopy(self.native)
        changed["cases"][0]["boundaries"].pop()
        with self.assertRaises(ValueError): validate_native(changed, self.original, self.raw)
        changed = copy.deepcopy(self.native)
        changed["cases"][0]["rng_offset"] = 4
        with self.assertRaises(ValueError): validate_native(changed, self.original, self.raw)

    def test_reject_framework_fetch_cleanup_or_reference_binding_changes(self):
        mutations = (
            lambda r: r["native_frameworks"]["torch"].update(installed=True),
            lambda r: r.update(python_http_disabled_during_native_execution=False),
            lambda r: r["memory_after_policy_close"]["device_used_bytes"].update({"0": 1}),
            lambda r: r.update(reference_report_sha256="0" * 64),
            lambda r: r["transaction_cleanup"].update(pending_recovery_owners=["window0"]),
        )
        for mutate in mutations:
            changed = copy.deepcopy(self.native); mutate(changed)
            with self.assertRaises(ValueError): validate_native(changed, self.original, self.raw)
        changed = copy.deepcopy(self.original)
        changed["reference_torch_device_allocated_after_cleanup_bytes"] += 1
        with self.assertRaises(ValueError): validate_original(changed)

    def test_rehashed_altered_original_is_not_automatically_qualified(self):
        changed = copy.deepcopy(self.original)
        changed["cases"][0]["boundaries"][0]["sha256"] = "0" * 64
        changed["complete_goldens_sha256"] = canonical(changed["cases"])
        with self.assertRaisesRegex(ValueError, "qualified"): validate_original(changed)


if __name__ == "__main__": unittest.main()
