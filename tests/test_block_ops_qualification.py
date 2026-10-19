import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from vesper import block_ops_qualification as qualification
from vesper.resources import GiB


class BlockOperationsQualificationTests(unittest.TestCase):
    root = Path(__file__).resolve().parents[1]

    def manifest(self):
        return json.loads((self.root / qualification.RESULT_ROOT / "evidence-manifest.json").read_bytes())

    def test_all_seventeen_original_goldens_and_fresh_native_runs_satisfy_original_five_checks(self):
        with patch("vesper.resources.physical_memory", return_value={"available": 64 * GiB}), \
                patch("vesper.block_ops_qualification.physical_memory", return_value={"available": 64 * GiB}):
            result = qualification.qualify(self.root, self.root / qualification.RESULT_ROOT / "evidence-manifest.json")
        self.assertEqual(result["qualified_evidence_groups"], 17)
        self.assertEqual([(c["id"], c["criterion"]) for c in result["acceptance_checks"]], list(qualification.CRITERIA))
        self.assertTrue(all(c["status"] == "pass" for c in result["acceptance_checks"]))
        self.assertFalse(result["M2_complete"])
        self.assertFalse(result["text_generation_available"])
        self.assertFalse(result["scope"]["all_forty_original_layers_numerically_executed"])
        self.assertEqual(result["memory"]["host_reserved_after_bytes"], 0)

    def test_manifest_cannot_drop_substitute_or_redefine_an_acceptance_gate(self):
        manifest = self.manifest()
        qualification.validate_manifest(manifest)
        for change in (lambda m: m["evidence"].pop(),
                lambda m: m["evidence"][0].update(validator="block_pair_validation"),
                lambda m: m["criteria"][0].update(criterion="Preserve approximate outputs"),
                lambda m: m["criteria"][0]["evidence"].pop(),
                lambda m: m["evidence"][0]["native_recheck"].update(file="other.json"),
                lambda m: m["evidence"][0]["reference"].update(bytes=65 * 1024**2)):
            bad = copy.deepcopy(manifest); change(bad)
            with self.assertRaises(ValueError): qualification.validate_manifest(bad)
            # Even a newly issued digest cannot authorize different coverage.
            with patch.object(qualification, "QUALIFIED_MANIFEST_SHA256", qualification.manifest_digest(bad)):
                with self.assertRaises(ValueError): qualification.validate_manifest(bad)

    def test_artifact_bytes_sha_and_root_confinement_are_checked_before_interpretation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = root / "run.json"; raw = b'{"status":"pass"}'
            path.write_bytes(raw)
            artifact = {"file": "run.json", "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            self.assertEqual(qualification.read_verified(root, artifact), {"status": "pass"})
            for change in (lambda a: a.update(file="../run.json"), lambda a: a.update(bytes=len(raw) + 1),
                           lambda a: a.update(sha256="0" * 64)):
                bad = dict(artifact); change(bad)
                with self.assertRaises(ValueError): qualification.read_verified(root, bad)

    def test_wrong_original_binding_or_framework_use_refuses_even_a_resealed_native_report(self):
        manifest = self.manifest()
        read = qualification.read_verified
        for change in (lambda n: n.update(reference_report_sha256="0" * 64),
                       lambda n: n.update(torch_imported=True), lambda n: n.update(tilelang_installed=True)):
            def load(root, artifact):
                data = read(root, artifact)
                if artifact == manifest["evidence"][0]["native_recheck"]: change(data)
                return data
            with patch("vesper.resources.physical_memory", return_value={"available": 64 * GiB}), \
                    patch("vesper.block_ops_qualification.physical_memory", return_value={"available": 64 * GiB}), \
                    patch("vesper.block_ops_qualification.read_verified", side_effect=load):
                with self.assertRaises(ValueError):
                    qualification.qualify(self.root, self.root / qualification.RESULT_ROOT / "evidence-manifest.json")


if __name__ == "__main__": unittest.main()
