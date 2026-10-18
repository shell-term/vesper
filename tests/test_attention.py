import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from vesper.attention import CANDIDATES, NativeAttention, argument_names, kernel_filename, load_bundle, validate_inputs
from vesper.attention_inputs import CASES, PROFILES, attention_inputs
from vesper.attention_memory import FUNCTION
from vesper.attention_storage import STAGES
from vesper.attention_validation import check_arrays, validate_goldens
from vesper.checkpoint import MODEL, REVISION, REFERENCE_HASHES

ROOT = Path(__file__).resolve().parents[1]


def rendered(value):
    return (json.dumps(value, indent=2) + "\n").encode()


def fake_bundle(directory):
    """CPU-only schema fixtures, never presented as qualified executable code."""
    receipt = json.loads((ROOT / "results/2026-10-18-attention-memory/attention-memory.json").read_bytes())
    profiles = []
    for (h, d, trace), (source, shared) in CANDIDATES.items():
        binary = f"CPU schema fixture {h} {d} {trace}".encode()
        filename = kernel_filename(h, d, trace)
        (directory / filename).write_bytes(binary)
        profiles.append({"heads": h, "dimension": d, "diagnostics": trace, "file": filename,
                         "function": FUNCTION, "argument_names": argument_names(trace),
                         "generated_cuda_sha256": source, "dynamic_shared_bytes": shared,
                         "cubin_sha256": hashlib.sha256(binary).hexdigest(), "binary_reproduction": True})
    receipt["native_candidate_exports"] = profiles
    receipt_raw = rendered(receipt)
    manifest = {"schema_version": 1, "kind": "vesper_attention_register_query_bundle", "model": MODEL,
                "revision": REVISION, "compute_capability": "8.9", "minimum_driver_api": 13040,
                "tilelang_version": "0.1.12", "nvrtc_version": [13, 4], "kernel_source_sha256": REFERENCE_HASHES["kernel.py"],
                "qualification_sha256": hashlib.sha256(receipt_raw).hexdigest(), "profiles": profiles}
    (directory / "qualification.json").write_bytes(receipt_raw)
    (directory / "manifest.json").write_bytes(rendered(manifest))
    # The upstream artifact is CRLF; Git's text checkout may normalize it.
    license_raw = (ROOT / "third_party/DeepSeek-LICENSE").read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    (directory / "DeepSeek-LICENSE").write_bytes(license_raw)
    return manifest, receipt


class NativeAttentionTests(unittest.TestCase):
    def setUp(self):
        # Replace executable hashes with the explicitly nonexecutable CPU fixture.
        hashes = {key: hashlib.sha256(f"CPU schema fixture {key[0]} {key[1]} {key[2]}".encode()).hexdigest()
                  for key in CANDIDATES}
        binaries = patch("vesper.attention.CANDIDATE_BINARIES", hashes)
        binaries.start()
        self.addCleanup(binaries.stop)
        # CI machines need not meet the real GPU probe's 32-GiB RAM floor.
        memory = patch("vesper.resources.physical_memory", return_value={"available": 128 * 1024**3, "total": 160 * 1024**3})
        memory.start()
        self.addCleanup(memory.stop)

    def test_numpy_inputs_reproduce_all_previous_oracle_hashes(self):
        receipt = json.loads((ROOT / "results/2026-10-18-attention-memory/attention-memory.json").read_bytes())
        for row in receipt["cases"]:
            arrays, provenance = attention_inputs(row["heads"], row["dimension"], CASES[row["case"]])
            self.assertEqual(provenance, row["input_sha256"])
            validate_inputs(*arrays, diagnostics=True)

    def test_input_contract_rejects_unbounded_wrong_layout_indices_and_nonfinite(self):
        arrays, _ = attention_inputs(64, 512, CASES[0])
        for position, bad in ((0, arrays[0].astype(np.float32)), (0, arrays[0][:, :, :, ::-1]),
                              (0, arrays[0][:, 0]), (1, arrays[1][:, :0]), (2, np.zeros(16, np.float32)),
                              (3, np.full_like(arrays[3], 257)), (3, np.full_like(arrays[3], -2))):
            invalid = list(arrays)
            invalid[position] = bad
            with self.assertRaises(ValueError):
                validate_inputs(*invalid)
        for position in range(3):
            invalid = [a.copy() for a in arrays]
            invalid[position].flat[0] = np.inf if position == 2 else 0x7F80
            with self.assertRaisesRegex(ValueError, "finite"):
                validate_inputs(*invalid)
        with self.assertRaises(ValueError):
            validate_inputs(*arrays, diagnostics=1)

    def test_bundle_binding_and_byte_corruption_are_checked_without_cuda(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            manifest, _ = fake_bundle(directory)
            loaded, binaries, sha = load_bundle(directory)
            self.assertEqual(loaded, manifest)
            self.assertEqual(len(binaries), 6)
            self.assertEqual(sha, hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest())
            filename = manifest["profiles"][0]["file"]
            raw = (directory / filename).read_bytes()
            (directory / filename).write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
            with self.assertRaisesRegex(ValueError, "CUBIN hash"):
                load_bundle(directory)
            with patch("vesper.attention.Driver", side_effect=AssertionError("CUDA must not be initialized")):
                runtime = NativeAttention(directory)
                with self.assertRaisesRegex(ValueError, "CUBIN hash"):
                    runtime.__enter__()
                self.assertEqual(runtime.budget.reserved, 0)

    def test_missing_receipt_changed_toolchain_and_forged_profile_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            manifest, receipt = fake_bundle(directory)
            for field, changed in (("nvrtc_version", [13, 3]), ("compute_capability", "8.6"), ("revision", "0" * 40)):
                broken = {**manifest, field: changed}
                (directory / "manifest.json").write_bytes(rendered(broken))
                with self.assertRaisesRegex(ValueError, "toolchain"):
                    load_bundle(directory)
            for change in (lambda p: p.update(file="../../outside.cubin"), lambda p: p.update(heads=32),
                           lambda p: p.update(dynamic_shared_bytes=0), lambda p: p.update(binary_reproduction=False),
                           lambda p: p.update(argument_names=list(reversed(p["argument_names"]))),
                           lambda p: p.update(cubin_sha256="0" * 64), lambda p: p.update(generated_cuda_sha256="0" * 64)):
                broken, proof = copy.deepcopy(manifest), copy.deepcopy(receipt)
                change(broken["profiles"][0])
                proof["native_candidate_exports"] = broken["profiles"]
                raw = rendered(proof)
                broken["qualification_sha256"] = hashlib.sha256(raw).hexdigest()
                (directory / "qualification.json").write_bytes(raw)
                (directory / "manifest.json").write_bytes(rendered(broken))
                with self.assertRaises(ValueError):
                    load_bundle(directory)

    def test_bitwise_check_distinguishes_signed_zero_and_nan_payload(self):
        a = np.array([0, 0x7FC00001], np.uint32).view(np.float32)
        b = np.array([0x80000000, 0x7FC00002], np.uint32).view(np.float32)
        self.assertEqual(check_arrays(a, b, "bits")["mismatches"], 2)

    def test_golden_recheck_refuses_missing_cases_or_boundaries(self):
        report = {"probe": "native_attention_parity", "status": "pass", "reference_goldens": [
            {"heads": h, "dimension": d, "case": i, "stage_sha256": {n: "0" * 64 for n in ("output_bf16", *STAGES)}}
            for h, d in PROFILES for i in range(len(CASES))]}
        self.assertEqual(len(validate_goldens(report)), 36)
        for change in (lambda r: r["reference_goldens"].pop(),
                       lambda r: r["reference_goldens"][0]["stage_sha256"].pop("qk_fp32"),
                       lambda r: r["reference_goldens"][0]["stage_sha256"].update(output_bf16="invalid")):
            broken = copy.deepcopy(report)
            change(broken)
            with self.assertRaises(ValueError):
                validate_goldens(broken)

    def test_call_requires_open_context_and_enter_failure_releases_allowance(self):
        runtime = NativeAttention(Path("unused"))
        with self.assertRaisesRegex(RuntimeError, "not open"):
            runtime(None, None, None, None)
        with patch("vesper.attention.load_bundle", side_effect=ValueError("invalid bundle")):
            with self.assertRaisesRegex(ValueError, "invalid bundle"):
                runtime.__enter__()
        self.assertEqual(runtime.stats()["host_reserved_bytes"], 0)
        with patch("vesper.resources.physical_memory", return_value={"available": 16 * 1024**3}):
            with self.assertRaisesRegex(MemoryError, "system memory reserve"):
                runtime.__enter__()
        self.assertEqual(runtime.stats()["host_reserved_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
