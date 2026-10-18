import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from vesper.checkpoint import MODEL, REVISION
from vesper.compression import CompressionSnapshot, NativeCompressedKV, NativeCompression, validate_compression_input, validate_quantized_kv
from vesper.compression_fixture import COMPRESSION_BYTES, COMPRESSION_LAYOUT, fetch_compression_fixture, load_compression_fixture
from vesper.compression_validation import component_cases, expected_metadata, validate_goldens
from vesper.normalization import NativeNormRoPE
from vesper.resources import GiB


def toy_backend():
    """Controlled outputs for publication-ownership checks, not a numeric oracle."""
    return SimpleNamespace(opened=True, arrays={"layers.20.attn.compressor.norm.weight": np.zeros(512, np.uint16)},
        project=lambda x, layer: {"wkv_bf16": x[..., :512].copy()},
        norm=SimpleNamespace(norm=lambda x, weight: {"output_bf16": x.copy()}, rope=lambda x, *a, **k: {"output_bf16": x.copy()}),
        round_kv=lambda x: {"rounded_bf16": x.copy()})


class CompressionTests(unittest.TestCase):
    def test_layout_position_and_bounded_quantization_refusal(self):
        value = np.zeros((2, 3, 5120), np.uint16)
        validate_compression_input(value, 2, 0, 0)
        for bad, batch, pos, expected in ((value.astype(np.float32), 2, 0, 0), (value[..., ::-1], 2, 0, 0),
                (value, 1, 0, 0), (value, True, 0, 0), (value, 2, 1, 1), (value, 2, 0, 1),
                (np.zeros((1, 17, 5120), np.uint16), 1, 0, 0), (np.full_like(value, 0x7fc0), 2, 0, 0)):
            with self.assertRaises(ValueError): validate_compression_input(bad, batch, pos, expected)
        validate_quantized_kv(np.full((1, 1, 512), 0xc528, np.uint16))
        for bad in (np.full((1, 1, 512), 0x4529, np.uint16), np.full((1, 1, 512), 0x7f80, np.uint16),
                    np.zeros((1, 0, 512), np.uint16), np.zeros((1, 1, 256), np.uint16), np.zeros((1, 1, 512), np.float32)):
            with self.assertRaises(ValueError): validate_quantized_kv(bad)

    def test_missing_weights_and_existing_destination_refuse_before_io(self):
        with patch("vesper.compression.Driver", side_effect=AssertionError("No driver access expected")):
            with self.assertRaises(ValueError): NativeCompression({})
        with tempfile.TemporaryDirectory() as tmp, patch("vesper.compression_fixture.read_small", side_effect=AssertionError("No network expected")):
            with self.assertRaises(FileExistsError): fetch_compression_fixture(Path(tmp))

    def test_pressure_refuses_before_allocating_state(self):
        state = NativeCompressedKV(toy_backend(), 20)
        with patch("vesper.resources.physical_memory", return_value={"available": 16 * GiB}), patch.object(state, "_clear", side_effect=AssertionError("No allocation expected")):
            with self.assertRaises(MemoryError): state.__enter__()
        self.assertFalse(state.opened)
        self.assertEqual(state.budget.reserved, 0)

    def test_diagnostics_cannot_mutate_committed_host_cache(self):
        with patch("vesper.resources.physical_memory", return_value={"available": 64 * GiB}):
            with NativeCompressedKV(toy_backend(), 20) as state:
                result = state.step(np.full((1, 1, 5120), 0x3f80, np.uint16))
                for key in ("cache_bf16", "visible_bf16", "latent_pre_rope_bf16"):
                    result[key].fill(0)
                self.assertTrue((state.inspect()["cache_bf16"][:, 0] == 0x3f80).all())
            state.__exit__()
            self.assertEqual(state.budget.reserved, 0)

    def test_reset_allocation_failure_keeps_position_cache_and_snapshot(self):
        with patch("vesper.resources.physical_memory", return_value={"available": 64 * GiB}):
            with NativeCompressedKV(toy_backend(), 20) as state:
                state.step(np.full((1, 1, 5120), 0x3f80, np.uint16))
                handle = state.snapshot()
                with patch("vesper.compression.np.zeros", side_effect=MemoryError("Injected host allocation failure")):
                    with self.assertRaises(MemoryError): state.reset()
                self.assertEqual(state.position, 1)
                self.assertIs(state.handle, handle)
                self.assertTrue((state.inspect()["cache_bf16"][:, 0] == 0x3f80).all())

    def test_snapshot_identity_reset_and_expiry(self):
        with patch("vesper.resources.physical_memory", return_value={"available": 64 * GiB}):
            with NativeCompressedKV(toy_backend(), 20) as state:
                handle = state.snapshot()
                with self.assertRaises(RuntimeError): state.snapshot()
                with self.assertRaises(ValueError): state.rollback(CompressionSnapshot(0))
                state.rollback(handle)
                with self.assertRaises(ValueError): state.discard(handle)
                handle = state.snapshot()
                state.reset()
                with self.assertRaises(ValueError): state.rollback(handle)

    def test_explicit_rotary_position_contract_before_kernel_launch(self):
        native = NativeNormRoPE()
        native._ready = lambda: None
        value = np.zeros((1, 2, 512), np.uint16)
        for positions in (np.array([0], np.int32), np.array([1, 2], np.int32), np.array([0, 2], np.int64), np.array([0, 131072], np.int32)):
            with self.assertRaises(ValueError): native.rope(value, positions=positions)

    def test_compressor_archive_rejects_forged_npy_shape_before_numpy_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entries = [{"name": n, "dtype": "BF16", "shape": list(s), "bytes": int(np.prod(s)) * 2, "sha256": "0" * 64}
                       for n, s in COMPRESSION_LAYOUT.items()]
            manifest = {"kind": "original_compression_fixture", "model": MODEL, "revision": REVISION,
                        "layers_and_ratios": {"2": 2, "20": 1}, "weights_fetched_bytes": COMPRESSION_BYTES, "tensors": entries}
            (root / "manifest.json").write_text(json.dumps(manifest))
            path = root / "native-compression.npz"
            with zipfile.ZipFile(path, "w") as archive:
                for name in COMPRESSION_LAYOUT:
                    with archive.open(name + ".npy", "w") as stream:
                        np.lib.format.write_array_header_1_0(stream, {"shape": (2**50,), "fortran_order": False, "descr": "<u2"})
            with patch("vesper.compression_fixture.np.load", side_effect=AssertionError("No array allocation expected")):
                with self.assertRaises(ValueError): load_compression_fixture(path)

    def test_published_source_complete_goldens_and_tamper_refusal(self):
        root = Path(__file__).resolve().parents[1]
        report = json.loads((root / "results/2026-10-18-compression/compression.json").read_bytes())
        self.assertEqual(len(list(component_cases())), 49)
        self.assertEqual(len(list(expected_metadata())), 481)
        self.assertEqual(len(validate_goldens(report)), 481)
        source = b"\n".join((root / "src/vesper/kernels" / n).read_bytes() for n in ("format_probe.cu", "compression_probe.cu"))
        self.assertEqual(hashlib.sha256(source.replace(b"\r\n", b"\n")).hexdigest(), report["compression_profile"]["compiler"]["concatenated_source_sha256"])
        for change in (lambda r: r["component_cases"].pop(), lambda r: r["state_cases"].pop(),
                       lambda r: r["state_cases"][0].update(position=1), lambda r: r["reference_goldens"][0]["stage_sha256"].popitem(),
                       lambda r: r["original_quant_cuda_sha256"].update({"True": "0" * 64}),
                       lambda r: r["unchanged_original_controls"][0]["checks"][0].update(mismatches=1),
                       lambda r: r["faults"][-1].update(position_unchanged=False),
                       lambda r: r["snapshot_events"][0].update(restored_position=4),
                       lambda r: r["memory"]["compression_cleanup"].update(device_owned_bytes=1),
                       lambda r: r["negative_controls"].pop()):
            changed = copy.deepcopy(report)
            change(changed)
            with self.assertRaises(ValueError): validate_goldens(changed)


if __name__ == "__main__":
    unittest.main()
