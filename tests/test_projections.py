import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from vesper.projections import FP8_LAYOUT, validate_values, validate_projection_weights
from vesper.projection_validation import component_plan, decode_real_inputs, validate_goldens
from vesper.window_attention import NativeWindowAttention, validate_layer_input
from vesper.resources import MemoryBudget


class FakeKV:
    """Model a committed position and one saved state without a CUDA dependency."""
    def __init__(self, batch, ordinal):
        self.position, self.handle, self.fail_rollback = 0, None, False
        self.budget=MemoryBudget(cap=16*1024**2,available_floor=0,owner='test_window_KV')

    def __enter__(self):
        self.position = 0
        return self

    def __exit__(self, *_):
        pass

    def reset(self):
        self.position, self.handle = 0, None

    def snapshot(self):
        self.handle = SimpleNamespace(position=self.position)
        return self.handle

    def append(self, value, *, start_pos):
        self.position += value.shape[1]
        return {"window_bf16": value, "indices_i32": np.zeros((*value.shape[:2], 1), np.int32)}

    def discard(self, handle):
        assert handle is self.handle
        self.handle = None

    def rollback(self, handle):
        if self.fail_rollback:
            raise OSError("Rollback transport failed")
        assert handle is self.handle
        self.position, self.handle = handle.position, None
        return {"position": self.position}


class FakeProjections:
    ordinal, opened, fail_output = 0, True, False

    def fp8(self, name, x):
        if name == "wo_b" and self.fail_output:
            raise OSError("Output publication failed")
        return {"output_bf16": np.zeros((*x.shape[:-1], FP8_LAYOUT[name][0]), np.uint16)}

    def grouped(self, x):
        return {"output_bf16": np.zeros((*x.shape[:-1], 1024), np.uint16)}


class FakeNorm:
    ordinal, opened = 0, True

    def norm(self, x, weight):
        return {"output_bf16": x.copy()}

    def rope(self, x, *args, **kwargs):
        return {"output_bf16": x.copy()}


class FakeAttention:
    ordinal, opened = 0, True

    def __call__(self, q, *args, **kwargs):
        return {"output_bf16": q.copy()}


def fake_layer():
    arrays = {"layers.0.attn.q_norm.weight": np.zeros(1280, np.uint16),
              "layers.0.attn.kv_norm.weight": np.zeros(512, np.uint16),
              "layers.0.attn.attn_sink": np.zeros(64, np.float32)}
    with patch("vesper.window_attention.NativeWindowKV", FakeKV):
        return NativeWindowAttention(arrays, FakeProjections(), FakeNorm(), FakeAttention())


class ProjectionTests(unittest.TestCase):
    def test_projection_input_bounds_dtype_layout_and_finite_bits(self):
        value = np.zeros((2, 3, 5120), np.uint16)
        self.assertEqual(validate_values(value, 5120), 6)
        for bad in (value.astype(np.float32), value[..., ::-1], value[0, 0], np.zeros((9, 5120), np.uint16),
                    np.zeros((0, 5120), np.uint16), np.zeros((1, 5121), np.uint16), np.full_like(value, 0x7F80)):
            with self.assertRaises(ValueError):
                validate_values(bad, 5120)

    def test_layer_shape_position_and_chunk_boundaries(self):
        value = np.zeros((2, 3, 5120), np.uint16)
        validate_layer_input(value, 2, 0)
        for bad, batch, position in ((value, 1, 0), (value, 2, 1), (value, 2, -1),
            (np.zeros((1, 4, 5120), np.uint16), 1, 0), (np.zeros((1, 1, 5120), np.uint16), 1, 4096)):
            with self.assertRaises(ValueError):
                validate_layer_input(bad, batch, position)

    def test_incomplete_or_wrong_weight_layout_is_refused(self):
        for arrays in ({}, {"layers.0.attn.wq_a.weight": np.zeros((1, 5120), np.uint8)},
                       {"layers.0.attn.wq_a.weight": np.zeros((1280, 5120), np.uint16)}):
            with self.assertRaises(ValueError):
                validate_projection_weights(arrays)

    def test_component_corpus_covers_all_five_original_projections(self):
        cases = list(component_plan())
        self.assertEqual(len(cases), 90)
        self.assertEqual(len({meta["id"] for meta, _ in cases}), 90)
        self.assertEqual({meta["projection"] for meta, _ in cases}, {*FP8_LAYOUT, "wo_a"})
        self.assertTrue(all(a.dtype == np.uint16 and a.flags.c_contiguous for _, a in cases))

    def test_closed_runtime_and_out_of_order_calls_are_refused(self):
        layer = fake_layer()
        value = np.zeros((1, 1, 5120), np.uint16)
        with self.assertRaisesRegex(RuntimeError, "not open"):
            layer.step(value)
        with layer:
            with self.assertRaises(ValueError):
                layer.step(value, start_pos=1)
            self.assertEqual(layer.step(value)["wo_b_output_bf16"].shape, value.shape)
            self.assertEqual(layer.position, 1)

    def test_output_failure_restores_committed_KV_and_can_retry(self):
        with fake_layer() as layer:
            value = np.zeros((1, 1, 5120), np.uint16)
            layer.step(value)
            layer.projections.fail_output = True
            with self.assertRaises(OSError):
                layer.step(value)
            self.assertEqual((layer.position, layer.kv.position, layer.pending), (1, 1, None))
            layer.projections.fail_output = False
            layer.step(value)
            self.assertEqual((layer.position, layer.kv.position), (2, 2))

    def test_failed_rollback_blocks_calls_until_explicit_recovery(self):
        with fake_layer() as layer:
            value = np.zeros((1, 1, 5120), np.uint16)
            layer.projections.fail_output, layer.kv.fail_rollback = True, True
            with self.assertRaisesRegex(RuntimeError, "recovery remains pending"):
                layer.step(value)
            self.assertEqual((layer.position, layer.kv.position), (0, 1))
            self.assertIsNotNone(layer.pending)
            with self.assertRaisesRegex(RuntimeError, "Recover"):
                layer.step(value)
            layer.projections.fail_output, layer.kv.fail_rollback = False, False
            layer.recover()
            self.assertEqual((layer.position, layer.kv.position, layer.pending), (0, 0, None))
            layer.step(value)
            self.assertEqual(layer.position, 1)

    def test_operator_device_agreement_reentry_and_reset(self):
        layer = fake_layer()
        layer.norm.ordinal = 1
        with self.assertRaises(ValueError):
            layer.__enter__()
        layer.norm.ordinal = 0
        with layer:
            layer.step(np.zeros((1, 1, 5120), np.uint16))
            with self.assertRaisesRegex(RuntimeError, "already open"):
                layer.__enter__()
        with layer:
            self.assertEqual(layer.position, 0)

    def test_whole_block_finalizer_failure_preserves_committed_window(self):
        with fake_layer() as layer:
            value = np.zeros((1, 1, 5120), np.uint16)
            def fail(stages):
                self.assertEqual((layer.position, layer.kv.position), (0, 1))
                raise OSError("MoE or residual failed after Attention")
            with self.assertRaises(OSError):
                layer.step(value, finalize=fail)
            self.assertEqual((layer.position, layer.kv.position, layer.pending), (0, 0, None))
            output = layer.step(value, finalize=lambda stages: {"block": stages["wo_b_output_bf16"]})
            self.assertEqual(output["block"].shape, value.shape)
            self.assertEqual(layer.position, 1)

    def test_invalid_finalizer_is_refused_before_KV_publication(self):
        with fake_layer() as layer:
            with self.assertRaises(ValueError):
                layer.step(np.zeros((1, 1, 5120), np.uint16), finalize=True)
            self.assertEqual((layer.position, layer.kv.position, layer.pending), (0, 0, None))

    def test_published_source_complete_goldens_and_tamper_controls(self):
        root = Path(__file__).resolve().parents[1]
        report = json.loads((root / "results/2026-10-18-attention-projections/projections.json").read_bytes())
        self.assertEqual(len(validate_goldens(report)), 399)
        source = b"\n".join((root / "src/vesper/kernels" / name).read_bytes()
                             for name in ("format_probe.cu", "kv_probe.cu", "projection_probe.cu"))
        self.assertEqual(hashlib.sha256(source.replace(b"\r\n", b"\n")).hexdigest(),
                         report["native_profile"]["compiler"]["concatenated_source_sha256"])
        for change in (lambda r: r["component_cases"].pop(), lambda r: r["layer_cases"].pop(),
                       lambda r: r["reference_goldens"][0]["meta"].update(seed=0),
                       lambda r: r["layer_cases"][0]["checks"][0].update(mismatches=1),
                       lambda r: r["faults"][-1].update(calls_blocked_until_recovery=False),
                       lambda r: r["captured_original_prefix_Attention_inputs"][0].update(input_bf16_sha256="0" * 64),
                       lambda r: r["negative_controls"].pop(), lambda r: r["native_profile"].update(cublas_version=130000)):
            broken = copy.deepcopy(report)
            change(broken)
            with self.assertRaises(ValueError):
                validate_goldens(broken)
        self.assertEqual(len(decode_real_inputs(report["captured_original_prefix_Attention_inputs"])), 4)


if __name__ == "__main__":
    unittest.main()
