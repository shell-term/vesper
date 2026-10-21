"""CPU-only streamed composition/ownership tests; no model quality claim."""

from contextlib import ExitStack, contextmanager
from dataclasses import FrozenInstanceError
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from vesper import spark_backbone_runtime as runtime


class Operator:
    def __init__(self, events, kind, layer=0, ordinal=0, *, fail_enter=False, fail_close=False):
        self.events, self.kind, self.layer, self.ordinal = events, kind, layer, ordinal
        self.fail_enter, self.fail_close, self.opened = fail_enter, fail_close, False

    def __enter__(self):
        self.events.append(("open", self.kind, self.layer))
        if self.fail_enter:
            raise ValueError("Injected constructor failure")
        self.opened = True
        return self

    def __exit__(self, *_):
        self.opened = False
        self.events.append(("close", self.kind, self.layer))
        if self.fail_close:
            raise RuntimeError("Injected close failure")

    def _call(self, *args, **kwargs):
        self.events.append(("call", self.kind, self.layer))
        return {"layer": self.layer, "args": args, "kwargs": kwargs}

    def stats(self):
        return {"opened":self.opened,"calls_completed":1}

    mixes = pre = post = fp8 = grouped = round_window = forward = _call


class StreamedPoolTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        @contextmanager
        def upload_arrays(layer):
            yield {"layer": layer}
        self.assets = SimpleNamespace(workspace=Mock(), arrays_for_layer=Mock(), upload_arrays=upload_arrays)
        self.factories = {}
        for kind in ("hc", "projection"):
            def factory(arrays, ordinal, *, layer, profile, execution_target, kind=kind):
                self.assertEqual(arrays, {"layer": layer})
                self.assertEqual(profile, "backbone_probe")
                self.assertEqual(execution_target, runtime.TARGET)
                return Operator(self.events, kind, layer, ordinal)
            self.factories[kind] = Mock(side_effect=factory)
        self.moe = Mock(side_effect=lambda layer: Operator(self.events, "moe", layer))
        self.pool = runtime.LayerOperatorPool(self.assets, hc_factory=self.factories["hc"],
            projection_factory=self.factories["projection"], moe_factory=self.moe)

    def test_proxies_have_immutable_lazy_identity_without_eager_weights(self):
        with self.pool:
            proxies = [self.pool.proxy(kind, layer) for kind in ("hc", "projection", "moe") for layer in range(40)]
            self.assertTrue(all(proxy.opened for proxy in proxies))
            self.assertTrue(all(proxy.ordinal == 0 for proxy in proxies))
            self.assets.arrays_for_layer.assert_not_called()
            with self.assertRaises(FrozenInstanceError):
                proxies[0].layer = 1
            with self.assertRaises(AttributeError):
                proxies[0].device
        self.assertFalse(proxies[0].opened)

    def test_each_kind_retains_one_operator_and_switch_closes_before_next_open(self):
        with self.pool:
            self.pool.proxy("hc", 0).mixes("input", "attn")
            self.pool.proxy("hc", 0).pre("input", "pre")
            self.pool.proxy("projection", 0).fp8("wq_a", "input")
            self.pool.proxy("hc", 1).mixes("input", "attn")
            self.assertEqual(self.pool.stats()["active_layers"], {"hc": 1, "projection": 0})
            self.pool.proxy("projection", 1).grouped("input")
            self.assertLess(self.events.index(("close", "hc", 0)), self.events.index(("open", "hc", 1)))
            self.assertLess(self.events.index(("close", "projection", 0)), self.events.index(("open", "projection", 1)))
            self.assertEqual(self.pool.stats()["maximum_resident_operators"], {"hc": 1, "projection": 1})
        self.assertEqual(self.pool.active, {})
        self.assertEqual(self.factories["hc"].call_count, 2)

    def test_moe_is_fresh_per_call_and_always_closed_without_extra_state(self):
        with self.pool:
            proxy = self.pool.proxy("moe", 39)
            self.assertEqual(proxy.forward("one", False)["layer"], 39)
            self.assertEqual(proxy.forward("two", False)["layer"], 39)
            self.assertEqual(self.events, [("open", "moe", 39), ("call", "moe", 39), ("close", "moe", 39)] * 2)
            self.assertEqual(self.pool.active, {})
            self.assertEqual(self.pool.stats()["moe_opened"], 2)

    def test_moe_cleanup_failure_poison_retains_owner_and_retries_cleanup_on_close(self):
        operator = Operator(self.events, "moe", 0, fail_close=True)
        self.moe.return_value, self.moe.side_effect = operator, None
        self.pool.__enter__()
        with self.assertRaises(RuntimeError):
            self.pool.proxy("moe", 0).forward("x", False)
        self.assertTrue(self.pool.poisoned)
        self.assertIs(self.pool.moe_owner, operator)
        self.assertEqual(self.pool.counts["calls"], 0)
        with self.assertRaises(RuntimeError):
            self.pool.proxy("moe", 1).forward("x", False)
        self.assertEqual(self.moe.call_count, 1)
        operator.fail_close = False
        self.pool.__exit__()
        self.assertIsNone(self.pool.moe_owner)
        self.assertEqual(self.events.count(("close", "moe", 0)), 2)

    def test_moe_failed_entry_respects_its_cleanup_and_preserves_original_failure(self):
        class EarlyFailure:
            stack = None
            def __enter__(self):
                raise ValueError("Original fixture unavailable")
            def __exit__(self, *_):
                raise AssertionError("Normal exit requires fields absent before successful entry")
        self.moe.side_effect = lambda layer: EarlyFailure()
        with self.pool:
            with self.assertRaisesRegex(ValueError, "Original fixture"):
                self.pool.proxy("moe", 0).forward("x", False)
            self.assertFalse(self.pool.poisoned)
            self.assertIsNone(self.pool.moe_owner)
            self.moe.side_effect = lambda layer: Operator(self.events, "moe", layer)
            self.pool.proxy("moe", 0).forward("x", False)

    def test_moe_entry_cleanup_failure_poison_retains_only_partial_stack_for_close(self):
        class PartialFailure:
            stack = Mock()
            def __enter__(self):
                raise RuntimeError("Entry cleanup failed")
            def __exit__(self, *_):
                raise AssertionError("Must close partial stack without post-entry diagnostics")
        owner = PartialFailure()
        self.moe.side_effect = lambda layer: owner
        self.pool.__enter__()
        with self.assertRaisesRegex(RuntimeError, "Entry cleanup"):
            self.pool.proxy("moe", 0).forward("x", False)
        self.assertTrue(self.pool.poisoned)
        self.assertIs(self.pool.moe_owner, owner)
        self.pool.__exit__()
        self.assertIsNone(owner.stack)
        self.assertIsNone(self.pool.moe_owner)

    def test_moe_forward_failure_closes_normally_and_allows_clean_fresh_retry(self):
        owner = Operator(self.events, "moe", 0)
        owner.forward = Mock(side_effect=ValueError("Injected forward failure"))
        self.moe.side_effect = None
        self.moe.return_value = owner
        with self.pool:
            with self.assertRaisesRegex(ValueError, "forward failure"):
                self.pool.proxy("moe", 0).forward("x", False)
            self.assertFalse(self.pool.poisoned)
            self.assertIsNone(self.pool.moe_owner)
            self.assertFalse(owner.opened)
            self.moe.side_effect = lambda layer: Operator(self.events, "moe", layer)
            self.pool.proxy("moe", 0).forward("x", False)

    def test_failed_new_operator_leaves_no_current_slot_and_healthy_retry_works(self):
        with self.pool:
            self.pool.proxy("hc", 0).pre("x", "p")
            self.factories["hc"].side_effect = lambda *args, **kw: Operator(self.events, "hc", kw["layer"], fail_enter=True)
            with self.assertRaises(ValueError):
                self.pool.proxy("hc", 1).pre("x", "p")
            self.assertNotIn("hc", self.pool.active)
            self.factories["hc"].side_effect = lambda *args, **kw: Operator(self.events, "hc", kw["layer"])
            self.pool.proxy("hc", 1).pre("x", "p")
            self.assertEqual(self.pool.active["hc"][0], 1)

    def test_close_failure_poison_prevents_new_operator_publication(self):
        self.factories["hc"].side_effect = lambda *a, **kw: Operator(self.events, "hc", kw["layer"], fail_close=True)
        with self.pool:
            self.pool.proxy("hc", 0).pre("x", "p")
            with self.assertRaises(RuntimeError):
                self.pool.proxy("hc", 1).pre("x", "p")
            self.assertTrue(self.pool.poisoned)
            with self.assertRaises(RuntimeError):
                self.pool.proxy("projection", 1).fp8("wq_a", "x")
            self.assertNotIn(("open", "hc", 1), self.events)

    def test_invalid_layer_kind_or_closed_pool_cannot_load_or_call(self):
        for layer in (-1, 40, True, 1.0):
            with self.assertRaises(ValueError):
                self.pool.proxy("hc", layer)
        with self.assertRaises(ValueError):
            self.pool.proxy("other", 0)
        with self.assertRaises(RuntimeError):
            self.pool.proxy("hc", 0).pre("x", "p")
        self.assets.arrays_for_layer.assert_not_called()

    def test_default_moe_preserves_trained_fixtures_and_full_original_source(self):
        self.assets.moe_paths = Mock(return_value=(Path("gate"), Path("router"), Path("shared")))
        self.assets.metadata, self.assets.originals, self.assets.integrity = map(Path, ("metadata", "originals", "integrity"))
        with patch.object(runtime, "ConnectedMoE") as constructor, patch.object(runtime, "OriginalShardExpertStore") as source:
            self.pool._moe(14)
            args, kwargs = constructor.call_args
            self.assertEqual(args[:4], (Path("originals"), Path("gate"), Path("router"), Path("shared")))
            self.assertEqual(kwargs["layer"], 14)
            config = args[4]
            self.assertIsNone(config.cache.helper_device)
            self.assertEqual(config.cache.ram_cache_bytes, 0)
            self.assertEqual(config.cache.primary_slots, 1)
            self.assertEqual(config.cache.integrity_mode, "readback")
            budget = object()
            kwargs["source_factory"](Path("unused"), budget)
            source.assert_called_once_with(Path("metadata"), Path("originals"), Path("integrity"), budget,
                                           layer=14, workspace=self.assets.workspace)


class EmbeddingTests(unittest.TestCase):
    def test_domain_is_lazy_and_actual_row_is_validated_at_each_lookup(self):
        read = Mock(return_value=np.zeros(5120, np.uint16))
        rows = runtime.EmbeddingProvider(read)
        self.assertIn(129279, rows)
        self.assertNotIn(-1, rows)
        self.assertNotIn(True, rows)
        read.assert_not_called()
        self.assertEqual(rows[129279].shape, (5120,))
        read.assert_called_once_with(129279)
        self.assertFalse(hasattr(rows, "items"))
        for bad in (np.zeros(5120, np.float32), np.zeros(5121, np.uint16),
                    np.full(5120, 0x7f80, np.uint16), np.zeros(10240, np.uint16)[::2]):
            read.return_value = bad
            with self.assertRaises(ValueError):
                rows[1]
        with self.assertRaises(KeyError):
            rows[129280]


class CompositionTests(unittest.TestCase):
    def test_foreign_workspace_model_revision_rejected_before_policy_or_cuda(self):
        assets = Mock()
        assets.workspace.context.return_value = {"platform": "dgx-spark", "profile": "native",
                                                  "model": runtime.MODEL, "revision": "0" * 40}
        with ExitStack() as stack, patch.object(runtime, "current_policy") as policy:
            with self.assertRaises(ValueError):
                runtime.open_spark_backbone(stack, assets, Path("bundle"))
            policy.assert_not_called()

    def test_window_constructor_explicit_target_reaches_its_kv(self):
        for layer in (0, 1):
            projection = SimpleNamespace(layer=layer, ordinal=0)
            window = runtime.NativeWindowAttention({}, projection, object(), object(),
                profile="backbone_probe", execution_target=runtime.TARGET)
            self.assertEqual(window.kv.execution_target.name, runtime.TARGET)

    def test_missing_or_discrete_policy_is_rejected_before_opening_any_cuda_operator(self):
        assets = Mock()
        assets.workspace.context.return_value = {"platform": "dgx-spark", "profile": "native", "model": runtime.MODEL, "revision": runtime.REVISION}
        for policy in (None, SimpleNamespace(config=SimpleNamespace(unified_device_memory=False))):
            with ExitStack() as stack, patch.object(runtime, "current_policy", return_value=policy), \
                    patch.object(runtime, "NativeNormRoPE") as norm:
                with self.assertRaises(ValueError):
                    runtime.open_spark_backbone(stack, assets, Path("bundle"))
                norm.assert_not_called()

    def test_default_spark_policy_has_one_shared_ceiling_and_system_floor(self):
        policy = runtime.spark_memory_policy()
        self.assertTrue(policy.unified_device_memory)
        self.assertEqual(policy.device_caps, {0:5 * runtime.GiB})
        self.assertEqual(policy.host_cap_bytes, 12 * runtime.GiB)
        self.assertEqual(policy.available_floor_bytes, 32 * runtime.GiB)

    def test_complete_composition_is_lazy_all40_and_explicitly_unqualified(self):
        events, opened = [], []
        assets = Mock()
        assets.arrays = {}
        assets.workspace.context.return_value = {"platform": "dgx-spark", "profile": "native", "model": runtime.MODEL, "revision": runtime.REVISION}
        assets.metadata, assets.originals, assets.integrity = map(Path, ("metadata", "originals", "integrity"))
        assets.norm_weight.return_value = np.zeros(5120, np.uint16)
        def op(name, **properties):
            value = Operator(events, name)
            for key, item in properties.items():
                setattr(value, key, item)
            opened.append(value)
            return value
        def window(arrays, projection, norm, sparse, **kwargs):
            self.assertEqual(kwargs, {"profile": "backbone_probe", "execution_target": runtime.TARGET})
            return op("window", layer=projection.layer, position=0)
        def sharing(arrays, projections, norm, comp, indexer, sparse, layers, **kwargs):
            self.assertEqual(set(projections), set(range(2, 40)))
            return op("sharing", layers=layers, binding_profile=kwargs["binding_profile"], position=0)
        with ExitStack() as patches:
            patches.enter_context(patch("vesper.resources.physical_memory", return_value={"available": 128 * runtime.GiB}))
            patches.enter_context(patch.object(runtime, "current_policy", return_value=SimpleNamespace(config=runtime.spark_memory_policy())))
            patches.enter_context(patch.object(runtime, "spark_kernel_cache", side_effect=lambda:ExitStack()))
            for name in ("NativeNormRoPE", "SparkAttentionCandidate", "NativeCompression", "NativeIndexer", "NativeSampler", "SparkOutputCandidate"):
                factory = lambda *a, name=name, **kw: op(name, **({"profile": "long_context"} if name == "SparkAttentionCandidate" else {}))
                patches.enter_context(patch.object(runtime, name, side_effect=factory))
            patches.enter_context(patch.object(runtime, "NativeWindowAttention", side_effect=window))
            patches.enter_context(patch.object(runtime, "SharedAttentionFrame", side_effect=sharing))
            patches.enter_context(patch.object(runtime, "OriginalShardRows", side_effect=lambda m, o, i, layer: op("rows", layer=layer)))
            patches.enter_context(patch.object(runtime, "NativeEngram", side_effect=lambda controls, rows, **kw: op("engram", rows=rows)))
            patches.enter_context(patch.object(runtime, "PinnedTokenizer", return_value="tokenizer"))
            patches.enter_context(patch.object(runtime, "compressed_token_map", return_value=(np.arange(1), {"identity": "mock"})))
            patches.enter_context(patch.object(runtime, "NgramState", return_value=SimpleNamespace(position=0, lock=threading.RLock())))
            with ExitStack() as stack:
                engine = runtime.open_spark_backbone(stack, assets, Path("candidate"))
                self.assertEqual(set(engine.hc), set(range(40)))
                self.assertEqual(set(engine.moe), set(range(40)))
                self.assertEqual(set(engine.engrams), {1, 14})
                self.assertIsInstance(engine.embeddings, runtime.EmbeddingProvider)
                self.assertFalse(engine.stats()["full_model_quality_qualified"])
                self.assertEqual(engine.operator_pool.active, {})
                assets.arrays_for_layer.assert_not_called()
            self.assertIsNone(engine.norm_weight)
        self.assertTrue(all(not op.opened for op in opened))


if __name__ == "__main__":
    unittest.main()
