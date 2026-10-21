"""CPU-only functional-driver tests using the real memory and chat adapters."""

from contextlib import redirect_stdout
import gc
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from vesper import spark_text_probe as probe
from vesper.memory import current_policy, RuntimeMemory
from vesper.messages import ASSISTANT, BOS, END_THINK, EOS
from vesper.resources import MemoryBudget, MiB, GiB


class Workspace:
    def __init__(self, root):
        self.root = Path(root)
        self.check = Mock()
        self.check_runtime_module = Mock()
        self.changed = False
        self.writes = []

    def context(self, run_id=None):
        return {"platform": "dgx-spark", "profile": "native", "model": probe.MODEL, "revision": probe.REVISION,
                "run_id": run_id, "git_head": "a" * 40, "dirty_diff_sha256": None}

    def artifact_path(self, relative, create_parent=False):
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("not an artifact-relative path")
        result = self.root / "artifacts/dgx-spark" / path
        if create_parent:
            result.parent.mkdir(parents=True, exist_ok=True)
        return result

    def result_path(self, relative, create_parent=False):
        path = self.root / "results/dgx-spark" / relative
        if create_parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def stamp_result(self, payload, run_id=None, *, initial_context=None):
        if self.changed:
            raise ValueError("source identity changed")
        return {**payload, "workspace_context": initial_context or self.context(run_id)}

    def write_result(self, relative, payload, run_id=None, *, initial_context=None):
        value = self.stamp_result(payload, run_id, initial_context=initial_context)
        path = self.result_path(relative, create_parent=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False)
        self.writes.append(value)
        return path

    def check_result(self, relative):
        raw = self.result_path(relative).read_bytes()
        value = json.loads(raw)
        context = value.get("workspace_context", {})
        if self.changed or context != self.context(context.get("run_id")):
            raise ValueError("source identity changed")
        return {"result_sha256": hashlib.sha256(raw).hexdigest()}


class Tokenizer:
    prompt_ids = (10, 11, 12)
    def __init__(self, metadata):
        self.metadata = metadata

    def __len__(self):
        return 129280

    def encode(self, text):
        return [1] if text == EOS else list(self.prompt_ids)

    def decode(self, ids):
        return "".join({1: EOS, 4: "4", 5: "bad", 6: "<think>", 7: "reason", 8: END_THINK}.get(i, "x") for i in ids)


class Assets:
    events = []
    failure = None
    missing_moe = False
    def __init__(self, metadata, originals, integrity, *, workspace, fixtures, budget):
        self.workspace, self.paths = workspace, (metadata, originals, integrity, fixtures)
        if budget.cap != GiB or budget.owner != "spark_model_assets":
            raise AssertionError("The real supplied asset budget differs from the S1 limit")
        self.budget = budget

    def __enter__(self):
        if current_policy() is None:
            raise AssertionError("Memory runtime must precede assets")
        self.events.append("assets_open")
        self.budget.acquire(self.budget.cap, category="metadata")
        return self

    def prepare_moe_fixtures(self):
        self.events.append("prepare")
        if self.failure:
            raise self.failure
        return {"layers": list(range(40)), "network_payload_bytes": 0}

    def stats(self):
        return {"live_arrays": 0, "network_payload_bytes": 0}

    def moe_paths(self, layer):
        if self.missing_moe:
            raise FileNotFoundError("unprepared original MoE fixture")
        return tuple(self.paths[-1] / f"layer{layer}" / name for name in ("gate", "router", "shared"))

    def __exit__(self, *_):
        self.events.append("assets_close")
        self.budget.release(self.budget.cap, category="metadata")


class MoE:
    def __init__(self, layer):
        self.layer, self.last_attempt = layer, None
        self.experts = SimpleNamespace(store=SimpleNamespace(stats=lambda: {
            "logical_expert_reads": 6, "logical_expert_bytes": 112803840, "physical_data_bytes_read": 113000448,
            "integrity_index_bytes_read": 8192, "network_payload_bytes_in_forward": 0}))

    def __enter__(self):
        return self

    def forward(self):
        self.last_attempt = {"status": "pass", "indices": list(range(6)), "completed_expert_ids": list(range(6)),
            "store_direct_read_calls": 6, "store_direct_read_bytes": 112803840, "wall_ms": 1.}
        return {"indices": np.arange(6, dtype=np.int32), "weights": np.ones(6, np.float32)}

    def __exit__(self, *_):
        pass


class Engine:
    selections = (4, 1)
    fail_at = None
    wrong_route = False
    close_failure = False
    change_source = False
    attention_mutation = None
    instances = []
    def __init__(self, assets, bundle, temperature, seed):
        self.assets, self.bundle, self.temperature, self.seed = assets, bundle, temperature, seed
        self.operator_pool = SimpleNamespace(moe_factory=lambda layer: MoE(layer))
        device = {"ordinal": 0, "name": "NVIDIA GB10", "compute_capability": "12.1", "total_bytes": 128 * GiB}
        attention = SimpleNamespace(manifest_sha256="c" * 64, device_info=device,
            driver=SimpleNamespace(inventory=lambda: {"driver_api_version": 13000, "devices": [device], "peers": []}))
        self.sharing = SimpleNamespace(attention=attention)
        self.windows = {layer: SimpleNamespace(attention=attention) for layer in (0, 1)}
        if type(self).attention_mutation is not None:
            type(self).attention_mutation(self)
        self.tokenizer_identity = {"compressed_map_sha256": "b" * 64}
        self.calls, self.sampled, self.inputs = 0, 0, []
        self.budget = MemoryBudget(cap=32 * MiB, owner="test_engine")
        self.instances.append(self)

    def __enter__(self):
        self.budget.acquire(self.budget.cap)
        Assets.events.append("engine_open")
        return self

    def step(self, ids, *, sample):
        self.calls += 1
        self.inputs.append((int(ids[0, 0]), sample))
        if self.fail_at == self.calls:
            raise OSError("private /machine/local/secret-model-path failed")
        hidden = np.zeros((1, 1, 4, 5120), np.uint16)
        result = {"output_bf16": hidden, "head": {"logits_fp32": np.zeros((1, 129280), np.float32)}}
        result.update({f"engram{layer}":{"output_bf16":hidden} for layer in (1,14)})
        for layer in range(40):
            with self.operator_pool.moe_factory(layer) as owner:
                route = owner.forward()
            indices = route["indices"].copy()
            if self.wrong_route and layer == 39:
                indices[0] = 8
            result[f"layer{layer}"] = {"moe_gate.indices": indices, "moe_gate.weights": route["weights"], "output_bf16": hidden}
        result["sampling"] = {}
        if sample:
            value = self.selections[min(self.sampled, len(self.selections) - 1)]
            result["sampling"]["ids_i64"] = np.array([value], np.int64)
            self.sampled += 1
        if self.change_source:
            self.assets.workspace.changed = True
        return result

    def stats(self):
        return {"all_backbone_layers": list(range(40)), "qualification": "unqualified_candidate", "calls": self.calls}

    def __exit__(self, *_):
        Assets.events.append("engine_close")
        self.budget.release(self.budget.cap)
        if self.close_failure:
            raise OSError("private close path")


class TextProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.addCleanup(self.temp.cleanup)
        self.ws = Workspace(self.temp.name)
        Assets.events, Assets.failure, Assets.missing_moe, Engine.instances = [], None, False, []
        Engine.selections, Engine.fail_at, Engine.wrong_route = (4, 1), None, False
        Engine.close_failure, Engine.change_source = False, False
        Engine.attention_mutation = None
        Tokenizer.prompt_ids = (10, 11, 12)
        self.inventory = Mock(return_value={"device_count": 1, "ordinal": 0, "name": "NVIDIA GB10", "compute_capability": [12, 1],
            "driver_api_version": 13000, "reported_cuda_total_bytes": 128 * GiB, "cuda_capacity_added_to_physical_ram": False})
        self.real_attention_check = probe.check_receipt
        self.attention_check = Mock(return_value={"status": "pass", "scope": "synthetic component gate only",
            "bundle_manifest_sha256": "c" * 64, "reference_result_sha256": "d" * 64,
            "receipt_result_sha256": "e" * 64,
            "observed_device": {"ordinal": 0, "name": "NVIDIA GB10", "compute_capability": "12.1", "total_bytes": 128 * GiB},
            "driver_api_version": 13000,
            "cases": [None] * 36, "faults": [None] * 2})
        def open_engine(stack, assets, bundle, *, temperature, seed, progress=None):
            return stack.enter_context(Engine(assets, bundle, temperature, seed))
        patches = [patch.object(probe, "SparkModelAssets", Assets), patch.object(probe, "PinnedTokenizer", Tokenizer),
            patch.object(probe, "check_receipt", self.attention_check),
            patch.object(probe, "load_candidate", return_value=({"qualification": "unqualified_candidate",
                "compute_capability": "12.1", "nvrtc_version": [13, 0], "profiles": []}, {}, "c" * 64)),
            patch.object(probe, "open_spark_backbone", side_effect=open_engine),
            patch.object(probe.Driver, "inventory", self.inventory), patch.object(probe, "Driver", return_value=SimpleNamespace(inventory=self.inventory)),
            patch.object(probe.platform, "system", return_value="Linux"), patch.object(probe.platform, "machine", return_value="aarch64"),
            patch.object(probe, "host_memory", return_value={"physical_total_bytes": 128 * GiB, "physical_available_bytes": 80 * GiB,
                "cgroup_headroom_bytes": None, "ram_plus_vram_is_not_capacity": True}),
            patch("vesper.resources.physical_memory", return_value={"available": 80 * GiB}),
            patch("vesper.memory.physical_memory", return_value={"available": 80 * GiB}),
            patch("vesper.memory.process_memory", return_value={"resident_bytes": MiB, "private_commit_bytes": MiB}),
            patch.object(probe, "process_memory", return_value={"resident_bytes": MiB}),
            patch.object(RuntimeMemory, "ENABLE_WATCHDOG", False),
            patch("urllib.request.urlopen", side_effect=AssertionError("provider must remain offline"))]
        for item in patches:
            item.start(); self.addCleanup(item.stop)
        self.addCleanup(self.assert_no_runtime)

    def assert_no_runtime(self):
        gc.collect()
        self.assertIsNone(current_policy())

    def generate(self, **kwargs):
        return probe.collect("generate", self.ws, run_id="synthetic-r01", **kwargs)

    def test_complete_chat_preserves_feedback_eos_routes_and_closed_memory(self):
        report = self.generate()
        self.assertEqual(report["status"], "functional_response_complete")
        self.assertTrue(report["functional_acceptance"])
        self.assertFalse(report["full_model_quality_qualified"])
        self.assertFalse(report["original_kernel_parity_qualified"])
        self.assertEqual(report["attention_qualification"]["case_count"], 36)
        self.assertEqual(report["attention_qualification"]["fault_count"], 2)
        self.attention_check.assert_called_once_with(self.ws, self.ws.artifact_path("s1/attention-candidate"),
            "s1/attention/reference-r02.json", "s1/attention/native-validation-r01.json")
        request, completed = report["request"], report["completion"]
        self.assertEqual(request["messages"], [{"role": "user", "content": probe.DEFAULT_PROMPT}])
        self.assertTrue(request["serialized_chat"].startswith(BOS))
        self.assertTrue(request["serialized_chat"].endswith(ASSISTANT + END_THINK))
        self.assertFalse(request["thinking_enabled"])
        self.assertEqual(request["prompt_ids"], [10, 11, 12])
        self.assertEqual(completed["selected_ids_including_eos"], [4, 1])
        self.assertEqual(completed["completion_text"], "4")
        self.assertEqual(report["completion_serialized_including_eos"], "4" + EOS)
        self.assertEqual(report["parsed_completion"]["content"], "4")
        self.assertTrue(report["default_prompt_answer_matches_four"])
        self.assertEqual(Engine.instances[0].inputs, [(10, False), (11, False), (12, True), (4, True)])
        self.assertEqual(len(report["steps"]), 4)
        for step in report["steps"]:
            self.assertEqual(set(step["layers"]), set(map(str, range(40))))
            self.assertEqual(len(step["route_io"]), 40)
            self.assertEqual(step["route_io"][0]["source_io"]["network_payload_bytes_in_forward"], 0)
        self.assertEqual(Assets.events, ["assets_open", "engine_open", "engine_close", "assets_close"])
        after = report["memory_after_runtime_close"]
        self.assertEqual(after["charged_host_bytes"], 0)
        self.assertEqual(after["live_leases"], 0)
        self.assertTrue(after["unified_device_memory"])
        self.assertEqual(report["limits"]["shared_host_cap_bytes"], after["host_cap_bytes"])
        self.assertEqual(report["limits"]["asset_host_cap_bytes"], GiB)
        self.assertEqual(report["limits"]["device_cap_bytes"], 5 * GiB)
        self.assertTrue(report["owned_resources_closed"])
        self.assertTrue(report["memory_runtime_closed"])
        self.assertLess(len(json.dumps(report)), probe.REPORT_LIMIT)
        marks = report["timing_marks_seconds_from_start"]
        self.assertLessEqual(marks["generation_start"], marks["first_selected_output"])
        self.assertLessEqual(marks["first_selected_output"], marks["generation_complete"])
        self.assertLessEqual(marks["owned_cleanup_start"], marks["owned_cleanup_complete"])
        self.assertLessEqual(marks["owned_cleanup_complete"], marks["runtime_cleanup_complete"])

    def test_s2_catalog_is_owned_before_assets_and_closes_after_engine_and_assets(self):
        from vesper import spark_artifacts
        catalogs=[]
        class Catalog:
            def __init__(self, metadata, originals, integrity, budget, *, workspace):
                self.budget, self.opened = budget, False
                catalogs.append(self)
            def __enter__(self):
                self.budget.acquire(self.budget.cap, category="metadata")
                self.opened = True; Assets.events.append("catalog_open")
                return self
            def stats(self): return {"opened":self.opened}
            def __exit__(self, *_):
                self.opened = False; Assets.events.append("catalog_close")
                self.budget.release(self.budget.cap, category="metadata")
        class ReusedAssets(Assets):
            def __init__(self, *args, catalog, **kwargs):
                if not catalog.opened: raise AssertionError("Catalog must precede assets")
                self.catalog = catalog
                super().__init__(*args, **kwargs)
            def __exit__(self, *error):
                if not self.catalog.opened: raise AssertionError("Catalog must outlive assets")
                return super().__exit__(*error)
        def open_reused(stack, assets, bundle, *, temperature, seed, progress=None, reuse_profile):
            self.assertEqual(reuse_profile,"s2")
            self.assertIs(assets.catalog,catalogs[0])
            return stack.enter_context(Engine(assets,bundle,temperature,seed))
        with patch.object(spark_artifacts,"AuthenticatedModelCatalog",Catalog), \
                patch.object(probe,"SparkModelAssets",ReusedAssets), \
                patch.object(probe,"open_spark_backbone",side_effect=open_reused):
            report=self.generate(reuse_profile="s2")
        self.assertTrue(report["functional_acceptance"])
        self.assertEqual(report["preparation_reuse_profile"],"s2")
        self.assertEqual(Assets.events,["catalog_open","assets_open","engine_open","engine_close","assets_close","catalog_close"])
        self.assertEqual(report["memory_after_runtime_close"]["charged_host_bytes"],0)
        self.assertFalse(catalogs[0].opened)

    def test_unknown_preparation_profile_refuses_before_hardware_and_model(self):
        with self.assertRaises(ValueError):self.generate(reuse_profile="unknown")
        self.inventory.assert_not_called()
        probe.open_spark_backbone.assert_not_called()

    def test_short_call_budget_records_partial_progress_and_never_claims_a_response(self):
        events=[]
        report=self.generate(max_model_calls=1, progress=events.append)
        self.assertEqual(report["status"], "partial_model_call_budget")
        self.assertFalse(report["functional_acceptance"])
        self.assertTrue(report["owned_resources_closed"])
        self.assertTrue(report["memory_runtime_closed"])
        self.assertEqual(report["completion"]["calls"],1)
        self.assertEqual(events[-1]["completed_model_calls"],1)
        self.assertEqual(events[-1]["generated_tokens"],0)
        self.assertEqual(set(report["steps"][0]["engrams"]),{"1","14"})
        self.assertNotIn("first_selected_output",report["timing_marks_seconds_from_start"])

    def test_thinking_uses_original_prefix_effort_parser_and_feedback(self):
        Engine.selections = (7, 8, 4, 1)
        report = self.generate(thinking_mode="thinking", reasoning_effort=100)
        self.assertTrue(report["functional_acceptance"])
        request = report["request"]
        self.assertTrue(request["serialized_chat"].endswith(ASSISTANT + "<think>"))
        self.assertIn("Reasoning Effort: 100 (range 1-100", request["serialized_chat"])
        self.assertEqual(request["reasoning_effort"], 100)
        self.assertTrue(request["thinking_enabled"])
        self.assertEqual(report["parsed_completion"]["reasoning_content"], "reason")
        self.assertEqual(report["parsed_completion"]["content"], "4")
        self.assertEqual(Engine.instances[0].inputs[-3:], [(7, True), (8, True), (4, True)])

    def test_thinking_default_effort_and_missing_terminator_fail_closed(self):
        report = self.generate(thinking_mode="thinking")
        self.assertEqual(report["request"]["reasoning_effort"], 75)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["failure_stage"], "completion_parse")
        self.assertTrue(report["memory_runtime_closed"])
        self.assertFalse(report["functional_acceptance"])

    def test_input_output_and_execution_budgets_are_separate_and_serialized(self):
        Tokenizer.prompt_ids = tuple([10] * 600)
        report = self.generate(context_tokens=1152, max_input_tokens=700, max_new_tokens=32,
                               max_model_calls=1, max_duration_seconds=60)
        self.assertEqual(report["status"], "partial_model_call_budget")
        limits = report["request"]["admitted_limits"]
        self.assertEqual(limits["actual_input_tokens"], 600)
        self.assertEqual(limits["maximum_required_model_calls"], 631)
        self.assertEqual(limits["input_tokens"], 700)
        self.assertEqual(limits["generated_tokens_including_eos"], 32)
        self.assertEqual(limits["model_calls"], 1)
        self.assertEqual(limits["duration_seconds"], 60)
        self.assertFalse(limits["silent_truncation"])
        self.assertEqual(report["sequence_scope"]["maximum_model_calls"], 1152)
        self.assertFalse(report["sequence_scope"]["extended_context_hardware_qualified"])
        self.assertEqual(report["limits"]["probe_host_admission_bytes"], 3 * probe.HOST_ADMISSION)
        self.assertEqual(report["limits"]["shared_host_cap_bytes"], 12 * GiB)
        self.assertEqual(report["completion"]["calls"], 1)

    def test_invalid_thinking_and_extended_limits_cannot_open_resources(self):
        for kwargs in ({"thinking_mode": "unknown"}, {"reasoning_effort": 100},
                       {"thinking_mode": "thinking", "reasoning_effort": True},
                       {"thinking_mode": "thinking", "reasoning_effort": 101},
                       {"context_tokens": 1153, "max_duration_seconds": 1},
                       {"context_tokens": 1152}, {"max_input_tokens": 513},
                       {"max_duration_seconds": float("inf")}, {"max_duration_seconds": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.generate(**kwargs)
        self.assertEqual(Assets.events, [])
        self.inventory.assert_not_called()

    def test_explicit_input_token_limit_never_truncates_the_prompt(self):
        Tokenizer.prompt_ids = (10, 11, 12)
        report = self.generate(max_input_tokens=2)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["failure_stage"], "tokenizer")
        self.assertIsNone(report["request"]["admitted_limits"])
        self.assertEqual(Engine.instances, [])

    def test_duration_budget_stops_between_calls_and_preserves_committed_progress(self):
        clock = [0.]
        original = Engine.step
        def slow(engine, ids, *, sample):
            result = original(engine, ids, sample=sample)
            clock[0] = 11.
            return result
        with patch.object(probe.time, "perf_counter", side_effect=lambda: clock[0]), \
             patch.object(Engine, "step", slow):
            report = self.generate(max_duration_seconds=10)
        self.assertEqual(report["status"], "partial_duration_budget")
        self.assertTrue(report["duration_budget_exceeded"])
        self.assertEqual(report["completion"]["calls"], 1)
        self.assertEqual(Engine.instances[0].calls, 1)
        self.assertTrue(report["memory_runtime_closed"])
        self.assertTrue(report["owned_resources_closed"])
        self.assertFalse(report["functional_acceptance"])

    def test_duration_in_cleanup_cannot_turn_a_late_response_into_a_budget_pass(self):
        clock = [0.]
        original = Engine.__exit__
        def slow_close(engine, *error):
            original(engine, *error)
            clock[0] = 11.
        with patch.object(probe.time, "perf_counter", side_effect=lambda: clock[0]), \
             patch.object(Engine, "__exit__", slow_close):
            report = self.generate(max_duration_seconds=10)
        self.assertTrue(report["eos_response_complete"])
        self.assertEqual(report["status"], "partial_duration_budget")
        self.assertFalse(report["functional_acceptance"])
        self.assertTrue(report["memory_runtime_closed"])

    def test_preparation_or_its_cleanup_overrun_retains_details_but_returns_partial(self):
        for delayed in ("prepare_moe_fixtures", "__exit__"):
            with self.subTest(delayed=delayed):
                clock = [0.]
                original = getattr(Assets, delayed)
                def slow(owner, *args):
                    value = original(owner, *args)
                    clock[0] = 11.
                    return value
                with patch.object(probe.time, "perf_counter", side_effect=lambda: clock[0]), \
                     patch.object(Assets, delayed, slow):
                    report = probe.collect("prepare", self.ws, run_id="prepare-budget-r01", max_duration_seconds=10)
                self.assertEqual(report["status"], "partial_duration_budget")
                self.assertTrue(report["duration_budget_exceeded"])
                self.assertEqual(report["preparation"]["layers"], list(range(40)))
                self.assertEqual(report["preparation"]["network_payload_bytes"], 0)
                self.assertTrue(report["memory_runtime_closed"])
                self.assertTrue(report["owned_resources_closed"])
                self.assertFalse(report["functional_acceptance"])
                self.assertEqual(report["memory_after_runtime_close"]["charged_host_bytes"], 0)
        self.inventory.assert_not_called()
        self.assertEqual(Engine.instances, [])

    def test_parser_character_budget_retains_output_and_never_claims_eos_success(self):
        Engine.selections = (5, 5, 1)
        with patch.object(probe, "MAX_COMPLETION_CHARS", 4):
            report = self.generate()
        self.assertEqual(report["status"], "partial_completion_character_budget")
        self.assertEqual(report["completion"]["selected_ids_including_eos"], [5, 5])
        self.assertEqual(report["completion_serialized_including_eos"], "badbad")
        self.assertIsNone(report["parsed_completion"])
        self.assertFalse(report["functional_acceptance"])
        self.assertTrue(report["memory_runtime_closed"])

    def test_atomic_live_progress_retains_counters_and_refuses_a_prior_run(self):
        live=probe.ProgressFile(self.ws,"live-r01")
        live({"phase":"layer_complete","layer":3,"completed_model_calls":2,"generated_tokens":0})
        path=self.ws.artifact_path("s1/text/live-r01/progress.json")
        first=json.loads(path.read_text())
        self.assertEqual(first["layer"],3)
        self.assertEqual(first["completed_model_calls"],2)
        live({"phase":"model_call_complete","completed_model_calls":3,"generated_tokens":1,"generated_ids":[4]})
        last=json.loads(path.read_text())
        self.assertEqual(last["generated_ids"],[4])
        self.assertEqual(last["completed_model_calls"],3)
        self.assertFalse(self.ws.artifact_path("s1/text/live-r01/progress.tmp").exists())
        with self.assertRaises(FileExistsError):probe.ProgressFile(self.ws,"live-r01")

    def test_interrupted_atomic_progress_removes_only_owned_temp_and_can_finish(self):
        live=probe.ProgressFile(self.ws,"live-r01")
        temporary=self.ws.artifact_path("s1/text/live-r01/progress.tmp")
        for error in (KeyboardInterrupt(), OSError("injected replacement")):
            with patch.object(probe.os,"replace",side_effect=error):
                with self.assertRaises(type(error)):live({"phase":"layer_complete"})
            self.assertFalse(temporary.exists())
        live({"phase":"finished","status":"cancelled"})
        self.assertEqual(json.loads(live.path.read_text())["status"],"cancelled")
        temporary.write_text("pre-existing foreign temp")
        with self.assertRaises(FileExistsError):live({"phase":"finished"})
        self.assertEqual(temporary.read_text(),"pre-existing foreign temp")

    def test_final_progress_failure_cannot_hide_successful_immutable_report(self):
        def progress(event):
            if event.get("phase")=="finished":raise OSError("injected progress failure")
        report=self.generate(progress=progress)
        self.assertTrue(report["functional_acceptance"])
        self.assertEqual(report["final_progress_failure_type"],"OSError")

    def test_keyboard_interrupt_preserves_completed_calls_and_cleanup(self):
        original=Engine.step
        def interrupted(engine, ids, *, sample):
            if engine.calls==1:raise KeyboardInterrupt()
            return original(engine,ids,sample=sample)
        with patch.object(Engine,"step",interrupted):
            report=self.generate()
        self.assertEqual(report["status"],"cancelled")
        self.assertEqual(report["completion"]["calls"],1)
        self.assertEqual(len(report["steps"]),1)
        self.assertTrue(report["owned_resources_closed"])
        self.assertTrue(report["memory_runtime_closed"])
        self.assertFalse(report["functional_acceptance"])

    def test_max_new_tokens_is_partial_and_never_a_functional_or_quality_pass(self):
        Engine.selections = (4,)
        report = self.generate(max_new_tokens=2, temperature=.7, seed=27)
        self.assertEqual(report["status"], "partial_max_new_tokens")
        self.assertFalse(report["functional_acceptance"])
        self.assertFalse(report["eos_response_complete"])
        self.assertIsNone(report["parsed_completion"])
        self.assertEqual(report["completion"]["selected_ids_including_eos"], [4, 4])
        self.assertEqual(Engine.instances[0].temperature, .7)
        self.assertEqual(Engine.instances[0].seed, 27)

    def test_unpassed_attention_reference_blocks_assets_and_cuda(self):
        from vesper import spark_attention_validation as bridge
        reference, receipt = "s1/attention/reference-r02.json", "s1/attention/native-validation-r01.json"
        self.ws.write_result(reference, {"status": "fail"}, run_id="ref")
        self.ws.write_result(receipt, {"bundle": "s1/attention-candidate", "reference_result": reference}, run_id="native")
        self.attention_check.side_effect = self.real_attention_check
        with patch.object(bridge, "load_candidate", return_value=({}, {}, "c" * 64)):
            report = self.generate()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["failure_stage"], "attention_qualification")
        self.assertFalse(report["functional_acceptance"])
        self.assertIsNone(report["attention_qualification"])
        self.assertEqual(Assets.events, [])
        self.inventory.assert_not_called()
        probe.open_spark_backbone.assert_not_called()

    def test_foreign_source_in_either_attention_result_blocks_cuda(self):
        reference, receipt = "s1/attention/reference-r02.json", "s1/attention/native-validation-r01.json"
        self.ws.write_result(reference, {}, run_id="ref")
        self.ws.write_result(receipt, {"bundle": "s1/attention-candidate", "reference_result": reference}, run_id="native")
        self.attention_check.side_effect = self.real_attention_check
        originals = {path: self.ws.result_path(path).read_bytes() for path in (reference, receipt)}
        for path in (reference, receipt):
            for target, raw in originals.items():
                self.ws.result_path(target).write_bytes(raw)
            value = json.loads(originals[path])
            value["workspace_context"]["git_head"] = "f" * 40
            self.ws.result_path(path).write_text(json.dumps(value), encoding="utf-8")
            report = self.generate()
            self.assertEqual(report["failure_stage"], "attention_qualification")
            self.assertFalse(report["functional_acceptance"])
        self.assertEqual(Assets.events, [])
        self.inventory.assert_not_called()
        probe.open_spark_backbone.assert_not_called()

    def test_changed_attention_bundle_rejected_by_receipt_loader_before_cuda(self):
        from vesper import spark_attention_validation as bridge
        reference, receipt = "s1/attention/reference-r02.json", "s1/attention/native-validation-r01.json"
        self.ws.write_result(reference, {}, run_id="ref")
        self.ws.write_result(receipt, {"bundle": "s1/attention-candidate", "reference_result": reference}, run_id="native")
        self.attention_check.side_effect = self.real_attention_check
        with patch.object(bridge, "load_candidate", side_effect=ValueError("Attention CUBIN hash differs")) as loader:
            report = self.generate()
        loader.assert_called_once_with(self.ws.artifact_path("s1/attention-candidate"))
        self.assertEqual(report["failure_stage"], "attention_qualification")
        self.assertFalse(report["functional_acceptance"])
        self.assertEqual(Assets.events, [])
        self.inventory.assert_not_called()

    def test_prepare_needs_no_attention_receipt_and_missing_gate_is_not_optional(self):
        report = probe.collect("prepare", self.ws, run_id="prepare", attention_reference_result=None,
                               attention_receipt_result=None)
        self.assertEqual(report["status"], "local_moe_prepared")
        self.assertIsNone(report["attention_qualification"])
        self.attention_check.assert_not_called()
        self.inventory.assert_not_called()
        for argument in ("attention_reference_result", "attention_receipt_result"):
            with self.assertRaises(ValueError):
                self.generate(**{argument: None})

    def test_bundle_replaced_after_receipt_validation_cannot_open_cuda(self):
        original = probe.load_candidate.return_value
        with patch.object(probe, "load_candidate", return_value=(*original[:2], "f" * 64)):
            report = self.generate()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["failure_stage"], "prepared_local_assets")
        self.inventory.assert_not_called()
        probe.open_spark_backbone.assert_not_called()
        self.assertFalse(report["functional_acceptance"])

    def test_inventory_hardware_or_driver_change_cannot_open_model(self):
        original = self.inventory.return_value.copy()
        for key, value in (("ordinal", 1), ("name", "Foreign GB10"), ("compute_capability", [12, 0]),
                           ("reported_cuda_total_bytes", 127 * GiB), ("driver_api_version", 13010), ("device_count", 2)):
            with self.subTest(key=key):
                self.inventory.return_value = {**original, key: value}
                report = self.generate()
                self.assertEqual(report["failure_stage"], "hardware_inventory")
                self.assertFalse(report["functional_acceptance"])
        probe.open_spark_backbone.assert_not_called()

    def test_actual_attention_bundle_device_driver_and_consumers_bound_before_forward(self):
        def bundle(engine):
            engine.sharing.attention.manifest_sha256 = "f" * 64
        def device(engine):
            engine.sharing.attention.device_info = {**engine.sharing.attention.device_info, "name": "Other GB10"}
        def driver(engine):
            attention = engine.sharing.attention
            attention.driver = SimpleNamespace(inventory=lambda: {"driver_api_version": 13010, "devices": [attention.device_info]})
        def enumerated_device(engine):
            attention = engine.sharing.attention
            attention.driver = SimpleNamespace(inventory=lambda: {"driver_api_version": 13000,
                "devices": [{**attention.device_info, "ordinal": 1}]})
        def window0(engine):
            engine.windows[0].attention = SimpleNamespace()
        def window1(engine):
            engine.windows[1].attention = SimpleNamespace()
        for mutation in (bundle, device, driver, enumerated_device, window0, window1):
            with self.subTest(mutation=mutation.__name__):
                Engine.attention_mutation = mutation
                report = self.generate()
                self.assertEqual(report["status"], "failed")
                self.assertEqual(report["failure_stage"], "model_attention_qualification")
                self.assertEqual(Engine.instances[-1].calls, 0)
                self.assertFalse(report["functional_acceptance"])
                self.assertTrue(report["owned_resources_closed"])
                self.assertTrue(report["memory_runtime_closed"])

    def test_empty_eos_and_unexpected_reasoning_fail_plain_response_contract(self):
        for sequence in ((1,), (6, 1)):
            with self.subTest(sequence=sequence):
                Engine.selections = sequence
                report = self.generate()
                self.assertEqual(report["status"], "failed")
                self.assertFalse(report["functional_acceptance"])
                self.assertTrue(report["memory_runtime_closed"])

    def test_failed_model_call_has_no_retry_or_fallback_and_keeps_private_reason(self):
        Engine.fail_at = 2
        report = self.generate()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(Engine.instances[0].calls, 2)
        self.assertEqual(report["completion"]["calls"], 1)
        self.assertEqual(report["completion"]["selected_ids_including_eos"], [])
        self.assertEqual(report["failed_step"]["driver_position_after_failure"], 1)
        self.assertTrue(report["private_failure"]["saved"])
        self.assertNotIn("secret-model-path", json.dumps(report))
        raw = self.ws.artifact_path(report["private_failure"]["artifact"]).read_text(encoding="utf-8")
        self.assertIn("secret-model-path", raw)
        self.assertTrue(report["owned_resources_closed"])
        self.assertTrue(report["memory_runtime_closed"])

    def test_actual_route_and_returned_route_mismatch_cannot_pass(self):
        Engine.wrong_route = True
        report = self.generate()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(Engine.instances[0].calls, 1)
        self.assertEqual(report["failed_step"]["route_io"][39]["indices"], list(range(6)))
        self.assertFalse(report["functional_acceptance"])

    def test_local_prepare_does_not_open_tokenizer_cuda_or_model(self):
        with patch.object(probe, "PinnedTokenizer", side_effect=AssertionError("not preparation")), \
             patch.object(probe, "open_spark_backbone", side_effect=AssertionError("not preparation")):
            report = probe.collect("prepare", self.ws, run_id="prepare-r01")
        self.assertEqual(report["status"], "local_moe_prepared")
        self.assertFalse(report["functional_acceptance"])
        self.assertFalse(report["cuda_inventory_requested"])
        self.inventory.assert_not_called()
        self.assertEqual(Assets.events, ["assets_open", "prepare", "assets_close"])
        self.assertEqual(report["memory_after_runtime_close"]["charged_host_bytes"], 0)

    def test_cleanup_failure_cannot_leave_a_successful_response_status(self):
        Engine.close_failure = True
        report = self.generate()
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["functional_acceptance"])
        self.assertTrue(report["memory_runtime_closed"])

    def test_remaining_owned_allocation_prevents_runtime_close_and_acceptance(self):
        def leave_lease(engine, *_):
            Assets.events.append("engine_close_with_retained_lease")
        with patch.object(Engine, "__exit__", new=leave_lease):
            report = self.generate()
        try:
            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["functional_acceptance"])
            self.assertFalse(report["owned_resources_closed"])
            self.assertFalse(report["memory_runtime_closed"])
            self.assertGreater(report["memory_after_runtime_close"]["charged_host_bytes"], 0)
        finally:
            # Deliberate leaked-allocation seam only: restore the process-wide
            # test governor after checking that publication refused success.
            owner = Engine.instances[0]
            owner.budget.release(owner.budget.reserved)
            current_policy().close()

    def test_model_budget_rejects_excess_prompt_before_engine_open(self):
        Tokenizer.prompt_ids = tuple([10] * 500)
        report = self.generate(max_new_tokens=64)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(Engine.instances, [])
        self.assertTrue(report["memory_runtime_closed"])

    def test_missing_local_moe_or_bundle_precedes_cuda_and_model(self):
        Assets.missing_moe = True
        report = self.generate()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["failure_stage"], "prepared_local_assets")
        self.inventory.assert_not_called()
        self.assertEqual(Engine.instances, [])
        Assets.missing_moe = False
        with patch.object(probe, "load_candidate", side_effect=ValueError("foreign Attention bundle")):
            report = self.generate()
        self.assertEqual(report["status"], "failed")
        self.inventory.assert_not_called()
        self.assertEqual(Engine.instances, [])

    def test_scope_and_path_refusals_precede_hardware_and_assets(self):
        with patch.object(self.ws, "context", return_value={**self.ws.context(), "revision": "f" * 40}):
            with self.assertRaises(ValueError):
                self.generate()
        with self.assertRaises(ValueError):
            self.generate(originals="../foreign-platform")
        self.ws.check_runtime_module.side_effect = ValueError("old installed module")
        with self.assertRaisesRegex(ValueError, "old installed module"):
            self.generate()
        self.assertEqual(Assets.events, [])
        self.inventory.assert_not_called()

    def test_invalid_request_configuration_cannot_open_resources(self):
        for kwargs in ({"max_new_tokens": 513}, {"temperature": float("nan")}, {"temperature": 1000.01}, {"seed": -1}, {"prompt": ""}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.generate(**kwargs)
        self.assertEqual(Assets.events, [])

    def test_physical_floor_refusal_does_not_use_cuda_free_as_extra_ram(self):
        with patch.object(probe, "host_memory", return_value={"physical_available_bytes": 32 * GiB,
                                                               "cgroup_headroom_bytes": None}):
            report = self.generate()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(Assets.events, [])
        self.inventory.assert_not_called()
        self.assertFalse(report["functional_acceptance"])

    def test_changed_source_context_refuses_publication_after_cleanup(self):
        Engine.change_source = True
        with self.assertRaisesRegex(ValueError, "source identity changed"):
            self.generate()
        self.assertEqual(self.ws.writes, [])
        self.assertEqual(Assets.events[-2:], ["engine_close", "assets_close"])

    def test_cli_writes_full_result_once_after_cleanup_and_prints_relative_path(self):
        with patch("vesper.workspace.Workspace.open", return_value=self.ws), redirect_stdout(io.StringIO()) as stream:
            code = probe.main(["generate", "--run-id", "cli-r01"])
        self.assertEqual(code, 0)
        status = json.loads(stream.getvalue())
        self.assertEqual(status["result"], "results/dgx-spark/s1/text/cli-r01.json")
        self.assertNotIn(str(self.ws.root), stream.getvalue())
        report = self.ws.writes[0]
        self.assertEqual(report["memory_after_runtime_close"]["charged_host_bytes"], 0)
        self.assertIn("serialized_chat", report["request"])
        with patch("vesper.workspace.Workspace.open", return_value=self.ws), self.assertRaises(FileExistsError):
            probe.main(["generate", "--run-id", "cli-r01"])
        self.assertEqual(len(Engine.instances), 1)

    def test_partial_cli_returns_nonzero_even_with_successful_resource_cleanup(self):
        Engine.selections = (4,)
        with patch("vesper.workspace.Workspace.open", return_value=self.ws), redirect_stdout(io.StringIO()):
            code = probe.main(["generate", "--run-id", "partial-r01", "--max-new-tokens", "1"])
        self.assertEqual(code, 2)
        self.assertFalse(self.ws.writes[0]["functional_acceptance"])

    def test_cli_records_explicit_thinking_and_independent_request_bounds(self):
        with patch("vesper.workspace.Workspace.open", return_value=self.ws), redirect_stdout(io.StringIO()):
            code = probe.main(["generate", "--run-id", "thinking-cli-r01", "--thinking-mode", "thinking",
                "--reasoning-effort", "25", "--context-tokens", "1152", "--max-input-tokens", "700",
                "--max-new-tokens", "32", "--max-model-calls", "1", "--max-duration-seconds", "60"])
        self.assertEqual(code, 2)
        report = self.ws.writes[0]
        self.assertEqual(report["status"], "partial_model_call_budget")
        self.assertEqual(report["request"]["thinking_mode"], "thinking")
        self.assertEqual(report["request"]["reasoning_effort"], 25)
        self.assertEqual(report["request"]["admitted_limits"]["model_calls"], 1)
        self.assertEqual(report["request"]["admitted_limits"]["input_tokens"], 700)
        self.assertEqual(report["request"]["admitted_limits"]["context_tokens"], 1152)


if __name__ == "__main__":
    unittest.main()
