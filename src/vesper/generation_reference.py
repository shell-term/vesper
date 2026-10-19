"""Unchanged original model methods driving their own text continuation."""

from contextlib import ExitStack
from pathlib import Path
import ast
import gc
import json
from types import SimpleNamespace
from typing import List
from unittest.mock import patch

from .attention_storage_reference import _numpy
from .backbone import LAYERS
from .backbone_assets import load as load_bindings
from .backbone_reference import BackboneOracle
from .generation_assets import load_token_rows
from .generation_validation import REQUEST, scope, load_plan, plan_identity, expert_plan, record, identity
from .memory import RuntimeMemory, MemoryPolicy
from .model_artifacts import canonical, digest, read_checked
from .output_assets import load as load_head
from .resources import GiB, MemoryBudget
from .tokenizer import PinnedTokenizer


def capture_original(root, run, progress=None):
    import torch
    root, run = Path(root), Path(run); plan = load_plan(root, run)
    bindings = json.loads((root / "backbone-controls.json").read_bytes())
    previous = torch.get_default_dtype(); torch.set_default_dtype(torch.bfloat16)
    try:
        torch.cuda.init(); baseline_gpu = torch.cuda.memory_allocated(0); torch.cuda.reset_peak_memory_stats(0)
        with RuntimeMemory(MemoryPolicy(host_cap_bytes=32 * GiB)) as policy:
            with ExitStack() as stack, torch.inference_mode(), torch.cuda.device(0), torch.device("cuda:0"):
                stack.enter_context(patch("urllib.request.urlopen", side_effect=RuntimeError("Original text generation must remain offline")))
                budget = MemoryBudget(cap=20 * GiB, owner="generation_original_fixture_and_framework")
                budget.acquire(budget.cap); stack.callback(budget.release, budget.cap)
                arrays = load_bindings(root, bindings); head_arrays, head_manifest = load_head(root / "output-fixture")
                stack.callback(arrays.clear); stack.callback(head_arrays.clear)
                tokens = load_token_rows(root, run, plan)
                oracle = BackboneOracle(stack, root, arrays, head_arrays, progress, token_rows=tokens,
                    rows_root=run / "rows", experts_root=run / "experts", max_seq_len=8)
                oracle.bindings = bindings
                if digest(_numpy(oracle.hash.token_map).astype("int64").tobytes()) != plan["tokenizer"]["compressed_map_sha256"]:
                    raise ValueError("Independent original compressed tokenizer differs")
                # Execute the pinned upstream generation loop itself. Its first
                # prompt chunk is serialized by an explicit memory adapter. The
                # adapter discards intermediate greedy IDs without consuming
                # RNG, and calls unchanged Transformer.forward for each query.
                # This qualifies serial ingestion, not original chunk-prefill.
                raw = (root / "reference/generate.py").read_bytes()
                if digest(raw) != oracle.p[0].source["files"]["generate.py"]: raise ValueError("Pinned original generation source differs")
                node = next(n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name == "generate")
                method_sha = digest(ast.dump(node).encode())
                namespace = {"torch": torch, "List": List, "Transformer": torch.nn.Module}
                exec(compile(ast.Module(body=[node], type_ignores=[]), "<pinned-original-generate>", "exec"), namespace)
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(str(root / "import-source"), local_files_only=True, trust_remote_code=False)
                if tokenizer.eos_token_id != REQUEST.eos_id: raise ValueError("Original tokenizer EOS differs from the request")
                prompt = tokenizer.encode(REQUEST.prompt)
                if prompt != PinnedTokenizer(root / "import-source").encode(REQUEST.prompt): raise ValueError("Original prompt encoding differs")
                cases = []
                def serial_forward(tokens, start_pos, images=None, token_types=None):
                    if images is not None or token_types is not None or start_pos != oracle.position: raise ValueError("Unexpected original generation input")
                    for i in range(tokens.shape[1]):
                        ids = _numpy(tokens[:, i:i + 1]).copy(); sample = i == tokens.shape[1] - 1
                        result, state = oracle.step(ids, sample=sample)
                        selected = int(result["sampling"]["ids_i64"][0]) if sample else None
                        meta = {"position": start_pos + i, "token_id": int(ids[0, 0]), "sampling_requested": sample, "selected_id": selected}
                        cases.append(record(meta, result, state))
                        cases[-1]["original_auxiliary_main_hidden"] = oracle.last_auxiliary
                        if progress: progress({"original_token": meta, "boundaries": len(cases[-1]["boundaries"])})
                    return torch.from_numpy(result["sampling"]["ids_i64"].copy()).to("cuda:0"), None, None
                adapter = SimpleNamespace(max_seq_len=8, forward=serial_forward)
                completion_ids = namespace["generate"](adapter, [prompt], REQUEST.max_new_tokens, REQUEST.eos_id)[0]
                selected_ids = [c["meta"]["selected_id"] for c in cases if c["meta"]["sampling_requested"]]
                completion = {"prompt": REQUEST.prompt, "prompt_ids": prompt, "max_new_tokens": REQUEST.max_new_tokens,
                    "eos_id": REQUEST.eos_id, "calls": len(cases), "selected_ids_including_eos": selected_ids,
                    "completion_ids": completion_ids, "completion_text": tokenizer.decode(completion_ids),
                    "stop_reason": "eos" if selected_ids[-1] == REQUEST.eos_id else "max_new_tokens"}
                report = {"schema_version": 1, "probe": "original_text_generation_goldens", "status": "pass", "scope": scope(),
                    "plan_sha256": plan_identity(plan), "bindings_sha256": bindings["bindings_sha256"], "output_fixture": head_manifest,
                    "expert_plan": expert_plan(plan), "expert_provenance": oracle.expert_provenance, "assets": plan["assets"], "tokenizer": plan["tokenizer"],
                    "cases": cases, "completion": completion, "reference_methods": oracle.methods, "original_source": oracle.p[0].source,
                    "original_generate_method_sha256": method_sha,
                    "original_ngram_source_sha256": digest(read_checked(root / "import-source", "inference/engram.py")),
                    "Engram_controls": oracle.engram.controls, "Engram_fixtures": oracle.engram.manifests,
                    "original_embedding_transport": oracle.virtual.stats(), "original_attention_adapter": oracle.qualification,
                    "minimum_sampled_reference_device_free_bytes": oracle.minimum_gpu_free,
                    "original_grouped_conversion": {str(l): (oracle.p[l] if l < 2 else oracle.sharing.p[l]).converted_sha256 for l in LAYERS},
                    "weight_payloads_fetched_in_reference": 0, "python_http_disabled_during_reference_execution": True}
                report["identity_sha256"], report["complete_goldens_sha256"] = identity(report), canonical(cases)
                oracle.close(); del oracle, adapter, serial_forward
            gc.collect(); torch.cuda.synchronize()
            report["reference_torch_device_allocated_before_workspace_cleanup_bytes"] = torch.cuda.memory_allocated(0)
            torch._C._cuda_clearCublasWorkspaces(); gc.collect(); torch.cuda.empty_cache()
            report["reference_torch_device_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
            report["reference_torch_device_allocated_baseline_bytes"] = baseline_gpu
            report["reference_torch_device_allocated_after_cleanup_bytes"] = torch.cuda.memory_allocated(0)
            if torch.cuda.memory_allocated(0) != baseline_gpu: raise RuntimeError("Original generation retained framework GPU allocations")
            report["memory_before_policy_close"] = policy.snapshot()
        report["memory_after_policy_close"] = policy.snapshot()
        return report
    finally: torch.set_default_dtype(previous)
