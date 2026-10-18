"""Optional unchanged quantization/window oracle at a post-RoPE boundary."""

import ast
from contextlib import ExitStack
import hashlib
import importlib.util
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .attention import NativeAttention
from .attention_memory_reference import AttentionMemoryReference, missing_dependencies
from .attention_storage import STAGES
from .attention_storage_reference import _numpy
from .attention_validation import array_digest, check_arrays
from .checkpoint import MODEL, REVISION, fetch_reference
from .kv import NativeWindowKV
from .kv_validation import exercise_window, kv_inputs, query_inputs
from .reference_runtime import configure_reference_runtime
from .resources import MiB, MemoryBudget


class WindowOracle:
    def __init__(self, directory, batch, ordinal):
        import torch
        import tilelang
        self.torch, self.batch, self.ordinal = torch, batch, ordinal
        self.source = fetch_reference(directory, include_model=True)
        spec = importlib.util.spec_from_file_location("vesper_window_official_kernel", directory / "kernel.py")
        self.kernel = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.kernel)
        self.workarounds = configure_reference_runtime(self.kernel, tilelang.__version__)
        tree = ast.parse((directory / "model.py").read_bytes())
        attention = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Attention")
        window = next(n for n in attention.body if isinstance(n, ast.FunctionDef) and n.name == "_window_kv")
        indices = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "get_window_topk_idxs")
        self.method_hashes = {n.name: hashlib.sha256(ast.dump(n).encode()).hexdigest() for n in (window, indices)}
        # Inputs start after these three operations. Their identity stand-ins
        # preserve that explicit boundary; they do not qualify those operations.
        namespace = {"torch": torch, "lru_cache": lru_cache, "act_quant": self.kernel.act_quant,
                     "fp8_block_size": 32, "scale_fmt": "ue8m0", "scale_dtype": torch.float8_e8m0fnu,
                     "apply_rotary_emb": lambda values, frequencies: values}
        exec(compile(ast.Module(body=[indices, window], type_ignores=[]), "<pinned-window-methods>", "exec"), namespace)
        self.window = namespace["_window_kv"]
        self.stub = SimpleNamespace(window_size=128, wkv=lambda x: x, kv_norm=lambda x: x,
                                    rope_head_dim=64, window_kv_cache=torch.zeros(batch, 128, 512, dtype=torch.bfloat16, device=f"cuda:{ordinal}"))
        self.saved, self.position = None, 0

    def control(self, event):
        if event == "reset":
            self.stub.window_kv_cache.zero_()
            self.saved, self.position = None, 0
        elif event == "snapshot":
            self.saved = self.stub.window_kv_cache.clone(), self.position
        elif event == "rollback":
            cache, self.position = self.saved
            self.stub.window_kv_cache.copy_(cache)
            self.saved = None
        elif event == "discard":
            self.saved = None
        else:
            raise ValueError("Unknown reference state event")

    def step(self, meta, values):
        torch = self.torch
        if meta["position"] != self.position:
            raise ValueError("Native/reference window positions diverged")
        with torch.inference_mode(), torch.cuda.device(self.ordinal), torch.device(f"cuda:{self.ordinal}"):
            x = torch.from_numpy(values).view(torch.bfloat16).to(f"cuda:{self.ordinal}")
            codes, scales = self.kernel.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
            window, indices = self.window(self.stub, x, None, self.position)
            self.position += values.shape[1]
            return {"rounded_bf16": _numpy(x), "codes_fp8": codes.view(torch.uint8).cpu().numpy().copy(),
                    "scales_e8m0": scales.view(torch.uint8).cpu().numpy().copy(),
                    "cache_bf16": _numpy(self.stub.window_kv_cache), "window_bf16": _numpy(window), "indices_i32": _numpy(indices)}

    def negative_controls(self):
        torch = self.torch
        values = kv_inputs(1, 3, 901234, "mixed_scales")
        with torch.inference_mode(), torch.cuda.device(self.ordinal):
            x = torch.from_numpy(values).view(torch.bfloat16).to(f"cuda:{self.ordinal}")
            expected = self.kernel.act_quant(x.clone(), 32, "ue8m0", torch.float8_e8m0fnu, True)
            whole = self.kernel.act_quant(x.clone(), 512, "ue8m0", torch.float8_e8m0fnu, True)
            skipped_tail = x.clone()
            self.kernel.act_quant(skipped_tail[..., :448], 32, "ue8m0", torch.float8_e8m0fnu, True)
            return [check_arrays(_numpy(expected), _numpy(actual), label) for label, actual in (
                ("omitted_rounding", x), ("one_scale_per_512_instead_of_per_32", whole), ("omitted_RoPE_tail_rounding", skipped_tail))]


def compare_window_kv(directory, attention_bundle, ordinal=0):
    missing = missing_dependencies()
    if missing:
        return {"probe": "native_window_kv_parity", "status": "blocked", "missing_dependencies": missing}
    import torch
    directory = Path(directory)
    goldens, phases, rows, faults, events = [], [], [], [], []
    with ExitStack() as stack:
        oracle_budget = MemoryBudget(cap=32 * MiB)
        oracle_budget.acquire(32 * MiB)
        stack.callback(oracle_budget.release, 32 * MiB)
        memory = stack.enter_context(AttentionMemoryReference(directory, ordinal))
        qualification = memory.qualify()
        if qualification["status"] != "pass":
            raise RuntimeError("Attention integration oracle failed its qualification")
        attention = stack.enter_context(NativeAttention(attention_bundle, ordinal))
        for batch in (1, 2):
            oracle = WindowOracle(directory, batch, ordinal)
            def checker(meta, values, actual):
                expected = oracle.step(meta, values)
                if meta["attention_checked"]:
                    q, sink = query_inputs(batch, meta["sequence"], meta["seed"])
                    with torch.inference_mode(), torch.cuda.device(ordinal):
                        tensors = (torch.from_numpy(q).view(torch.bfloat16).to(f"cuda:{ordinal}"),
                                   torch.from_numpy(expected["window_bf16"]).view(torch.bfloat16).to(f"cuda:{ordinal}"),
                                   torch.from_numpy(sink).to(f"cuda:{ordinal}"),
                                   torch.from_numpy(expected["indices_i32"]).to(f"cuda:{ordinal}"))
                        output, traces, _ = memory._run(64, 512, True, tensors)
                    expected.update(("attention_" + name, _numpy(a)) for name, a in zip(("output_bf16", *STAGES), (output, *traces)))
                if set(expected) != set(actual):
                    raise ValueError("Window/Attention boundary set differs")
                goldens.append({"meta": meta, "stage_sha256": {name: array_digest(a) for name, a in expected.items()}})
                return [check_arrays(a, actual[name], name) for name, a in expected.items()]
            with NativeWindowKV(batch, ordinal) as native:
                records, failed, operations, minimum = exercise_window(native, attention, checker, oracle.control)
                profile = {"compiler": native.compiler, "cubin_sha256": native.cubin_sha256,
                           "kernel_source_sha256": native.kernel_source_sha256, "device": native.device_info}
            phases.append({"batch": batch, "minimum_sampled_host_available_bytes": minimum, "cleanup": native.stats()})
            rows.extend(records)
            faults.extend({"batch": batch, **f} for f in failed)
            events.extend({"batch": batch, **e} for e in operations)
        controls = oracle.negative_controls()
        attention_manifest_sha = attention.manifest_sha256
        source, methods, workarounds = oracle.source, oracle.method_hashes, oracle.workarounds
    passed = (all(r["status"] == "pass" for r in rows + faults + events)
              and all(c["mismatches"] > 0 and c["status"] == "fail" for c in controls)
              and all(not p["cleanup"]["host_reserved_bytes"] and not p["cleanup"]["device_owned_bytes"] for p in phases)
              and not attention.stats()["device_owned_bytes"] and oracle_budget.reserved == 0)
    return {"schema_version": 1, "probe": "native_window_kv_parity", "status": "pass" if passed else "fail",
            "model": MODEL, "revision": REVISION, "reference": source, "reference_methods_ast_sha256": methods,
            "runtime_workarounds": workarounds, "native_profile": profile, "attention_bundle_manifest_sha256": attention_manifest_sha,
            "cases": rows, "faults": faults, "state_events": events, "reference_goldens": goldens,
            "negative_controls": controls, "attention_reference_qualification": qualification,
            "memory": {"phases": phases, "attention_cleanup": attention.stats(), "oracle_host_reserved_after_close": oracle_budget.reserved,
                       "combined_native_and_reference_host_allowance_bytes": 144 * MiB, "reservation_is_RSS_limit": False},
            "weights_fetched_bytes": 0, "projection_normalization_and_RoPE": "identity_standins_at_post_RoPE_boundary",
            "rollback_reference": "exact saved original cache bits and logical position; runtime extension, not an upstream rollback method",
            "native_compressed_KV_indexing_and_sharing": "not_run", "full_model_quality": "not_run", "end_to_end_speed": "not_measured",
            "scope": "Bounded original per-32 post-RoPE quant+dequant, pure sliding-window state and selected sparse Attention integration"}
