"""Pinned original projections and unchanged Attention.forward as an oracle."""

import ast
import base64
from contextlib import ExitStack
from functools import lru_cache
import hashlib
import importlib.util
import os
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np

from .attention import NativeAttention
from .attention_memory_reference import AttentionMemoryReference, missing_dependencies
from .attention_storage import STAGES
from .attention_storage_reference import _numpy
from .attention_validation import check_arrays
from .checkpoint import MODEL, REVISION
from .gate_reference import fetch_gate_reference
from .normalization import NativeNormRoPE
from .normalization_fixture import NORM_WEIGHTS
from .normalization_reference import ReferenceMath
from .prefix import PREFIX_IDS, PREFIX_SEQUENCES, load_prefix_fixture
from .prefix_reference import fetch_prefix_reference
from .projections import FP8_LAYOUT, NativeProjections
from .reference_runtime import configure_reference_runtime
from .resources import MemoryBudget, MiB
from .window_attention import NativeWindowAttention


class ProjectionOracle:
    def __init__(self, directory, arrays, ordinal, *, layer=0):
        self.layer = layer
        import torch
        import tilelang
        if (str(torch.backends.cuda.preferred_blas_library()) != "_BlasBackend.Cublas"
                or torch.cuda.tunable.is_enabled() or torch.get_float32_matmul_precision() != "highest"
                or not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
                or os.environ.get("CUBLAS_WORKSPACE_CONFIG") is not None):
            raise ValueError("Unregistered reference BLAS settings; qualification required")
        self.torch, self.ordinal = torch, ordinal
        self.source = fetch_prefix_reference(Path(directory))
        self.blas_source = fetch_gate_reference(Path(directory) / "pytorch-gate")
        spec = importlib.util.spec_from_file_location("vesper_original_projection_kernel", Path(directory) / "kernel.py")
        self.kernel = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.kernel)
        self.workarounds = configure_reference_runtime(self.kernel, tilelang.__version__)
        norm_names = NORM_WEIGHTS if layer == 0 else (f"layers.{layer}.attn.q_norm.weight", f"layers.{layer}.attn.kv_norm.weight")
        if layer == 1:
            norm_names += ("layers.1.attn_norm.weight", "layers.1.ffn_norm.weight")
        self.math = ReferenceMath(directory, {n: arrays[n] for n in norm_names}, ordinal)
        self.tree = ast.parse((Path(directory) / "model.py").read_bytes())
        self.scope = {"torch": torch, "F": torch.nn.functional, "lru_cache": lru_cache, "world_size": 1, "fp8_block_size": 32,
                      "scale_fmt": "ue8m0", "scale_dtype": torch.float8_e8m0fnu, "act_quant": self.kernel.act_quant,
                      "fp8_gemm": self.kernel.fp8_gemm, "fp4_gemm": self.kernel.fp4_gemm,
                      "apply_rotary_emb": self.math.scope["apply_rotary_emb"], "hc_split_sinkhorn": self.kernel.hc_split_sinkhorn}
        self.methods = {}
        for name in ("linear", "get_window_topk_idxs", "make_identity_pre_mix"):
            node = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            self.methods[name] = hashlib.sha256(ast.dump(node).encode()).hexdigest()
            exec(compile(ast.Module(body=[node], type_ignores=[]), "<pinned-projection-model>", "exec"), self.scope)
        self.weights = {}
        for name in (*FP8_LAYOUT, "wo_a"):
            w = torch.from_numpy(arrays[f"layers.{layer}.attn.{name}.weight"]).to(f"cuda:{ordinal}").view(torch.float8_e4m3fn)
            s = torch.from_numpy(arrays[f"layers.{layer}.attn.{name}.scale"]).to(f"cuda:{ordinal}").view(torch.float8_e8m0fnu)
            w.scale = s
            self.weights[name] = w
        state = {f"layers.{layer}.attn.wo_a.weight": self.weights["wo_a"], f"layers.{layer}.attn.wo_a.scale": self.weights["wo_a"].scale}
        conversion = ast.parse((Path(directory) / "convert.py").read_bytes())
        node = next(n for n in ast.walk(conversion) if isinstance(n, ast.If) and isinstance(n.test, ast.Call)
                    and ast.unparse(n.test) == "name.endswith('wo_a.weight')")
        self.methods["wo_a_conversion"] = hashlib.sha256(ast.dump(node).encode()).hexdigest()
        local = {"torch": torch, "state_dicts": [state], "i": 0, "name": f"layers.{layer}.attn.wo_a.weight"}
        exec(compile(ast.Module(body=node.body, type_ignores=[]), "<pinned-wo-a-conversion>", "exec"), local)
        self.weights["wo_a"] = state[f"layers.{layer}.attn.wo_a.weight"]
        self.converted_sha256 = hashlib.sha256(_numpy(self.weights["wo_a"]).tobytes()).hexdigest()
        self.sink = torch.from_numpy(arrays[f"layers.{layer}.attn.attn_sink"]).to(f"cuda:{ordinal}")
        self.fp32_kernels, self.fp8_controls, self.layer_controls = {}, 0, []

    def method(self, cls, name):
        parent = next(n for n in self.tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
        node = next(n for n in parent.body if isinstance(n, ast.FunctionDef) and n.name == name)
        self.methods[cls + "." + name] = hashlib.sha256(ast.dump(node).encode()).hexdigest()
        local = dict(self.scope)
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<pinned-projection-method>", "exec"), local)
        return local[name]

    def fp8(self, name, values):
        torch = self.torch
        x, weight = self.math.tensor(values), self.weights[name]
        q, scales = self.kernel.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
        output = self.scope["linear"](x, weight)
        rows, width = FP8_LAYOUT[name]
        if name not in self.fp32_kernels:
            self.fp32_kernels[name] = self.kernel.fp8_gemm_kernel(rows, width, block_size=32, out_dtype="float32", scale_dtype="float8_e8m0fnu")
        trace = torch.empty((*values.shape[:-1], rows), dtype=torch.float32, device=x.device)
        self.fp32_kernels[name](q.view(-1, width), weight, trace.view(-1, rows), scales.view(-1, width // 32), weight.scale)
        if not torch.equal(output.view(torch.uint16), trace.bfloat16().view(torch.uint16)):
            raise RuntimeError("FP32 diagnostic output differs from unchanged original linear BF16 output")
        self.fp8_controls += 1
        return {"input_fp8": _numpy(q.view(torch.uint8)), "scale_e8m0": _numpy(scales.view(torch.uint8)),
                "output_fp32": _numpy(trace), "output_bf16": _numpy(output)}

    def grouped(self, values):
        torch = self.torch
        x = self.math.tensor(values)
        q, s = self.kernel.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
        rounded = self.kernel.act_quant(x.clone(), 32, "ue8m0", torch.float8_e8m0fnu, True)
        output = torch.einsum("bsgd,grd->bsgr", rounded, self.weights["wo_a"].view(8, 1024, 4096))
        return {"input_fp8": _numpy(q.view(torch.uint8)), "scale_e8m0": _numpy(s.view(torch.uint8)),
                "rounded_bf16": _numpy(rounded), "output_bf16": _numpy(output)}

    def round_window(self, values):
        x = self.math.tensor(values)
        q, s = self.kernel.act_quant(x, 32, "ue8m0", self.torch.float8_e8m0fnu)
        rounded = self.kernel.act_quant(x.clone(), 32, "ue8m0", self.torch.float8_e8m0fnu, True)
        return {"rounded_bf16": _numpy(rounded), "codes_fp8": _numpy(q.view(self.torch.uint8)),
                "scales_e8m0": _numpy(s.view(self.torch.uint8))}

    def stub(self, batch, memory):
        self.scope["sparse_attn"] = memory
        projections = {}
        for name in FP8_LAYOUT:
            fn = self.method("RowParallelLinear" if name == "wo_b" else "Linear", "forward")
            obj = SimpleNamespace(weight=self.weights[name], bias=None)
            projections[name] = lambda x, f=fn, o=obj: f(o, x)
        stub = SimpleNamespace(n_local_heads=64, n_local_groups=8, head_dim=512, rope_head_dim=64, o_lora_rank=1024,
            window_size=128, compress_ratio=0, softmax_scale=512**-.5, attn_sink=self.sink,
            q_norm=self.math.modules[f"layers.{self.layer}.attn.q_norm.weight"], kv_norm=self.math.modules[f"layers.{self.layer}.attn.kv_norm.weight"],
            freqs_cis=self.math.tables["window"], window_kv_cache=self.torch.zeros(batch, 128, 512, device=f"cuda:{self.ordinal}"),
            wo_a=SimpleNamespace(weight=self.weights["wo_a"]), **projections)
        stub._window_kv = MethodType(self.method("Attention", "_window_kv"), stub)
        return stub

    def reset(self, batch, memory):
        self.batch, self.memory, self.position = batch, memory, 0
        self.original = self.stub(batch, memory)
        self.full_forward = self.method("Attention", "forward")

    def window_step(self, values, position):
        if position != self.position:
            raise ValueError("Reference/native Attention positions differ")
        expected = {}
        def add(prefix, result):
            expected.update((prefix + n, a) for n, a in result.items())
            return result["output_bf16"]
        q = add("wq_a_", self.fp8("wq_a", values))
        q = add("q_norm_", self.math.norm(q, f"layers.{self.layer}.attn.q_norm.weight"))
        q = add("wq_b_", self.fp8("wq_b", q)).reshape(*values.shape[:2], 64, 512)
        q = add("query_rope_", self.math.rope(q, position, "window", False))
        kv = add("wkv_", self.fp8("wkv", values))
        kv = add("kv_norm_", self.math.norm(kv, f"layers.{self.layer}.attn.kv_norm.weight"))
        kv = add("kv_rope_", self.math.rope(kv, position, "window", False))
        diagnostic = self.stub(self.batch, self.memory)
        diagnostic.window_kv_cache.copy_(self.original.window_kv_cache)
        freq = self.math.tables["window"][position:position + values.shape[1]]
        window, indices = diagnostic._window_kv(self.math.tensor(values), freq, position)
        qkv = self.math.tensor(kv)
        codes, scales = self.kernel.act_quant(qkv, 32, "ue8m0", self.torch.float8_e8m0fnu)
        rounded = self.kernel.act_quant(qkv.clone(), 32, "ue8m0", self.torch.float8_e8m0fnu, True)
        cached = {"rounded_bf16": _numpy(rounded), "codes_fp8": _numpy(codes.view(self.torch.uint8)),
            "scales_e8m0": _numpy(scales.view(self.torch.uint8)), "cache_bf16": _numpy(diagnostic.window_kv_cache),
            "window_bf16": _numpy(window), "indices_i32": _numpy(indices)}
        expected.update(("kv_" + n, a) for n, a in cached.items())
        out, traces, _ = self.memory._run(64, 512, True, (self.math.tensor(q), window, self.sink, indices))
        o = add("attention_", dict(zip(("output_bf16", *STAGES), map(_numpy, (out, *traces)))))
        o = add("inverse_rope_", self.math.rope(o, position, "window", True))
        o = add("wo_a_", self.grouped(o.reshape(*values.shape[:2], 8, 4096)))
        add("wo_b_", self.fp8("wo_b", o.reshape(*values.shape[:2], 8192)))
        original = self.full_forward(self.original, self.math.tensor(values), position)
        controls = [check_arrays(_numpy(original), expected["wo_b_output_bf16"], "unchanged_Attention_forward_BF16"),
                    check_arrays(_numpy(self.original.window_kv_cache), cached["cache_bf16"], "unchanged_Attention_forward_KV")]
        if any(c["status"] != "pass" for c in controls):
            raise RuntimeError("Instrumented chain differs from unchanged Attention.forward")
        self.layer_controls.append(controls)
        self.position += values.shape[1]
        return expected

    def capture_real_inputs(self, arrays, memory):
        torch = self.torch
        block = SimpleNamespace(norm_eps=1e-20, hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6,
            attn_norm=self.math.modules["layers.0.attn_norm.weight"], ffn_norm=self.math.modules["layers.0.ffn_norm.weight"])
        for name in ("hc_attn_fn", "hc_attn_scale", "hc_attn_base", "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base"):
            setattr(block, name, torch.from_numpy(arrays["layers.0." + name]).to(f"cuda:{self.ordinal}"))
        for name in ("hc_mixes", "hc_pre", "hc_post"):
            setattr(block, name, MethodType(self.method("Block", name), block))
        forward = self.method("Block", "forward")
        class Captured(Exception):
            pass
        def stop(*args):
            raise Captured()
        block.ffn = stop
        captured = []
        for seq, tokens in enumerate(PREFIX_SEQUENCES):
            stub = self.stub(1, memory)
            attn_forward = self.method("Attention", "forward")
            def attention(x, position):
                raw = _numpy(x).tobytes()
                captured.append({"sequence": seq, "position": position, "token_id": tokens[position],
                    "shape": list(x.shape), "input_bf16_sha256": hashlib.sha256(raw).hexdigest(),
                    "input_bf16_base64": base64.b64encode(raw).decode()})
                return attn_forward(stub, x, position)
            block.attn = attention
            for position, token in enumerate(tokens):
                h = self.math.tensor(arrays["embedding_rows"][PREFIX_IDS.index(token)]).view(1, 1, 1, 5120).repeat(1, 1, 4, 1)
                mix = self.scope["make_identity_pre_mix"](h, 4)
                before = len(captured)
                try:
                    forward(block, h, position, mix, None)
                except Captured:
                    pass
                if len(captured) != before + 1:
                    raise RuntimeError("Original prefix did not reach exactly one Attention input")
        return captured

    def negative_controls(self):
        from .normalization_validation import inputs
        torch = self.torch
        bits = inputs((1, 2, 8, 4096), 867)
        x = self.math.tensor(bits)
        expected = self.grouped(bits)["output_bf16"]
        weight = self.weights["wo_a"].view(8, 1024, 4096)
        omit = torch.einsum("bsgd,grd->bsgr", x, weight)
        rounded = self.kernel.act_quant(x.clone(), 32, "ue8m0", torch.float8_e8m0fnu, True)
        promoted = torch.einsum("bsgd,grd->bsgr", rounded.float(), weight.float()).bfloat16()
        return [check_arrays(expected, _numpy(omit), "omit_output_activation_rounding"),
                check_arrays(expected, _numpy(promoted), "replace_grouped_BF16_with_FP32")]


def compare_projections(directory, prefix_fixture, attention_bundle, ordinal=0):
    missing = missing_dependencies()
    if missing:
        return {"probe": "native_attention_projection_parity", "status": "blocked", "missing_dependencies": missing}
    import torch
    from .projection_validation import exercise_components, exercise_layer, exercise_failures
    goldens, layer_rows, faults, phases = [], [], [], []
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    budget = MemoryBudget(cap=512 * MiB)
    try:
        budget.acquire(512 * MiB)
        arrays, fixture = load_prefix_fixture(prefix_fixture)
        with ExitStack() as stack, torch.inference_mode(), torch.cuda.device(ordinal), torch.device(f"cuda:{ordinal}"):
            reference = ProjectionOracle(directory, arrays, ordinal)
            memory = stack.enter_context(AttentionMemoryReference(Path(directory), ordinal))
            qualification = memory.qualify()
            if qualification["status"] != "pass":
                raise RuntimeError("Attention oracle qualification failed")
            real_inputs = reference.capture_real_inputs(arrays, memory)
            native = stack.enter_context(NativeProjections(arrays, ordinal))
            norm = stack.enter_context(NativeNormRoPE(ordinal))
            attention = stack.enter_context(NativeAttention(attention_bundle, ordinal))
            if native.converted_sha256 != reference.converted_sha256:
                raise RuntimeError("Original/native wo_a conversion differs")
            def collect(meta, expected, actual):
                if set(expected) != set(actual):
                    raise ValueError("Projection/layer trace boundary set differs")
                goldens.append({"meta": meta, "stage_sha256": {n: hashlib.sha256(a.tobytes()).hexdigest() for n, a in expected.items()}})
                return [check_arrays(a, actual[n], n) for n, a in expected.items()]
            def component(meta, values, actual):
                expected = reference.grouped(values) if meta["projection"] == "wo_a" else reference.fp8(meta["projection"], values)
                return collect(meta, expected, actual)
            components, minimum = exercise_components(native, component)
            for batch in (1, 2):
                def reset():
                    reference.reset(batch, memory)
                def check(meta, values, actual):
                    return collect(meta, reference.window_step(values, meta["position"]), actual)
                with NativeWindowAttention(arrays, native, norm, attention, batch) as layer:
                    rows, low = exercise_layer(layer, real_inputs, check, reset)
                    if batch == 1:
                        retries, failures = exercise_failures(layer, check)
                        rows.extend(retries)
                        faults.extend(failures)
                layer_rows.extend(rows)
                minimum = min(minimum, low)
                phases.append({"batch": batch, "KV_cleanup": layer.kv.stats()})
            negatives = reference.negative_controls()
            profile = {"compiler": native.compiler, "cubin_sha256": native.cubin_sha256, "device": native.device_info,
                       "cublas_version": native.blas.version, "cublas_math_mode": native.blas.math_mode,
                       "converted_wo_a_bf16_sha256": native.converted_sha256}
            attention_sha = attention.manifest_sha256
        passed = (all(r["status"] == "pass" for r in components + layer_rows + faults)
                  and all(n["mismatches"] > 0 for n in negatives)
                  and all(not p["KV_cleanup"]["device_owned_bytes"] for p in phases)
                  and not native.stats()["device_owned_bytes"] and not norm.stats()["device_owned_bytes"] and not attention.stats()["device_owned_bytes"])
        return {"schema_version": 1, "probe": "native_attention_projection_parity", "status": "pass" if passed else "fail",
            "model": MODEL, "revision": REVISION, "reference": reference.source, "blas_reference": reference.blas_source,
            "reference_methods_ast_sha256": reference.methods, "fixture": fixture, "native_profile": profile,
            "attention_reference_qualification": qualification, "attention_bundle_manifest_sha256": attention_sha,
            "component_cases": components, "layer_cases": layer_rows, "faults": faults, "reference_goldens": goldens,
            "captured_original_prefix_Attention_inputs": real_inputs, "negative_controls": negatives,
            "instrumentation_controls": {"FP32_projection_to_original_BF16": reference.fp8_controls,
                "unchanged_Attention_forward_output_and_KV": reference.layer_controls},
            "reference_runtime_workarounds": reference.workarounds,
            "memory": {"native_cleanup": native.stats(), "normalization_cleanup": norm.stats(), "attention_cleanup": attention.stats(),
                "KV_phases": phases, "combined_native_host_allowance_bytes": 656 * MiB,
                "combined_native_and_reference_host_allowance_bytes": 1232 * MiB, "minimum_sampled_host_available_bytes": minimum,
                "reservation_is_RSS_limit": False}, "weights_fetched_bytes": 0,
            "torch_version": torch.__version__, "torch_git_version": torch.version.git_version, "torch_cuda_version": torch.version.cuda,
            "scope": "Layer-0 window Attention, original weights and unchanged forward; generated and original-prefix inputs; one rank; bounded batch/prefill/decode",
            "native_HyperConnections_compressed_KV_indexing": "not_implemented", "full_model_quality": "not_run", "end_to_end_speed": "not_measured"}
    finally:
        if budget.reserved:
            budget.release(budget.reserved)
        torch.set_default_dtype(previous)
