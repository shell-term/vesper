"""Pinned original RMS/RoPE methods and projected-input Attention oracle."""

import ast
from contextlib import ExitStack
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import urllib.request

import numpy as np

from .attention import NativeAttention
from .attention_memory_reference import AttentionMemoryReference, missing_dependencies
from .attention_storage import STAGES
from .attention_storage_reference import _numpy
from .attention_validation import array_digest, check_arrays
from .checkpoint import MODEL, REVISION
from .kv import NativeWindowKV
from .kv_reference import WindowOracle
from .kv_validation import query_inputs
from .normalization import FREQUENCY_STAGES, MAX_POSITION, NORM_EPS, NORM_STAGES, ROPE_PROFILES, NativeNormRoPE
from .normalization_fixture import load_norm_weights
from .normalization_validation import exercise_chain, exercise_components, exercise_failures, inputs
from .prefix_reference import fetch_prefix_reference
from .resources import MemoryBudget, MiB
from .router import ROUTER_TORCH_REVISION

PYTORCH_SOURCES = {
    "Reduce.cuh": ("aten/src/ATen/native/cuda/Reduce.cuh", "5042ab10a27342625f5e3f469684ab446b9922e8ad16607e9ec673d27623e91c"),
    "ReduceMomentKernel.cu": ("aten/src/ATen/native/cuda/ReduceMomentKernel.cu", "f563fdfc101667cb7d637de622cceea7a8d0e40c29cac8c342c3439db3727f4c"),
    "SharedReduceOps.h": ("aten/src/ATen/native/SharedReduceOps.h", "582740ea0b91a3ef01387fc7aedec801310fd374f8d839419555d762e74be009"),
    "BinaryDivTrueKernel.cu": ("aten/src/ATen/native/cuda/BinaryDivTrueKernel.cu", "db07343723996d6b4d9931b19a93db9e5bdeacb96fca81538a496c5bf05b7bec"),
    "BinaryMulKernel.cu": ("aten/src/ATen/native/cuda/BinaryMulKernel.cu", "2a6325d254dc598ca00c1d359fbb01352b44afd2f803fb2cb761fdfeada8f1e0"),
    "PowKernel.cu": ("aten/src/ATen/native/cuda/PowKernel.cu", "51497f62d1abcf9568419bd63290e0d6784f66d6b0dac9e6e9fa643c6edcb9fe"),
    "UnaryOpsKernel.cu": ("aten/src/ATen/native/cuda/UnaryOpsKernel.cu", "ae9e870121de229ffb5cac1cfccab0bde4184d1c6da91dcd2c10de3ddaa9a4e5"),
    "ComplexKernel.cu": ("aten/src/ATen/native/cuda/ComplexKernel.cu", "6fa3a68c495485f18b30ebf85c1e559ddee57acbb693d59cfa6edf0b8bc08348"),
    "ComplexHeader.h": ("torch/headeronly/util/complex.h", "7f7d82f1def3117bb95100d5710d388a9083cd676f995875d73a2f37b1c286d1"),
    "LICENSE": ("LICENSE", "bd018feef8825e88181c84eb7e3aa4eafb8f08a20d9fd6ef948569610c4a3e43"),
}


def fetch_pytorch_sources(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, (source, expected) in PYTORCH_SOURCES.items():
        path = directory / name
        url = f"https://raw.githubusercontent.com/pytorch/pytorch/{ROUTER_TORCH_REVISION}/{source}"
        if path.exists():
            with path.open("rb") as stream:
                raw = stream.read(262145)
        else:
            with urllib.request.urlopen(url, timeout=30) as response:
                raw = response.read(262145)
        if len(raw) > 262144 or hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("Normalization reference source differs from its pinned hash/bound")
        if not path.exists():
            path.write_bytes(raw)
        records[name] = {"url": url, "bytes": len(raw), "sha256": expected}
    return {"revision": ROUTER_TORCH_REVISION, "files": records}


class ReferenceMath:
    def __init__(self, directory, weights, ordinal):
        import torch
        self.torch, self.ordinal = torch, ordinal
        if torch.version.git_version != ROUTER_TORCH_REVISION:
            raise ValueError("Unqualified PyTorch RMS/reduction/complex arithmetic build")
        self.source = fetch_prefix_reference(Path(directory))
        tree = ast.parse((Path(directory) / "model.py").read_bytes())
        names = {"RMSNorm", "precompute_freqs_cis", "apply_rotary_emb"}
        nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
        self.method_hashes = {n.name: hashlib.sha256(ast.dump(n).encode()).hexdigest() for n in nodes}
        self.scope = {"torch": torch, "nn": torch.nn, "math": math, "lru_cache": lru_cache}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "<pinned-RMS-RoPE>", "exec"), self.scope)
        self.modules, self.tables, self.inverses = {}, {}, {}
        self.norm_controls, self.rope_controls = 0, 0
        with torch.inference_mode(), torch.cuda.device(ordinal), torch.device(f"cuda:{ordinal}"):
            for name, weight in weights.items():
                module = self.scope["RMSNorm"](len(weight), NORM_EPS)
                module.weight = torch.nn.Parameter(torch.from_numpy(weight).view(torch.bfloat16).to(f"cuda:{ordinal}"), requires_grad=False)
                self.modules[name] = module
            for profile, (original, base) in ROPE_PROFILES.items():
                self.tables[profile] = self.scope["precompute_freqs_cis"](64, MAX_POSITION, original, base, 16, 32, 1)
                freqs = 1.0 / (base ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
                if original:
                    low = max(math.floor(64 * math.log(original / (32 * 2 * math.pi)) / (2 * math.log(base))), 0)
                    high = min(math.ceil(64 * math.log(original / (2 * math.pi)) / (2 * math.log(base))), 63)
                    ramp = ((torch.arange(32, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
                    smooth = 1 - ramp
                    freqs = freqs / 16 * (1 - smooth) + freqs * smooth
                self.inverses[profile] = freqs

    def tensor(self, values):
        return self.torch.from_numpy(values).view(self.torch.bfloat16).to(f"cuda:{self.ordinal}")

    def norm(self, values, name):
        torch = self.torch
        x = self.tensor(values)
        square = x.float().square()
        mean = square.mean(-1, keepdim=True)
        shifted = mean + NORM_EPS
        rstd = torch.rsqrt(shifted)
        normalized = x.float() * rstd
        weighted = self.modules[name].weight * normalized
        output = self.modules[name](x)
        if not torch.equal(output.view(torch.uint16), weighted.to(torch.bfloat16).view(torch.uint16)):
            raise RuntimeError("RMS diagnostic expressions diverged from the unchanged method")
        self.norm_controls += 1
        return dict(zip(NORM_STAGES, map(_numpy, (square, mean, shifted, rstd, normalized, weighted, output))))

    def frequencies(self, positions, profile):
        torch = self.torch
        pos = torch.from_numpy(positions).to(f"cuda:{self.ordinal}")
        inverse = self.inverses[profile]
        angles = torch.outer(pos, inverse)
        table = self.tables[profile][pos.to(torch.long)]
        return dict(zip(FREQUENCY_STAGES, map(_numpy, (inverse, angles, torch.view_as_real(table)))))

    def rope(self, values, start, profile, inverse):
        torch = self.torch
        x = self.tensor(values)
        freqs = self.tables[profile][start:start + values.shape[1]]
        complex_x = torch.view_as_complex(x[..., -64:].float().unflatten(-1, (-1, 2)))
        f = freqs.conj() if inverse else freqs
        shaped = f.view((1, values.shape[1], 32) if values.ndim == 3 else (1, values.shape[1], 1, 32))
        rotated = torch.view_as_real(complex_x * shaped).flatten(-2)
        self.scope["apply_rotary_emb"](x[..., -64:], freqs, inverse)
        if not torch.equal(x[..., -64:].view(torch.uint16), rotated.to(torch.bfloat16).view(torch.uint16)):
            raise RuntimeError("Rotary diagnostics diverged from the unchanged method")
        self.rope_controls += 1
        return {"frequency_fp32": _numpy(torch.view_as_real(freqs)), "rotated_fp32": _numpy(rotated), "output_bf16": _numpy(x)}

    def negative_controls(self, weights):
        torch = self.torch
        value = inputs((17, 5120), 2468)
        x = self.tensor(value)
        mean = x.float().square().mean(-1, keepdim=True)
        promoted = x.float().square().double().mean(-1, keepdim=True).float()
        mean_check = check_arrays(_numpy(mean), _numpy(promoted), "higher_precision_mean_replacement")
        tiny = self.tensor(inputs((2, 512), 4321, "near_eps"))
        exact = self.modules["layers.0.attn.kv_norm.weight"](tiny)
        wrong = (self.modules["layers.0.attn.kv_norm.weight"].weight * tiny.float()
                 * torch.rsqrt(tiny.float().square().mean(-1, keepdim=True) + 1e-6)).to(torch.bfloat16)
        eps_check = check_arrays(_numpy(exact), _numpy(wrong), "default_epsilon_replacement")
        profile, start = "window", 65535
        bits = inputs((2, 2, 64, 512), 65535)
        t = self.tensor(bits)[..., -64:].float().unflatten(-1, (-1, 2))
        f = torch.view_as_real(self.tables[profile][start:start + 2]).view(1, 2, 1, 32, 2)
        r, im, c, d = t[..., 0], t[..., 1], f[..., 0], f[..., 1]
        separated = torch.stack((r * c - im * d, r * d + im * c), -1).flatten(-2)
        expected = self.rope(bits, start, profile, False)["rotated_fp32"]
        fma_check = check_arrays(expected, _numpy(separated), "unfused_rotary_replacement")
        original, base = ROPE_PROFILES["compressed-yarn"]
        low = max(math.floor(64 * math.log(original / (32 * 2 * math.pi)) / (2 * math.log(base))), 0)
        high = min(math.ceil(64 * math.log(original / (2 * math.pi)) / (2 * math.log(base))), 63)
        # A double division then FP32 cast models a directly rounded FP32
        # quotient for these small integer operands, exposing the rejected path.
        wrong_ramp = ((torch.arange(32, dtype=torch.float64, device=f"cuda:{self.ordinal}") - low) / (high - low)).float().clamp(0, 1)
        smooth = 1 - wrong_ramp
        freq = 1.0 / (base ** (torch.arange(0, 64, 2, dtype=torch.float32, device=f"cuda:{self.ordinal}") / 64))
        wrong_freq = freq / 16 * (1 - smooth) + freq * smooth
        yarn_check = check_arrays(_numpy(self.inverses["compressed-yarn"]), _numpy(wrong_freq), "direct_YaRN_division_replacement")
        return [mean_check, eps_check, fma_check, yarn_check]


def compare_normalization(directory, prefix_fixture, attention_bundle, ordinal=0):
    missing = missing_dependencies()
    if missing:
        return {"probe": "native_normalization_rotary_parity", "status": "blocked", "missing_dependencies": missing}
    import torch
    weights, provenance = load_norm_weights(prefix_fixture)
    pytorch_source = fetch_pytorch_sources(Path(directory) / "pytorch-normalization")
    goldens, chains, phases = [], [], []
    with ExitStack() as stack, torch.inference_mode(), torch.cuda.device(ordinal), torch.device(f"cuda:{ordinal}"):
        budget = MemoryBudget(cap=128 * MiB)
        budget.acquire(128 * MiB)
        stack.callback(budget.release, 128 * MiB)
        reference = ReferenceMath(directory, weights, ordinal)
        native = stack.enter_context(NativeNormRoPE(ordinal))
        def collect(meta, expected, actual):
            if set(expected) != set(actual):
                raise ValueError("Normalization/rotary boundary set differs")
            goldens.append({"meta": meta, "stage_sha256": {name: array_digest(a) for name, a in expected.items()}})
            return [check_arrays(a, actual[name], name) for name, a in expected.items()]
        def component_checker(meta, values, actual):
            if meta["operation"] == "norm":
                expected = reference.norm(values, meta["weight"])
            elif meta["operation"] == "frequencies":
                expected = reference.frequencies(values, meta["profile"])
            else:
                expected = reference.rope(values, meta["start"], meta["profile"], meta["inverse"])
            return collect(meta, expected, actual)
        records, minimum = exercise_components(native, weights, component_checker)
        failures = exercise_failures(native, weights, component_checker)
        memory = stack.enter_context(AttentionMemoryReference(Path(directory), ordinal))
        qualification = memory.qualify()
        if qualification["status"] != "pass":
            raise RuntimeError("Attention integration oracle failed its qualification")
        attention = stack.enter_context(NativeAttention(attention_bundle, ordinal))
        for batch in (1, 2):
            window = WindowOracle(Path(directory), batch, ordinal)
            window.stub.kv_norm = reference.modules["layers.0.attn.kv_norm.weight"]
            window.window.__globals__["apply_rotary_emb"] = reference.scope["apply_rotary_emb"]
            def chain_checker(meta, projected, actual):
                position, sequence = meta["position"], meta["sequence"]
                if position != window.position:
                    raise ValueError("Native/reference logical positions diverged in the normalization chain")
                normalized = reference.norm(projected, "layers.0.attn.kv_norm.weight")
                rotated = reference.rope(normalized["output_bf16"], position, "window", False)
                tensor = reference.tensor(rotated["output_bf16"])
                codes, scales = window.kernel.act_quant(tensor, 32, "ue8m0", torch.float8_e8m0fnu)
                rounded = window.kernel.act_quant(tensor.clone(), 32, "ue8m0", torch.float8_e8m0fnu, True)
                freqs = reference.tables["window"][position:position + sequence]
                output_window, indices = window.window(window.stub, reference.tensor(projected), freqs, position)
                window.position += sequence
                cached = {"rounded_bf16": _numpy(rounded), "codes_fp8": codes.view(torch.uint8).cpu().numpy().copy(),
                          "scales_e8m0": scales.view(torch.uint8).cpu().numpy().copy(), "cache_bf16": _numpy(window.stub.window_kv_cache),
                          "indices_i32": _numpy(indices), "window_bf16": _numpy(output_window)}
                expected = {"norm_" + name: a for name, a in normalized.items()}
                expected.update(("rope_" + name, a) for name, a in rotated.items())
                expected.update(("kv_" + name, a) for name, a in cached.items())
                if meta["attention_checked"]:
                    q, sink = query_inputs(batch, sequence, meta["seed"])
                    query = reference.rope(q, position, "window", False)
                    args = (reference.tensor(query["output_bf16"]), reference.tensor(cached["window_bf16"]),
                            torch.from_numpy(sink).to(f"cuda:{ordinal}"), indices)
                    output, traces, _ = memory._run(64, 512, True, args)
                    attn = dict(zip(("output_bf16", *STAGES), map(_numpy, (output, *traces))))
                    inverse = reference.rope(attn["output_bf16"], position, "window", True)
                    for prefix, result in (("query_rope_", query), ("attention_", attn), ("inverse_rope_", inverse)):
                        expected.update((prefix + name, a) for name, a in result.items())
                return collect(meta, expected, actual)
            with NativeWindowKV(batch, ordinal) as kv:
                rows, chain_minimum = exercise_chain(native, kv, attention, weights, chain_checker, lambda: window.control("reset"))
            minimum = min(minimum, chain_minimum)
            chains.extend(rows)
            phases.append({"batch": batch, "KV_cleanup": kv.stats()})
        negatives = reference.negative_controls(weights)
        profile = {"compiler": native.compiler, "cubin_sha256": native.cubin_sha256,
                   "kernel_source_sha256": native.kernel_source_sha256, "device": native.device_info}
        attention_sha = attention.manifest_sha256
        source, methods = reference.source, reference.method_hashes
        controls = {"RMS_method_diagnostic_identity": reference.norm_controls, "rotary_method_diagnostic_identity": reference.rope_controls}
    passed = (all(r["status"] == "pass" for r in records + chains + failures)
              and all(n["status"] == "fail" and n["mismatches"] > 0 for n in negatives)
              and all(not p["KV_cleanup"]["device_owned_bytes"] for p in phases)
              and not native.stats()["device_owned_bytes"] and not native.stats()["host_reserved_bytes"]
              and not attention.stats()["device_owned_bytes"] and budget.reserved == 0)
    return {"schema_version": 1, "probe": "native_normalization_rotary_parity", "status": "pass" if passed else "fail",
            "model": MODEL, "revision": REVISION, "reference": source, "pytorch_source": pytorch_source,
            "reference_methods_ast_sha256": methods, "original_weight_fixture": provenance, "native_profile": profile,
            "attention_bundle_manifest_sha256": attention_sha, "component_cases": records, "chain_cases": chains, "faults": failures,
            "reference_goldens": goldens, "instrumentation_controls": controls, "negative_controls": negatives,
            "attention_reference_qualification": qualification,
            "memory": {"native_cleanup": native.stats(), "attention_cleanup": attention.stats(), "KV_phases": phases,
                       "oracle_host_reserved_after_close": budget.reserved, "minimum_sampled_host_available_bytes": minimum,
                       "combined_native_and_reference_host_allowance_bytes": 336 * MiB, "reservation_is_RSS_limit": False},
            "torch_version": torch.__version__, "torch_git_version": torch.version.git_version,
            "torch_cuda_version": torch.version.cuda, "weights_fetched_bytes": 0,
            "chain_input_boundary": "original learned KV norm, pre-RoPE projected KV/query inputs; actual projections not implemented",
            "compressed_rope_scope": "frequency/rotation arithmetic only; native compressed KV/indexing/sharing not implemented",
            "HyperConnections_and_full_block": "not_run", "full_model_quality": "not_run", "end_to_end_speed": "not_measured"}
