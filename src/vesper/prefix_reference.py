"""Capture real layer-0 MoE inputs with unchanged official prefix methods.

This optional oracle is not a native attention/Hyper-Connections implementation.
"""

from __future__ import annotations

import ast
import base64
import ctypes as C
from functools import lru_cache
import hashlib
import importlib.util
import math
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np

from .checkpoint import MODEL, REVISION, fetch_reference, read_small
from .conformance import compare
from .cuda import Driver, _bind
from .formats import bf16_bits
from .prefix import PREFIX_IDS, PREFIX_SEQUENCES, load_prefix_fixture
from .reference_runtime import configure_reference_runtime
from .resources import GiB, MemoryBudget, MiB, physical_memory

PREFIX_SOURCE_HASHES = {"convert.py": "035028340479145594a81d6084a8424e57363adf83c0d5983914783d95614d76",
                        "generate.py": "8668d67f7d108e32b90d50cb0d8606889ceb2219bfe95741d84e22f70768e9f0",
                        "config.json": "2e84f45cf1dac8c7fcbb200e96667d4b913275690668ed496f24c7747207a809"}


def fetch_prefix_reference(directory):
    source = fetch_reference(directory, include_model=True)
    for name, expected in PREFIX_SOURCE_HASHES.items():
        path = directory / name
        raw = path.read_bytes() if path.exists() else read_small(f"inference/{name}")
        if len(raw) > MiB or hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("Prefix preparation source differs from its pinned hash")
        if not path.exists():
            path.write_bytes(raw)
        source["files"][name] = expected
    return source


def prefix_preflight(directory: Path, ordinal=0):
    """Compile the unchanged attention specialization; inspect resources, no launch."""
    missing = [n for n in ("torch", "tilelang") if importlib.util.find_spec(n) is None]
    if missing:
        return {"probe": "official_prefix_resource_preflight", "status": "blocked", "missing_dependencies": missing}
    import torch
    import tilelang
    directory = Path(directory)
    source = fetch_prefix_reference(directory)
    spec = importlib.util.spec_from_file_location("vesper_prefix_resource_probe", directory / "kernel.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with torch.cuda.device(ordinal):
        compiled = module.sparse_attn_kernel(64, 512, 512**-0.5)
    adapter = compiled.adapter
    host_source = adapter.host_func
    if not isinstance(host_source, str):
        host_source = Path(adapter.pymodule.__file__).read_text(encoding="utf-8")
    required = required_shared_bytes(host_source)
    driver = Driver()
    query = _bind(driver.lib, "cuDeviceGetAttribute", [C.POINTER(C.c_int), C.c_int, C.c_int])
    limit = C.c_int()
    driver.check(query(C.byref(limit), 97, ordinal))  # CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
    return {"probe": "official_prefix_resource_preflight", "status": "blocked" if required > limit.value else "pass",
            "model": MODEL, "revision": REVISION, "reference": source,
            "device": next(d for d in driver.inventory()["devices"] if d["ordinal"] == ordinal),
            "attention_heads": 64, "head_dim": 512, "required_dynamic_shared_bytes": required,
            "device_max_optin_shared_bytes": limit.value,
            "compiled_host_wrapper_sha256": hashlib.sha256(host_source.encode()).hexdigest(),
            "tilelang_version": tilelang.__version__, "torch_version": torch.__version__, "kernel_launches": 0,
            "source_or_arithmetic_modified": False,
            "scope": "Original layer-0 sparse attention resource check; no input capture or native attention qualification"}


def required_shared_bytes(host_source):
    tree = ast.parse(host_source)
    sizes = [arg.value for node in ast.walk(tree) if isinstance(node, ast.Call)
             and ((isinstance(node.func, ast.Attribute) and node.func.attr in ("cuKernelSetAttribute", "cuFuncSetAttribute"))
                  or (isinstance(node.func, ast.Name) and node.func.id in ("cuKernelSetAttribute", "cuFuncSetAttribute")))
             for arg in node.args if isinstance(arg, ast.Constant) and type(arg.value) is int and arg.value > 1024]
    if not sizes:
        raise RuntimeError("Unable to inspect the original kernel's dynamic shared-memory requirement")
    return max(sizes)


def capture_prefix(fixture: Path, directory: Path, ordinal=0):
    preflight = prefix_preflight(directory, ordinal)
    if preflight["status"] != "pass":
        return {"probe": "original_prefix_moe_inputs", "status": "blocked", "model": MODEL, "revision": REVISION,
                "preflight": preflight, "cases": [], "reason": "Missing optional reference dependencies" if preflight.get("missing_dependencies")
                else "Unchanged official attention exceeds device shared-memory capacity"}
    import torch
    import torch.nn.functional as F
    import tilelang

    budget = MemoryBudget(cap=512 * MiB)
    budget.acquire(512 * MiB)
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        directory = Path(directory)
        source = fetch_prefix_reference(directory)
        arrays, manifest = load_prefix_fixture(fixture)
        spec = importlib.util.spec_from_file_location("vesper_prefix_official_kernel", directory / "kernel.py")
        kernel = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(kernel)
        workarounds = configure_reference_runtime(kernel, tilelang.__version__)
        scope = {"torch": torch, "F": F, "math": math, "lru_cache": lru_cache,
                 "world_size": 1, "fp8_block_size": 32, "scale_fmt": "ue8m0", "scale_dtype": torch.float8_e8m0fnu,
                 "act_quant": kernel.act_quant, "fp8_gemm": kernel.fp8_gemm, "fp4_gemm": kernel.fp4_gemm,
                 "sparse_attn": kernel.sparse_attn, "hc_split_sinkhorn": kernel.hc_split_sinkhorn}
        tree = ast.parse((directory / "model.py").read_text(encoding="utf-8"))
        names = ("linear", "precompute_freqs_cis", "apply_rotary_emb", "get_window_topk_idxs", "make_identity_pre_mix")
        for name in names:
            node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            exec(compile(ast.Module(body=[node], type_ignores=[]), "pinned_prefix_model.py", "exec"), scope)
        def method(class_name, name):
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
            node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
            local = dict(scope)
            exec(compile(ast.Module(body=[node], type_ignores=[]), "pinned_prefix_model.py", "exec"), local)
            return local[name]
        records, repeated, minimum = [], [], physical_memory()["available"]
        with torch.cuda.device(ordinal), torch.device(f"cuda:{ordinal}"), torch.inference_mode():
            if torch.cuda.mem_get_info(ordinal)[0] < 2 * GiB:
                raise MemoryError("Insufficient prefix oracle VRAM reserve")
            tensors = {}
            for name, array in arrays.items():
                tensor = torch.from_numpy(array.copy()).to(f"cuda:{ordinal}")
                if name == "embedding_rows" or name.endswith("norm.weight"):
                    tensor = tensor.view(torch.bfloat16)
                elif array.dtype == np.uint8:
                    tensor = tensor.view(torch.float8_e8m0fnu if name.endswith(".scale") else torch.float8_e4m3fn)
                tensors[name] = tensor
            for name, tensor in tensors.items():
                if name.endswith(".weight") and name.replace(".weight", ".scale") in tensors:
                    tensor.scale = tensors[name.replace(".weight", ".scale")]
            # Execute the unchanged wo_a conversion statements from convert.py.
            conversion = ast.parse((directory / "convert.py").read_text(encoding="utf-8"))
            node = next(n for n in ast.walk(conversion) if isinstance(n, ast.If)
                        and isinstance(n.test, ast.Call) and ast.unparse(n.test) == "name.endswith('wo_a.weight')")
            state = {name: tensor for name, tensor in tensors.items() if name.startswith("layers.0.attn.wo_a.")}
            local = {"torch": torch, "state_dicts": [state], "i": 0, "name": "layers.0.attn.wo_a.weight"}
            exec(compile(ast.Module(body=node.body, type_ignores=[]), "pinned_wo_a_conversion.py", "exec"), local)
            tensors["layers.0.attn.wo_a.weight"] = state["layers.0.attn.wo_a.weight"]
            conversion_hash = hashlib.sha256(tensors["layers.0.attn.wo_a.weight"].view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
            def projection(name, row_parallel=False):
                obj = SimpleNamespace(weight=tensors[f"layers.0.attn.{name}.weight"], bias=None)
                fn = method("RowParallelLinear" if row_parallel else "Linear", "forward")
                return lambda x: fn(obj, x)
            def norm(key):
                obj = SimpleNamespace(weight=tensors[f"layers.0.{key}.weight"], eps=1e-20)
                fn = method("RMSNorm", "forward")
                return lambda x: fn(obj, x)
            attn = SimpleNamespace(n_local_heads=64, n_local_groups=8, head_dim=512, rope_head_dim=64,
                                   o_lora_rank=1024, window_size=128, compress_ratio=0, softmax_scale=512**-0.5,
                                   attn_sink=tensors["layers.0.attn.attn_sink"],
                                   wq_a=projection("wq_a"), wq_b=projection("wq_b"), wkv=projection("wkv"),
                                   wo_a=SimpleNamespace(weight=tensors["layers.0.attn.wo_a.weight"]), wo_b=projection("wo_b", True),
                                   q_norm=norm("attn.q_norm"), kv_norm=norm("attn.kv_norm"),
                                   freqs_cis=scope["precompute_freqs_cis"](64, 2, 0, 10000, 16, 32, 1),
                                   window_kv_cache=torch.zeros(1, 128, 512))
            attn._window_kv = MethodType(method("Attention", "_window_kv"), attn)
            attention_forward = method("Attention", "forward")
            class Captured(Exception):
                pass
            captured = []
            def stop_at_moe(x, mask):
                captured.append(x.clone())
                raise Captured()
            block = SimpleNamespace(norm_eps=1e-20, hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6,
                                    attn=lambda x, start: attention_forward(attn, x, start),
                                    attn_norm=norm("attn_norm"), ffn_norm=norm("ffn_norm"), ffn=stop_at_moe)
            for name in ("hc_attn_fn", "hc_attn_scale", "hc_attn_base", "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base"):
                setattr(block, name, tensors[f"layers.0.{name}"])
            for name in ("hc_mixes", "hc_pre", "hc_post"):
                setattr(block, name, MethodType(method("Block", name), block))
            block_forward = method("Block", "forward")
            for repeat in range(2):
                for sequence_index, sequence in enumerate(PREFIX_SEQUENCES):
                    attn.window_kv_cache.zero_()
                    for position, token in enumerate(sequence):
                        available = physical_memory()["available"]
                        minimum = min(minimum, available)
                        budget.acquire(0, available)
                        h = tensors["embedding_rows"][PREFIX_IDS.index(token)].view(1, 1, 5120).unsqueeze(2).repeat(1, 1, 4, 1)
                        pre_mix = scope["make_identity_pre_mix"](h, 4)
                        captured.clear()
                        try:
                            block_forward(block, h, position, pre_mix, None)
                        except Captured:
                            pass
                        if len(captured) != 1:
                            raise RuntimeError("Original Block.forward did not reach exactly one MoE boundary")
                        bits = captured[0].view(torch.uint16).reshape(5120).cpu().numpy().copy()
                        if repeat:
                            expected = np.frombuffer(base64.b64decode(records[len(repeated)]["activation_bf16_base64"]), np.uint16)
                            repeated.append(compare(expected, bits, "repeat_original_prefix_input"))
                        else:
                            raw = bits.tobytes()
                            records.append({"input": f"sequence{sequence_index}_position{position}_token{token}",
                                            "sequence_token_ids": list(sequence), "position": position, "token_id": token,
                                            "phase": "single_token_prefill" if position == 0 else "single_token_decode",
                                            "activation_bf16_sha256": hashlib.sha256(raw).hexdigest(),
                                            "activation_bf16_base64": base64.b64encode(raw).decode("ascii")})
        return {"probe": "original_prefix_moe_inputs", "status": "pass" if records and all(c["status"] == "pass" for c in repeated) else "fail",
                "model": MODEL, "revision": REVISION, "layer": 0, "fixture": manifest, "reference": source,
                "cases": records, "repeat_checks": repeated, "wo_a_original_conversion_bf16_sha256": conversion_hash,
                "reference_runtime_workarounds": workarounds, "torch_version": torch.__version__, "torch_cuda_version": torch.version.cuda,
                "torch_git_version": torch.version.git_version, "tilelang_version": tilelang.__version__, "numpy_version": np.__version__,
                "minimum_observed_host_available_bytes": minimum, "host_reservation_bytes": budget.peak,
                "scope": "Unchanged layer-0 prefix methods; original embedding rows and weights; stop before MoE; no native attention qualification",
                "tokenizer_quality": "not_run", "full_model_quality": "not_run"}
    finally:
        torch.set_default_dtype(previous)
        budget.release(512 * MiB)
