"""Optional pinned original compressor and compressed-cache oracle.

Index selection is explicitly excluded: its callback records the unrotated
latent and returns empty indices. Original compressor/cache arithmetic remains
unchanged; this oracle does not qualify a complete compressed Attention layer.
"""

import ast
import hashlib
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import urllib.request

import numpy as np

from .attention_storage_reference import _numpy
from .attention_validation import check_arrays
from .compression import LAYERS, MAX_POSITION
from .gate_reference import fetch_gate_reference
from .normalization_reference import ReferenceMath
from .prefix_reference import fetch_prefix_reference
from .reference_runtime import configure_reference_runtime, isolate_nvrtc_handles
from .router import ROUTER_TORCH_REVISION

SOFTMAX_SOURCES = {
    "SoftMax.cu": "17bf86e61f8c0f0bc2a37eed314bf412a1992826141bdcb6d9202c5066765b90",
    "PersistentSoftmax.cuh": "5abdcb3359480795f96d528d184dd98ecbe98d913061e63c764dcffe49906d4d",
}
COMPRESSION_METHODS = {
    "linear": "269a2858eac23cd6e35d246e37b3ec5b20b06161bce0675049811e5a284146c3",
    "Compressor.forward": "f64470bf198a1772424894240d26065a19be2d0e733217b8efa1b113d944aca9",
    "Attention._compress_kv": "0480c387247e2cdabfd3b20773eb89507b5733c83d5082ed70b65f10d5a50dd9",
}
QUANTIZATION_CUDA = {
    "False": "3fe084116f9081d14708406e188b4980ae0900976858015d307b087e17ce04d7",
    "True": "12d0efae567db5db22f69279526f1ed5b910d4ae8c4578209e6d1b1a025b75ff",
}


def fetch_softmax_sources(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, digest in SOFTMAX_SOURCES.items():
        url = f"https://raw.githubusercontent.com/pytorch/pytorch/{ROUTER_TORCH_REVISION}/aten/src/ATen/native/cuda/{name}"
        path = directory / name
        if path.exists():
            with path.open("rb") as stream:
                raw = stream.read(262145)
        else:
            with urllib.request.urlopen(url, timeout=30) as response:
                raw = response.read(262145)
        if len(raw) > 262144 or hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("Unregistered pinned softmax source")
        if not path.exists():
            path.write_bytes(raw)
        records[name] = {"url": url, "sha256": digest, "bytes": len(raw)}
    return {"revision": ROUTER_TORCH_REVISION, "files": records}


class CompressionOracle:
    def __init__(self, directory, arrays, ordinal=0):
        import torch
        import tilelang
        if (torch.version.git_version != ROUTER_TORCH_REVISION
                or str(torch.backends.cuda.preferred_blas_library()) != "_BlasBackend.Cublas"
                or torch.get_float32_matmul_precision() != "highest" or torch.cuda.tunable.is_enabled()
                or not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
                or os.environ.get("CUBLAS_WORKSPACE_CONFIG") is not None):
            raise ValueError("Unregistered compressor reference build or BLAS configuration")
        self.torch, self.ordinal = torch, ordinal
        self.source = fetch_prefix_reference(Path(directory))
        self.blas_source = fetch_gate_reference(Path(directory) / "pytorch-gate")
        self.softmax_source = fetch_softmax_sources(Path(directory) / "pytorch-compression")
        spec = importlib.util.spec_from_file_location("vesper_original_compression_kernel", Path(directory) / "kernel.py")
        self.kernel = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.kernel)
        self.workarounds = configure_reference_runtime(self.kernel, tilelang.__version__)
        self.kernel.fp4_quant_kernel = isolate_nvrtc_handles(self.kernel.fp4_quant_kernel)
        self.workarounds.append("TileLang 0.1.12 NVRTC: isolate FP4 launch handles per specialization; arithmetic unchanged")
        norms = {n: a for n, a in arrays.items() if ".norm.weight" in n}
        self.math = ReferenceMath(directory, norms, ordinal)
        tree = ast.parse((Path(directory) / "model.py").read_bytes())
        self.scope = {"torch": torch, "F": torch.nn.functional, "fp4_act_quant": self.kernel.fp4_act_quant,
                      "apply_rotary_emb": self.math.scope["apply_rotary_emb"]}
        self.functions, self.methods = {}, {}
        for cls, name in ((None, "linear"), ("Compressor", "forward"), ("Attention", "_compress_kv")):
            parent = tree if cls is None else next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
            node = next(n for n in parent.body if isinstance(n, ast.FunctionDef) and n.name == name)
            label = (cls + "." if cls else "") + name
            self.methods[label] = hashlib.sha256(ast.dump(node).encode()).hexdigest()
            local = dict(self.scope)
            exec(compile(ast.Module(body=[node], type_ignores=[]), "<pinned-compression-method>", "exec"), local)
            self.functions[label] = local[name]
        if self.methods != COMPRESSION_METHODS:
            raise ValueError("Unregistered original compressor method AST")
        self.weights = {}
        for layer in LAYERS:
            for part in (("wkv", "wgate") if layer == 2 else ("wkv",)):
                w = self.tensor(arrays[f"layers.{layer}.attn.compressor.{part}.weight"])
                self.weights[layer, part] = w.float() if layer == 2 else w
        self.controls, self.quantization_controls = [], []

    def tensor(self, values):
        t = self.torch.from_numpy(values).to(f"cuda:{self.ordinal}")
        return t.view(self.torch.bfloat16) if values.dtype == np.uint16 else t

    def project(self, values, layer):
        x = self.tensor(values)
        if layer == 2:
            x = x.float()
        return {part + ("_fp32" if layer == 2 else "_bf16"): _numpy(self.functions["linear"](x, self.weights[layer, part]))
                for part in (("wkv", "wgate") if layer == 2 else ("wkv",))}

    def pool(self, kv, scores):
        torch = self.torch
        k, s = self.tensor(kv), self.tensor(scores)
        maximum = s.amax(dim=2)
        exp = (s - maximum.unsqueeze(2)).exp()
        total = exp.sum(dim=2)
        weights = s.softmax(dim=2)
        products = k * weights
        pooled = products.sum(dim=2)
        return dict(zip(("maximum_fp32", "exponents_fp32", "sum_exp_fp32", "weights_fp32", "products_fp32", "pooled_fp32", "pooled_bf16"),
                        map(_numpy, (maximum, exp, total, weights, products, pooled, pooled.bfloat16()))))

    def round_kv(self, values):
        torch = self.torch
        x = self.tensor(values)
        packed, scales = self.kernel.fp4_act_quant(x, 16, False, torch.float8_e4m3fn)
        rounded = self.kernel.fp4_act_quant(x.clone(), 16, True, torch.float8_e4m3fn)
        # Original packed bytes/scales must also reconstruct the in-place result.
        codes = packed.view(torch.uint8)
        lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.], device=x.device, dtype=torch.float32)
        unpacked = torch.stack((codes & 15, codes >> 4), dim=-1).flatten(-2).long()
        rebuilt = (lut[unpacked] * scales.float().repeat_interleave(16, dim=-1)).bfloat16()
        control = check_arrays(_numpy(rounded), _numpy(rebuilt), "unchanged_FP4_packed_vs_inplace")
        if control["status"] != "pass":
            raise RuntimeError("Original packed FP4 and in-place quantization differ")
        self.quantization_controls.append(control)
        return {"packed_fp4": _numpy(codes), "scales_e4m3": _numpy(scales.view(torch.uint8)), "rounded_bf16": _numpy(rounded)}

    def rope(self, values, positions):
        torch = self.torch
        x = self.tensor(values)
        freqs = self.math.tables["compressed-yarn"][torch.from_numpy(positions).to(x.device).long()]
        cx = torch.view_as_complex(x[..., -64:].float().unflatten(-1, (-1, 2)))
        rotated = torch.view_as_real(cx * freqs.view(1, len(positions), 32)).flatten(-2)
        self.math.scope["apply_rotary_emb"](x[..., -64:], freqs)
        return {"frequency_fp32": _numpy(torch.view_as_real(freqs)), "rotated_fp32": _numpy(rotated), "output_bf16": _numpy(x)}

    def reset(self, layer=2, batch=1):
        torch = self.torch
        self.layer, self.batch, self.ratio = layer, batch, LAYERS[layer]
        self.original = SimpleNamespace(compress_ratio=self.ratio, head_dim=512,
            norm=self.math.modules[f"layers.{layer}.attn.compressor.norm.weight"],
            wkv=lambda x: self.functions["linear"](x, self.weights[layer, "wkv"]))
        if layer == 2:
            self.original.wgate = lambda x: self.functions["linear"](x, self.weights[layer, "wgate"])
            self.original.kv_state = torch.zeros((batch, 2, 512), dtype=torch.float32, device=f"cuda:{self.ordinal}")
            self.original.score_state = torch.full_like(self.original.kv_state, -torch.inf)
        self.cache = torch.zeros((batch, MAX_POSITION // self.ratio, 512), dtype=torch.bfloat16, device=f"cuda:{self.ordinal}")
        self.kv_state = np.zeros((batch, 2 if self.ratio == 2 else 0, 512), np.float32)
        self.score_state = np.full_like(self.kv_state, -np.inf)
        self.original_cache_method = self.functions["Attention._compress_kv"]
        self.original_cache_method.__globals__["shared_attn"] = SimpleNamespace(compress_kv=None)

    def step(self, values, position):
        torch = self.torch
        sequence = values.shape[1]
        stages = self.project(values, self.layer)
        if self.ratio == 2:
            kv, scores = stages["wkv_fp32"], stages["wgate_fp32"]
            if position == 0:
                cutoff = sequence - sequence % 2
                if sequence % 2:
                    self.kv_state[:, 0], self.score_state[:, 0] = kv[:, -1], scores[:, -1]
                pairs = [np.ascontiguousarray(a[:, :cutoff]).reshape(self.batch, cutoff // 2, 2, 512) for a in (kv, scores)]
            else:
                self.kv_state[:, position % 2], self.score_state[:, position % 2] = kv[:, 0], scores[:, 0]
                pairs = [np.ascontiguousarray(a[:, None]) for a in (self.kv_state, self.score_state)]
            emits = sequence >= 2 if position == 0 else (position + 1) % 2 == 0
            if emits:
                stages.update(("pool_" + n, a) for n, a in self.pool(*pairs).items())
                pooled = stages["pool_pooled_bf16"]
            else:
                pooled = None
        else:
            pooled = stages["wkv_bf16"]
        if pooled is not None:
            norm = self.math.norm(pooled, f"layers.{self.layer}.attn.compressor.norm.weight")
            stages.update(("norm_" + n, a) for n, a in norm.items())
            latent = norm["output_bf16"]
            positions = np.arange(latent.shape[1], dtype=np.int32) * self.ratio if position == 0 else np.array([position + 1 - self.ratio], np.int32)
            stages["latent_pre_rope_bf16"], stages["latent_positions_i32"] = latent.copy(), positions
            rope = self.rope(latent, positions)
            stages.update(("rope_" + n, a) for n, a in rope.items())
            stages.update(("quant_" + n, a) for n, a in self.round_kv(rope["output_bf16"]).items())
        else:
            stages["latent_pre_rope_bf16"] = np.empty((self.batch, 0, 512), np.uint16)
            stages["latent_positions_i32"] = np.empty((0,), np.int32)
        observed = []
        def index_boundary(x, qr, latent, start, offset, count):
            observed.append(None if latent is None else _numpy(latent).copy())
            return torch.empty((self.batch, sequence, 0), dtype=torch.int32, device=x.device)
        stub = SimpleNamespace(compress_ratio=self.ratio, is_kv_source=True, compress_kv_cache=self.cache,
            compressor=lambda x, start: self.functions["Compressor.forward"](self.original, x, start),
            freqs_cis=self.math.tables["compressed-yarn"], rope_head_dim=64, _compress_topk_idxs=index_boundary)
        visible, indices = self.original_cache_method(stub, self.tensor(values), None, position, 0)
        if len(observed) != 1 or (observed[0] is None) != (pooled is None):
            raise RuntimeError("Original compressor emission/Indexer boundary differs")
        controls = []
        if pooled is not None:
            controls.append(check_arrays(stages["latent_pre_rope_bf16"], observed[0], "unchanged_Compressor_forward_latent"))
            start = position // self.ratio
            controls.append(check_arrays(stages["quant_rounded_bf16"], _numpy(self.cache[:, start:start + pooled.shape[1]]), "unchanged_Attention_compress_KV"))
        if self.ratio == 2:
            controls.extend((check_arrays(self.kv_state, _numpy(self.original.kv_state), "unchanged_Compressor_KV_state"),
                             check_arrays(self.score_state, _numpy(self.original.score_state), "unchanged_Compressor_score_state")))
        if any(c["status"] != "pass" for c in controls):
            raise RuntimeError("Diagnostic compression differs from unchanged original methods")
        self.controls.append({"emits": pooled is not None, "empty_index_callback": True, "checks": controls})
        stages.update(kv_state_fp32=self.kv_state.copy(), score_state_fp32=self.score_state.copy(),
                      cache_bf16=_numpy(self.cache), visible_bf16=_numpy(visible))
        return stages


def negative_controls(reference):
    from .normalization_validation import inputs
    torch = reference.torch
    x = inputs((1, 3, 5120), 34001)
    exact = reference.project(x, 2)["wkv_fp32"]
    early = torch.nn.functional.linear(reference.tensor(x), reference.weights[2, "wkv"].bfloat16()).float()
    projection = check_arrays(exact, _numpy(early), "replace_ratio2_FP32_projection_with_BF16")
    rng = np.random.default_rng(34002)
    kv, score = [rng.standard_normal((1, 3, 2, 512)).astype(np.float32) for _ in range(2)]
    pooled = reference.pool(kv, score)
    rounded = (reference.tensor(pooled["products_fp32"]).bfloat16().float().sum(dim=2)).bfloat16()
    pool = check_arrays(pooled["pooled_bf16"], _numpy(rounded), "round_pool_products_to_BF16_early")
    values = inputs((1, 3, 512), 34003)
    original = reference.kernel.fp4_act_quant(reference.tensor(values), 16, True, torch.float8_e4m3fn)
    group32 = reference.kernel.fp4_act_quant(reference.tensor(values), 32, True, torch.float8_e4m3fn)
    e8m0 = reference.kernel.fp4_act_quant(reference.tensor(values), 16, True, torch.float8_e8m0fnu)
    grouping = check_arrays(_numpy(original), _numpy(group32), "replace_compressed_KV_group16_with_group32")
    scales = check_arrays(_numpy(original), _numpy(e8m0), "replace_compressed_KV_E4M3_scales_with_E8M0")
    first = reference.rope(values, np.array([0, 2, 4], np.int32))["output_bf16"]
    last = reference.rope(values, np.array([1, 3, 5], np.int32))["output_bf16"]
    positions = check_arrays(first, last, "rotate_compressed_group_at_last_token")
    return [projection, pool, grouping, scales, positions]


def compare_compression(directory, fixture_path, ordinal=0):
    from .attention_memory_reference import missing_dependencies
    from .checkpoint import MODEL, REVISION
    from .compression import NativeCompression
    from .compression_fixture import load_compression_fixture
    from .compression_validation import exercise_components, exercise_states, exercise_failures, exercise_snapshots
    from .normalization_reference import fetch_pytorch_sources
    from .resources import MemoryBudget, MiB
    missing = missing_dependencies()
    if missing:
        return {"probe": "native_compression_parity", "status": "blocked", "missing_dependencies": missing}
    import torch
    import tilelang
    budget = MemoryBudget(cap=256 * MiB)
    budget.acquire(256 * MiB)
    try:
        arrays, fixture = load_compression_fixture(fixture_path)
        goldens = []
        with torch.inference_mode(), torch.cuda.device(ordinal), torch.device(f"cuda:{ordinal}"):
            reference = CompressionOracle(directory, arrays, ordinal)
            def collect(meta, expected, actual):
                if set(expected) != set(actual):
                    raise ValueError("Compressed-KV diagnostic boundary set differs")
                goldens.append({"meta": meta, "stage_sha256": {n: hashlib.sha256(a.tobytes()).hexdigest() for n, a in expected.items()}})
                return [check_arrays(a, actual[n], n) for n, a in expected.items()]
            def component_checker(meta, args, actual):
                expected = reference.round_kv(*args) if meta["operation"] == "quant" else reference.pool(*args)
                return collect(meta, expected, actual)
            def state_checker(meta, value, actual):
                return collect(meta, reference.step(value, meta["position"]), actual)
            with NativeCompression(arrays, ordinal) as native:
                components, minimum = exercise_components(native, component_checker)
                states, low = exercise_states(native, state_checker, reference.reset)
                retries, faults = exercise_failures(native, state_checker, reference.reset)
                snapshots, events = exercise_snapshots(native, state_checker, reference.reset, lambda prefix: reference.step(prefix, 0))
                states.extend(retries + snapshots)
                minimum = min(minimum, low)
                controls, quant_controls = list(reference.controls), list(reference.quantization_controls)
                negatives = negative_controls(reference)
                profile = {"compiler": native.compiler, "cubin_sha256": native.cubin_sha256,
                           "device": native.info, "cublas_version": native.blas.version, "cublas_math_mode": native.blas.math_mode}
            original_quant = {}
            for inplace in (False, True):
                kernel = reference.kernel.fp4_quant_kernel(512, 16, scale_dtype="float8_e4m3fn", inplace=inplace)
                original_quant[str(inplace)] = hashlib.sha256(kernel.get_kernel_source().encode()).hexdigest()
            if original_quant != QUANTIZATION_CUDA:
                raise ValueError("Unregistered original FP4 quantization generated CUDA")
        passed = (all(r["status"] == "pass" for r in components + states + faults + events)
                  and all(c["mismatches"] > 0 for c in negatives)
                  and not native.stats()["device_owned_bytes"] and not native.norm.stats()["device_owned_bytes"])
        return {"schema_version": 1, "probe": "native_compression_parity", "status": "pass" if passed else "fail",
            "model": MODEL, "revision": REVISION, "reference": reference.source,
            "blas_reference": reference.blas_source, "reduction_reference": fetch_pytorch_sources(Path(directory) / "pytorch-normalization"),
            "softmax_reference": reference.softmax_source, "fixture": fixture, "compression_profile": profile,
            "normalization_profile": {"compiler": native.norm.compiler, "cubin_sha256": native.norm.cubin_sha256},
            "original_methods_ast_sha256": reference.methods, "original_quant_cuda_sha256": original_quant,
            "component_cases": components, "state_cases": states, "faults": faults, "snapshot_events": events,
            "reference_goldens": goldens, "unchanged_original_controls": controls,
            "original_packed_vs_inplace_controls": quant_controls, "negative_controls": negatives,
            "reference_runtime_workarounds": reference.workarounds,
            "memory": {"compression_cleanup": native.stats(), "normalization_cleanup": native.norm.stats(),
                "combined_native_host_allowance_bytes": 240 * MiB, "combined_native_and_reference_host_allowance_bytes": 496 * MiB,
                "all_state_reservations_released": True, "minimum_sampled_host_available_bytes": minimum,
                "reservation_is_RSS_limit": False, "publication_cache": "bounded_host_diagnostic_copy"},
            "weights_fetched_bytes": 0, "fixture_original_weight_bytes": fixture["weights_fetched_bytes"],
            "torch_version": torch.__version__, "torch_git_version": torch.version.git_version, "torch_cuda_version": torch.version.cuda,
            "tilelang_version": tilelang.__version__,
            "scope": "Original layer-2 ratio-2 and layer-20 ratio-1 compressor/quantized cache, generated inputs, batches one/two, one rank",
            "index_selection": "empty callback records pre-RoPE latent only; not qualified",
            "source_layer_sharing": "not_implemented", "complete_compressed_attention": "not_implemented",
            "full_model_quality": "not_run", "end_to_end_speed": "not_measured"}
    finally:
        budget.release(256 * MiB)
