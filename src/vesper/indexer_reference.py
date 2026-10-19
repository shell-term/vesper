"""Optional original indexer operators and unchanged forward/selection controls."""

import ast
import copy
import hashlib
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import urllib.request

import numpy as np

from .attention_storage_reference import _numpy
from .attention_validation import check_arrays
from .gate_reference import fetch_gate_reference
from .normalization_reference import ReferenceMath, fetch_pytorch_sources
from .prefix_reference import fetch_prefix_reference
from .reference_runtime import configure_reference_runtime, isolate_nvrtc_handles
from .router import ROUTER_TORCH_REVISION
from .router_reference import fetch_topk_reference

INDEX_SOURCE_HASHES = {
    "ReduceSumProdKernel.cu": "6c913046db78b3577ae9d44f5a657b21654ae1f141fbe1cd51fffaa39b10b6de",
    "ActivationThresholdKernel.cu": "06e6c5f9bb13852ec55be7833165670fcf98fe67a38b29a537e19724188c5b8c",
}
INDEX_METHODS = {
    "linear": "269a2858eac23cd6e35d246e37b3ec5b20b06161bce0675049811e5a284146c3",
    "select_candidate_blocks": "766de6c8ea9cd09638410fe1200f75adaf86869a961e24f4e9d96e6533bfd099",
    "Indexer.forward": "87f5d3af5e696b381697c42b69b94c3cb5443c68c5c9c66b3e650155a7798c7a",
    "Attention._compress_topk_idxs": "20b150bbc385158746f9cc40716000fd7a301f01d2f9e4b5276889cab0f06be4",
}
INDEX_QUANT_CUDA = {"False": "5cdc72a828e8e8503501034192dfb8ad37ad86fb5a80e21f2f870666ec7743ff",
                    "True": "9a4629e9336d8534d82cf758f039ab41dab770237c5a637562b7c36afeb493ee"}
INDEX_QUERY_TRACE_CUDA = "cea81befa8f205dd78e0dced7aaef0945c186329008d545eda5a10338914c0b4"


def fetch_index_sources(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, digest in INDEX_SOURCE_HASHES.items():
        path = directory / name
        url = f"https://raw.githubusercontent.com/pytorch/pytorch/{ROUTER_TORCH_REVISION}/aten/src/ATen/native/cuda/{name}"
        if path.exists():
            with path.open("rb") as stream: raw = stream.read(262145)
        else:
            with urllib.request.urlopen(url, timeout=30) as response: raw = response.read(262145)
        if len(raw) > 262144 or hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("Unregistered pinned indexer reduction/activation source")
        if not path.exists(): path.write_bytes(raw)
        records[name] = {"url": url, "bytes": len(raw), "sha256": digest}
    return {"revision": ROUTER_TORCH_REVISION, "files": records}


class IndexerOracle:
    def __init__(self, directory, arrays, ordinal=0, *, layers=(2, 20, 24)):
        from .attention_topology import indexer_layout
        self.layout, self.layers = indexer_layout(layers), layers
        import torch
        import tilelang
        if (torch.version.git_version != ROUTER_TORCH_REVISION or torch.get_default_dtype() != torch.bfloat16
                or str(torch.backends.cuda.preferred_blas_library()) != "_BlasBackend.Cublas"
                or torch.get_float32_matmul_precision() != "highest" or torch.cuda.tunable.is_enabled()
                or not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
                or os.environ.get("CUBLAS_WORKSPACE_CONFIG") is not None):
            raise ValueError("Unregistered original indexer reference profile")
        self.torch, self.ordinal = torch, ordinal
        self.source = fetch_prefix_reference(Path(directory))
        self.blas_source = fetch_gate_reference(Path(directory) / "pytorch-gate")
        self.topk_source = fetch_topk_reference(Path(directory) / "pytorch-topk")
        self.reduction_source = fetch_pytorch_sources(Path(directory) / "pytorch-normalization")
        self.index_source = fetch_index_sources(Path(directory) / "pytorch-indexer")
        spec = importlib.util.spec_from_file_location("vesper_original_indexer_kernel", Path(directory) / "kernel.py")
        self.kernel = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.kernel)
        self.workarounds = configure_reference_runtime(self.kernel, tilelang.__version__)
        self.kernel.fp4_quant_kernel = isolate_nvrtc_handles(self.kernel.fp4_quant_kernel)
        self.workarounds.append("TileLang 0.1.12 NVRTC: isolate FP4 specialization launch handles; arithmetic unchanged")
        self.math = ReferenceMath(directory, {n: a for n, a in arrays.items() if n.endswith("k_norm.weight")}, ordinal)
        tree = ast.parse((Path(directory) / "model.py").read_bytes())
        scope = {"torch": torch, "F": torch.nn.functional, "world_size": 1,
                 "fp8_block_size": 32, "fp4_block_size": 32, "scale_fmt": "ue8m0", "scale_dtype": torch.float8_e8m0fnu,
                 "act_quant": self.kernel.act_quant, "fp4_act_quant": self.kernel.fp4_act_quant,
                 "fp8_gemm": self.kernel.fp8_gemm, "fp4_gemm": self.kernel.fp4_gemm,
                 "apply_rotary_emb": self.math.scope["apply_rotary_emb"], "shared_attn": SimpleNamespace()}
        self.functions, self.methods = {}, {}
        for cls, name in ((None, "linear"), (None, "select_candidate_blocks"), ("Indexer", "forward"), ("Attention", "_compress_topk_idxs")):
            parent = tree if cls is None else next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
            node = next(n for n in parent.body if isinstance(n, ast.FunctionDef) and n.name == name)
            label = (cls + "." if cls else "") + name
            self.methods[label] = hashlib.sha256(ast.dump(node).encode()).hexdigest()
            exec(compile(ast.Module(body=[node], type_ignores=[]), "<pinned-indexer-method>", "exec"), scope)
            self.functions[label] = scope[name]
        if self.methods != INDEX_METHODS: raise ValueError("Unregistered original indexer method AST")
        self.scope = scope
        self.weights = {}
        for name in self.layout:
            a = arrays[name]
            self.weights[name] = self.tensor(a).view(torch.float8_e4m3fn if name.endswith("wq_b.weight") else torch.float8_e8m0fnu) if a.dtype == np.uint8 else self.tensor(a)
        for layer in self.layers:
            self.weights[f"layers.{layer}.attn.indexer.wq_b.weight"].scale = self.weights[f"layers.{layer}.attn.indexer.wq_b.scale"]
        self.trace_kernel = self.kernel.fp8_gemm_kernel(4096, 1280, block_size=32, out_dtype="float32", scale_dtype="float8_e8m0fnu")
        self.projection_controls, self.sum_controls, self.candidate_controls, self.quant_controls, self.forward_controls = [], [], [], [], []

    def tensor(self, values):
        t = self.torch.from_numpy(values).to(f"cuda:{self.ordinal}")
        return t.view(self.torch.bfloat16) if values.dtype == np.uint16 else t

    def round(self, values):
        torch = self.torch
        x = self.tensor(values)
        packed, scales = self.kernel.fp4_act_quant(x, 32, False)
        rounded = self.kernel.fp4_act_quant(x.clone(), 32, True)
        codes = packed.view(torch.uint8)
        lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.], dtype=torch.float32, device=x.device)
        nibble = torch.stack((codes & 15, codes >> 4), dim=-1).flatten(-2).long()
        rebuilt = (lut[nibble] * scales.float().repeat_interleave(32, dim=-1)).bfloat16()
        control = check_arrays(_numpy(rounded), _numpy(rebuilt), "original_index_FP4_packed_vs_inplace")
        if control["status"] != "pass": raise RuntimeError("Original index FP4 packed/in-place results differ")
        self.quant_controls.append(control)
        return {"packed_fp4": _numpy(codes), "scales_e8m0": _numpy(scales.view(torch.uint8)), "rounded_bf16": _numpy(rounded)}

    def rope(self, values, positions):
        torch = self.torch
        x = self.tensor(values)
        f = self.math.tables["compressed-yarn"][torch.from_numpy(positions).to(x.device).long()]
        shape = (1, len(positions), 32) if values.ndim == 3 else (1, len(positions), 1, 32)
        rotated = torch.view_as_real(torch.view_as_complex(x[..., -64:].float().unflatten(-1, (-1, 2))) * f.view(shape)).flatten(-2)
        self.math.scope["apply_rotary_emb"](x[..., -64:], f)
        return {"frequency_fp32": _numpy(torch.view_as_real(f)), "rotated_fp32": _numpy(rotated), "output_bf16": _numpy(x)}

    def key(self, latent, layer, positions):
        out = self.functions["linear"](self.tensor(latent), self.weights[f"layers.{layer}.attn.indexer.wk.weight"])
        stages = {"wk_bf16": _numpy(out)}
        norm = self.math.norm(stages["wk_bf16"], f"layers.{layer}.attn.indexer.k_norm.weight")
        stages.update(("norm_" + n, a) for n, a in norm.items())
        rope = self.rope(norm["output_bf16"], positions)
        stages.update(("rope_" + n, a) for n, a in rope.items())
        stages.update(("quant_" + n, a) for n, a in self.round(rope["output_bf16"]).items())
        return stages

    def queries(self, x, qr, layer, position):
        torch = self.torch
        a, weight = self.tensor(qr), self.weights[f"layers.{layer}.attn.indexer.wq_b.weight"]
        quant, scale = self.kernel.act_quant(a, 32, "ue8m0", torch.float8_e8m0fnu)
        output = self.functions["linear"](a, weight)
        trace = torch.empty((*qr.shape[:2], 4096), dtype=torch.float32, device=a.device)
        self.trace_kernel(quant.view(-1, 1280), weight, trace.view(-1, 4096), scale.view(-1, 40), weight.scale)
        control = check_arrays(_numpy(output), _numpy(trace.bfloat16()), "original_index_query_linear_BF16")
        if control["status"] != "pass": raise RuntimeError("Index query diagnostic differs from original linear")
        self.projection_controls.append(control)
        raw = self.functions["linear"](self.tensor(x), self.weights[f"layers.{layer}.attn.indexer.weights_proj.weight"])
        scaled = raw * (128**-0.5 * 32**-0.5)
        q = _numpy(output).reshape(*qr.shape[:2], 32, 128)
        stages = {"input_fp8": _numpy(quant.view(torch.uint8)), "input_scale_e8m0": _numpy(scale.view(torch.uint8)),
                  "projection_fp32": _numpy(trace).reshape(q.shape), "projection_bf16": q,
                  "weights_raw_bf16": _numpy(raw), "weights_scaled_bf16": _numpy(scaled)}
        rope = self.rope(q, np.arange(position, position + x.shape[1], dtype=np.int32))
        stages.update(("rope_" + n, a) for n, a in rope.items())
        stages.update(("quant_" + n, a) for n, a in self.round(rope["output_bf16"]).items())
        return stages

    def score(self, queries, keys, weights):
        torch = self.torch
        dot = torch.einsum("bshd,btd->bsht", self.tensor(queries), self.tensor(keys))
        relu = dot.clone().relu_()
        weighted = relu * self.tensor(weights).unsqueeze(-1)
        accum, out = weighted.sum(dim=2, dtype=torch.float32), weighted.sum(dim=2)
        control = check_arrays(_numpy(out), _numpy(accum.bfloat16()), "original_index_sum_diagnostic")
        if control["status"] != "pass": raise RuntimeError("Index sum diagnostic changes original BF16 output")
        self.sum_controls.append(control)
        return dict(zip(("dot_bf16", "relu_bf16", "weighted_bf16", "sum_fp32", "score_bf16"), map(_numpy, (dot, relu, weighted, accum, out))))

    def select(self, scores, lengths, offset=128, *, candidate_source=False, candidates=None):
        torch = self.torch
        x, lens = self.tensor(scores), self.tensor(lengths).unsqueeze(-1)
        causal = x.masked_fill(torch.arange(x.shape[-1], device=x.device) >= lens, -torch.inf)
        result = {"causal_bf16": _numpy(causal)}
        if candidate_source:
            mask = self.functions["select_candidate_blocks"](causal, lens, 2048, 8)
            blocks = torch.nn.functional.pad(causal, (0, -x.shape[-1] % 8), value=-torch.inf).unflatten(-1, (-1, 8)).amax(-1)
            pinned = blocks.masked_fill(torch.arange(blocks.shape[-1], device=x.device) == (lens - 1) // 8, torch.inf)
            result["candidate_blocks_bf16"], result["candidates_bool"] = _numpy(pinned), _numpy(mask)
        elif candidates is not None:
            result["candidates_bool"] = candidates.copy()
            causal = causal.masked_fill(~self.tensor(candidates), -torch.inf)
        result["masked_bf16"] = _numpy(causal)
        idx = causal.topk(min(512, x.shape[-1]), dim=-1, sorted=False).indices.sort(-1).values
        result["indices_i32"] = _numpy(torch.where(idx < lens, idx + offset, -1).int())
        return result

    def reset(self, batch=1, capacity=128):
        from .attention_topology import role
        torch = self.torch
        forward, weights = self.functions["Indexer.forward"], self.weights
        class IndexCall(SimpleNamespace):
            def __call__(stub, *args):
                return forward(stub, *args)
        self.states = {}
        self.scope["shared_attn"] = SimpleNamespace()
        for layer in self.layers:
            prefix = f"layers.{layer}.attn.indexer."
            stub = IndexCall(owns_k=role(layer).owns_kv, compress_ratio=role(layer).ratio,
                is_candidate_source=layer == 20, uses_candidates=layer > 20,
                candidate_topk_blocks=2048, candidate_block_size=8, n_heads=32, n_local_heads=32,
                index_head_dim=128, rope_head_dim=64, index_topk=512, softmax_scale=128**-0.5,
                freqs_cis=self.math.tables["compressed-yarn"],
                wq_b=lambda x, p=prefix: self.functions["linear"](x, weights[p + "wq_b.weight"]),
                weights_proj=lambda x, p=prefix: self.functions["linear"](x, weights[p + "weights_proj.weight"]))
            if role(layer).owns_kv:
                stub.wk = lambda x, p=prefix: self.functions["linear"](x, weights[p + "wk.weight"])
                stub.k_norm = self.math.modules[prefix + "k_norm.weight"]
                stub.k_cache = torch.zeros((batch, capacity, 128), dtype=torch.bfloat16, device=f"cuda:{self.ordinal}")
            self.states[layer] = stub
        self.diagnostic_caches = {layer: np.zeros((batch, capacity, 128), np.uint16) for layer in self.layers if role(layer).owns_kv}
        self.batch, self.capacity, self.position = batch, capacity, 0

    def prime(self, keys2, keys20, position):
        for layer, value in ((2, keys2), (20, keys20)):
            self.diagnostic_caches[layer][:, :value.shape[1]] = value
            self.states[layer].k_cache[:, :value.shape[1]].copy_(self.tensor(value))
        self.position = position

    def snapshot(self):
        shared = self.scope["shared_attn"]
        return (copy.deepcopy(self.diagnostic_caches), self.position,
                {layer: self.states[layer].k_cache.clone() for layer in (2, 20)},
                {n: getattr(shared, n).clone() for n in ("candidates", "topk_idxs") if hasattr(shared, n)})

    def rollback(self, saved):
        caches, position, originals, shared = saved
        self.diagnostic_caches, self.position = copy.deepcopy(caches), position
        for layer in (2, 20): self.states[layer].k_cache.copy_(originals[layer])
        self.scope["shared_attn"] = SimpleNamespace(index_k=self.states[20].k_cache, **{n: v.clone() for n, v in shared.items()})

    def frame(self, xs, qrs, latents, position):
        sequence = xs[2].shape[1]
        offset, stages, shared = sequence if position == 0 else 128, {}, {}
        for layer in (2, 20, 24):
            ratio = 2 if layer == 2 else 1
            n = (position + sequence) // ratio
            def add(prefix, values):
                stages.update((f"layer{layer}_" + prefix + name, a) for name, a in values.items())
            if layer != 24:
                latent = latents[layer]
                if latent is not None:
                    positions = (np.arange(latent.shape[1], dtype=np.int32) * ratio if position == 0
                                 else np.array([position + 1 - ratio], np.int32))
                    key = self.key(latent, layer, positions)
                    add("key_", key)
                    self.diagnostic_caches[layer][:, position // ratio:position // ratio + latent.shape[1]] = key["quant_rounded_bf16"]
                keys = self.diagnostic_caches[layer][:, :n].copy(order="C")
            else:
                keys = shared["keys20"].copy()
                stages["shared20_to24_keys_bf16"] = keys.copy()
                stages["shared20_to24_candidates_bool"] = shared["candidates20"].copy()
            stages[f"layer{layer}_index_keys_bf16"] = keys.copy()
            query = self.queries(xs[layer], qrs[layer], layer, position)
            add("query_", query)
            score = self.score(query["quant_rounded_bf16"], keys, query["weights_scaled_bf16"])
            add("score_", score)
            lengths = (np.broadcast_to(np.arange(1, sequence + 1, dtype=np.int32)[None, :] // ratio, (self.batch, sequence)).copy()
                       if position == 0 else np.full((self.batch, sequence), n, np.int32))
            selection = self.select(score["score_bf16"], lengths, offset, candidate_source=layer == 20,
                                    candidates=shared["candidates20"] if layer == 24 else None)
            add("selection_", selection)
            if layer == 2: stages["shared2_to3_indices_i32"] = selection["indices_i32"].copy()
            elif layer == 20: shared["keys20"], shared["candidates20"] = keys.copy(), selection["candidates_bool"].copy()
            else: stages["shared24_to25_indices_i32"] = selection["indices_i32"].copy()
        for layer in (2, 20): stages[f"cache{layer}_bf16"] = self.diagnostic_caches[layer].copy()
        controls = []
        wrapper = self.functions["Attention._compress_topk_idxs"]
        for layer in (2, 20, 24):
            n = (position + sequence) // (2 if layer == 2 else 1)
            stub = SimpleNamespace(is_index_source=True, indexer=self.states[layer], freqs_cis=self.math.tables["compressed-yarn"])
            latent = self.tensor(latents[layer]) if layer != 24 and latents[layer] is not None else None
            actual = wrapper(stub, self.tensor(xs[layer]), self.tensor(qrs[layer]), latent, position, offset, n)
            controls.append(check_arrays(stages[f"layer{layer}_selection_indices_i32"], _numpy(actual), f"unchanged_Indexer_{layer}_and_Attention_index_wrapper"))
            if layer in (2, 24):
                consumer = 3 if layer == 2 else 25
                read = wrapper(SimpleNamespace(is_index_source=False), None, None, None, position, offset, n)
                controls.append(check_arrays(stages[f"shared{layer}_to{consumer}_indices_i32"], _numpy(read), f"unchanged_shared_indices_{layer}_to_{consumer}"))
        for layer in (2, 20):
            controls.append(check_arrays(stages[f"cache{layer}_bf16"], _numpy(self.states[layer].k_cache), f"unchanged_Indexer_{layer}_key_cache"))
        original = self.scope["shared_attn"]
        controls.extend((check_arrays(stages["shared20_to24_keys_bf16"], _numpy(original.index_k[:, :(position + sequence)]), "unchanged_shared20_to24_keys"),
                         check_arrays(stages["shared20_to24_candidates_bool"], _numpy(original.candidates), "unchanged_shared20_to24_candidates")))
        if any(c["status"] != "pass" for c in controls):
            raise RuntimeError("Diagnostic index frame differs from unchanged original forwards/sharing")
        self.forward_controls.append(controls)
        self.position = position + sequence
        return stages


def negative_controls(reference):
    from .normalization_validation import inputs
    torch = reference.torch
    values = inputs((1, 3, 128), 91000)
    fp4 = reference.round(values)["rounded_bf16"]
    fp8 = reference.kernel.act_quant(reference.tensor(values), 32, "ue8m0", torch.float8_e8m0fnu, True)
    quant = check_arrays(fp4, _numpy(fp8), "replace_index_FP4_with_FP8")
    q, k, w = inputs((1, 2, 32, 128), 91001), inputs((1, 513, 128), 91002), inputs((1, 2, 32), 91003)
    score = reference.score(q, k, w)
    promoted = (reference.tensor(score["relu_bf16"]).float() * reference.tensor(w).float().unsqueeze(-1)).sum(dim=2).bfloat16()
    product = check_arrays(score["score_bf16"], _numpy(promoted), "remove_BF16_weighted_product_boundary")
    cancellation = torch.zeros((1, 1, 32, 3), dtype=torch.bfloat16, device=f"cuda:{reference.ordinal}")
    cancellation[:, :, 0], cancellation[:, :, 1], cancellation[:, :, 4] = 2**26, 1, -(2**26)
    correct = cancellation.sum(dim=2)
    sequential = torch.zeros((1, 1, 3), dtype=torch.float32, device=cancellation.device)
    for h in range(32): sequential += cancellation[:, :, h].float()
    reduction = check_arrays(_numpy(correct), _numpy(sequential.bfloat16()), "replace_ATen_head_reduction_with_sequential_sum")
    signed = np.zeros((1, 1, 1025), np.uint16)
    signed[..., 1::2] = 0x8000
    expected = reference.select(signed, np.array([[1025]], np.int32))["indices_i32"]
    wrong = reference.tensor(signed).argsort(dim=-1, descending=True, stable=True)[..., :512].sort(-1).values.int() + 128
    ties = check_arrays(expected, _numpy(wrong), "replace_radix_signed_zero_membership_with_stable_float_ties")
    tied = torch.ones((1, 1, 16385), dtype=torch.bfloat16, device=cancellation.device)
    lens = torch.tensor([[[16385]]], dtype=torch.int32, device=cancellation.device)
    pinned = reference.functions["select_candidate_blocks"](tied, lens, 2048, 8)
    blocks = torch.nn.functional.pad(tied, (0, 7), value=-torch.inf).unflatten(-1, (-1, 8)).amax(-1)
    top = blocks.topk(2048, dim=-1)
    keep = torch.zeros_like(blocks, dtype=torch.bool).scatter_(-1, top.indices, top.values > -torch.inf).repeat_interleave(8, dim=-1)[..., :16385]
    newest = check_arrays(_numpy(pinned), _numpy(keep), "omit_newest_candidate_block_pin")
    latent = inputs((1, 3, 512), 91004)
    first = reference.key(latent, 2, np.array([0, 2, 4], np.int32))["quant_rounded_bf16"]
    last = reference.key(latent, 2, np.array([1, 3, 5], np.int32))["quant_rounded_bf16"]
    position = check_arrays(first, last, "rotate_index_key_at_last_group_token")
    return [quant, product, reduction, ties, newest, position]


def compare_indexer(directory, fixture_path, ordinal=0):
    from .attention_memory_reference import missing_dependencies
    from .checkpoint import MODEL, REVISION
    from .indexer import NativeIndexer
    from .indexer_fixture import load_indexer_fixture
    from .indexer_validation import execute_component, exercise_components, exercise_frames, exercise_failures, exercise_snapshot, prime_keys
    from .resources import MemoryBudget, MiB
    missing = missing_dependencies()
    if missing:
        return {"probe": "native_indexer_parity", "status": "blocked", "missing_dependencies": missing}
    import torch
    import tilelang
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    budget = MemoryBudget(cap=512 * MiB)
    try:
        budget.acquire(512 * MiB)
        arrays, fixture = load_indexer_fixture(fixture_path)
        goldens, prime_controls = [], []
        with torch.inference_mode(), torch.cuda.device(ordinal), torch.device(f"cuda:{ordinal}"):
            reference = IndexerOracle(directory, arrays, ordinal)
            def collect(meta, expected, actual):
                if set(expected) != set(actual): raise ValueError("Indexer diagnostic boundary set differs")
                goldens.append({"meta": meta, "stage_sha256": {n: hashlib.sha256(a.tobytes()).hexdigest() for n, a in expected.items()}})
                return [check_arrays(a, actual[n], n) for n, a in expected.items()]
            def component_checker(meta, args, actual): return collect(meta, execute_component(reference, meta, args), actual)
            def frame_checker(meta, args, actual): return collect(meta, reference.frame(*args, meta["position"]), actual)
            def prime(keys, position):
                expected = prime_keys(reference, keys[0].shape[0], position)
                checks = [check_arrays(a, b, f"original_primed_key_{layer}") for layer, a, b in zip((2, 20), expected, keys)]
                if any(c["status"] != "pass" for c in checks): raise RuntimeError("Diagnostic history differs from original key arithmetic")
                prime_controls.append(checks)
                reference.prime(*keys, position)
            saved = None
            def lifecycle(event, *args):
                nonlocal saved
                if event == "reset": reference.reset(*args)
                elif event == "snapshot": saved = reference.snapshot()
                elif event == "rollback": reference.rollback(saved)
            with NativeIndexer(arrays, ordinal) as native:
                components, minimum = exercise_components(native, component_checker)
                frames, low = exercise_frames(native, frame_checker, reference.reset, prime)
                retries, faults = exercise_failures(native, frame_checker, reference.reset)
                snapshots, events = exercise_snapshot(native, frame_checker, lifecycle)
                frames.extend(retries + snapshots)
                minimum = min(minimum, low)
                controls = {"projection": list(reference.projection_controls), "sum": list(reference.sum_controls),
                            "packed_vs_inplace": list(reference.quant_controls), "forward_sharing": list(reference.forward_controls),
                            "primed_keys": prime_controls}
                negatives = negative_controls(reference)
                profile = {"compiler": native.compiler, "cubin_sha256": native.cubin_sha256, "device": native.info,
                           "cublas_version": native.blas.version, "cublas_math_mode": native.blas.math_mode}
                norm_profile = {"compiler": native.norm.compiler, "cubin_sha256": native.norm.cubin_sha256}
            quant_cuda = {str(flag): hashlib.sha256(reference.kernel.fp4_quant_kernel(128, 32, inplace=flag).get_kernel_source().encode()).hexdigest() for flag in (False, True)}
            trace_sha = hashlib.sha256(reference.trace_kernel.get_kernel_source().encode()).hexdigest()
            if quant_cuda != INDEX_QUANT_CUDA or trace_sha != INDEX_QUERY_TRACE_CUDA:
                raise ValueError("Unregistered original index quantization/query diagnostic CUDA")
        passed = (all(r["status"] == "pass" for r in components + frames + faults + events) and all(n["mismatches"] > 0 for n in negatives)
                  and not native.stats()["device_owned_bytes"] and not native.norm.stats()["device_owned_bytes"])
        return {"schema_version": 1, "probe": "native_indexer_parity", "status": "pass" if passed else "fail", "model": MODEL, "revision": REVISION,
            "reference": reference.source, "blas_reference": reference.blas_source, "topk_reference": reference.topk_source,
            "reduction_reference": reference.reduction_source, "index_reference": reference.index_source, "fixture": fixture,
            "indexer_profile": profile, "normalization_profile": norm_profile, "original_methods_ast_sha256": reference.methods,
            "original_quant_cuda_sha256": quant_cuda, "original_query_FP32_cuda_sha256": trace_sha,
            "component_cases": components, "frame_cases": frames, "faults": faults, "snapshot_events": events,
            "reference_goldens": goldens, "unchanged_original_controls": controls, "negative_controls": negatives,
            "reference_runtime_workarounds": reference.workarounds, "torch_version": torch.__version__,
            "torch_git_version": torch.version.git_version, "torch_cuda_version": torch.version.cuda, "tilelang_version": tilelang.__version__,
            "memory": {"indexer_cleanup": native.stats(), "normalization_cleanup": native.norm.stats(),
                "combined_native_host_allowance_bytes": 480 * MiB, "combined_native_and_reference_host_allowance_bytes": 992 * MiB,
                "minimum_sampled_host_available_bytes": minimum, "all_session_reservations_released": True,
                "state_storage": "bounded_host_diagnostic_key_caches", "reservation_is_RSS_limit": False},
            "fixture_original_weight_bytes": fixture["weights_fetched_bytes"], "weights_fetched_during_comparison_bytes": 0,
            "scope": "Original indexers 2/20/24, selected-index sharing 2-to-3 and 24-to-25, key/candidate sharing 20-to-24, generated inputs, one rank",
            "primed_history": "Long cases repeat original first-key component outputs; not an autoregressive history or model quality evaluation",
            "compressed_KV_sharing": "not_implemented", "complete_compressed_attention": "not_implemented", "full_model_quality": "not_run", "end_to_end_speed": "not_measured"}
    finally:
        if budget.reserved: budget.release(budget.reserved)
        torch.set_default_dtype(previous)
