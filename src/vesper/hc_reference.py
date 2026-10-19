"""Original Hyper-Connection methods and store-only Sinkhorn instrumentation."""

import ast
import copy
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from .attention_storage_reference import _numpy
from .attention_validation import check_arrays
from .gate_reference import fetch_gate_reference
from .normalization_reference import fetch_pytorch_sources
from .prefix_reference import fetch_prefix_reference
from .reference_runtime import configure_reference_runtime, isolate_nvrtc_handles
from .router import ROUTER_TORCH_REVISION

SINK_STAGES = ("affine_fp32", "row_max_fp32", "exp_fp32", "reductions_fp32", "iterations_fp32")
HC_INSTRUMENTATION = {
    "original_factory_ast_sha256": "3b0f8bf173937b068cfabb0729ba31b8fe9e29681b9502ef2ead04e8a9a7a81e",
    "instrumented_factory_ast_sha256": "632d7b4edf39bb96326bbb14fa46240e7e8c4ca019fd7fea22ad140f1d5636dd",
    "arithmetic_AST_restoration": True,
    "original_cuda_sha256": "d1863b136c88c7f17d3767610a0f2424dda54df8481a4956e08b92870c357f19",
    "instrumented_cuda_sha256": "ad93138ce5515955e321f62414508000a535791d46ccba9bfa282c8ba4d668fa",
}


def instrument_hc(tree):
    original = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "hc_split_sinkhorn_kernel")
    node = copy.deepcopy(original)
    node.name = "_hc_trace_factory"
    inner = next(n for n in node.body if isinstance(n, ast.FunctionDef))
    for name, shape in zip(SINK_STAGES, ("n,hc,hc", "n,hc", "n,hc,hc", "n,40,hc", "n,40,hc,hc")):
        inner.args.args.append(ast.arg(arg=name, annotation=ast.parse("T.Tensor[(" + shape + "),FP32]", mode="eval").body))
    launch = next(n for n in inner.body if isinstance(n, ast.With))
    def store(value, target):
        return ast.parse(f"T.copy({value}, {target})").body
    body, parallel = [], 0
    for statement in launch.body:
        body.append(statement)
        if isinstance(statement, ast.For):
            if isinstance(statement.iter, ast.Call) and isinstance(statement.iter.func, ast.Attribute) and statement.iter.func.attr == "serial":
                loop, side = [], 0
                for step in statement.body:
                    loop.append(step)
                    index = "2+2*_" if side == 0 else "3+2*_"
                    if isinstance(step, ast.Expr) and isinstance(step.value, ast.Call) and isinstance(step.value.func, ast.Attribute) and step.value.func.attr == "reduce_sum":
                        value = ast.unparse(step.value.args[1])
                        loop.extend(store(value, "reductions_fp32[i," + index + ",:]"))
                    elif isinstance(step, ast.For):
                        loop.extend(store("comb_frag", "iterations_fp32[i," + index + ",:,:]"))
                        side += 1
                if side != 2:
                    raise ValueError("Original Sinkhorn iteration body changed")
                statement.body = loop
            elif isinstance(statement.iter, ast.Call) and len(statement.iter.args) == 2:
                target = ("affine_fp32[i,:,:]", "exp_fp32[i,:,:]", "iterations_fp32[i,0,:,:]", "iterations_fp32[i,1,:,:]")[parallel]
                body.extend(store("comb_frag", target))
                parallel += 1
        elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call) and isinstance(statement.value.func, ast.Attribute):
            name = statement.value.func.attr
            if name == "reduce_max":
                body.extend(store("row_max", "row_max_fp32[i,:]"))
            elif name == "reduce_sum":
                value = ast.unparse(statement.value.args[1])
                stage = 0 if value == "row_sum" else 1
                body.extend(store(value, f"reductions_fp32[i,{stage},:]"))
    if parallel != 4:
        raise ValueError("Original HC affine/normalization layout changed")
    launch.body = body
    # Removing only the added stores/arguments/name must recover the pinned AST.
    stripped = copy.deepcopy(node)
    stripped.name = original.name
    stripped_inner = next(n for n in stripped.body if isinstance(n, ast.FunctionDef))
    stripped_inner.args.args = stripped_inner.args.args[:-5]
    class RemoveStores(ast.NodeTransformer):
        def visit_Expr(self, item):
            if isinstance(item.value, ast.Call) and any(isinstance(n, ast.Name) and n.id in SINK_STAGES for n in ast.walk(item.value)):
                return None
            return item
    stripped = RemoveStores().visit(stripped)
    if ast.dump(stripped) != ast.dump(original):
        raise ValueError("HC instrumentation changed an arithmetic/source node")
    ast.fix_missing_locations(node)
    return node, {"original_factory_ast_sha256": hashlib.sha256(ast.dump(original).encode()).hexdigest(),
                  "instrumented_factory_ast_sha256": hashlib.sha256(ast.dump(node).encode()).hexdigest(),
                  "arithmetic_AST_restoration": True}


class HCOracle:
    def __init__(self, directory, arrays, ordinal=0, *, layer=0, profile="qualified"):
        from .prefix_profile import validate_layer
        validate_layer(layer, (0, 2, 3, 20, 21, 22, 23, 24, 25), profile)
        self.layer = layer
        import torch
        import tilelang
        from tilelang.contrib.nvrtc import get_nvrtc_version
        if (torch.version.git_version != ROUTER_TORCH_REVISION or torch.get_float32_matmul_precision() != "highest"
                or str(torch.backends.cuda.preferred_blas_library()) != "_BlasBackend.Cublas" or torch.cuda.tunable.is_enabled()
                or tilelang.__version__ != "0.1.12" or get_nvrtc_version() != (13, 4)
                or os.environ.get("CUBLAS_WORKSPACE_CONFIG") is not None
                or any(os.environ.get(n) not in (None,"0") for n in ("NVIDIA_TF32_OVERRIDE","TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"))):
            raise ValueError("Unregistered HC PyTorch/cuBLAS profile")
        self.torch, self.ordinal, self.directory = torch, ordinal, Path(directory)
        self.source = fetch_prefix_reference(self.directory)
        self.blas_source = fetch_gate_reference(self.directory / "pytorch-gate")
        self.reduce_source = fetch_pytorch_sources(self.directory / "pytorch-normalization")
        spec = importlib.util.spec_from_file_location("vesper_original_hc_kernel", self.directory / "kernel.py")
        self.kernel = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.kernel)
        self.workarounds = configure_reference_runtime(self.kernel, tilelang.__version__)
        self.kernel.hc_split_sinkhorn_kernel = isolate_nvrtc_handles(self.kernel.hc_split_sinkhorn_kernel)
        self.workarounds.append("TileLang 0.1.12 NVRTC: isolate original/instrumented HC launch handles; arithmetic unchanged")
        source_tree = ast.parse((self.directory / "kernel.py").read_bytes())
        instrumented, self.instrumentation = instrument_hc(source_tree)
        configs = next(n for n in source_tree.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "pass_configs" for t in n.targets))
        text = "import tilelang\nimport tilelang.language as T\nFP32='float32'\n" + ast.unparse(configs) + "\n" + ast.unparse(instrumented) + "\n"
        generated = self.directory / "hc-variants" / (self.instrumentation["instrumented_factory_ast_sha256"] + ".py")
        generated.parent.mkdir(exist_ok=True)
        if generated.exists() and generated.read_text(encoding="utf-8") != text:
            raise ValueError("Cached HC diagnostic source differs")
        if not generated.exists():
            generated.write_text(text, encoding="utf-8")
        name = "vesper_hc_trace_" + generated.stem
        spec = importlib.util.spec_from_file_location(name, generated)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        self.traced = isolate_nvrtc_handles(getattr(module, instrumented.name))(4, 20, 1e-6)
        self.original = self.kernel.hc_split_sinkhorn_kernel(4, 20, 1e-6)
        self.instrumentation.update(original_cuda_sha256=hashlib.sha256(self.original.get_kernel_source().encode()).hexdigest(),
                                    instrumented_cuda_sha256=hashlib.sha256(self.traced.get_kernel_source().encode()).hexdigest())
        if self.instrumentation != HC_INSTRUMENTATION:
            raise ValueError("Unregistered HC oracle source/codegen profile; qualification required")
        self.stub = SimpleNamespace(norm_eps=1e-20, hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6)
        tree = ast.parse((self.directory / "model.py").read_bytes())
        block = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Block")
        self.methods, self.functions = {}, {}
        scope = {"torch": torch, "F": torch.nn.functional, "hc_split_sinkhorn": self.kernel.hc_split_sinkhorn}
        for name in ("hc_mixes", "hc_pre", "hc_post"):
            node = next(n for n in block.body if isinstance(n, ast.FunctionDef) and n.name == name)
            self.methods[name] = hashlib.sha256(ast.dump(node).encode()).hexdigest()
            exec(compile(ast.Module(body=[node], type_ignores=[]), "<pinned-HC-method>", "exec"), scope)
            self.functions[name] = scope[name]
        self.weights = {p + "_" + s: torch.from_numpy(arrays[f"layers.{layer}.hc_{p}_{s}"]).to(f"cuda:{ordinal}")
                        for p in ("attn", "ffn") for s in ("fn", "scale", "base")}
        self.controls = []

    def tensor(self, a):
        value = self.torch.from_numpy(a).to(f"cuda:{self.ordinal}")
        return value.view(self.torch.bfloat16) if a.dtype == np.uint16 else value

    def mixes(self, values, part="attn"):
        torch = self.torch
        x = self.tensor(values)
        promoted = x.flatten(2).float()
        square = promoted.square()
        mean = square.mean(-1, keepdim=True)
        shifted = mean + 1e-20
        rstd = torch.rsqrt(shifted)
        fn, scale, base = [self.weights[part + "_" + s] for s in ("fn", "scale", "base")]
        raw = torch.nn.functional.linear(promoted, fn)
        mixed = raw * rstd
        outputs = self.functions["hc_mixes"](self.stub, x, fn, scale, base)
        b,s = values.shape[:2]
        traces = [torch.empty(shape, dtype=torch.float32, device=x.device) for shape in ((b,s,4,4),(b,s,4),(b,s,4,4),(b,s,40,4),(b,s,40,4,4))]
        instrumented = [torch.empty_like(a) for a in outputs]
        self.traced(mixed.view(-1,24), scale, base, *[a.view(-1,4) if i < 2 else a.view(-1,4,4) for i,a in enumerate(instrumented)],
                    *[a.view(-1,*a.shape[2:]) for a in traces])
        controls = [check_arrays(_numpy(a), _numpy(b), "unchanged_HC_" + n) for n,a,b in zip(("pre","post","comb"), outputs,instrumented)]
        if any(c["status"] != "pass" for c in controls):
            raise RuntimeError("HC diagnostic instrumentation differs from unchanged original method")
        self.controls.append(controls)
        names = ("promoted_fp32","square_fp32","mean_fp32","mean_eps_fp32","rstd_fp32","projection_fp32","mixes_fp32","pre_fp32","post_fp32","comb_fp32",*SINK_STAGES)
        return dict(zip(names, map(_numpy,(promoted,square,mean,shifted,rstd,raw,mixed,*outputs,*traces))))

    def pre(self, values, pre):
        x, p = self.tensor(values), self.tensor(pre)
        products = p.unsqueeze(-1) * x.float()
        summed = products.sum(dim=2)
        output = self.functions["hc_pre"](self.stub,x,p)
        control = check_arrays(_numpy(output), _numpy(summed.to(x.dtype)), "unchanged_HC_pre")
        if control["status"] != "pass":
            raise RuntimeError("HC pre diagnostics differ from unchanged method")
        self.controls.append([control])
        return dict(zip(("products_fp32","sum_fp32","output_bf16"),map(_numpy,(products,summed,output))))

    def post(self, values, residual, post, comb):
        x,r,p,c = map(self.tensor,(values,residual,post,comb))
        product = p.unsqueeze(-1) * x.unsqueeze(-2)
        residual_products = c.unsqueeze(-1) * r.unsqueeze(-2)
        summed = residual_products.sum(dim=2)
        combined = product + summed
        output = self.functions["hc_post"](self.stub,x,r,p,c)
        control = check_arrays(_numpy(output), _numpy(combined.type_as(x)), "unchanged_HC_post")
        if control["status"] != "pass":
            raise RuntimeError("HC residual diagnostics differ from unchanged method")
        self.controls.append([control])
        return dict(zip(("post_product_fp32","residual_products_fp32","residual_sum_fp32","combined_fp32","output_bf16"),map(_numpy,(product,residual_products,summed,combined,output))))
