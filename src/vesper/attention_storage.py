"""Explicit optional attention storage experiment; never a reference fallback.

The generated code is adapted from the pinned MIT-licensed DeepSeek source.
See third_party/DeepSeek-LICENSE. Only the Query allocation's storage scope
changes. Diagnostics add stores without replacing arithmetic expressions.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
from pathlib import Path
import sys

from .checkpoint import LIMIT, REFERENCE_HASHES
from .reference_runtime import isolate_nvrtc_handles

STAGES = ("qk_fp32", "probability_fp32", "softmax_state_fp32", "value_accumulator_fp32", "output_fp32")


def _query_allocation(node):
    matches = [n for n in ast.walk(node) if isinstance(n, ast.Assign) and len(n.targets) == 1
               and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "q_shared"]
    if len(matches) != 1:
        raise ValueError("Expected exactly one original Query allocation")
    call = matches[0].value
    expected = ast.parse("T.alloc_shared((h, d), BF16)", mode="eval").body
    if ast.dump(call) != ast.dump(expected):
        raise ValueError("Query allocation differs from the pinned storage layout")
    return call


def _call(node, name, first=None):
    return (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute) and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id == "T" and node.value.func.attr == name
            and (first is None or (node.value.args and isinstance(node.value.args[0], ast.Name)
                                   and node.value.args[0].id == first)))


def _diagnostics(node):
    inner = next(n for n in node.body if isinstance(n, ast.FunctionDef))
    annotations = ("T.Tensor[(b,m,num_blocks,h,block),FP32]", "T.Tensor[(b,m,num_blocks,h,block),FP32]",
                   "T.Tensor[(b,m,num_blocks,5,h),FP32]", "T.Tensor[(b,m,num_blocks,h,d),FP32]",
                   "T.Tensor[(b,m,h,d),FP32]")
    for name, annotation in zip(STAGES, annotations):
        inner.args.args.append(ast.arg(arg=name, annotation=ast.parse(annotation, mode="eval").body))
    launch = next(n for n in inner.body if isinstance(n, ast.With))
    loop = next(n for n in launch.body if isinstance(n, ast.For) and isinstance(n.iter, ast.Call)
                and isinstance(n.iter.func, ast.Attribute) and n.iter.func.attr == "Pipelined")
    stores = {
        "qk": "T.copy(acc_s, qk_fp32[by,bx,t,:,:])",
        "probability": "T.copy(acc_s, probability_fp32[by,bx,t,:,:])\n" + "\n".join(
            f"T.copy({value}, softmax_state_fp32[by,bx,t,{i},:])"
            for i, value in enumerate(("scores_max_prev", "scores_max", "scores_scale", "scores_sum", "sum_exp"))),
        "value": "T.copy(acc_o, value_accumulator_fp32[by,bx,t,:,:])",
        "output": "T.copy(acc_o, output_fp32[by,bx,:,:])",
    }
    inserted = []
    body = []
    for statement in loop.body:
        if _call(statement, "copy", "acc_s"):
            body.extend(ast.parse(stores["probability"]).body)
            inserted.append("probability")
        body.append(statement)
        if _call(statement, "gemm", "q_shared"):
            body.extend(ast.parse(stores["qk"]).body)
            inserted.append("qk")
        elif _call(statement, "gemm", "acc_s_cast"):
            body.extend(ast.parse(stores["value"]).body)
            inserted.append("value")
    loop.body = body
    output_index = next(i for i, n in enumerate(launch.body) if _call(n, "copy", "acc_o"))
    launch.body[output_index:output_index] = ast.parse(stores["output"]).body
    if inserted != ["qk", "probability", "value"]:
        raise ValueError("Original attention diagnostic boundaries changed")


def variant_source(raw: bytes, storage="register-query", diagnostics=False):
    """Hash-check before generating any executable source; leave input intact."""
    if storage not in ("shared", "register-query") or type(diagnostics) is not bool:
        raise ValueError("Unsupported attention storage experiment")
    if len(raw) > LIMIT or hashlib.sha256(raw).hexdigest() != REFERENCE_HASHES["kernel.py"]:
        raise ValueError("Attention source differs from the pinned hash or exceeds its bound")
    tree = ast.parse(raw.decode("utf-8"))
    original = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "sparse_attn_kernel")
    node = copy.deepcopy(original)
    call = _query_allocation(node)
    if storage == "register-query":
        call.func.attr = "alloc_fragment"
    # Verify the inverse transformation before adding instrumentation. No
    # head partition, tile shape, pipeline, reduction, cast or formula changed.
    inverse = copy.deepcopy(node)
    for n in ast.walk(inverse):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "q_shared":
            n.value.func.attr = "alloc_shared"
    if ast.dump(inverse) != ast.dump(original):
        raise ValueError("Attention transformation changed more than Query storage")
    if diagnostics:
        _diagnostics(node)
    ast.fix_missing_locations(node)
    config = next(n for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "pass_configs" for t in n.targets))
    prelude = ("# Adapted from DeepSeek (c) 2023, MIT; see third_party/DeepSeek-LICENSE.\n"
               "import tilelang\nimport tilelang.language as T\n"
               "BF16='bfloat16'\nFP32='float32'\nINT32='int32'\n")
    rendered = (prelude + ast.unparse(config) + "\n\n" + ast.unparse(node) + "\n").encode("utf-8")
    return rendered, {"storage": storage, "diagnostic_stores": diagnostics,
                      "query_storage_only_ast_check": "pass", "head_partition_changed": False,
                      "original_function_ast_sha256": hashlib.sha256(ast.dump(original).encode()).hexdigest(),
                      "generated_python_sha256": hashlib.sha256(rendered).hexdigest()}


def load_variant(raw: bytes, directory: Path, storage="register-query", diagnostics=False):
    """Import a content-addressed generated module in an explicit artifact dir."""
    rendered, metadata = variant_source(raw, storage, diagnostics)
    digest = metadata["generated_python_sha256"]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"attention_{digest}.py"
    if path.exists():
        with path.open("rb") as stream:
            if stream.read(len(rendered) + 1) != rendered:
                raise ValueError("Generated attention source was modified")
    else:
        path.write_bytes(rendered)
    name = f"vesper_attention_{digest}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return isolate_nvrtc_handles(module.sparse_attn_kernel), metadata


def trace_shapes(batch, sequence, heads, dimension, topk, *, profile="baseline"):
    if profile not in ("baseline", "long_context"):
        raise ValueError("Unknown Attention shape profile")
    maximum = 640 if profile == "long_context" and sequence == 1 and (heads, dimension) == (64, 512) else 193
    if (any(type(v) is not int for v in (batch, sequence, heads, dimension, topk))
            or not 1 <= batch <= 2 or not 1 <= sequence <= 3 or heads not in (16, 32, 64)
            or dimension not in (128, 256, 512) or not 1 <= topk <= maximum):
        raise ValueError("Attention diagnostics exceed the bounded probe shapes")
    blocks = (topk + 63) // 64
    return ((batch, sequence, blocks, heads, 64), (batch, sequence, blocks, heads, 64),
            (batch, sequence, blocks, 5, heads), (batch, sequence, blocks, heads, dimension),
            (batch, sequence, heads, dimension))
