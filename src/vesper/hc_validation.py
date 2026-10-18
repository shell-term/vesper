"""Deterministic HC component corpus shared by paired and native-only checks."""

import numpy as np

from .attention_validation import array_digest
from .formats import bf16_bits
from .resources import physical_memory

MODES = ("normal", "zeros", "signed_zeros", "wide_scales", "near_eps", "cancellation")
ROWS = (1,2,3,4,6,8)


def hc_inputs(rows, mode, seed):
    rng = np.random.default_rng(20261018 + seed)
    shape = (2, rows // 2, 4, 5120) if rows in (4,6,8) else (1,rows,4,5120)
    x = rng.standard_normal(shape).astype(np.float32)
    y = rng.standard_normal((*shape[:2],5120)).astype(np.float32)
    if mode == "wide_scales":
        x = np.ldexp(x, rng.integers(-20,21,(*shape[:3],1)))
    elif mode == "near_eps":
        x *= np.float32(1e-10)
    elif mode == "cancellation":
        x[...,0,:],x[...,1,:],x[...,2,:],x[...,3,:] = 2.**26,1.,-2.**26,2.**-8
    x,y = bf16_bits(x),bf16_bits(y)
    pre = rng.standard_normal((*shape[:2],4)).astype(np.float32)
    post = rng.standard_normal((*shape[:2],4)).astype(np.float32)
    comb = rng.standard_normal((*shape[:2],4,4)).astype(np.float32)
    if mode in ("zeros","signed_zeros"):
        x.fill(0)
        y.fill(0)
        if mode == "signed_zeros":
            x[...,1::2],y[...,1::2] = 0x8000,0x8000
    return x,pre,y,post,comb


def component_plan():
    for mode in MODES:
        for rows in ROWS if mode == "normal" else (1,3,8):
            seed = 9000 + rows + 100 * MODES.index(mode)
            x,pre,y,post,comb = hc_inputs(rows, mode, seed)
            for part in ("attn","ffn"):
                meta = {"id": f"hc/mixes/{part}/{mode}/{rows}", "kind": "hc", "operation": "mixes", "part": part,
                        "rows": rows, "shape": list(x.shape), "seed": seed, "mode": mode, "input_sha256": [array_digest(x)]}
                yield meta,(x,)
            for operation,args in (("pre",(x,pre)),("post",(y,x,post,comb))):
                yield {"id": f"hc/{operation}/{mode}/{rows}", "kind": "hc", "operation": operation,
                       "rows": rows,"shape": list(x.shape),"seed": seed,"mode": mode,"input_sha256": [array_digest(a) for a in args]},args


def exercise_hc(native,checker):
    rows,minimum = [],physical_memory()["available"]
    for meta,args in component_plan():
        actual = native.mixes(args[0],meta["part"]) if meta["operation"] == "mixes" else getattr(native,meta["operation"])(*args)
        checks = checker(meta,args,actual)
        if sum(native.device.allocations.values()) != native.resident_bytes:
            raise RuntimeError("HC call leaked temporary GPU buffers")
        rows.append({**meta,"checks":checks,"status":"pass" if all(c["status"]=="pass" for c in checks) else "fail"})
        minimum = min(minimum,physical_memory()["available"])
    return rows,minimum
