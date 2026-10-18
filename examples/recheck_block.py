"""Replay full original-weight layer-0 blocks with no Torch/TileLang runtime."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from vesper.attention import NativeAttention,read_bounded
from vesper.attention_validation import array_digest
from vesper.block import NativeBlock
from vesper.block_validation import exercise_blocks,exercise_block_failures,validate_goldens
from vesper.connected import ConnectedMoE
from vesper.hc import NativeHC
from vesper.hc_validation import exercise_hc
from vesper.normalization import NativeNormRoPE
from vesper.prefix import load_prefix_fixture
from vesper.projections import NativeProjections
from vesper.resources import MemoryBudget,MiB


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("reference-report","prefix-fixture","attention-bundle","store","gate-fixture","router-fixture","shared-fixture","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    raw=read_bounded(args.reference_report,32*MiB)
    report=json.loads(raw)
    goldens=validate_goldens(report)
    budget=MemoryBudget(cap=512*MiB)
    budget.acquire(512*MiB)
    try:
        arrays,fixture=load_prefix_fixture(args.prefix_fixture)
    finally:
        budget.release(512*MiB)
    if fixture!=report["fixture"] or hashlib.sha256(read_bounded(args.store/"manifest.json",2*MiB)).hexdigest()!=report["store_manifest_sha256"]:
        raise ValueError("Original block fixture or expert-store provenance differs")
    def checker(meta,args,actual):
        golden=goldens[meta["id"]]
        if meta!=golden["meta"] or set(actual)!=set(golden["stage_sha256"]):
            raise ValueError("HC/block input identity or boundary set differs")
        return [{"check":n,"status":"pass" if array_digest(a)==golden["stage_sha256"][n] else "fail","elements":int(a.size),
                 "expected_sha256":golden["stage_sha256"][n],"actual_sha256":array_digest(a)} for n,a in actual.items()]
    with NativeHC(arrays) as hc,NativeNormRoPE() as norm,NativeProjections(arrays) as projection,NativeAttention(args.attention_bundle) as attention,ConnectedMoE(args.store,args.gate_fixture,args.router_fixture,args.shared_fixture) as moe:
        profile={"compiler":hc.compiler,"cubin_sha256":hc.cubin_sha256,"device":hc.info,"cublas_version":hc.blas.version,"cublas_math_mode":hc.blas.math_mode}
        if (profile!=report["hc_profile"] or attention.manifest_sha256!=report["attention_bundle_manifest_sha256"]
                or moe.gate_manifest!=report["original_moe_gate_fixture"] or moe.router_manifest!=report["original_moe_router_fixture"]
                or moe.shared_manifest!=report["original_moe_shared_fixture"]):
            raise ValueError("HC compiler/BLAS, Attention bundle or original MoE fixtures differ")
        components,minimum=exercise_hc(hc,checker)
        with NativeBlock(arrays,hc,norm,projection,attention,moe) as block:
            def block_checker(meta,h,p,result):
                return checker(meta,(h,p),result)
            blocks,low=exercise_blocks(block,arrays,block_checker,lambda:None)
            retries,faults=exercise_block_failures(block,arrays,block_checker,lambda:None)
            blocks.extend(retries)
            cache=moe.snapshot()
            if cache["compiled_cubin_sha256"]!=report["MoE_cache"]["compiled_cubin_sha256"] or cache["cublas_version"]!=report["MoE_cache"]["cublas_version"]:
                raise ValueError("MoE compiled profile differs")
        minimum=min(minimum,low)
    frameworks={n+"_installed":importlib.util.find_spec(n) is not None for n in ("torch","tilelang")}
    frameworks.update({n+"_imported":any(m==n or m.startswith(n+".") for m in sys.modules) for n in ("torch","tilelang")})
    passed=(all(r["status"]=="pass" for r in components+blocks+faults) and not frameworks["torch_imported"] and not frameworks["tilelang_imported"]
        and not hc.stats()["device_owned_bytes"] and not norm.stats()["device_owned_bytes"] and not projection.stats()["device_owned_bytes"]
        and not attention.stats()["device_owned_bytes"] and not block.attention.kv.stats()["device_owned_bytes"]
        and not block.budget.reserved and all(v==0 for v in moe.cleanup.values()))
    result={"schema_version":1,"probe":"native_layer0_block_recheck","status":"pass" if passed else "fail",
        "reference_report_sha256":hashlib.sha256(raw).hexdigest(),"hc_profile":profile,**frameworks,
        "HC_cases":components,"block_cases":blocks,"faults":faults,"MoE_cache":cache,
        "memory":{"HC_cleanup":hc.stats(),"normalization_cleanup":norm.stats(),"projection_cleanup":projection.stats(),"attention_cleanup":attention.stats(),
            "KV_cleanup":block.attention.kv.stats(),"block_cleanup":block.stats(),"MoE_cleanup":moe.cleanup,
            "combined_native_host_allowance_bytes":1136*MiB,"minimum_sampled_host_available_bytes":minimum,"reservation_is_RSS_limit":False},
        "weights_fetched_bytes":0,"full_model_quality":"not_run","end_to_end_speed":"not_measured",
        "scope":"Independent framework-free single-token layer-0 complete block replay, original embedding streams/masks and audited heterogeneous offload"}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_bytes((json.dumps(result,indent=2,allow_nan=False)+"\n").encode())
    print(json.dumps({"status":result["status"],"HC_cases":len(components),"block_calls":len(blocks),**frameworks}))
    return 0 if passed else 2


if __name__=="__main__":
    raise SystemExit(main())
