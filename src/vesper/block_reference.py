"""Complete original layer-0 block oracle, including original MoE callbacks."""

from contextlib import ExitStack
import ctypes as C
import hashlib
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np

from .attention import NativeAttention
from .attention_memory_reference import AttentionMemoryReference, missing_dependencies
from .attention_storage_reference import _numpy
from .attention_validation import check_arrays
from .block import NativeBlock
from .block_validation import exercise_blocks, exercise_block_failures
from .checkpoint import MODEL, REVISION
from .connected import ConnectedMoE
from .connected_reference import OfficialConnected
from .formats import bf16_values
from .hc import NativeHC
from .hc_reference import HCOracle
from .hc_validation import exercise_hc, hc_inputs
from .normalization import NativeNormRoPE
from .prefix import load_prefix_fixture
from .projection_reference import ProjectionOracle
from .projections import NativeProjections
from .resources import MemoryBudget, MiB


class BlockOracle:
    def __init__(self, arrays, hc, projection, memory, moe):
        self.arrays,self.hc,self.projection,self.memory,self.moe=arrays,hc,projection,memory,moe
        self.controls=[]

    def reset(self):
        self.projection.reset(1,self.memory)
        self.original_attention=self.projection.stub(1,self.memory)
        block=SimpleNamespace(norm_eps=1e-20,hc_mult=4,hc_sinkhorn_iters=20,hc_eps=1e-6,
            attn_norm=self.projection.math.modules["layers.0.attn_norm.weight"],ffn_norm=self.projection.math.modules["layers.0.ffn_norm.weight"])
        for part in ("attn","ffn"):
            for suffix in ("fn","scale","base"):
                setattr(block,"hc_"+part+"_"+suffix,self.hc.weights[part+"_"+suffix])
        for name in ("hc_mixes","hc_pre","hc_post"):
            setattr(block,name,MethodType(self.hc.functions[name],block))
        block.attn=lambda x,pos: self.projection.full_forward(self.original_attention,x,pos)
        self.original_block,self.forward=block,self.projection.method("Block","forward")

    def step(self, hidden, pre_mix, image_span, position):
        expected={}
        def add(prefix,values):
            expected.update((prefix+n,a) for n,a in values.items())
            return values.get("output_bf16")
        attn_mixes=self.hc.mixes(hidden,"attn")
        add("hc_attn_",attn_mixes)
        x=add("attn_pre_",self.hc.pre(hidden,pre_mix))
        x=add("attn_norm_",self.projection.math.norm(x,"layers.0.attn_norm.weight"))
        attention=self.projection.layer(x,position)
        add("attention_",attention)
        after=add("attn_post_",self.hc.post(attention["wo_b_output_bf16"],hidden,attn_mixes["post_fp32"],attn_mixes["comb_fp32"]))
        ffn_mixes=self.hc.mixes(after,"ffn")
        add("hc_ffn_",ffn_mixes)
        x=add("ffn_pre_",self.hc.pre(after,attn_mixes["pre_fp32"]))
        x=add("ffn_norm_",self.projection.math.norm(x,"layers.0.ffn_norm.weight"))
        moe=self.moe.forward(bf16_values(x.reshape(5120)),image_span)
        add("moe_",moe)
        out=add("ffn_post_",self.hc.post(moe["aggregation.output_bf16"].reshape(1,1,5120),after,ffn_mixes["post_fp32"],ffn_mixes["comb_fp32"]))
        expected["output_bf16"],expected["next_pre_mix_fp32"]=out,ffn_mixes["pre_fp32"]
        called=[]
        def original_moe(x,mask):
            called.append(1)
            value=self.moe.forward(bf16_values(_numpy(x).reshape(5120)),image_span)
            return self.hc.tensor(value["aggregation.output_bf16"].reshape(1,1,5120))
        self.original_block.ffn=original_moe
        torch=self.hc.torch
        mask=torch.full((1,1),image_span,dtype=torch.bool,device=f"cuda:{self.hc.ordinal}")
        output,next_pre=self.forward(self.original_block,self.hc.tensor(hidden),position,self.hc.tensor(pre_mix),mask)
        controls=[check_arrays(_numpy(output),out,"unchanged_Block_forward_output_BF16"),
                  check_arrays(_numpy(next_pre),ffn_mixes["pre_fp32"],"unchanged_Block_forward_next_pre_FP32"),
                  check_arrays(_numpy(self.original_attention.window_kv_cache),attention["kv_cache_bf16"],"unchanged_Block_forward_KV")]
        if len(called)!=1 or any(c["status"]!="pass" for c in controls):
            raise RuntimeError("Instrumented full block differs from unchanged original Block.forward")
        self.controls.append(controls)
        return expected


def negative_controls(native,reference):
    torch=reference.torch
    x,pre,y,post,comb=hc_inputs(8,"normal",9721)
    expected=reference.mixes(x,"attn")
    # Deliberately collapse ATen's block.x-then-block.y tree into one flat tree.
    with ExitStack() as stack:
        squares=native._upload(stack,expected["square_fp32"])
        means=native._allocate(stack,8*4)
        native.device.launch(native.module,"norm_mean",[C.c_uint64(squares),C.c_uint64(means),C.c_int(20480),C.c_float(1/20480)],8*512,threads=512)
        wrong=native.device.download(means,expected["mean_fp32"].shape,np.float32)
    tree=check_arrays(expected["mean_fp32"],wrong,"flatten_HC_mean_reduction_tree")
    mixed=reference.tensor(expected["mixes_fp32"])
    scale,base=reference.weights["attn_scale"],reference.weights["attn_base"]
    before,after,comb_short=[torch.empty_like(reference.tensor(expected[n])) for n in ("pre_fp32","post_fp32","comb_fp32")]
    reference.kernel.hc_split_sinkhorn_kernel(4,1,1e-6)(mixed.view(-1,24),scale,base,before.view(-1,4),after.view(-1,4),comb_short.view(-1,4,4))
    short=check_arrays(expected["comb_fp32"],_numpy(comb_short),"replace_20_Sinkhorn_iterations_with_one")
    correct=reference.post(y,x,post,comb)
    wrong=reference.post(y,x,post,comb.swapaxes(-1,-2).copy())
    transpose=check_arrays(correct["output_bf16"],wrong["output_bf16"],"transpose_residual_comb_source_target")
    p=reference.tensor(correct["post_product_fp32"]).bfloat16().float()
    rp=reference.tensor(correct["residual_products_fp32"]).bfloat16().float()
    rounded=(p+rp.sum(dim=2)).bfloat16()
    early=check_arrays(correct["output_bf16"],_numpy(rounded),"round_residual_products_to_BF16_early")
    return [tree,short,transpose,early]


def compare_block(directory,prefix_fixture,attention_bundle,store,gate_fixture,router_fixture,shared_fixture,expert_directory,ordinal=0):
    missing=missing_dependencies()
    if missing:
        return {"probe":"native_layer0_block_parity","status":"blocked","missing_dependencies":missing}
    for path in (prefix_fixture,gate_fixture,router_fixture,shared_fixture,Path(store)/"manifest.json",Path(attention_bundle)/"manifest.json"):
        if not Path(path).is_file():
            raise FileNotFoundError("Required bounded block fixture or bundle is missing")
    import torch
    previous=torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    budget=MemoryBudget(cap=768*MiB)
    goldens=[]
    try:
        budget.acquire(768*MiB)
        arrays,fixture=load_prefix_fixture(prefix_fixture)
        with ExitStack() as stack,torch.inference_mode(),torch.cuda.device(ordinal),torch.device(f"cuda:{ordinal}"):
            hc_reference=HCOracle(directory,arrays,ordinal)
            projection_reference=ProjectionOracle(directory,arrays,ordinal)
            memory=stack.enter_context(AttentionMemoryReference(Path(directory),ordinal))
            qualification=memory.qualify()
            if qualification["status"]!="pass":
                raise RuntimeError("Block Attention oracle failed qualification")
            moe_reference=OfficialConnected(gate_fixture,router_fixture,shared_fixture,expert_directory,Path(directory),ordinal)
            reference=BlockOracle(arrays,hc_reference,projection_reference,memory,moe_reference)
            hc=stack.enter_context(NativeHC(arrays,ordinal))
            norm=stack.enter_context(NativeNormRoPE(ordinal))
            projection=stack.enter_context(NativeProjections(arrays,ordinal))
            attention=stack.enter_context(NativeAttention(attention_bundle,ordinal))
            moe=stack.enter_context(ConnectedMoE(store,gate_fixture,router_fixture,shared_fixture))
            def collect(meta,expected,actual):
                if set(expected)!=set(actual):
                    raise ValueError("HC/block trace boundaries differ")
                goldens.append({"meta":meta,"stage_sha256":{n:hashlib.sha256(a.tobytes()).hexdigest() for n,a in expected.items()}})
                return [check_arrays(a,actual[n],n) for n,a in expected.items()]
            def check_hc(meta,args,actual):
                expected=hc_reference.mixes(args[0],meta["part"]) if meta["operation"]=="mixes" else getattr(hc_reference,meta["operation"])(*args)
                return collect(meta,expected,actual)
            components,minimum=exercise_hc(hc,check_hc)
            def check_block(meta,h,p,actual):
                return collect(meta,reference.step(h,p,meta["image_span"],meta["position"]),actual)
            with NativeBlock(arrays,hc,norm,projection,attention,moe) as block:
                blocks,low=exercise_blocks(block,arrays,check_block,reference.reset)
                retries,faults=exercise_block_failures(block,arrays,check_block,reference.reset)
                blocks.extend(retries)
                cache_state=moe.snapshot()
            minimum=min(minimum,low)
            negatives=negative_controls(hc,hc_reference)
            hc_profile={"compiler":hc.compiler,"cubin_sha256":hc.cubin_sha256,"device":hc.info,"cublas_version":hc.blas.version,"cublas_math_mode":hc.blas.math_mode}
            attention_sha=attention.manifest_sha256
            store_sha=hashlib.sha256((Path(store)/"manifest.json").read_bytes()).hexdigest()
        passed=(all(r["status"]=="pass" for r in components+blocks+faults) and all(n["mismatches"]>0 for n in negatives)
            and not hc.stats()["device_owned_bytes"] and not norm.stats()["device_owned_bytes"] and not projection.stats()["device_owned_bytes"]
            and not attention.stats()["device_owned_bytes"] and not block.attention.kv.stats()["device_owned_bytes"]
            and not block.budget.reserved and all(v==0 for v in moe.cleanup.values()))
        return {"schema_version":1,"probe":"native_layer0_block_parity","status":"pass" if passed else "fail","model":MODEL,"revision":REVISION,
            "reference":hc_reference.source,"blas_reference":hc_reference.blas_source,"reduction_reference":hc_reference.reduce_source,
            "fixture":fixture,"hc_profile":hc_profile,"hc_instrumentation":hc_reference.instrumentation,
            "original_methods_ast_sha256":{**hc_reference.methods,**projection_reference.methods},
            "attention_bundle_manifest_sha256":attention_sha,"attention_reference_qualification":qualification,
            "store_manifest_sha256":store_sha,"original_moe_gate_fixture":moe.gate_manifest,"original_moe_router_fixture":moe.router_manifest,
            "original_moe_shared_fixture":moe.shared_manifest,"used_original_experts":moe_reference.provenance,
            "HC_cases":components,"block_cases":blocks,"faults":faults,"reference_goldens":goldens,"negative_controls":negatives,
            "unchanged_Block_forward_controls":reference.controls,"unchanged_HC_controls":hc_reference.controls,
            "unchanged_Attention_forward_controls":projection_reference.layer_controls,
            "reference_runtime_workarounds":hc_reference.workarounds,
            "memory":{"HC_cleanup":hc.stats(),"normalization_cleanup":norm.stats(),"projection_cleanup":projection.stats(),
                "attention_cleanup":attention.stats(),"KV_cleanup":block.attention.kv.stats(),"block_cleanup":block.stats(),"MoE_cleanup":moe.cleanup,
                "combined_native_host_allowance_bytes":1136*MiB,"combined_native_and_reference_host_allowance_bytes":1968*MiB,
                "minimum_sampled_host_available_bytes":minimum,"reservation_is_RSS_limit":False},"MoE_cache":cache_state,
            "weights_fetched_bytes":0,"torch_version":torch.__version__,"torch_git_version":torch.version.git_version,"torch_cuda_version":torch.version.cuda,
            "scope":"Original layer-0 uncompressed single-token Transformer block, two two-token embedding sequences, both original gate masks, one rank",
            "compressed_KV_indexing_sharing_Engram":"not_implemented","full_model_quality":"not_run","end_to_end_speed":"not_measured"}
    finally:
        if budget.reserved:
            budget.release(budget.reserved)
        torch.set_default_dtype(previous)
