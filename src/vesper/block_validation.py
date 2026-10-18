"""Original-embedding block calls, image masks, and whole-call refusal/retry."""

import hashlib

import numpy as np

from .attention_validation import array_digest
from .prefix import PREFIX_IDS, PREFIX_SEQUENCES
from .resources import physical_memory

FAULTS = ("host_pressure", "MoE_after_Attention", "final_residual_download", "combined_device_cap", "rollback_download")


def block_input(arrays, sequence, position):
    token = PREFIX_SEQUENCES[sequence][position]
    hidden = np.repeat(arrays["embedding_rows"][PREFIX_IDS.index(token)].reshape(1,1,1,5120),4,axis=2)
    pre = np.zeros((1,1,4),np.float32)
    pre[...,0] = 1
    return hidden,pre


def block_meta(identity, hidden, pre, *, sequence, position, image_span, repeat=0):
    return {"id":identity,"kind":"block","sequence":sequence,"position":position,"image_span":image_span,"repeat":repeat,
            "token_id":PREFIX_SEQUENCES[sequence][position],"origin":"original_layer0_embedding_streams_and_initial_one_hot_mix",
            "input_bf16_sha256":array_digest(hidden),"incoming_pre_sha256":array_digest(pre)}


def block_record(meta,checks,actual):
    return {**meta,"checks":checks,"status":"pass" if all(c["status"]=="pass" for c in checks) else "fail",
            "selected_expert_ids":actual["moe_gate.indices"].tolist(),
            "routing_weight_bits":actual["moe_gate.weights"].view(np.uint32).tolist()}


def exercise_blocks(native, arrays, checker, reset_reference):
    rows,minimum = [],physical_memory()["available"]
    for repeat in (0,1):
        for image_span in (False,True):
            for sequence in (0,1):
                native.reset()
                reset_reference()
                for position in (0,1):
                    h,p = block_input(arrays,sequence,position)
                    meta=block_meta(f"block/repeat{repeat}/image{int(image_span)}/seq{sequence}/p{position}",h,p,
                                    sequence=sequence,position=position,image_span=image_span,repeat=repeat)
                    actual=native.step(h,p,image_span,start_pos=position)
                    checks=checker(meta,h,p,actual)
                    rows.append(block_record(meta,checks,actual))
                    minimum=min(minimum,physical_memory()["available"])
    return rows,minimum


def exercise_block_failures(native, arrays, checker, reset_reference):
    rows,faults=[],[]
    for ordinal,mode in enumerate(FAULTS):
        native.reset()
        reset_reference()
        image_span=bool(ordinal%2)
        h,p=block_input(arrays,0,0)
        meta=block_meta("block/fault-prefill/"+mode,h,p,sequence=0,position=0,image_span=image_span)
        actual=native.step(h,p,image_span)
        checks=checker(meta,h,p,actual)
        rows.append(block_record(meta,checks,actual))
        h,p=block_input(arrays,0,1)
        before=native.attention.kv.inspect()
        acquire,moe,post=native.budget.acquire,native.moe.forward,native.hc.post
        download,kv_download=native.hc.device.download,native.attention.kv.device.download
        count=0
        def refuse(*args,**kwargs):
            raise MemoryError("Injected whole-block host pressure")
        def refuse_download(*args,**kwargs):
            raise OSError("Injected final-residual or rollback download failure")
        def fail_moe(*args,**kwargs):
            if mode == "rollback_download":
                native.attention.kv.device.download=refuse_download
                raise OSError("Injected MoE failure followed by rollback-publication failure")
            if mode == "combined_device_cap":
                result=moe(*args,**kwargs)
                native.hc.device.allocate(512*1024**2)
                return result
            raise OSError("Injected MoE failure after Attention produced its output")
        def final_post(*args,**kwargs):
            nonlocal count
            count+=1
            if count==2:
                native.hc.device.download=refuse_download
            return post(*args,**kwargs)
        if mode == "host_pressure":
            native.budget.acquire=refuse
        elif mode == "final_residual_download":
            native.hc.post=final_post
        else:
            native.moe.forward=fail_moe
        caught=None
        try:
            native.step(h,p,image_span)
        except (OSError,MemoryError,RuntimeError) as exc:
            caught=type(exc).__name__
        finally:
            native.budget.acquire,native.moe.forward,native.hc.post=acquire,moe,post
            native.hc.device.download,native.attention.kv.device.download=download,kv_download
        blocked=False
        if mode == "rollback_download":
            if native.attention.pending is None:
                raise RuntimeError("Whole-block recovery handle was lost")
            try:
                native.step(h,p,image_span)
            except RuntimeError:
                blocked=True
            if not blocked:
                raise RuntimeError("Whole block accepted work with uncommitted KV")
            native.recover()
        after=native.attention.kv.inspect()
        if (caught is None or native.position!=1 or after["position"]!=before["position"]
                or array_digest(after["cache_bf16"])!=array_digest(before["cache_bf16"])
                or native.attention.pending is not None or native.norm.device.allocations or native.sparse.device.allocations
                or sum(native.hc.device.allocations.values())!=native.hc.resident_bytes
                or sum(native.projections.device.allocations.values())!=native.projections.resident_bytes
                or sum(native.attention.kv.device.allocations.values())!=native.attention.kv.cache_bytes*3):
            raise RuntimeError("Failed block published partial model state or leaked temporary buffers")
        meta=block_meta("block/retry/"+mode,h,p,sequence=0,position=1,image_span=image_span)
        actual=native.step(h,p,image_span)
        checks=checker(meta,h,p,actual)
        status="pass" if all(c["status"]=="pass" for c in checks) else "fail"
        rows.append(block_record(meta,checks,actual))
        faults.append({"condition":mode,"caught":caught,"partial_output_returned":False,"committed_position_preserved":1,
                       "committed_cache_preserved":True,"temporary_buffers_released":True,
                       "calls_blocked_until_recovery":blocked,"retry_id":meta["id"],"status":status})
    return rows,faults


def expected_block_meta():
    pre=np.array([1.,0.,0.,0.],np.float32).reshape(1,1,4)
    def meta(identity,sequence,position,image_span,repeat=0):
        return {"id":identity,"kind":"block","sequence":sequence,"position":position,"image_span":image_span,"repeat":repeat,
                "token_id":PREFIX_SEQUENCES[sequence][position],"origin":"original_layer0_embedding_streams_and_initial_one_hot_mix",
                "incoming_pre_sha256":array_digest(pre)}
    for repeat in (0,1):
        for image_span in (False,True):
            for sequence in (0,1):
                for position in (0,1):
                    yield meta(f"block/repeat{repeat}/image{int(image_span)}/seq{sequence}/p{position}",sequence,position,image_span,repeat)
    for ordinal,mode in enumerate(FAULTS):
        yield meta("block/fault-prefill/"+mode,0,0,bool(ordinal%2))
        yield meta("block/retry/"+mode,0,1,bool(ordinal%2))


def block_stage_names(ids):
    from .attention_storage import STAGES
    from .hc import MIX_STAGES
    from .normalization import NORM_STAGES
    from .projections import PROJECTION_STAGES
    names={p+n for p in ("hc_attn_","hc_ffn_") for n in MIX_STAGES}
    names|={p+n for p in ("attn_pre_","ffn_pre_") for n in ("products_fp32","sum_fp32","output_bf16")}
    names|={p+n for p in ("attn_post_","ffn_post_") for n in ("post_product_fp32","residual_products_fp32","residual_sum_fp32","combined_fp32","output_bf16")}
    names|={p+n for p in ("attn_norm_","ffn_norm_") for n in NORM_STAGES}
    attention={p+n for p in ("wq_a_","wq_b_","wkv_","wo_b_") for n in PROJECTION_STAGES}
    attention|={p+n for p in ("q_norm_","kv_norm_") for n in NORM_STAGES}
    attention|={p+n for p in ("query_rope_","kv_rope_","inverse_rope_") for n in ("frequency_fp32","rotated_fp32","output_bf16")}
    attention|={"kv_"+n for n in ("rounded_bf16","codes_fp8","scales_e8m0","cache_bf16","window_bf16","indices_i32")}
    attention|={"attention_"+n for n in ("output_bf16",*STAGES)}
    attention|={"wo_a_"+n for n in ("input_fp8","scale_e8m0","rounded_bf16","output_bf16")}
    names|={"attention_"+n for n in attention}
    names|={"moe_gate."+n for n in ("input_promoted","logits","scores","selection","raw","denominator","weights","indices")}
    expert=("q","s","w1","w3","down_input","down_q","down_s","w2","clamped_gate","clamped_up","silu_fp32","product_fp32","routed_fp32")
    names|={f"moe_expert_{i}."+n for i in ids for n in expert}
    names|={"moe_shared."+n for n in expert}
    names|={f"moe_aggregation.after_expert_{i}_fp32" for i in ids}
    names|={"moe_aggregation."+n for n in ("before_shared_fp32","after_shared_fp32","output_bf16")}
    return names|{"output_bf16","next_pre_mix_fp32"}


def validate_goldens(report):
    from .attention_memory import validate_qualification
    from .checkpoint import MODEL,REVISION,REFERENCE_HASHES
    from .gate_reference import GATE_SOURCE_HASHES
    from .hc import HC_CUBIN_SHA256,MIX_STAGES
    from .hc_reference import HC_INSTRUMENTATION
    from .hc_validation import component_plan
    from .normalization_reference import PYTORCH_SOURCES
    from .prefix_reference import PREFIX_SOURCE_HASHES
    from .router import ROUTER_TORCH_REVISION
    profile=report.get("hc_profile",{})
    if (report.get("status")!="pass" or report.get("probe")!="native_layer0_block_parity" or report.get("model")!=MODEL or report.get("revision")!=REVISION
            or report.get("torch_git_version")!=ROUTER_TORCH_REVISION or profile.get("cubin_sha256")!=HC_CUBIN_SHA256
            or profile.get("cublas_version")!=120402 or profile.get("cublas_math_mode")!=0
            or profile.get("compiler",{}).get("nvrtc_version")!=[12,4]
            or profile.get("compiler",{}).get("concatenated_source_sha256")!="8ce8aa29cc89488ab1238894154554d04a5448d34e8674e0954078d271f4ec13"
            or any(report.get("reference",{}).get("files",{}).get(n)!=h for n,h in {**REFERENCE_HASHES,**PREFIX_SOURCE_HASHES}.items() if n in ("kernel.py","model.py","convert.py","config.json","generate.py"))
            or any(report.get("blas_reference",{}).get("files",{}).get(n,{}).get("sha256")!=h for n,h in GATE_SOURCE_HASHES.items())
            or any(report.get("reduction_reference",{}).get("files",{}).get(n,{}).get("sha256")!=h for n,(_,h) in PYTORCH_SOURCES.items())
            or report.get("hc_instrumentation")!=HC_INSTRUMENTATION):
        raise ValueError("Expected passing pinned original/native block and HC evidence")
    validate_qualification(report.get("attention_reference_qualification"))
    hc,blocks,failures=report.get("HC_cases",[]),report.get("block_cases",[]),report.get("faults",[])
    if len(hc)!=84 or len(blocks)!=26 or [f.get("condition") for f in failures]!=list(FAULTS):
        raise ValueError("HC/block/failure coverage is incomplete")
    if any(f.get("status")!="pass" or f.get("partial_output_returned") is not False or f.get("temporary_buffers_released") is not True
           or f.get("committed_cache_preserved") is not True or f.get("committed_position_preserved")!=1 for f in failures) or failures[-1].get("calls_blocked_until_recovery") is not True:
        raise ValueError("Whole-block refusal/recovery evidence differs")
    goldens=report.get("reference_goldens",[])
    indexed={g["meta"]["id"]:g for g in goldens}
    if len(goldens)!=110 or len(indexed)!=110 or len({r["id"] for r in hc+blocks})!=110:
        raise ValueError("HC/block golden identities are incomplete or duplicated")
    expected_meta={m["id"]:m for m,_ in component_plan()}
    for meta in expected_block_meta():
        golden=indexed.get(meta["id"],{}).get("meta",{})
        if {k:v for k,v in golden.items() if k!="input_bf16_sha256"}!=meta or not isinstance(golden.get("input_bf16_sha256"),str) or len(golden["input_bf16_sha256"])!=64:
            raise ValueError("Original embedding block input/mask/position identity differs")
        expected_meta[meta["id"]]=golden
    if set(indexed)!=set(expected_meta):
        raise ValueError("Original block or component input coverage differs")
    used=set()
    for row in hc+blocks:
        golden=indexed[row["id"]]
        meta=golden["meta"]
        if meta!=expected_meta[row["id"]] or {k:v for k,v in row.items() if k not in ("checks","status","selected_expert_ids","routing_weight_bits")}!=meta:
            raise ValueError("HC/block metadata differs from its golden")
        if row["kind"]=="hc":
            names=set(MIX_STAGES) if row["operation"]=="mixes" else (set(("products_fp32","sum_fp32","output_bf16")) if row["operation"]=="pre" else set(("post_product_fp32","residual_products_fp32","residual_sum_fp32","combined_fp32","output_bf16")))
        else:
            ids,bits=row.get("selected_expert_ids",[]),row.get("routing_weight_bits",[])
            if (len(ids)!=6 or len(set(ids))!=6 or any(type(i) is not int or not 0<=i<384 for i in ids)
                    or len(bits)!=6 or any(type(i) is not int or not 0<=i<=0xffffffff for i in bits)):
                raise ValueError("Original ordered route metadata differs")
            hashes=golden.get("stage_sha256",{})
            if (hashlib.sha256(np.array(ids,np.int32).tobytes()).hexdigest()!=hashes.get("moe_gate.indices")
                    or hashlib.sha256(np.array(bits,np.uint32).tobytes()).hexdigest()!=hashes.get("moe_gate.weights")):
                raise ValueError("Original route IDs/weights differ from exact reference bits")
            used.update("0:"+str(i) for i in ids)
            names=block_stage_names(ids)
        hashes=golden.get("stage_sha256",{})
        checks=row.get("checks",[])
        if (row.get("status")!="pass" or set(hashes)!=names or len(checks)!=len(names) or {c["check"] for c in checks}!=names
                or any(c.get("status")!="pass" or c.get("mismatches")!=0 or c.get("expected_sha256")!=c.get("actual_sha256")
                       or c.get("expected_sha256")!=hashes.get(c["check"]) for c in checks)
                or any(not isinstance(h,str) or len(h)!=64 or any(c not in "0123456789abcdef" for c in h) for h in hashes.values())):
            raise ValueError("HC/block golden boundaries or exact comparisons differ")
    if used!=set(report.get("used_original_experts",{})) or len(used)!=27:
        raise ValueError("Original expert provenance union differs from selected routes")
    for key,count in (("unchanged_Block_forward_controls",26),("unchanged_Attention_forward_controls",26),("unchanged_HC_controls",243)):
        controls=report.get(key,[])
        if len(controls)!=count or any(not row or any(c.get("status")!="pass" or c.get("mismatches")!=0 for c in row) for row in controls):
            raise ValueError("Unchanged original method controls are incomplete")
    negatives=report.get("negative_controls",[])
    if ([n.get("check") for n in negatives]!=["flatten_HC_mean_reduction_tree","replace_20_Sinkhorn_iterations_with_one","transpose_residual_comb_source_target","round_residual_products_to_BF16_early"]
            or any(n.get("mismatches",0)<=0 or n.get("status")!="fail" for n in negatives)):
        raise ValueError("HC arithmetic negative controls are missing")
    memory=report.get("memory",{})
    for key in ("HC_cleanup","normalization_cleanup","projection_cleanup","attention_cleanup","KV_cleanup"):
        record=memory.get(key,{})
        if record.get("device_owned_bytes")!=0 or record.get("host_reserved_bytes")!=0 or record.get("owned_context_active") is not False:
            raise ValueError("Owned native block resources were not released")
    if memory.get("block_cleanup",{}).get("host_reserved_bytes")!=0 or not memory.get("MoE_cleanup") or any(memory["MoE_cleanup"].values()):
        raise ValueError("Block/MoE cleanup is incomplete")
    return indexed
