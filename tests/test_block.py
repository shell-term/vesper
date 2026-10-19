from contextlib import ExitStack
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from vesper.block import NativeBlock,validate_block_input
from vesper.block_validation import block_stage_names,expected_block_meta,validate_goldens
from vesper.hc import NativeHC,coefficients,hidden_shape
from vesper.hc_validation import component_plan
from vesper.resources import GiB,MiB


class LedgerDevice:
    free_floor=0
    def __init__(self):
        self.allocations={1:4096}
        self.calls=0
    def allocate(self,size):
        self.calls+=1
        pointer=max(self.allocations)+1
        self.allocations[pointer]=size
        return pointer
    def memory(self):
        return {"free":8*GiB}


class WindowContext:
    def __init__(self,*args):
        self.kv=SimpleNamespace(device=LedgerDevice())
        self.position=0
    def __enter__(self):
        return self
    def __exit__(self,*args):
        pass


def ledger_block():
    operators=[SimpleNamespace(device=LedgerDevice(),ordinal=0,opened=True,layer=0) for _ in range(4)]
    hc,norm,projection,sparse=operators
    moe=SimpleNamespace(stack=ExitStack(),config=SimpleNamespace(cache=SimpleNamespace(primary_device=0)),devices={"moe":LedgerDevice()},layer=0)
    with patch("vesper.block.NativeWindowAttention",WindowContext):
        return NativeBlock({},hc,norm,projection,sparse,moe)


class BlockTests(unittest.TestCase):
    def test_hidden_layout_and_finite_storage_contract(self):
        value=np.zeros((2,4,4,5120),np.uint16)
        self.assertEqual(hidden_shape(value),8)
        for bad in (value.astype(np.float32),value[...,::-1],np.zeros((1,9,4,5120),np.uint16),
                    np.zeros((1,1,3,5120),np.uint16),np.zeros((1,0,4,5120),np.uint16),np.full_like(value,0x7fc0)):
            with self.assertRaises(ValueError):
                hidden_shape(bad)

    def test_coefficients_and_missing_weights_fail_before_CUDA(self):
        coefficients(np.zeros((1,1,4),np.float32),(1,1,4))
        for bad in (None,np.zeros((4,),np.float32),np.zeros((1,1,4),np.float64),np.full((1,1,4),np.inf,np.float32)):
            with self.assertRaises(ValueError):
                coefficients(bad,(1,1,4))
        with patch("vesper.hc.Driver",side_effect=AssertionError("No CUDA access expected")):
            with self.assertRaises(ValueError):
                NativeHC({})

    def test_block_retains_single_token_mask_and_mixture_scope(self):
        hidden=np.zeros((1,1,4,5120),np.uint16)
        pre=np.zeros((1,1,4),np.float32)
        validate_block_input(hidden,pre,False)
        for h,p,image in ((hidden,pre,1),(np.zeros((1,2,4,5120),np.uint16),np.zeros((1,2,4),np.float32),False),
                           (hidden,np.zeros((1,1,4),np.uint16),False)):
            with self.assertRaises(ValueError):
                validate_block_input(h,p,image)

    def test_owned_GPU_cap_rejects_before_external_allocation_and_restores_binding(self):
        block=ledger_block()
        original=block.hc.device.allocate
        with patch("vesper.resources.physical_memory",return_value={"available":64*GiB}),block:
            calls=block.hc.device.calls
            with self.assertRaises(MemoryError):
                block.hc.device.allocate(512*MiB)
            self.assertEqual(block.hc.device.calls,calls)
            pointer=block.hc.device.allocate(32*MiB)
            self.assertEqual(block.hc.device.allocations[pointer],32*MiB)
            self.assertGreaterEqual(block.stats()["peak_owned_primary_device_bytes"],32*MiB)
        self.assertEqual(block.hc.device.allocate,original)
        self.assertEqual(block.budget.reserved,0)

    def test_duplicate_ledger_ownership_refuses_and_releases_reservation(self):
        block=ledger_block()
        block.norm.device=block.hc.device
        with patch("vesper.resources.physical_memory",return_value={"available":64*GiB}):
            with self.assertRaises(ValueError):
                block.__enter__()
        self.assertFalse(block.opened)
        self.assertEqual(block.budget.reserved,0)

    def test_HC_entry_respects_host_headroom_before_driver_creation(self):
        arrays={f"layers.0.hc_{part}_{suffix}":np.zeros(shape,np.float32) for part in ("attn","ffn")
                for suffix,shape in (("fn",(24,20480)),("base",(24,)),("scale",(3,)))}
        native=NativeHC(arrays)
        with patch("vesper.resources.physical_memory",return_value={"available":16*GiB}),patch("vesper.hc.Driver",side_effect=AssertionError("No CUDA access expected")):
            with self.assertRaises(MemoryError):
                native.__enter__()
        self.assertEqual(native.budget.reserved,0)

    def test_corpus_and_block_boundary_coverage_are_explicit(self):
        cases=list(component_plan())
        self.assertEqual(len(cases),84)
        self.assertEqual(len({m["id"] for m,_ in cases}),84)
        self.assertEqual({m["operation"] for m,_ in cases},{"mixes","pre","post"})
        self.assertEqual(len(list(expected_block_meta())),26)
        self.assertEqual(len(block_stage_names([0,1,2,3,4,5])),225)

    def test_published_pinned_source_and_complete_block_goldens(self):
        root=Path(__file__).resolve().parents[1]
        report=json.loads((root/"results/2026-10-18-layer0-block/block.json").read_bytes())
        self.assertEqual(len(validate_goldens(report)),110)
        source=b"\n".join((root/"src/vesper/kernels"/name).read_bytes() for name in ("format_probe.cu","normalization_probe.cu","hc_probe.cu"))
        self.assertEqual(hashlib.sha256(source.replace(b"\r\n",b"\n")).hexdigest(),report["hc_profile"]["compiler"]["concatenated_source_sha256"])
        for change in (lambda r:r["HC_cases"].pop(),lambda r:r["block_cases"].pop(),
                       lambda r:r["block_cases"][0]["selected_expert_ids"].reverse(),
                       lambda r:r["block_cases"][0]["routing_weight_bits"].__setitem__(0,0),
                       lambda r:r["hc_instrumentation"].update(arithmetic_AST_restoration=False),
                       lambda r:r["unchanged_Block_forward_controls"][0][0].update(mismatches=1),
                       lambda r:r["memory"]["HC_cleanup"].update(device_owned_bytes=1),
                       lambda r:r["faults"][-1].update(calls_blocked_until_recovery=False),
                       lambda r:r["negative_controls"].pop()):
            broken=copy.deepcopy(report)
            change(broken)
            with self.assertRaises(ValueError):
                validate_goldens(broken)


if __name__=="__main__":
    unittest.main()
