import hashlib
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from vesper.attention_memory import _edit, inspect_codegen, validate_qualification
from vesper.attention_ptx import arithmetic_lines, control_lines, global_matrix_load, virtualize_ptx


PROFILE = {"original_generated_cuda_sha256": "d8f6dbef7f2ddea62ce38d2956f3b9699f71f607425a8bd4d77aff9318a4d1ba",
           "heads": 64, "dimension": 512, "diagnostics": False, "original_shared_bytes": 141312,
           "kernel_argument_names": ["attn_sink", "kv", "o", "q", "topk_idxs", "b", "m", "n", "topk", "q_workspace"]}
PARAMETERS = ",\n".join(f" .param .{'u64' if i < 5 else 'u32'} sparse_attn_kernel__kernel_param_{i}" for i in range(9))
PTX = '''.version 9.4
.target sm_89
.address_size 64
.extern .shared .align 1024 .b8 buf_dyn_shmem[];
.visible .entry sparse_attn_kernel__kernel(
''' + PARAMETERS + '''
)
.maxntid 256,1,1
{
\t.reg .pred %p<2>;
\t.reg .b16 %rs<2>;
\t.reg .f32 %f<4>;
\t.reg .b32 %r<8>;
\t.reg .b64 %rd<4>;
 mov.u32 %r0, buf_dyn_shmem;
 st.shared.v4.u32 [%r0], {%r1, %r2, %r3, %r4};
 bar.sync 0;
 ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%r1, %r2, %r3, %r4}, [%r0];
 mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%f0,%f1,%f2,%f3}, {%r1,%r2,%r3,%r4}, {%r5,%r6}, {%f0,%f1,%f2,%f3};
 mul.f32 %f0, %f1, %f2;
 @%p0 bra $loop;
$loop:
 ret;
}
'''


class AttentionMemoryTests(unittest.TestCase):
    def test_ptx_adapter_preserves_literal_arithmetic_registers_and_control(self):
        adapted, audit = virtualize_ptx(PTX, PROFILE)
        self.assertEqual(arithmetic_lines(adapted), arithmetic_lines(PTX))
        self.assertEqual(control_lines(adapted), control_lines(PTX))
        self.assertEqual(audit["adapted_shared_access_sites"], 2)
        self.assertEqual(audit["floating_point_instruction_count"], 2)
        self.assertTrue(audit["floating_point_instructions_unchanged"])
        self.assertTrue(audit["control_instructions_unchanged"])
        self.assertEqual(audit["original_ptx_sha256"], hashlib.sha256(PTX.encode()).hexdigest())
        self.assertIn("st.global.v4.u32", adapted)
        self.assertIn("shfl.sync.idx.b32", adapted)
        self.assertIn(".param .u64 vesper_qworkspace", adapted)

    def test_unknown_cuda_source_and_changed_profile_metadata_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unqualified generated"):
            inspect_codegen("float arithmetic_changed = 1;")
        for key, value in (("heads", 16), ("dimension", 256), ("original_shared_bytes", 65536),
                           ("original_generated_cuda_sha256", "0" * 64)):
            with self.assertRaises(ValueError):
                virtualize_ptx(PTX, {**PROFILE, key: value})

    def test_new_compiler_target_and_predicated_or_unknown_memory_ops_fail_closed(self):
        for changed in (PTX.replace(".version 9.4", ".version 9.3"), PTX.replace(".target sm_89", ".target sm_90"),
                        PTX.replace(" st.shared.v4.u32", " @%p0 st.shared.v4.u32"),
                        PTX.replace(" st.shared.v4.u32", " st.shared.v4.u16"),
                        PTX.replace("[%r0]", "[%rd0]")):
            with self.assertRaises(ValueError):
                virtualize_ptx(changed, PROFILE)

    def test_wrong_parameter_type_and_existing_adapter_namespace_are_rejected(self):
        for changed in (PTX.replace(".param .u32 sparse_attn_kernel__kernel_param_6", ".param .u64 sparse_attn_kernel__kernel_param_6"),
                        PTX.replace("%r0", "%vesper32_0")):
            with self.assertRaises(ValueError):
                virtualize_ptx(changed, PROFILE)

    def test_memory_edit_recovery_rejects_overlap(self):
        with self.assertRaisesRegex(ValueError, "overlapping"):
            _edit("abcdefgh", [(1, 5, "x", "first"), (3, 6, "y", "second")])
        self.assertEqual(_edit("abcdefgh", [(1, 3, "XYZ", "first"), (5, 7, "q", "second")]), "aXYZdeqh")

    def test_load_emitter_only_transports_integer_bits(self):
        code = global_matrix_load([f"r{i}" for i in range(4)], "pointer", [f"t{i}" for i in range(9)], [f"a{i}" for i in range(3)])
        self.assertEqual(code.count("ld.global.cg.u32"), 4)
        self.assertEqual(code.count("shfl.sync.idx.b32"), 8)
        self.assertEqual(arithmetic_lines(code), [])
        self.assertEqual(control_lines(code), [])
        with self.assertRaises(ValueError):
            global_matrix_load(["r0"], "pointer", [], [])

    def test_published_qualification_requires_all_profiles_checks_and_negative_control(self):
        path = Path(__file__).resolve().parents[1] / "results/2026-10-18-attention-memory/attention-memory.json"
        report = json.loads(path.read_bytes())
        self.assertIs(validate_qualification(report), report)
        mutations = (lambda r: r["profiles"][0].update(heads=16),
                     lambda r: r["profiles"][0]["ptx_audit"].update(floating_point_instructions_unchanged=False),
                     lambda r: r["profiles"][0].update(original_binary_reproduction=False),
                     lambda r: r["load_checks"][0].update(actual_sha256="0" * 64),
                     lambda r: r["store_checks"][0].update(mismatches=1),
                     lambda r: r["cases"].pop(),
                     lambda r: r["cases"][0]["checks"].pop(),
                     lambda r: r["negative_control"].update(mismatches=0))
        for change in mutations:
            broken = copy.deepcopy(report)
            change(broken)
            with self.assertRaises(ValueError):
                validate_qualification(broken)

    def test_reference_refuses_execution_before_qualification(self):
        from vesper.attention_memory_reference import AttentionMemoryReference
        reference = AttentionMemoryReference(Path("unused"))
        with self.assertRaisesRegex(RuntimeError, "not qualified"):
            reference(None, None, None, None, 512**-.5)
        reference.opened = True
        with self.assertRaisesRegex(RuntimeError, "already open"):
            reference.__enter__()
        reference.torch = object()
        with self.assertRaisesRegex(ValueError, "three-dimensional indices"):
            reference._run(64, 512, False, (SimpleNamespace(ndim=4), None, None, SimpleNamespace(ndim=2)))


if __name__ == "__main__":
    unittest.main()
