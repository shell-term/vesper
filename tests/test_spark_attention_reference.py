"""CPU policy/transform tests; synthetic inputs issue no GPU qualification."""

import copy
import hashlib
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from vesper import attention_memory as pc_memory
from vesper import attention_ptx as pc_ptx
from vesper import spark_attention_reference as spark
from vesper.attention_storage import STAGES


SOURCE_SHA = "9488e9f8b61b841ab752fc5238f4dbf1955f0909fc2f60519889327d0352104e"
PROFILE = {"original_generated_cuda_sha256": SOURCE_SHA, "heads": 64, "dimension": 512,
           "diagnostics": False, "original_shared_bytes": 141312,
           "query_output_lifetime_alias": "pass",
           "kernel_argument_names": ["attn_sink", "kv", "o", "q", "topk_idxs", "b", "m", "n", "topk", "q_workspace"]}
PARAMETERS = ",\n".join(f" .param .{'u64' if i < 5 else 'u32'} p_{i}" for i in range(9))
PTX = '''.version 9.0
.target sm_121a
.address_size 64
.extern .shared .align 1024 .b8 buf_dyn_shmem[];
.visible .entry sparse_attn_kernel__kernel(
''' + PARAMETERS + '''
)
.maxntid 256,1,1
{
\t.reg .pred %p<2>;
\t.reg .b16 %rs<2>;
\t.reg .b32 %r<8>;
\t.reg .b32 %f<4>;
\t.reg .b64 %rd<4>;
 mov.u32 %r0, buf_dyn_shmem;
$loop:
 st.shared.v4.u32 [%r0], {%r1, %r2, %r3, %r4};
 bar.sync 0;
 ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%r1, %r2, %r3, %r4}, [%r0];
 mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%f0,%f1,%f2,%f3}, {%r1,%r2,%r3,%r4}, {%r5,%r6}, {%f0,%f1,%f2,%f3};
 mul.f32 %f0, %f1, %f2;
 @%p0 bra $loop;
 div.rn.f32 %f0, %f1, %f2;
''' + "\n".join(" stmatrix.sync.aligned.x4.m8n8.shared.b16 [%r0], {%r1,%r2,%r3,%r4};" for _ in range(16)) + "\n" + \
    "\n".join(" ld.shared.v4.u32 {%r1,%r2,%r3,%r4}, [%rd0];" for _ in range(16)) + '''
 ret;
}
'''


def passing_report():
    def check(name):
        return {"check": name, "status": "pass", "mismatches": 0, "elements": 8,
                "expected_sha256": "a" * 64, "actual_sha256": "a" * 64}
    report = {"probe": spark.PROBE, "status": "pass", "model": spark.checkpoint.MODEL,
              "revision": spark.checkpoint.REVISION, "official_arithmetic_memory_adapter_gate": "pass",
              "negative_control_detected": True, "tilelang_version": "0.1.12", "nvrtc_version": [13, 0],
              "reference": {"files": {"kernel.py": spark.checkpoint.REFERENCE_HASHES["kernel.py"]}},
              "device": {"compute_capability": "12.1", "name": "NVIDIA GB10"},
              "driver_api_version": 13000, "execution_target": spark.SPARK_TARGET,
              "torch_version": "2.14.1+cu130", "torch_git_version": "5c4886908584029761b579af026dcfb627c84070",
              "torch_cuda_version": "13.0", "independent_Spark_qualification": True,
              "negative_control": {"status": "fail", "mismatches": 1}, "profiles": [], "cases": []}
    for digest, (h, d, diag, required) in spark.CODEGEN_PROFILES.items():
        report["profiles"].append({"original_generated_cuda_sha256": digest, "heads": h,
            "dimension": d, "diagnostics": diag, "original_shared_bytes": required,
            "original_binary_reproduction": True, "query_slab_inspection": "pass",
            "query_output_lifetime_alias": "pass",
            "original_cubin_sha256": spark.CUBIN_PROFILES[h, d, diag],
            "ptx_audit": {"inverse_ptx_recovery": "pass", "floating_point_instructions_unchanged": True,
                          "control_instructions_unchanged": True, "ptx_version": "9.0", "ptx_target": "sm_121a",
                          "query_output_lifetime_alias": "pass", "original_ptx_sha256": spark.PTX_PROFILES[h, d, diag],
                          "output_shared_alias_sites": 2 * (h * d * 2 // (256 * 16))}})
    for name in ("load_checks", "store_checks"):
        report[name] = [{**check(name), "pattern": i} for i in range(4)]
    common = {"PTX_instrumentation_BF16", "register_instrumentation_BF16", "PTX_original_arithmetic_vs_register_BF16"}
    common |= {"PTX_vs_register_" + stage for stage in STAGES}
    original = {"unmodified_instrumentation_BF16", "unmodified_vs_PTX_adapter_BF16"}
    original |= {"unmodified_vs_PTX_" + stage for stage in STAGES}
    for h, d in ((64, 256), (16, 512), (64, 512)):
        for i in range(12):
            names = common if (h, d) == (64, 512) else common | original
            report["cases"].append({"heads": h, "dimension": d, "case": i,
                                    "checks": [check(name) for name in sorted(names)]})
    return report


class SparkAttentionReferenceTests(unittest.TestCase):
    def test_spark_profile_set_is_distinct_and_exactly_six(self):
        self.assertEqual(len(spark.CODEGEN_PROFILES), 6)
        self.assertFalse(set(spark.CODEGEN_PROFILES) & set(pc_memory.CODEGEN_PROFILES))
        self.assertEqual(set(spark.CUBIN_PROFILES), {(h, d, trace) for h, d in ((64, 256), (16, 512), (64, 512))
                                                  for trace in (False, True)})

    def test_spark_ptx_changes_only_integer_query_transport(self):
        with patch.dict(spark.PTX_PROFILES, {(64, 512, False): hashlib.sha256(PTX.encode()).hexdigest()}):
            adapted, audit = spark.virtualize_ptx(PTX, PROFILE)
        self.assertEqual(pc_ptx.arithmetic_lines(adapted), pc_ptx.arithmetic_lines(PTX))
        self.assertEqual(pc_ptx.control_lines(adapted), pc_ptx.control_lines(PTX))
        self.assertEqual(audit["inverse_ptx_recovery"], "pass")
        self.assertTrue(audit["floating_point_instructions_unchanged"])
        self.assertEqual(audit["adapted_shared_access_sites"], 34)
        self.assertEqual(audit["output_shared_alias_sites"], 32)
        self.assertIn("ld.global.cg.u32", adapted)
        self.assertIn("st.global.v4.u32", adapted)
        self.assertIn("selp.u64", adapted)
        self.assertIn(".target sm_121a", adapted)

    def test_pc_public_ptx_entry_still_rejects_spark(self):
        with self.assertRaises(ValueError):
            pc_ptx.virtualize_ptx(PTX, PROFILE)
        pc_profile = {**PROFILE, "original_generated_cuda_sha256":
                      "d8f6dbef7f2ddea62ce38d2956f3b9699f71f607425a8bd4d77aff9318a4d1ba"}
        with self.assertRaises(ValueError):
            pc_ptx.virtualize_ptx(PTX, pc_profile)

    def test_foreign_target_version_profile_and_memory_access_refused(self):
        for text in (PTX.replace(".target sm_121a", ".target sm_121"),
                     PTX.replace(".version 9.0", ".version 9.4"),
                     PTX.replace("st.shared.v4.u32", "@%p0 st.shared.v4.u32"),
                     PTX.replace("st.shared.v4.u32", "st.shared.v4.u16")):
            with self.subTest(text=text[:30]), patch.dict(spark.PTX_PROFILES,
                    {(64, 512, False): hashlib.sha256(text.encode()).hexdigest()}), self.assertRaises(ValueError):
                spark.virtualize_ptx(text, PROFILE)
        with self.assertRaises(ValueError):
            spark.virtualize_ptx(PTX, {**PROFILE, "original_shared_bytes": 100000})

    def test_whole_ptx_pin_and_output_lifetime_coverage_fail_closed(self):
        with self.assertRaises(ValueError):
            spark.virtualize_ptx(PTX, PROFILE)  # Synthetic bytes are not the real PTX pin.
        with patch.dict(spark.PTX_PROFILES, {(64, 512, False): hashlib.sha256(PTX.encode()).hexdigest()}):
            with self.assertRaises(ValueError):
                spark.virtualize_ptx(PTX, {**PROFILE, "query_output_lifetime_alias": "not_checked"})
        for text in (PTX.replace(" div.rn.f32", " mul.f32"),
                     PTX.replace(" ret;", " ld.shared.f32 %f0, [%rd0];\n ret;"),
                     PTX.replace("stmatrix.sync.aligned.x4", "stmatrix.sync.aligned.x1")):
            with patch.dict(spark.PTX_PROFILES, {(64, 512, False): hashlib.sha256(text.encode()).hexdigest()}):
                with self.assertRaises(ValueError):
                    spark.virtualize_ptx(text, PROFILE)

    def test_64bit_shared_pointer_and_signed_offset_keep_integer_only_address_math(self):
        text = PTX.replace(" mul.f32", " ld.shared.f32 %f0, [%rd0+-1024];\n mul.f32")
        with patch.dict(spark.PTX_PROFILES, {(64, 512, False): hashlib.sha256(text.encode()).hexdigest()}):
            adapted, _ = spark.virtualize_ptx(text, PROFILE)
        self.assertIn("add.u64 %vesper64_5, %rd0, -1024;", adapted)
        self.assertEqual(pc_ptx.arithmetic_lines(text), pc_ptx.arithmetic_lines(adapted))

    def test_source_output_alias_must_be_after_the_query_loop(self):
        params = "float* attn_sink, void* kv, void* o, void* q, int* topk_idxs, int b, int m, int n, int topk"
        declarations = "\n".join(f"  void* {name} = ((void*)((char*)buf_dyn_shmem + {offset}));" for name, offset in
            (("o_shared", 0), ("q_shared", 0), ("kv_shared", 65536), ("acc_s_cast", 131072),
             ("workspace", 139264), ("workspace_1", 140288)))
        source = (f'extern "C" void {spark.FUNCTION}({params});\nvoid {spark.FUNCTION}({params}) {{\n'
            ' extern __shared__ unsigned char buf_dyn_shmem[];\n' + declarations + '\n'
            '  copy(q_shared);\n  for (int t = 0; t < topk; ++t) {\n'
            '  tl::ptx_ldmatrix_x4(q_shared, registers);\n  }\n'
            '  store(o_shared);\n  load(o_shared);\n}\n')
        digest = hashlib.sha256(source.encode()).hexdigest()
        with patch.dict(spark.CODEGEN_PROFILES, {digest: (64, 512, False, 141312)}):
            self.assertEqual(spark.inspect_codegen(source)["query_output_lifetime_alias"], "pass")
        changed = source.replace('  }\n  store(o_shared);', '  store(o_shared);\n  }')
        digest = hashlib.sha256(changed.encode()).hexdigest()
        with patch.dict(spark.CODEGEN_PROFILES, {digest: (64, 512, False, 141312)}):
            with self.assertRaises(ValueError):
                spark.inspect_codegen(changed)

    def test_report_needs_new_spark_execution_identity_and_full_coverage(self):
        report = passing_report()
        self.assertIs(spark.validate_qualification(report), report)
        mutations = [lambda r: r.update(execution_target="pc-sm89"),
                     lambda r: r.update(nvrtc_version=[13, 4]),
                     lambda r: r.update(torch_git_version="0" * 40),
                     lambda r: r.update(independent_Spark_qualification=False),
                     lambda r: r["profiles"].pop(), lambda r: r["cases"].pop(),
                     lambda r: r["profiles"][0]["ptx_audit"].update(ptx_target="sm_89"),
                     lambda r: r["profiles"][0].update(original_cubin_sha256="0" * 64),
                     lambda r: r["load_checks"][0].update(mismatches=1),
                     lambda r: r["store_checks"].pop(),
                     lambda r: r["negative_control"].update(status="pass", mismatches=0),
                     lambda r: r["cases"][24]["checks"].pop()]
        for mutate in mutations:
            modified = copy.deepcopy(report)
            mutate(modified)
            with self.assertRaises(ValueError):
                spark.validate_qualification(modified)

    def test_pc_validator_rejects_spark_receipt(self):
        with self.assertRaises(ValueError):
            pc_memory.validate_qualification(passing_report())

    def test_unknown_generated_source_is_rejected_without_any_edits(self):
        with self.assertRaises(ValueError):
            spark.inspect_codegen("changed original math")
        with self.assertRaises(ValueError):
            spark.loader_check_cuda("changed original math")

    def workspace(self):
        workspace = Mock()
        workspace.context.return_value = {"platform": "dgx-spark", "profile": "native",
            "model": spark.checkpoint.MODEL, "revision": spark.checkpoint.REVISION}
        workspace.artifact_path.side_effect = lambda relative: Path.cwd() / "artifacts" / "dgx-spark" / relative
        return workspace

    def test_foreign_workspace_or_unsafe_directory_refused_before_cuda(self):
        workspace = self.workspace()
        workspace.context.return_value["revision"] = "0" * 40
        with patch.object(spark, "Driver") as driver:
            with self.assertRaises(ValueError):
                spark.SparkAttentionMemoryReference(Path.cwd() / "artifacts/dgx-spark/reference", workspace=workspace)
            driver.assert_not_called()
        workspace = self.workspace()
        with self.assertRaises(ValueError):
            spark.SparkAttentionMemoryReference(Path.cwd() / "artifacts/pc-heterogeneous/reference", workspace=workspace)

    def test_physical_admission_refuses_before_cuda_or_optional_frameworks(self):
        workspace = self.workspace()
        reference = spark.SparkAttentionMemoryReference(Path.cwd() / "artifacts/dgx-spark/reference", workspace=workspace)
        memory = {"physical_available_bytes": 32 * GiB + MiB, "physical_total_bytes": 128 * GiB,
                  "cgroup_headroom_bytes": None}
        with patch.object(spark.platform, "system", return_value="Linux"), \
                patch.object(spark.platform, "machine", return_value="aarch64"), \
                patch.object(spark, "host_memory", return_value=memory), patch.object(spark, "Driver") as driver:
            with self.assertRaises((MemoryError, RuntimeError)):
                reference._open()
            driver.assert_not_called()

    def test_foreign_os_is_refused_before_memory_or_cuda_preflight(self):
        workspace = self.workspace()
        reference = spark.SparkAttentionMemoryReference(Path.cwd() / "artifacts/dgx-spark/reference", workspace=workspace)
        with patch.object(spark.platform, "system", return_value="Windows"), \
                patch.object(spark, "host_memory") as memory, patch.object(spark, "Driver") as driver:
            with self.assertRaisesRegex(ValueError, "Linux AArch64"):
                reference._open()
            memory.assert_not_called()
            driver.assert_not_called()

    def test_qualification_fails_closed_on_wrong_receipt_after_common_checks(self):
        workspace = self.workspace()
        reference = spark.SparkAttentionMemoryReference(Path.cwd() / "artifacts/dgx-spark/reference", workspace=workspace)
        reference.initial_context, reference.framework_profile = {}, {}
        reference._checkpoint = Mock()
        report = passing_report()
        report["profiles"][0]["ptx_audit"]["floating_point_instructions_unchanged"] = False
        with patch.object(spark.AttentionMemoryReference, "qualify", return_value=report):
            with self.assertRaises(ValueError):
                reference.qualify()
        self.assertFalse(reference.qualified)

    def test_real_model_capture_remains_locked_before_qualification(self):
        workspace = self.workspace()
        reference = spark.SparkAttentionMemoryReference(Path.cwd() / "artifacts/dgx-spark/reference", workspace=workspace)
        with self.assertRaisesRegex(RuntimeError, "not qualified"):
            reference(None, None, None, None, 512**-.5)


from vesper.resources import GiB, MiB


if __name__ == "__main__":
    unittest.main()
