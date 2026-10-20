"""CPU-only candidate-loader/builder contracts, never executable GPU evidence."""

import copy
from contextlib import ExitStack, nullcontext
import ctypes as C
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from vesper import spark_attention as spark


def rendered(value):
    return (json.dumps(value, sort_keys=True) + "\n").encode()


class SparkAttentionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / "bundle"
        self.directory.mkdir()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        # Nonexecutable schema fixtures deliberately replace official hashes.
        # This tests provenance binding/negative controls, not real codegen.
        self.original, self.license = b"CPU-only source fixture", b"CPU-only license fixture"
        self.hashes = {"kernel.py": spark.sha(self.original), "LICENSE": spark.sha(self.license)}
        self.stack.enter_context(patch.object(spark, "REFERENCE_HASHES", self.hashes))
        self.stack.enter_context(patch.object(spark, "variant_source", side_effect=self.variant))
        self.stack.enter_context(patch("vesper.resources.physical_memory", return_value={"available": 128 * spark.GiB}))
        self.manifest = self.bundle()

    def variant(self, original, storage, diagnostics):
        self.assertEqual(original, self.original)
        self.assertEqual(storage, "register-query")
        return f"CPU-only rendered variant {diagnostics}".encode(), {"query_storage_only_ast_check": "pass", "diagnostics": diagnostics}

    def bundle(self):
        (self.directory / "kernel.py").write_bytes(self.original)
        (self.directory / "DeepSeek-LICENSE").write_bytes(self.license)
        profiles = []
        for heads, dimension, diagnostics in spark.CANDIDATES:
            filename = spark.kernel_filename(heads, dimension, diagnostics)
            args = spark.argument_names(diagnostics)
            cuda = ("void " + spark.FUNCTION + "(" + ", ".join("int " + arg for arg in args) + ") {}\n").encode()
            shared = spark.CANDIDATES[heads, dimension, diagnostics][1]
            host = f"def call():\n    cuFuncSetAttribute(0, 8, {shared})\n".encode()
            binary = bytearray(64)
            binary[:6] = b"\x7fELF\x02\x01"
            binary[18:20] = (190).to_bytes(2, "little")
            binary[-8:] = bytes([heads // 16, dimension // 256, diagnostics, 0, 0, 0, 0, 0])
            (self.directory / filename).write_bytes(binary)
            (self.directory / (filename + ".cu")).write_bytes(cuda)
            (self.directory / (filename + ".host.py")).write_bytes(host)
            source, audit = self.variant(self.original, "register-query", diagnostics)
            (self.directory / (filename + ".py")).write_bytes(source)
            profiles.append({"heads": heads, "dimension": dimension, "diagnostics": diagnostics,
                "file": filename, "function": spark.FUNCTION, "argument_names": spark.argument_names(diagnostics),
                "dynamic_shared_bytes": shared, "generated_host_sha256": spark.sha(host),
                "binary_reproduction": True, "storage_transformation": audit,
                "generated_cuda_sha256": spark.sha(cuda), "cubin_sha256": spark.sha(binary)})
        manifest = {"schema_version": 1, "kind": "vesper_spark_attention_candidate",
            "model": spark.MODEL, "revision": spark.REVISION, "compute_capability": "12.1",
            "execution_target": "spark-sm121-candidate", "qualification": "unqualified_candidate",
            "minimum_driver_api": 13000, "nvrtc_version": [13, 0], "tilelang_version": "0.1.12",
            "kernel_source_sha256": self.hashes["kernel.py"], "original_kernel_parity_qualified": False,
            "profiles": profiles}
        self.save(manifest)
        return manifest

    def save(self, manifest):
        (self.directory / "manifest.json").write_bytes(rendered(manifest))

    def test_native_import_has_no_framework_side_effect(self):
        code = "import sys; import vesper.spark_attention; assert 'torch' not in sys.modules; assert 'tilelang' not in sys.modules"
        result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_all_six_profile_receipts_load_without_cuda_or_qualification(self):
        with patch.object(spark, "Driver", side_effect=AssertionError("CUDA must not load")):
            manifest, binaries, digest = spark.load_candidate(self.directory)
        self.assertEqual(set(binaries), set(spark.CANDIDATES))
        self.assertEqual(digest, spark.sha(rendered(self.manifest)))
        self.assertFalse(manifest["original_kernel_parity_qualified"])
        self.assertEqual(manifest["qualification"], "unqualified_candidate")

    def test_foreign_identity_toolchain_or_promoted_claim_refused_before_cuda(self):
        for field, value in (("model", "foreign/model"), ("revision", "0" * 40),
                ("compute_capability", "8.9"), ("nvrtc_version", [13, 4]),
                ("tilelang_version", "0.1.13"), ("execution_target", "pc-sm89"),
                ("original_kernel_parity_qualified", True), ("qualification", "qualified")):
            with self.subTest(field=field):
                self.save({**self.manifest, field: value})
                candidate = spark.SparkAttentionCandidate(self.directory)
                with patch.object(spark, "Driver") as driver:
                    with self.assertRaises(ValueError):
                        candidate.__enter__()
                    driver.assert_not_called()
                self.assertEqual(candidate.budget.reserved, 0)

    def test_missing_duplicate_abi_and_filename_profiles_refused(self):
        mutations = (lambda p: p.pop(), lambda p: p.__setitem__(1, p[0]),
            lambda p: p[0].update(file="../../other.cubin"),
            lambda p: p[0].update(argument_names=list(reversed(p[0]["argument_names"]))),
            lambda p: p[0].update(function="foreign_kernel"),
            lambda p: p[0].update(storage_transformation={}),
            lambda p: p[0].update(diagnostics=1),
            lambda p: p[0].update(binary_reproduction=False))
        for mutation in mutations:
            broken = copy.deepcopy(self.manifest)
            mutation(broken["profiles"])
            self.save(broken)
            with self.assertRaises(ValueError):
                spark.load_candidate(self.directory)

    def test_payload_and_rendered_python_tamper_are_detected(self):
        first = self.manifest["profiles"][0]["file"]
        for name in ("kernel.py", "DeepSeek-LICENSE", first, first + ".cu", first + ".py", first + ".host.py"):
            path = self.directory / name
            original = path.read_bytes()
            path.write_bytes(original + b"tamper")
            with self.assertRaises(ValueError):
                spark.load_candidate(self.directory)
            path.write_bytes(original)

    def test_dynamic_shared_metadata_mismatch_refused_even_with_unchanged_payload_hashes(self):
        self.manifest["profiles"][0]["dynamic_shared_bytes"] = 1
        self.save(self.manifest)
        with self.assertRaisesRegex(ValueError, "shared-memory contract"):
            spark.load_candidate(self.directory)

    def test_rehashed_generated_cuda_abi_mismatch_refused_before_cuda(self):
        first = self.manifest["profiles"][0]
        path = self.directory / (first["file"] + ".cu")
        cuda = path.read_bytes().replace(b"int attn_sink", b"int wrong_argument")
        path.write_bytes(cuda)
        first["generated_cuda_sha256"] = spark.sha(cuda)
        self.save(self.manifest)
        with self.assertRaisesRegex(ValueError, "launch signature"):
            spark.load_candidate(self.directory)

    def test_rehashed_foreign_elf_machine_still_refused(self):
        first = self.manifest["profiles"][0]
        path = self.directory / first["file"]
        binary = bytearray(path.read_bytes())
        binary[18:20] = (183).to_bytes(2, "little")
        path.write_bytes(binary)
        first["cubin_sha256"] = spark.sha(binary)
        self.save(self.manifest)
        with self.assertRaisesRegex(ValueError, "ELF CUDA"):
            spark.load_candidate(self.directory)

    def builder_context(self, *, argument_mismatch=False, binary_mismatch=False):
        stack = ExitStack()
        self.addCleanup(stack.close)
        root = Path(self.temp.name)
        workspace = Mock(root=root)
        workspace.context.return_value = {"platform": "dgx-spark"}
        def artifact(relative, create_parent=False):
            path = root / relative
            if create_parent:
                path.parent.mkdir(parents=True, exist_ok=True)
            return path
        workspace.artifact_path.side_effect = artifact
        info = {"ordinal": 0, "name": "NVIDIA GB10", "compute_capability": "12.1"}
        driver = Mock(inventory=Mock(return_value={"driver_api_version": 13000, "devices": [info]}))
        driver.check.side_effect = lambda status: None if status == 0 else (_ for _ in ()).throw(ValueError("status"))
        def query(pointer, attribute, ordinal):
            C.cast(pointer, C.POINTER(C.c_int)).contents.value = 128 * 1024
            return 0
        stack.enter_context(patch.object(spark, "Driver", return_value=driver))
        stack.enter_context(patch.object(spark, "_bind", return_value=query))
        stack.enter_context(patch("vesper.execution_target.platform.system", return_value="Linux"))
        stack.enter_context(patch("vesper.execution_target.platform.machine", return_value="aarch64"))
        torch = ModuleType("torch")
        torch.cuda = SimpleNamespace(device=lambda *_: nullcontext())
        tilelang = ModuleType("tilelang")
        tilelang.__version__ = "0.1.12"
        nvrtc = ModuleType("tilelang.contrib.nvrtc")
        nvrtc.get_nvrtc_version = lambda: (13, 0)
        binary = b"CPU nonexecutable compiled fixture"
        nvrtc.compile_cuda = Mock(return_value=binary + b"different" if binary_mismatch else binary)
        env = ModuleType("tilelang.env")
        env.TILELANG_TEMPLATE_PATH = env.CUTLASS_INCLUDE_DIR = env.CUDA_HOME = "/CPU-fixture"
        stack.enter_context(patch.dict(sys.modules, {"torch": torch, "tilelang": tilelang,
            "tilelang.contrib": ModuleType("tilelang.contrib"), "tilelang.contrib.nvrtc": nvrtc, "tilelang.env": env}))
        def fetch(directory):
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "kernel.py").write_bytes(self.original)
            (directory / "LICENSE").write_bytes(self.license)
        stack.enter_context(patch("vesper.checkpoint.fetch_reference", side_effect=fetch))
        cached = root / "CPU-fixture.cubin"
        cached.write_bytes(binary)
        def load_variant(raw, directory, storage, diagnostics):
            def factory(heads, dimension, scale):
                args = spark.argument_names(diagnostics)
                if argument_mismatch:
                    args = list(reversed(args))
                source = "void " + spark.FUNCTION + "(" + ", ".join("int " + arg for arg in args) + ") {}"
                return SimpleNamespace(get_kernel_source=lambda: source,
                    adapter=SimpleNamespace(lib_generator=SimpleNamespace(compile_flags=[], libpath=cached)))
            return factory, self.variant(raw, storage, diagnostics)[1]
        stack.enter_context(patch("vesper.attention_storage.load_variant", side_effect=load_variant))
        stack.enter_context(patch("vesper.attention_storage_reference._host_source", return_value="def call():\n    cuFuncSetAttribute(0, 8, 43008)\n"))
        self.loader = stack.enter_context(patch.object(spark, "load_candidate"))
        return workspace, nvrtc

    def test_builder_generated_abi_mismatch_refused_before_compilation(self):
        workspace, nvrtc = self.builder_context(argument_mismatch=True)
        with self.assertRaisesRegex(ValueError, "ABI differs"):
            spark.build_candidate(workspace, "new-build")
        nvrtc.compile_cuda.assert_not_called()
        self.loader.assert_not_called()
        self.assertFalse((Path(self.temp.name) / "new-build/manifest.json").exists())

    def test_builder_unreproduced_binary_refused_without_manifest_publication(self):
        workspace, nvrtc = self.builder_context(binary_mismatch=True)
        with self.assertRaisesRegex(ValueError, "reproduction failed"):
            spark.build_candidate(workspace, "new-build")
        nvrtc.compile_cuda.assert_called_once()
        self.loader.assert_not_called()
        self.assertFalse((Path(self.temp.name) / "new-build/manifest.json").exists())

    def test_builder_shared_memory_excess_refused_before_compile(self):
        workspace, nvrtc = self.builder_context()
        with patch("vesper.attention_storage_reference._host_source", return_value="def call():\n    cuFuncSetAttribute(0, 8, 150000)\n"):
            with self.assertRaisesRegex(ValueError, "shared-memory capacity"):
                spark.build_candidate(workspace, "new-build")
        nvrtc.compile_cuda.assert_not_called()
        self.assertFalse((Path(self.temp.name) / "new-build/manifest.json").exists())

    def test_builder_memory_admission_refusal_precedes_cuda_initialization(self):
        workspace = Mock()
        workspace.context.return_value = {"platform": "dgx-spark"}
        with patch("vesper.resources.physical_memory", return_value={"available": 33 * spark.GiB}), \
                patch.object(spark, "Driver") as driver:
            with self.assertRaises(MemoryError):
                spark.build_candidate(workspace, "new-build")
        driver.assert_not_called()

    def test_builder_all_six_profiles_remain_explicitly_unqualified(self):
        workspace, nvrtc = self.builder_context()
        manifest = spark.build_candidate(workspace, "new-build")
        self.assertEqual(len(manifest["profiles"]), 6)
        self.assertEqual(nvrtc.compile_cuda.call_count, 6)
        for call in nvrtc.compile_cuda.call_args_list:
            self.assertEqual(call.args[1:3], ("cubin", 121))
        self.assertFalse(manifest["original_kernel_parity_qualified"])
        self.assertEqual(manifest["qualification"], "unqualified_candidate")
        self.loader.assert_called_once()


if __name__ == "__main__":
    unittest.main()
