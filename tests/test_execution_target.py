"""CPU/mock checks for explicit targets; these do not qualify GB10 arithmetic."""

from contextlib import ExitStack
from copy import deepcopy
import hashlib
import importlib
import inspect
from unittest import TestCase, main
from unittest.mock import Mock, patch

from vesper.execution_target import (PC_TARGET, SPARK_TARGET, ExecutionTarget,
    compile_execution_kernel, kernel_source_sha256, resolve_execution_target)


BINARY = b"mock-not-a-real-CUBIN"
BINARY_HASH = hashlib.sha256(BINARY).hexdigest()


def inventory(cc="12.1", name="NVIDIA GB10", api=13000):
    return {"driver_api_version": api, "devices": [
        {"ordinal": 0, "name": name, "compute_capability": cc, "total_bytes": 128 * 1024**3}]}


def compiler_metadata(architecture="121", version=(13, 0), **flags):
    return {"nvrtc_version": list(version), "architecture": architecture,
        "concatenated_source_sha256": kernel_source_sha256(execution_target=SPARK_TARGET if architecture=='121' else PC_TARGET,**flags),
        "options": [f"--gpu-architecture=sm_{architecture}", "--std=c++17", "--fmad=false",
                    "--ftz=false", "--include-path=<CUDA>/include"], "include_path_redacted": True}


def mock_compile(architecture, *, compiler_info, **flags):
    version = (13, 0) if architecture == "121" else (12, 4)
    compiler_info.update(compiler_metadata(architecture, version, **flags))
    return BINARY


class Budget:
    def __init__(self, cap, **_):
        self.cap, self.reserved, self.peak = cap, 0, 0

    def acquire(self, count):
        self.reserved += count
        self.peak = max(self.peak, self.reserved)

    def release(self, count):
        self.reserved -= count


class Device:
    def __init__(self):
        self.context, self.allocations, self.peak = True, {}, 0
        self.load = Mock(return_value=1)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.context = False


class ExecutionTargetTests(TestCase):
    def test_default_and_explicit_pc_select_the_same_strict_profile(self):
        self.assertEqual(resolve_execution_target().name, PC_TARGET)
        self.assertEqual(resolve_execution_target(PC_TARGET).architecture, "89")
        self.assertFalse(resolve_execution_target().is_candidate)

    def test_unknown_or_implicit_boolean_targets_are_refused(self):
        for value in (True, False, 121, {}, "spark", "12.1", "pc-sm89-permissive"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_execution_target(value)

    def test_candidate_identity_never_claims_reference_qualification(self):
        target = resolve_execution_target(SPARK_TARGET)
        self.assertTrue(target.is_candidate)
        self.assertEqual(target.architecture, "121")
        first = target.describe()
        self.assertEqual(first["qualification"], "unqualified_candidate")
        self.assertEqual(first["spark_reference_qualification"], "not_run")
        self.assertFalse(first["qualification_receipt_issued"])
        first["qualification"] = "pass"
        self.assertEqual(target.describe()["qualification"], "unqualified_candidate")

    def test_pc_target_does_not_accept_a_spark_device(self):
        target = resolve_execution_target()
        with self.assertRaises(ValueError):
            target.validate_device(inventory()["devices"][0], inventory())
        self.assertEqual(target.validate_device(inventory("8.9")["devices"][0])["compute_capability"], "8.9")

    def test_spark_requires_actual_linux_arm_single_gb10_and_driver_profile(self):
        target = resolve_execution_target(SPARK_TARGET)
        with patch("vesper.execution_target.platform.system", return_value="Linux"), \
             patch("vesper.execution_target.platform.machine", return_value="aarch64"):
            good = inventory()
            evidence = target.validate_device(good["devices"][0], good)
            self.assertEqual(evidence["driver_api_version"], 13000)
            for bad in (inventory("8.9"), inventory(name="NVIDIA RTX 5090"), inventory(api=12090)):
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    target.validate_device(bad["devices"][0], bad)
            duplicate = inventory(); duplicate["devices"].append(deepcopy(duplicate["devices"][0]))
            with self.assertRaises(ValueError): target.validate_device(duplicate["devices"][0], duplicate)
            with self.assertRaises(ValueError): target.validate_device(good["devices"][0])
        with patch("vesper.execution_target.platform.system", return_value="Windows"), \
             patch("vesper.execution_target.platform.machine", return_value="aarch64"):
            with self.assertRaises(ValueError): target.validate_device(good["devices"][0], good)
        with patch("vesper.execution_target.platform.system", return_value="Linux"), \
             patch("vesper.execution_target.platform.machine", return_value="x86_64"):
            with self.assertRaises(ValueError): target.validate_device(good["devices"][0], good)

    def test_device_evidence_excludes_unknown_private_metadata(self):
        target = resolve_execution_target()
        info = {**inventory("8.9")["devices"][0], "uuid": "private-device-id", "path": "private/path"}
        evidence = target.validate_device(info)
        self.assertNotIn("uuid", evidence)
        self.assertNotIn("path", evidence)

    def test_pc_compiler_and_binary_gate_is_not_relaxed(self):
        metadata = {}
        result, evidence = compile_execution_kernel(mock_compile, compiler_info=metadata,
            pc_cubin_sha256=BINARY_HASH, normalization=True)
        self.assertEqual(result, BINARY)
        self.assertEqual(metadata["nvrtc_version"], [12, 4])
        self.assertEqual(evidence["execution_target"], PC_TARGET)
        with self.assertRaises(ValueError):
            compile_execution_kernel(mock_compile, compiler_info={}, pc_cubin_sha256="0" * 64, normalization=True)
        def wrong_version(*args, **kwargs):
            value = mock_compile(*args, **kwargs)
            kwargs["compiler_info"]["nvrtc_version"] = [13, 0]
            return value
        with self.assertRaises(ValueError):
            compile_execution_kernel(wrong_version, compiler_info={}, pc_cubin_sha256=BINARY_HASH, normalization=True)

    def test_candidate_compiles_its_actual_architecture_and_records_source_and_binary(self):
        compiler = Mock(side_effect=mock_compile)
        metadata = {}
        value, evidence = compile_execution_kernel(compiler, SPARK_TARGET, compiler_info=metadata,
            pc_cubin_sha256="0" * 64, engram=True)
        self.assertEqual(value, BINARY)
        self.assertEqual(compiler.call_args.args, ("121",))
        self.assertEqual(evidence["compiler"]["nvrtc_version"], [13, 0])
        self.assertEqual(evidence["compiler"]["concatenated_source_sha256"], kernel_source_sha256(execution_target=SPARK_TARGET,engram=True))
        self.assertEqual(evidence["cubin_sha256"], BINARY_HASH)
        self.assertEqual(evidence["qualification"], "unqualified_candidate")
        metadata["nvrtc_version"][0] = 99
        self.assertEqual(evidence["compiler"]["nvrtc_version"], [13, 0])

    def test_candidate_refuses_foreign_compiler_source_or_math_options(self):
        mutations = ({"nvrtc_version": [13, 1]}, {"architecture": "89"},
            {"concatenated_source_sha256": "0" * 64}, {"options": ["--use_fast_math"]},
            {"include_path_redacted": False})
        for mutation in mutations:
            def compiler(architecture, *, compiler_info, **flags):
                mock_compile(architecture, compiler_info=compiler_info, **flags)
                compiler_info.update(mutation)
                return BINARY
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                compile_execution_kernel(compiler, SPARK_TARGET, compiler_info={},
                    pc_cubin_sha256=BINARY_HASH, kv=True)

    def test_changed_source_during_compilation_is_refused(self):
        def compiler(architecture, *, compiler_info, **flags):
            compiler_info.update(compiler_metadata(architecture, **flags))
            compiler_info["concatenated_source_sha256"] = "a" * 64
            return BINARY
        with patch("vesper.execution_target.kernel_source_sha256", side_effect=["a" * 64, "b" * 64, "c" * 64]):
            with self.assertRaises(ValueError):
                compile_execution_kernel(compiler, SPARK_TARGET, compiler_info={}, pc_cubin_sha256=BINARY_HASH, kv=True)

    def test_unknown_compile_flags_and_invalid_binary_are_refused(self):
        compiler = Mock(side_effect=mock_compile)
        with self.assertRaises(ValueError):
            compile_execution_kernel(compiler, SPARK_TARGET, compiler_info={}, pc_cubin_sha256=BINARY_HASH, approximate=True)
        compiler.assert_not_called()
        with self.assertRaises(ValueError):
            compile_execution_kernel(lambda *_a, **_k: b"", compiler_info={}, pc_cubin_sha256=BINARY_HASH)

    def test_blas_keeps_pc_version_and_fp32_math_strict(self):
        pc = resolve_execution_target()
        self.assertEqual(pc.validate_blas(120402, 0)["version"], 120402)
        for version, mode in ((130101, 0), (130000, 0), (120402, 1), (120402, False), (True, 0)):
            with self.subTest(version=version, mode=mode), self.assertRaises(ValueError):
                pc.validate_blas(version, mode)

    def test_spark_blas_pins_observed_library_independently_of_nvrtc(self):
        spark = resolve_execution_target(SPARK_TARGET)
        for version, mode in ((120402, 0), (130000, 0), (130001, 0), (130100, 0),
                              (130102, 0), (130199, 0), (130200, 0), (140000, 0),
                              (130101, 1), (130101, 2), (130101, False),
                              (130101.0, 0), (True, 0)):
            with self.subTest(version=version, mode=mode), self.assertRaises(ValueError):
                spark.validate_blas(version, mode)
        result = spark.validate_blas(130101, 0)
        self.assertEqual(result["version"], 130101)
        self.assertEqual(result["math_mode"], 0)
        self.assertFalse(result["TF32_enabled"])
        self.assertEqual(result["qualification"], "unqualified_candidate")
        self.assertFalse(spark.describe()["qualification_receipt_issued"])

    def test_sampling_policy_preserves_pc_and_records_actual_spark_geometry(self):
        pc, spark = resolve_execution_target(), resolve_execution_target(SPARK_TARGET)
        self.assertEqual(pc.validate_sampling_policy(128, 1536)["multiprocessors"], 128)
        with self.assertRaises(ValueError): pc.validate_sampling_policy(48, 1536)
        result = spark.validate_sampling_policy(48, 1536)
        self.assertEqual(result["multiprocessors"], 48)
        self.assertEqual(result["qualification"], "unqualified_candidate")
        for values in ((0, 1536), (48, 0), (48, 255), (True, 1536)):
            with self.assertRaises(ValueError): spark.validate_sampling_policy(*values)

    def test_all_eight_operator_constructors_require_an_explicit_registered_target(self):
        specifications = (("normalization", "NativeNormRoPE", ()), ("kv", "NativeWindowKV", ()),
            ("hc", "NativeHC", ({},)), ("projections", "NativeProjections", ({},)),
            ("compression", "NativeCompression", ({},)), ("indexer", "NativeIndexer", ({},)),
            ("engram", "NativeEngram", ({}, None)), ("sampling", "NativeSampler", ()))
        for module, name, args in specifications:
            constructor = getattr(importlib.import_module("vesper." + module), name)
            with self.subTest(module=module):
                self.assertIsNone(inspect.signature(constructor).parameters["execution_target"].default)
                with self.assertRaises(ValueError): constructor(*args, execution_target="implicit-spark")

    def open_norm(self, *, target=SPARK_TARGET, metadata_mutation=None, cc="12.1"):
        from vesper import normalization
        stack = ExitStack(); self.addCleanup(stack.close)
        driver, device = Mock(), Device()
        driver.inventory.return_value = inventory(cc, "NVIDIA GB10" if cc == "12.1" else "NVIDIA RTX 4090")
        def compiler(architecture, *, compiler_info, **flags):
            value = mock_compile(architecture, compiler_info=compiler_info, **flags)
            if metadata_mutation: compiler_info.update(metadata_mutation)
            return value
        for name, replacement in (("Driver", Mock(return_value=driver)), ("Device", Mock(return_value=device)),
                                  ("MemoryBudget", Budget), ("compile_probe", compiler), ("NORM_CUBIN_SHA256", BINARY_HASH)):
            stack.enter_context(patch.object(normalization, name, replacement))
        stack.enter_context(patch("vesper.execution_target.platform.system", return_value="Linux"))
        stack.enter_context(patch("vesper.execution_target.platform.machine", return_value="aarch64"))
        return normalization.NativeNormRoPE(execution_target=target), device

    def test_real_operator_wiring_opens_only_explicit_candidate_and_keeps_provenance(self):
        operator, device = self.open_norm()
        with operator:
            self.assertTrue(operator.opened)
            self.assertEqual(operator.compiler["architecture"], "121")
            evidence = operator.stats()["execution"]
            self.assertEqual(evidence["qualification"], "unqualified_candidate")
            self.assertEqual(evidence["device"]["compute_capability"], "12.1")
            self.assertEqual(evidence["cubin_sha256"], BINARY_HASH)
            evidence["compiler"]["nvrtc_version"] = [99, 0]
            self.assertEqual(operator.stats()["execution"]["compiler"]["nvrtc_version"], [13, 0])
        self.assertFalse(device.context)
        self.assertEqual(operator.budget.reserved, 0)

    def test_default_pc_operator_still_rejects_spark_before_module_load(self):
        operator, device = self.open_norm(target=None)
        with self.assertRaises(ValueError): operator.__enter__()
        device.load.assert_not_called()
        self.assertEqual(operator.budget.reserved, 0)

    def test_wrong_candidate_compiler_closes_owned_device_before_any_module_load(self):
        operator, device = self.open_norm(metadata_mutation={"nvrtc_version": [13, 1]})
        with self.assertRaises(ValueError): operator.__enter__()
        device.load.assert_not_called()
        self.assertFalse(device.context)
        self.assertEqual(operator.budget.reserved, 0)

    def test_pc_operator_still_opens_with_its_original_pinned_gate(self):
        operator, device = self.open_norm(target=None, cc="8.9")
        with operator:
            self.assertEqual(operator.compiler["architecture"], "89")
            self.assertEqual(operator.compiler["nvrtc_version"], [12, 4])
            self.assertEqual(operator.stats()["execution"]["execution_target"], PC_TARGET)
        self.assertFalse(device.context)


if __name__ == "__main__":
    main()
