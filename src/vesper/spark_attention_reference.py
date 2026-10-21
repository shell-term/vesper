"""GB10-specific qualification of original Attention's Query-storage adapter.

The six generated CUDA sources and compiler profile are Spark observations,
not transferred PC receipts. Only integer address/bit transport is adapted in
the original PTX. The unchanged full 64x512 kernel remains a separate resource
gate when its 141312 shared bytes exceed the device's measured capacity.
"""

from __future__ import annotations

import ctypes as C
import hashlib
from pathlib import Path
import platform
import re

from . import checkpoint
from .attention_memory import (FUNCTION, _inspect_codegen_for, _loader_check_cuda_for,
                               _validate_qualification_for, _edit)
from .attention_memory_reference import AttentionMemoryReference
from .attention_ptx import arithmetic_lines, control_lines, global_matrix_load
from .attention_storage import load_variant
from .attention_storage_reference import _host_source
from .cuda import Device, Driver, _bind
from .execution_target import resolve_execution_target, SPARK_TARGET
from .resources import GiB, MiB, MemoryBudget
from .spark_probe import host_memory, require_capacity

PROBE = "spark_official_attention_PTX_memory_adapter"
CODEGEN_PROFILES = {
    "d110e09be04f9c2a94f0fb7f7d1e67fc90c19f131cb620c397466391e113e7b7": (64, 256, False, 75776),
    "c55528258c74b74aa3df4fb3cbe14068d921e86f4ecb329d2e4ec71ac3ee1cab": (64, 256, True, 75776),
    "6ab9e1a8a7ade6e8cf5a4d58a81bbdc0baaefba14fb73b9b77a47cbe3dfe8541": (16, 512, False, 86016),
    "6915d85ed87143666e3199c31d93d238968dbc6cbd657ddb65c122059eecacf1": (16, 512, True, 86016),
    "9488e9f8b61b841ab752fc5238f4dbf1955f0909fc2f60519889327d0352104e": (64, 512, False, 141312),
    "4e34de4d9f1c0094a3edb4b09614fb588bceb0bfa65be5eb0be13e613feda5e5": (64, 512, True, 141312),
}
CUBIN_PROFILES = {
    (64, 256, False): "59d50eeb3e3d1523407d5fff1216dec2599ff5b428fb424849faeb9a6f0a0fd4",
    (64, 256, True): "7272105e3d6bd69925ac88ca1785715b78b01fe6c938e2ff3a1679e421dde8ac",
    (16, 512, False): "01ed374cf4fc70d623ca7c69ac67f4f9bd95a685fd6f9a596b89305becd96197",
    (16, 512, True): "1addbd6458ed01288cbf2737309064266d249b5450f32cfdae0dfcf4b9de974d",
    (64, 512, False): "2a5fd8e898485b6cf6d4bca7167817943ef65248b9b2bdb60633a9d4f0036a2d",
    (64, 512, True): "77efeb25df32e4d072cc7d9eb35df42d629d8cc2b0ac3a521663c8b4117b1ee7",
}
PTX_PROFILES = {
    (64, 256, False): "f06345adae9626efd53782291ee7622586bdcf8e7ba4f53e29163e916b37179f",
    (64, 256, True): "5286d0be7457244fbd28e5a152e5695db0f8c0ddfb947673e9a05f98864a538e",
    (16, 512, False): "a5145cbfd739e63f93a85f5c6ab2123ac60ab493abfa27dfb7721c84f8fecf08",
    (16, 512, True): "57079c692e3d0f35ebd67ed664fe5e9c3c2bf0a0cc636b81171846ab38261ac3",
    (64, 512, False): "cd104e01c015a6d76d315f92395bd069cece5b7f18fc1a4fd4f7f75e8d53aec0",
    (64, 512, True): "191eb831eedc8e4121d38c0524b1fb388cdb5935b01ecacf47f93eaa061831d7",
}


def inspect_codegen(source):
    profile = _inspect_codegen_for(source, CODEGEN_PROFILES, output_alias=True)
    # The pinned Spark compiler reuses Query's slab for output, after all Query
    # reads. Preserve that lifetime alias instead of relocating output below 0.
    if source.count("o_shared") != 3 or "void* o_shared = ((void*)((char*)buf_dyn_shmem + 0));" not in source:
        raise ValueError("Spark output/Query shared lifetime alias differs")
    loop = source.index("  for (int t =")
    opened, depth, end = source.index("{", loop), 0, None
    for position in range(opened, len(source)):
        depth += (source[position] == "{") - (source[position] == "}")
        if depth == 0:
            end = position
            break
    uses = [match.start() for match in re.finditer("o_shared", source)]
    if end is None or not loop < source.rindex("q_shared") < end < uses[1] < uses[2]:
        raise ValueError("Spark Query/output lifetime boundary differs")
    if profile["adapted_shared_bytes"] < profile["workspace_bytes_per_CTA"]:
        raise ValueError("Spark output slab no longer fits the compacted shared allocation")
    profile["query_output_lifetime_alias"] = "pass"
    return profile


def virtualize_ptx(ptx, profile):
    """Original FP/control stays literal; Query and later output share a slab.

    Only Query lives in the added global buffer. The output epilogue reuses
    shared offset zero after all Query/KV matrix reads have finished. Exact PTX
    pins prevent this lifetime boundary from becoming a compiler heuristic.
    """
    key = tuple(profile.get(k) for k in ("heads", "dimension", "diagnostics"))
    identity = profile.get("original_generated_cuda_sha256")
    if (not isinstance(ptx, str) or len(ptx) > 2 * MiB or identity not in CODEGEN_PROFILES
            or CODEGEN_PROFILES[identity] != (*key, profile.get("original_shared_bytes"))
            or hashlib.sha256(ptx.encode()).hexdigest() != PTX_PROFILES.get(key)
            or profile.get("query_output_lifetime_alias") != "pass"):
        raise ValueError("Unregistered Spark generated CUDA/PTX/lifetime profile")
    if (re.findall(r"(?m)^\.version ([\d.]+)\s*$", ptx) != ["9.0"]
            or re.findall(r"(?m)^\.target ([\w]+)\s*$", ptx) != ["sm_121a"] or "%vesper" in ptx):
        raise ValueError("Foreign Spark PTX target/version/namespace")
    entry = re.search(rf"\.visible \.entry {FUNCTION}\((.*?)\n\)", ptx, re.S)
    if entry is None or ptx.count(".entry ") != 1:
        raise ValueError("Expected one pinned Spark Attention entry")
    parameters = re.findall(r"\.param \.(u64|u32)(?: \.ptr \.align 1)? (\w+)", entry[1])
    names = profile["kernel_argument_names"][:-1]
    if len(parameters) != len(names) or any(kind != ("u32" if name in ("b", "m", "n", "topk") else "u64")
                                           for (kind, _), name in zip(parameters, names)):
        raise ValueError("Spark PTX parameter layout differs")
    declarations = list(re.finditer(r"\t\.reg [^\n]+;", ptx))
    if len(declarations) != 5:
        raise ValueError("Spark PTX register declarations differ")
    labels = {m[1]: m.start() for m in re.finditer(r"(?m)^([\w$.]+):", ptx)}
    backedges = [m for m in re.finditer(r"(?m)^\s*(?:@!?%\w+\s+)?bra(?:\.uni)?\s+([\w$.]+);", ptx)
                 if labels.get(m[1], len(ptx)) < m.start()]
    if not backedges:
        raise ValueError("Missing original Spark Attention loop")
    tail = backedges[-1].end()
    if "ldmatrix." in ptx[tail:] or "div.rn.f32" not in ptx[tail:]:
        raise ValueError("Spark output epilogue lifetime boundary differs")
    q_bytes = key[0] * key[1] * 2
    expected_output_sites = q_bytes // (256 * 16)
    tail_ops = re.findall(r"(?:stmatrix\.[\w.]+|(?:ld|st)\.shared\.[\w.]+)", ptx[tail:])
    if (tail_ops.count("stmatrix.sync.aligned.x4.m8n8.shared.b16") != expected_output_sites
            or tail_ops.count("ld.shared.v4.u32") != expected_output_sites
            or len(tail_ops) != expected_output_sites * 2):
        raise ValueError("Spark output slab has unregistered memory uses")
    insertion = declarations[-1].end()
    m_parameter = parameters[names.index("m")][1]
    prolog = f'''
 .reg .b32 %vesper32_<16>;
 .reg .b64 %vesper64_<8>;
 .reg .pred %vesperp<2>;
 ld.param.u64 %vesper64_7, [vesper_qworkspace];
 ld.param.u32 %vesper32_0, [{m_parameter}];
 mov.u32 %vesper32_1, %ctaid.y;
 mov.u32 %vesper32_2, %ctaid.x;
 mad.lo.u32 %vesper32_0, %vesper32_1, %vesper32_0, %vesper32_2;
 mul.wide.u32 %vesper64_0, %vesper32_0, {q_bytes};
 add.u64 %vesper64_7, %vesper64_7, %vesper64_0;
 mov.u64 %vesper64_6, buf_dyn_shmem;
'''
    edits = [(entry.start(1), entry.end(1), entry[1] + ",\n\t.param .u64 vesper_qworkspace", "workspace_parameter"),
             (insertion - 1, insertion, ";" + prolog, "integer_registers_and_prolog")]
    sites, tail_sites = [], 0
    allowed = {"st.shared.v4.u32", "st.shared.f32", "ld.shared.f32", "ld.shared.v4.u32",
               "ldmatrix.sync.aligned.x4.m8n8.shared.b16", "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16",
               "ldmatrix.sync.aligned.x2.m8n8.shared.b16", "stmatrix.sync.aligned.x2.m8n8.shared.b16",
               "stmatrix.sync.aligned.x4.m8n8.shared.b16"}
    for line in re.finditer(r"^[ \t]+[^\n]+;", ptx, re.M):
        text = line[0]
        if ".shared" not in text or text.lstrip().startswith(".extern"):
            continue
        instruction = re.match(r"\s*((?:ldmatrix|stmatrix)\.[\w.]+|(?:ld|st)\.shared\.[\w.]+)\s+", text)
        address = re.search(r"\[(%(?:rd|r)\d+)(?:\+(-?\d+))?\]", text)
        if not instruction or not address or instruction[1] not in allowed:
            raise ValueError("Unsupported or predicated Spark PTX shared access")
        opcode, pointer = instruction[1], address[1]
        if pointer.startswith("%rd"):
            prefix = f"add.u64 %vesper64_5, {pointer}, {address[2] or '0'};\n"
        else:
            prefix = f"cvt.u64.u32 %vesper64_5, {pointer};\nadd.u64 %vesper64_5, %vesper64_5, {address[2] or '0'};\n"
        prefix += "sub.u64 %vesper64_4, %vesper64_5, %vesper64_6;\n"
        prefix += f"sub.u64 %vesper64_1, %vesper64_5, {q_bytes};\n"
        is_tail = line.start() > tail
        if is_tail:
            prefix += f"setp.lo.u64 %vesperp0, %vesper64_4, {q_bytes};\n"
            prefix += "selp.u64 %vesper64_1, %vesper64_5, %vesper64_1, %vesperp0;\n"
            tail_sites += 1
        if pointer.startswith("%rd"):
            relocated = text[:address.start()] + "[%vesper64_1]" + text[address.end():]
        else:
            prefix += "cvt.u32.u64 %vesper32_12, %vesper64_1;\n"
            relocated = text[:address.start()] + "[%vesper32_12]" + text[address.end():]
        query = not is_tail and opcode in ("st.shared.v4.u32", "ldmatrix.sync.aligned.x4.m8n8.shared.b16")
        if query:
            prefix += f"setp.lo.u64 %vesperp0, %vesper64_4, {q_bytes};\n"
            prefix += "add.u64 %vesper64_0, %vesper64_7, %vesper64_4;\n"
            replacement = prefix + "@!%vesperp0 " + relocated.strip() + "\n"
            if opcode == "st.shared.v4.u32":
                global_store = re.sub(r"\[[^]]+\]", "[%vesper64_0]", text.replace("st.shared", "st.global"))
                replacement += "@%vesperp0 " + global_store.strip()
            else:
                destinations = [value.strip() for value in re.search(r"\{([^}]+)\}", text)[1].split(",")]
                replacement += global_matrix_load(destinations, "%vesper64_0",
                    [f"%vesper32_{i}" for i in range(3, 12)], [f"%vesper64_{i}" for i in range(1, 4)], predicate="@%vesperp0")
        else:
            replacement = prefix + relocated
        edits.append((line.start(), line.end(), replacement, "integer_shared_memory_access"))
        sites.append(opcode)
    adapted = _edit(ptx, edits)
    if arithmetic_lines(adapted) != arithmetic_lines(ptx) or control_lines(adapted) != control_lines(ptx):
        raise ValueError("Spark adaptation changed literal floating-point/control instructions")
    return adapted, {"original_ptx_sha256": hashlib.sha256(ptx.encode()).hexdigest(),
        "adapted_ptx_sha256": hashlib.sha256(adapted.encode()).hexdigest(), "inverse_ptx_recovery": "pass",
        "floating_point_instructions_unchanged": True, "control_instructions_unchanged": True,
        "floating_point_instruction_count": len(arithmetic_lines(ptx)), "control_instruction_count": len(control_lines(ptx)),
        "adapted_shared_access_sites": len(sites), "output_shared_alias_sites": tail_sites,
        "query_output_lifetime_alias": "pass", "ptx_version": "9.0", "ptx_target": "sm_121a",
        "arithmetic_instruction_sha256": hashlib.sha256("\n".join(arithmetic_lines(ptx)).encode()).hexdigest()}


def loader_check_cuda(source):
    return _loader_check_cuda_for(source, CODEGEN_PROFILES)


def validate_qualification(report):
    _validate_qualification_for(report, profiles=CODEGEN_PROFILES, probe=PROBE,
                                nvrtc=(13, 0), capability="12.1")
    if (report.get("execution_target") != SPARK_TARGET or report.get("driver_api_version", 0) < 13000
            or report.get("torch_version") != "2.14.1+cu130"
            or report.get("torch_git_version") != "5c4886908584029761b579af026dcfb627c84070"
            or report.get("torch_cuda_version") != "13.0" or "GB10" not in report["device"].get("name", "")
            or report.get("independent_Spark_qualification") is not True):
        raise ValueError("Foreign Spark Attention execution/framework identity")
    for profile in report["profiles"]:
        key = tuple(profile[k] for k in ("heads", "dimension", "diagnostics"))
        audit = profile["ptx_audit"]
        if (profile.get("original_cubin_sha256") != CUBIN_PROFILES[key]
                or audit.get("original_ptx_sha256") != PTX_PROFILES[key]
                or audit.get("query_output_lifetime_alias") != "pass"
                or profile.get("query_output_lifetime_alias") != "pass"
                or audit.get("output_shared_alias_sites") != 2 * (key[0] * key[1] * 2 // (256 * 16))
                or audit.get("ptx_version") != "9.0" or audit.get("ptx_target") != "sm_121a"):
            raise ValueError("Spark Attention binary or PTX identity differs")
    return report


class SparkAttentionMemoryReference(AttentionMemoryReference):
    """Independent Spark oracle. Call qualify() before using real model inputs."""

    def __init__(self, directory, ordinal=0, *, workspace=None, profile="baseline"):
        super().__init__(directory, ordinal, profile=profile)
        if workspace is None:
            from .workspace import Workspace
            workspace = Workspace.open()
        self.workspace = workspace
        self.memory_observations = []
        self._guard()
        base = workspace.artifact_path("namespace-check").parent
        relative = Path(directory).absolute().relative_to(base.absolute()).as_posix()
        if workspace.artifact_path(relative).absolute() != Path(directory).absolute():
            raise ValueError("Spark Attention reference directory differs from its bound namespace")

    def _guard(self):
        self.workspace.check()
        context = self.workspace.context()
        if tuple(context[k] for k in ("platform", "profile", "model", "revision")) != (
                "dgx-spark", "native", checkpoint.MODEL, checkpoint.REVISION):
            raise ValueError("The pinned Spark/native workspace is required")
        self.workspace.check_runtime_module(__file__, "src/vesper/spark_attention_reference.py")
        from . import workspace as workspace_module
        self.workspace.check_runtime_module(workspace_module.__file__, "src/vesper/workspace.py")

    def _checkpoint(self, pending=0):
        self._guard()
        memory = host_memory()
        self.memory_observations.append(memory)
        require_capacity(memory, pending)

    def _open(self):
        if platform.system() != "Linux" or platform.machine().lower() not in ("aarch64", "arm64"):
            raise ValueError("Spark Attention reference requires Linux AArch64")
        self._checkpoint(3 * GiB)
        self.initial_context = self.workspace.context()
        self.budget = MemoryBudget(cap=3 * GiB, owner="spark_attention_reference")
        self.budget.acquire(self.budget.cap)
        self.stack.callback(self.budget.release, self.budget.cap)
        import torch
        import tilelang
        from tilelang.contrib.nvrtc import compile_cuda, get_nvrtc_version
        from tilelang.env import TILELANG_TEMPLATE_PATH, CUTLASS_INCLUDE_DIR, CUDA_HOME
        from .spark_operator_probe import _torch_profile
        if tilelang.__version__ != "0.1.12" or tuple(get_nvrtc_version()) != (13, 0):
            raise ValueError("Spark Attention requires TileLang 0.1.12 / NVRTC 13.0")
        self.torch, self.compile_cuda, self.tilelang = torch, compile_cuda, tilelang
        self.nvrtc_version = tuple(get_nvrtc_version())
        self.source = checkpoint.fetch_reference(self.directory)
        raw = (self.directory / "kernel.py").read_bytes()
        self.driver = Driver()
        inventory = self.driver.inventory()
        self.device_info = next(d for d in inventory["devices"] if d["ordinal"] == self.ordinal)
        resolve_execution_target(SPARK_TARGET).validate_device(self.device_info, inventory)
        self.framework_profile = _torch_profile(torch, {"name": self.device_info["name"]})
        self.driver_version = inventory["driver_api_version"]
        query = _bind(self.driver.lib, "cuDeviceGetAttribute", [C.POINTER(C.c_int), C.c_int, C.c_int])
        shared_limit = C.c_int()
        self.driver.check(query(C.byref(shared_limit), 97, self.ordinal))
        self.shared_limit = shared_limit.value
        self.device = self.stack.enter_context(Device(self.driver, self.ordinal, cap=64 * MiB))
        self.set_attribute = _bind(self.driver.lib, "cuFuncSetAttribute", [C.c_void_p, C.c_int, C.c_int])
        self.options = [f"-I{TILELANG_TEMPLATE_PATH}", f"-I{CUTLASS_INCLUDE_DIR}", f"-I{CUDA_HOME}/include",
                        f"-I{CUDA_HOME}/include/cccl", "-D__CUDACC_VER_MAJOR__=13"]
        self.factories = {(storage, diagnostic): load_variant(raw, self.directory / "variants", storage, diagnostic)[0]
                          for storage in ("shared", "register-query") for diagnostic in (False, True)}
        for factory in self.factories.values():
            factory.__wrapped__.execution_backend = "nvrtc"
            factory.__wrapped__.target = {"kind": "cuda", "arch": "sm_121"}
        self.profiles, self.kernels, self.originals, self.candidates = [], {}, {}, {}
        with torch.cuda.device(self.ordinal):
            for heads, dimension in ((64, 256), (16, 512), (64, 512)):
                for diagnostic in (False, True):
                    self._checkpoint()
                    key = heads, dimension, diagnostic
                    original = self.factories["shared", diagnostic](heads, dimension, dimension**-.5)
                    candidate = self.factories["register-query", diagnostic](heads, dimension, dimension**-.5)
                    if type(original.adapter).__name__ != "NVRTCKernelAdapter":
                        raise ValueError("Spark original Attention selected another execution backend")
                    cuda = original.get_kernel_source()
                    meta = inspect_codegen(cuda)
                    from .prefix_reference import required_shared_bytes
                    if required_shared_bytes(_host_source(original)) != meta["original_shared_bytes"]:
                        raise ValueError("Original Spark launch shared allocation differs")
                    if meta["adapted_shared_bytes"] > self.shared_limit:
                        raise ValueError("Adapted Spark Attention exceeds measured shared capacity")
                    flags = original.adapter.lib_generator.compile_flags or []
                    extra = [item for flag in flags for item in flag.split() if item not in self.options]
                    binary = bytes(compile_cuda(cuda, "cubin", 121, self.options + extra))
                    with Path(original.adapter.lib_generator.libpath).open("rb") as stream:
                        cached = stream.read(8 * MiB + 1)
                    binary_sha = hashlib.sha256(binary).hexdigest()
                    if len(cached) > 8 * MiB or binary != cached or binary_sha != CUBIN_PROFILES[key]:
                        raise ValueError("Original Spark CUBIN does not reproduce its pinned compiler observation")
                    ptx = bytes(compile_cuda(cuda, "ptx", 121, self.options + extra)).decode()
                    adapted, audit = virtualize_ptx(ptx, meta)
                    audit.update(ptx_version="9.0", ptx_target="sm_121a")
                    module = self.device.load(adapted.encode() + b"\0")
                    meta.update(ptx_audit=audit, original_binary_reproduction=True,
                                original_cubin_sha256=binary_sha,
                                compile_flags=["paths redacted; fresh binary/cache reproduction required"],
                                compiler_options=["include:<TileLang>", "include:<CUTLASS>", "include:<CUDA>",
                                                  "include:<CUDA>/cccl", "-D__CUDACC_VER_MAJOR__=13"])
                    self.profiles.append(meta)
                    self.kernels[key] = module, meta
                    self.originals[key], self.candidates[key] = original, candidate
                    if key == (64, 512, False):
                        self.full_source = cuda
            loader_source, self.loader_meta = loader_check_cuda(self.full_source)
            loader = bytes(compile_cuda(loader_source, "cubin", 121, self.options))
            self.loader_meta["cubin_sha256"] = hashlib.sha256(loader).hexdigest()
            self.loader = self.device.load(loader)
        self._checkpoint()

    def _run(self, *args, **kwargs):
        self._checkpoint()
        return super()._run(*args, **kwargs)

    def qualify(self):
        self.qualified = False
        self._checkpoint()
        try:
            report = super().qualify()
            report.update(probe=PROBE, execution_target=SPARK_TARGET, independent_Spark_qualification=True,
                          framework_profile=self.framework_profile,
                          memory_observations=self.memory_observations,
                          full_model_quality_qualified=False, generation_speed_claim=False,
                          qualification_scope="GB10 original PTX arithmetic with qualified integer Query storage; fixed component corpus only")
            self._checkpoint()
            self.workspace.stamp_result({}, initial_context=self.initial_context)
            if report["status"] == "pass":
                validate_qualification(report)
            self.qualified = report["status"] == "pass"
            return report
        except BaseException:
            self.qualified = False
            raise


def compare_attention_memory(directory, ordinal=0, *, workspace=None):
    with SparkAttentionMemoryReference(directory, ordinal, workspace=workspace) as reference:
        return reference.qualify()
