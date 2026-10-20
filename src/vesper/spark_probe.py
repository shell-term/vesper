"""Guarded, framework-free GB10 inventory and a bounded integer CUDA probe.

This qualifies only discovery, NVRTC compilation, and a small copy/kernel round
trip. It does not qualify model arithmetic, original-kernel parity, or speed.
No context, library, compiler, or allocation is opened at import time.
"""

from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import struct
import time

MiB, GiB = 1024**2, 1024**3
MAX_BUFFER_BYTES = 8 * MiB
DEFAULT_BUFFER_BYTES = MiB
AVAILABLE_FLOOR_BYTES = 32 * GiB
RUNTIME_HEADROOM_BYTES = 256 * MiB
SOURCE = b'''extern "C" __global__ void spark_roundtrip(unsigned int *p, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] ^= 0x5a5a5a5au;
}
'''
ATTRIBUTES = {"integrated": 18, "unified_addressing": 41,
              "managed_memory": 83, "pageable_memory_access": 88,
              "concurrent_managed_access": 89, "host_page_tables": 100}


class Unsupported(RuntimeError):
    """The narrowly qualified Spark profile could not be established."""


def _bind(lib, name, args):
    fn = getattr(lib, name)
    fn.argtypes, fn.restype = args, C.c_int
    return fn


def _check(code, operation):
    if code:
        raise Unsupported(f"{operation} returned CUDA/NVRTC status {code}")


def host_memory(proc=Path("/proc"), cgroup_root=Path("/sys/fs/cgroup")):
    fields = {}
    for line in (proc / "meminfo").read_text(encoding="ascii").splitlines():
        name, value = line.split(":", 1)
        parts = value.split()
        if len(parts) != 2 or parts[1] != "kB":
            continue
        fields[name] = int(parts[0]) * 1024
    total, available = fields["MemTotal"], fields["MemAvailable"]
    if not 0 <= available <= total or total <= AVAILABLE_FLOOR_BYTES:
        raise Unsupported("Invalid physical-memory observation")
    result = {"physical_total_bytes": total, "physical_available_bytes": available,
              "shared_pool_budget_bytes": min(96 * GiB, total - AVAILABLE_FLOOR_BYTES),
              "available_floor_bytes": AVAILABLE_FLOOR_BYTES,
              "ram_plus_vram_is_not_capacity": True,
              "cgroup_headroom_bytes": None}
    memberships = (proc / "self/cgroup").read_text(encoding="ascii").splitlines()
    unified = [line[3:] for line in memberships if line.startswith("0::")]
    if len(unified) != 1:
        raise Unsupported("A single readable cgroup-v2 hierarchy is required")
    member = PurePosixPath(unified[0])
    if not member.is_absolute() or ".." in member.parts:
        raise Unsupported("Unresolved cgroup-v2 membership")
    root = cgroup_root.resolve(strict=True)
    current = root.joinpath(*member.parts[1:]).resolve(strict=True)
    if current != root and root not in current.parents:
        raise Unsupported("Cgroup-v2 membership escapes its mount")
    caps = []
    while True:
        limit_file, usage_file = current / "memory.max", current / "memory.current"
        if limit_file.exists():
            limit = limit_file.read_text(encoding="ascii").strip()
            if limit != "max":
                maximum, used = int(limit), int(usage_file.read_text(encoding="ascii").strip())
                if maximum <= 0 or used < 0:
                    raise Unsupported("Invalid cgroup-v2 memory constraint")
                caps.append(max(0, maximum - used))
        if current == root:
            break
        current = current.parent
    if caps:
        result["cgroup_headroom_bytes"] = min(caps)
    return result


def require_capacity(memory, pending_bytes=0):
    """One physical pool: never add CUDA's free/total figures to host RAM."""
    requested = pending_bytes + RUNTIME_HEADROOM_BYTES
    if memory["physical_available_bytes"] - requested < AVAILABLE_FLOOR_BYTES:
        raise Unsupported("The 32 GiB physical available-memory floor lacks headroom")
    cg = memory.get("cgroup_headroom_bytes")
    if cg is not None and cg < requested:
        raise Unsupported("The cgroup memory headroom is insufficient")


def elf_architecture(path):
    with Path(path).open("rb") as stream:
        header = stream.read(64)
    if len(header) < 64 or header[:6] != b"\x7fELF\x02\x01":
        raise Unsupported("Expected a 64-bit little-endian ELF CUDA library")
    machine = struct.unpack_from("<H", header, 18)[0]
    if machine != 183:
        raise Unsupported("CUDA library is not AArch64")
    return "aarch64"


def library_candidates(kind):
    """Include both Linux ARM CUDA layouts; never load toolkit driver stubs."""
    if kind == "driver":
        return [(Path(p), "system_aarch64") for p in (
            "/usr/lib/aarch64-linux-gnu/libcuda.so.1", "/lib/aarch64-linux-gnu/libcuda.so.1",
            "/usr/lib/aarch64-linux-gnu/nvidia/current/libcuda.so.1",
            "/usr/local/nvidia/lib64/libcuda.so.1")]
    if kind != "nvrtc":
        raise ValueError("Unknown CUDA library kind")
    roots = []
    for name in ("CUDA_HOME", "CUDA_PATH"):
        value = os.environ.get(name)
        if value:
            root = Path(value)
            if not root.is_absolute():
                raise Unsupported("CUDA toolkit roots must be absolute")
            roots.append(root)
    roots += [Path("/usr/local/cuda")]
    roots += sorted(Path("/usr/local").glob("cuda-[0-9]*"), reverse=True)
    result = []
    for root in roots:
        for relative in ("targets/aarch64-linux/lib", "targets/sbsa-linux/lib", "lib64"):
            directory = root / relative
            for path in sorted(directory.glob("libnvrtc.so*"), reverse=True):
                result.append((path, relative))
    for path in sorted(Path("/usr/lib/aarch64-linux-gnu").glob("libnvrtc.so*"), reverse=True):
        result.append((path, "system_aarch64"))
    return result


def load_library(kind):
    seen = set()
    for candidate, layout in library_candidates(kind):
        if not candidate.is_file():
            continue
        path = candidate.resolve(strict=True)
        if path in seen:
            continue
        seen.add(path)
        try:
            architecture = elf_architecture(path)
            library = C.CDLL(str(path))
        except (OSError, Unsupported):
            continue
        return library, {"kind": kind, "basename": path.name, "elf_architecture": architecture,
                         "layout": layout, "file_bytes": path.stat().st_size,
                         "absolute_path_redacted": True}
    raise Unsupported(f"No loadable AArch64 {kind} library in supported system/toolkit layouts")


class Driver:
    def __init__(self):
        self.lib, self.library_info = load_library("driver")
        ptr, integer, size, dp = C.c_void_p, C.c_int, C.c_size_t, C.c_uint64
        signatures = {
            "cuInit": [C.c_uint], "cuDriverGetVersion": [C.POINTER(integer)],
            "cuDeviceGetCount": [C.POINTER(integer)], "cuDeviceGet": [C.POINTER(integer), integer],
            "cuDeviceGetName": [ptr, integer, integer],
            "cuDeviceComputeCapability": [C.POINTER(integer), C.POINTER(integer), integer],
            "cuDeviceGetAttribute": [C.POINTER(integer), integer, integer],
            "cuDeviceTotalMem_v2": [C.POINTER(size), integer],
            "cuCtxGetCurrent": [C.POINTER(ptr)], "cuCtxSetCurrent": [ptr],
            "cuCtxCreate_v2": [C.POINTER(ptr), C.c_uint, integer], "cuCtxDestroy_v2": [ptr],
            "cuCtxSynchronize": [], "cuMemGetInfo_v2": [C.POINTER(size), C.POINTER(size)],
            "cuMemAlloc_v2": [C.POINTER(dp), size], "cuMemFree_v2": [dp],
            "cuMemHostAlloc": [C.POINTER(ptr), size, C.c_uint], "cuMemFreeHost": [ptr],
            "cuMemcpyHtoD_v2": [dp, ptr, size], "cuMemcpyDtoH_v2": [ptr, dp, size],
            "cuModuleLoadData": [C.POINTER(ptr), ptr], "cuModuleUnload": [ptr],
            "cuModuleGetFunction": [C.POINTER(ptr), ptr, C.c_char_p],
            "cuLaunchKernel": [ptr, C.c_uint, C.c_uint, C.c_uint, C.c_uint, C.c_uint, C.c_uint,
                               C.c_uint, ptr, C.POINTER(ptr), C.POINTER(ptr)],
        }
        for name, signature in signatures.items():
            setattr(self, name, _bind(self.lib, name, signature))
        _check(self.cuInit(0), "cuInit")

    def inventory(self):
        count, version, device = C.c_int(), C.c_int(), C.c_int()
        _check(self.cuDeviceGetCount(C.byref(count)), "cuDeviceGetCount")
        _check(self.cuDriverGetVersion(C.byref(version)), "cuDriverGetVersion")
        if count.value != 1:
            raise Unsupported("This profile requires exactly one CUDA device")
        _check(self.cuDeviceGet(C.byref(device), 0), "cuDeviceGet")
        name, major, minor, total = C.create_string_buffer(256), C.c_int(), C.c_int(), C.c_size_t()
        _check(self.cuDeviceGetName(name, len(name), device), "cuDeviceGetName")
        _check(self.cuDeviceComputeCapability(C.byref(major), C.byref(minor), device), "cuDeviceComputeCapability")
        _check(self.cuDeviceTotalMem_v2(C.byref(total), device), "cuDeviceTotalMem_v2")
        attributes = {}
        for key, enum in ATTRIBUTES.items():
            value = C.c_int()
            _check(self.cuDeviceGetAttribute(C.byref(value), enum, device), "cuDeviceGetAttribute")
            attributes[key] = value.value
        info = {"device_count": count.value, "ordinal": 0, "name": name.value.decode("utf-8", "replace"),
                "compute_capability": [major.value, minor.value], "driver_api_version": version.value,
                "reported_cuda_total_bytes": total.value, "attributes": attributes,
                "cuda_capacity_added_to_physical_ram": False, "library": self.library_info}
        validate_device(info)
        return info


def validate_device(info):
    if info["device_count"] != 1 or "GB10" not in info["name"] or info["compute_capability"] != [12, 1]:
        raise Unsupported("The CUDA device does not match the single GB10 / sm_121 profile")
    a = info["attributes"]
    if not a["unified_addressing"] or not a["managed_memory"]:
        raise Unsupported("GB10 unified-addressing/managed-memory capabilities were not established")
    if not (a["integrated"] or (a["pageable_memory_access"] and a["host_page_tables"])):
        raise Unsupported("The observed GB10 attributes do not establish the expected shared-memory profile")


class Compiler:
    def __init__(self):
        self.lib, self.library_info = load_library("nvrtc")
        version = _bind(self.lib, "nvrtcVersion", [C.POINTER(C.c_int), C.POINTER(C.c_int)])
        major, minor = C.c_int(), C.c_int()
        _check(version(C.byref(major), C.byref(minor)), "nvrtcVersion")
        self.info = {"nvrtc_version": [major.value, minor.value], "architecture": "sm_121",
                     "library": self.library_info, "source_sha256": hashlib.sha256(SOURCE).hexdigest(),
                     "options": ["--gpu-architecture=sm_121", "--std=c++17", "--fmad=false"]}
        if major.value < 13:
            raise Unsupported("NVRTC 13 or newer is required for the sm_121 profile")

    def compile(self):
        ptr, size = C.c_void_p, C.c_size_t
        create = _bind(self.lib, "nvrtcCreateProgram", [C.POINTER(ptr), C.c_char_p, C.c_char_p, C.c_int, ptr, ptr])
        compile_ = _bind(self.lib, "nvrtcCompileProgram", [ptr, C.c_int, C.POINTER(C.c_char_p)])
        destroy = _bind(self.lib, "nvrtcDestroyProgram", [C.POINTER(ptr)])
        get_size = _bind(self.lib, "nvrtcGetCUBINSize", [ptr, C.POINTER(size)])
        get_image = _bind(self.lib, "nvrtcGetCUBIN", [ptr, ptr])
        program = ptr()
        _check(create(C.byref(program), SOURCE, b"spark_roundtrip.cu", 0, None, None), "nvrtcCreateProgram")
        try:
            options = [x.encode("ascii") for x in self.info["options"]]
            arguments = (C.c_char_p * len(options))(*options)
            _check(compile_(program, len(options), arguments), "nvrtcCompileProgram")
            length = size()
            _check(get_size(program, C.byref(length)), "nvrtcGetCUBINSize")
            if not 0 < length.value <= MiB:
                raise Unsupported("Unexpectedly large small-probe CUDA image")
            image = C.create_string_buffer(length.value)
            _check(get_image(program, image), "nvrtcGetCUBIN")
            return image.raw
        finally:
            _check(destroy(C.byref(program)), "nvrtcDestroyProgram")


def integer_probe(driver, image, buffer_bytes, checkpoint):
    """Own context, one device buffer and one pinned buffer; no GPU reset."""
    context, previous, module, host = (C.c_void_p() for _ in range(4))
    device = C.c_uint64()
    errors, report = [], {"device_peak_explicit_bytes": 0, "pinned_peak_explicit_bytes": 0,
                          "driver_context_module_memory_not_in_explicit_caps": True}
    _check(driver.cuCtxGetCurrent(C.byref(previous)), "cuCtxGetCurrent")
    try:
        checkpoint(2 * buffer_bytes)
        _check(driver.cuCtxCreate_v2(C.byref(context), 4, 0), "cuCtxCreate_v2")
        checkpoint(2 * buffer_bytes)
        free, total = C.c_size_t(), C.c_size_t()
        _check(driver.cuMemGetInfo_v2(C.byref(free), C.byref(total)), "cuMemGetInfo_v2")
        if free.value - 2 * buffer_bytes < AVAILABLE_FLOOR_BYTES:
            raise Unsupported("CUDA's shared-pool free-memory observation is below reserve")
        report["cuda_memory_before"] = {"free_bytes": free.value, "total_bytes": total.value,
                                         "counted_as_additional_capacity": False}
        _check(driver.cuMemAlloc_v2(C.byref(device), buffer_bytes), "cuMemAlloc_v2")
        report["device_peak_explicit_bytes"] = buffer_bytes
        checkpoint(buffer_bytes)
        _check(driver.cuMemHostAlloc(C.byref(host), buffer_bytes, 0), "cuMemHostAlloc")
        report["pinned_peak_explicit_bytes"] = buffer_bytes
        checkpoint()
        n = buffer_bytes // 4
        values = (C.c_uint32 * n).from_address(host.value)
        for i in range(n):
            values[i] = ((i * 2654435761) ^ 0xA5A5A5A5) & 0xFFFFFFFF
        report["input_sha256"] = hashlib.sha256(memoryview(values).cast("B")).hexdigest()
        _check(driver.cuMemcpyHtoD_v2(device, host, buffer_bytes), "cuMemcpyHtoD_v2")
        checkpoint()
        image_buffer = C.create_string_buffer(image)
        _check(driver.cuModuleLoadData(C.byref(module), image_buffer), "cuModuleLoadData")
        checkpoint()
        function = C.c_void_p()
        _check(driver.cuModuleGetFunction(C.byref(function), module, b"spark_roundtrip"), "cuModuleGetFunction")
        count = C.c_uint(n)
        arguments = (C.c_void_p * 2)(C.cast(C.byref(device), C.c_void_p), C.cast(C.byref(count), C.c_void_p))
        start = time.perf_counter()
        _check(driver.cuLaunchKernel(function, (n + 127) // 128, 1, 1, 128, 1, 1, 0, None, arguments, None), "cuLaunchKernel")
        _check(driver.cuCtxSynchronize(), "cuCtxSynchronize")
        report["single_launch_and_synchronize_seconds"] = time.perf_counter() - start
        _check(driver.cuMemcpyDtoH_v2(host, device, buffer_bytes), "cuMemcpyDtoH_v2")
        checkpoint()
        mismatches = sum(values[i] != ((((i * 2654435761) ^ 0xA5A5A5A5) & 0xFFFFFFFF) ^ 0x5A5A5A5A) for i in range(n))
        report.update(checked_uint32_elements=n, mismatches=mismatches,
                      output_sha256=hashlib.sha256(memoryview(values).cast("B")).hexdigest(),
                      cubin_sha256=hashlib.sha256(image).hexdigest())
        if mismatches:
            raise Unsupported("The small integer CUDA round trip did not match")
        return report
    except Exception as exc:
        exc.probe_evidence = report
        raise
    finally:
        for pointer, name in ((module, "cuModuleUnload"), (device, "cuMemFree_v2"), (host, "cuMemFreeHost")):
            if pointer.value:
                status = getattr(driver, name)(pointer)
                if status:
                    errors.append({"operation": name, "status": status})
        if context.value:
            status = driver.cuCtxDestroy_v2(context)
            if status:
                errors.append({"operation": "cuCtxDestroy_v2", "status": status})
        status = driver.cuCtxSetCurrent(previous)
        if status:
            errors.append({"operation": "cuCtxSetCurrent", "status": status})
        report["cleanup"] = {"owned_resources_release_calls_succeeded": not errors,
                             "previous_context_restore_requested": True, "errors": errors,
                             "gpu_reset_or_other_workload_stop": False}
        if errors:
            failure = Unsupported("Probe-owned CUDA cleanup failed")
            failure.probe_evidence = report
            raise failure


def collect(mode, workspace, *, buffer_bytes=DEFAULT_BUFFER_BYTES):
    if mode not in ("inventory", "probe"):
        raise ValueError("Unknown Spark probe mode")
    if type(buffer_bytes) is not int or buffer_bytes <= 0 or buffer_bytes > MAX_BUFFER_BYTES or buffer_bytes % 4:
        raise ValueError("Buffers must be a positive multiple of four, at most 8 MiB each")
    workspace.check()
    context = workspace.stamp_result({})["workspace_context"]
    if context["platform"] != "dgx-spark" or context["profile"] != "native":
        raise Unsupported("An explicitly bound dgx-spark/native workspace is required")
    report = {"schema": "vesper.spark-probe.v1", "mode": mode, "status": "unsupported",
              "scope": "GB10 inventory and bounded integer CUDA round trip only",
              "model_checkpoint_access": False, "model_kernel_parity_qualified": False,
              "full_model_quality_qualified": False, "general_performance_claim": False,
              "limits": {"device_buffer_cap_bytes": MAX_BUFFER_BYTES, "pinned_buffer_cap_bytes": MAX_BUFFER_BYTES,
                         "requested_buffer_bytes_each": buffer_bytes, "physical_available_floor_bytes": AVAILABLE_FLOOR_BYTES,
                         "floor_enforcement": "admission_and_stage_observation_not_OS_reservation"},
              "observed_hardware": {"system": platform.system(), "machine": platform.machine()},
              "memory_observations": []}
    try:
        if platform.system() != "Linux" or platform.machine().lower() not in ("aarch64", "arm64"):
            raise Unsupported("This probe executes only on Linux AArch64")
        def checkpoint(pending=0):
            workspace.check()
            memory = host_memory()
            report["memory_observations"].append(memory)
            require_capacity(memory, pending)
        checkpoint(2 * buffer_bytes if mode == "probe" else 0)
        driver = Driver()
        report["observed_hardware"]["cuda"] = driver.inventory()
        checkpoint()
        compiler = Compiler()
        report["compiler"] = compiler.info
        checkpoint()
        if mode == "probe":
            image = compiler.compile()
            checkpoint(2 * buffer_bytes)
            report["integer_roundtrip"] = integer_probe(driver, image, buffer_bytes, checkpoint)
            checkpoint()
            report["status"] = "bounded_roundtrip_pass"
        else:
            report["status"] = "supported_inventory"
    except (Unsupported, OSError, KeyError, ValueError, AttributeError) as exc:
        report["reason"] = str(exc) if isinstance(exc, Unsupported) else "Required CUDA, memory, or platform metadata is unavailable or invalid"
        report["failure_type"] = type(exc).__name__
        if hasattr(exc, "probe_evidence"):
            report["integer_roundtrip"] = exc.probe_evidence
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inventory", "probe"))
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--buffer-bytes", type=int, default=DEFAULT_BUFFER_BYTES)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", args.run_id):
        parser.error("Expected a simple lowercase run ID")
    from . import workspace as workspace_module
    workspace = workspace_module.Workspace.open(args.workspace)
    workspace.check_runtime_module(__file__, "src/vesper/spark_probe.py")
    workspace.check_runtime_module(workspace_module.__file__, "src/vesper/workspace.py")
    target = workspace.result_path(f"s0/{args.run_id}.json")
    if target.exists():
        raise FileExistsError("Refusing to rerun or overwrite existing Spark evidence")
    initial_context = workspace.context(args.run_id)
    report = collect(args.mode, workspace, buffer_bytes=args.buffer_bytes)
    path = workspace.write_result(f"s0/{args.run_id}.json", report, run_id=args.run_id,
                                  initial_context=initial_context)
    print(json.dumps({"status": report["status"], "result": path.relative_to(workspace.root).as_posix(), "mode": args.mode}))
    return 0 if report["status"] in ("bounded_roundtrip_pass", "supported_inventory") else 2


if __name__ == "__main__":
    raise SystemExit(main())
