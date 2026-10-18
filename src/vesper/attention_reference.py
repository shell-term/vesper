"""Build qualified local kernels, then compare the framework-free executor."""

import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .attention import CANDIDATES, CANDIDATE_BINARIES, NativeAttention, argument_names, kernel_filename, read_bounded
from .attention_memory import FUNCTION, validate_qualification
from .attention_memory_reference import AttentionMemoryReference, missing_dependencies
from .attention_storage import STAGES
from .attention_storage_reference import _host_source, _numpy
from .attention_validation import array_digest, check_arrays, exercise_attention
from .checkpoint import MODEL, REVISION, REFERENCE_HASHES
from .prefix_reference import required_shared_bytes
from .resources import MiB


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def export_bundle(reference, qualification, destination):
    """Bind reproduced candidate binaries to this successful oracle qualification."""
    if not reference.qualified:
        raise RuntimeError("Refusing to export before Attention oracle qualification")
    validate_qualification(qualification)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("Use a new Attention bundle directory; previous artifacts are immutable")
    profiles, binaries = [], {}
    for (heads, dimension, diagnostic), (expected_hash, expected_shared) in CANDIDATES.items():
        candidate = reference.candidates[heads, dimension, diagnostic]
        source = candidate.get_kernel_source()
        shared = required_shared_bytes(_host_source(candidate))
        if hashlib.sha256(source.encode()).hexdigest() != expected_hash or shared != expected_shared:
            raise ValueError("Candidate generated CUDA or shared-memory profile changed")
        signature = re.search(r"void " + FUNCTION + r"\(([^)]*)\)", source)
        if signature is None:
            raise ValueError("Missing candidate launch signature")
        arguments = [part.strip().split()[-1].lstrip("*") for part in signature[1].split(",")]
        if arguments != argument_names(diagnostic):
            raise ValueError("Candidate parameter order differs from the qualified ABI")
        flags = candidate.adapter.lib_generator.compile_flags or []
        extra = [item for flag in flags for item in flag.split() if item not in reference.options]
        binary = bytes(reference.compile_cuda(source, "cubin", 89, reference.options + extra))
        if binary != read_bounded(candidate.adapter.lib_generator.libpath, 2 * MiB):
            raise ValueError("Candidate CUBIN does not reproduce the qualified loaded binary")
        if hashlib.sha256(binary).hexdigest() != CANDIDATE_BINARIES[heads, dimension, diagnostic]:
            raise ValueError("Candidate compiler binary changed; a new qualification profile is required")
        filename = kernel_filename(heads, dimension, diagnostic)
        profiles.append({"heads": heads, "dimension": dimension, "diagnostics": diagnostic, "file": filename,
                         "function": FUNCTION, "argument_names": arguments, "dynamic_shared_bytes": shared,
                         "generated_cuda_sha256": expected_hash, "cubin_sha256": hashlib.sha256(binary).hexdigest(),
                         "binary_reproduction": True})
        binaries[filename] = binary
    qualification = {**qualification, "native_candidate_exports": profiles}
    receipt = json_bytes(qualification)
    manifest = {"schema_version": 1, "kind": "vesper_attention_register_query_bundle", "model": MODEL,
                "revision": REVISION, "compute_capability": "8.9", "minimum_driver_api": 13040,
                "tilelang_version": "0.1.12", "nvrtc_version": [13, 4], "kernel_source_sha256": REFERENCE_HASHES["kernel.py"],
                "qualification_sha256": hashlib.sha256(receipt).hexdigest(), "profiles": profiles}
    destination.mkdir(parents=True, exist_ok=False)
    for name, binary in binaries.items():
        (destination / name).write_bytes(binary)
    (destination / "DeepSeek-LICENSE").write_bytes(read_bounded(reference.directory / "LICENSE", 64 * 1024))
    (destination / "qualification.json").write_bytes(receipt)
    # Publish the entry point last. Incomplete exports cannot be opened.
    (destination / "manifest.json").write_bytes(json_bytes(manifest))
    return manifest


def compare_native_attention(directory, bundle, ordinal=0):
    missing = missing_dependencies()
    if missing:
        return {"probe": "native_attention_parity", "status": "blocked", "missing_dependencies": missing}
    if Path(bundle).exists():
        raise FileExistsError("Use a new Attention bundle directory; previous artifacts are immutable")
    import torch
    goldens = {}
    with AttentionMemoryReference(Path(directory), ordinal) as reference:
        qualification = reference.qualify()
        manifest = export_bundle(reference, qualification, bundle)
        def checker(heads, dimension, index, inputs, provenance, actual):
            tensors = tuple(torch.from_numpy(a).view(torch.bfloat16).to(f"cuda:{ordinal}") if a.dtype == np.uint16
                            else torch.from_numpy(a).to(f"cuda:{ordinal}") for a in inputs)
            with torch.inference_mode(), torch.cuda.device(ordinal):
                output, traces, _ = reference._run(heads, dimension, True, tensors)
            expected = dict(zip(("output_bf16", *STAGES), map(_numpy, (output, *traces))))
            goldens[heads, dimension, index] = {"heads": heads, "dimension": dimension, "case": index,
                                              "input_sha256": provenance,
                                              "stage_sha256": {name: array_digest(a) for name, a in expected.items()}}
            return [check_arrays(a, actual[name], name) for name, a in expected.items()]
        with NativeAttention(bundle, ordinal) as native:
            rows, faults, minimum = exercise_attention(native, checker)
            device, manifest_hash = native.device_info, native.manifest_sha256
        cleanup = native.stats()
    passed = (all(r["status"] == "pass" for r in rows + faults) and len(goldens) == 36
              and not cleanup["host_reserved_bytes"] and not cleanup["device_owned_bytes"])
    return {"schema_version": 1, "probe": "native_attention_parity", "status": "pass" if passed else "fail",
            "model": MODEL, "revision": REVISION, "device": device,
            "reference_profile": "qualified_official_PTX_memory_adapter", "qualification_sha256": manifest["qualification_sha256"],
            "bundle_manifest": manifest, "bundle_manifest_sha256": manifest_hash,
            "cases": rows, "faults": faults, "reference_goldens": list(goldens.values()),
            "memory": {**cleanup, "minimum_sampled_host_available_bytes": minimum,
                       "combined_native_and_reference_host_allowance_bytes": 96 * MiB},
            "kernel_execution": "CUDA Driver, no Torch/TileLang dispatch", "optional_build_and_oracle_require_torch": True,
            "kv_state_and_projections": "not_run", "full_model_quality": "not_run", "end_to_end_speed": "not_measured",
            "scope": "Bounded sparse Attention with original BF16/FP32 boundaries; not a complete Attention layer or model"}
