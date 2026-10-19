"""Reproducible short text-continuation gate: python -m vesper.generation_probe."""

import argparse
import ctypes
import json
import os
from pathlib import Path
import re

from .memory import RuntimeMemory, MemoryPolicy
from .model_artifacts import digest
from .resources import GiB


def read_evidence(path):
    limit = 64 * 1024**2
    if path.stat().st_size > limit: raise ValueError("Excessive generation evidence metadata")
    with path.open("rb") as file: raw = file.read(limit + 1)
    if len(raw) > limit: raise ValueError("Excessive generation evidence metadata")
    return raw


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fixed eight-call-scope original text continuation probe")
    parser.add_argument("mode", choices=("prepare", "reference", "native", "check"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--run", default="story-v1")
    parser.add_argument("--reference", type=Path, help="Sealed original report (default: run/original.json)")
    parser.add_argument("--nvrtc-bin", type=Path, help="Optional reference-only Windows CUDA 13 NVRTC directory")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", args.run): parser.error("Expected a simple generation run name")
    if args.nvrtc_bin is not None and args.mode != "reference": parser.error("--nvrtc-bin is reference-only")
    run = args.artifacts / "generation" / args.run
    reference = args.reference if args.reference is not None else run / "original.json"
    def progress(value): print(json.dumps(value), flush=True)
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as file: file.write((json.dumps(value, indent=2) + "\n").encode())
    if args.mode == "check":
        from .generation_validation import validate_native
        raw = read_evidence(reference)
        validate_native(json.loads(read_evidence(run / "native.json")), json.loads(raw), raw)
        progress({"status": "pass", "probe": "complete_text_generation_evidence"})
        return
    target = run / {"prepare": "preparation.json", "reference": "original.json", "native": "native.json"}[args.mode]
    if target.exists(): raise FileExistsError("Refusing to replace completed generation evidence")
    if args.mode == "reference":
        handle = None
        if args.nvrtc_bin is not None:
            if os.name != "nt": parser.error("--nvrtc-bin is only used by the Windows reference profile")
            handle = os.add_dll_directory(str(args.nvrtc_bin.resolve()))
            nvrtc = ctypes.CDLL(str(args.nvrtc_bin.resolve() / "nvrtc64_130_0.dll"))
        try:
            from .generation_reference import capture_original
            report = capture_original(args.artifacts, run, progress)
        finally:
            if handle is not None: handle.close()
    else:
        with RuntimeMemory(MemoryPolicy(host_cap_bytes=64 * GiB)) as policy:
            if args.mode == "prepare":
                from .generation_prepare import prepare
                from .generation_validation import REQUEST
                report = prepare(args.artifacts, run, REQUEST, progress)
            else:
                from .generation_validation import run_native
                raw = read_evidence(reference)
                report = run_native(args.artifacts, run, json.loads(raw), progress)
                report["reference_report_sha256"] = digest(raw)
            report["memory_before_policy_close"] = policy.snapshot()
        report["memory_after_policy_close"] = policy.snapshot()
    save(target, report)
    progress({"status": "pass", "probe": report["probe"], "completion": report.get("completion")})


if __name__ == "__main__": main()
