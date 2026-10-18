"""Inspect optional TileLang launch-handle isolation without launching kernels."""

import argparse
import importlib.util
import json
from pathlib import Path

from vesper.checkpoint import fetch_reference
from vesper.reference_runtime import isolate_nvrtc_handles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-directory", type=Path, default=Path("artifacts/reference"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--isolate", action="store_true")
    args = parser.parse_args()
    import torch
    import tilelang

    if tilelang.__version__ != "0.1.12":
        raise ValueError("This diagnostic is scoped to TileLang 0.1.12")
    manifest = fetch_reference(args.reference_directory, include_model=True)
    spec = importlib.util.spec_from_file_location("vesper_fp8_handle_diagnostic", args.reference_directory / "kernel.py")
    kernel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernel)
    factory = isolate_nvrtc_handles(kernel.fp8_gemm_kernel) if args.isolate else kernel.fp8_gemm_kernel
    with torch.cuda.device(args.device):
        first = factory(2304, 5120, block_size=32, scale_dtype=kernel.FE8M0)
        before = dict(first.adapter.kernels)
        second = factory(5120, 2304, block_size=32, scale_dtype=kernel.FE8M0)
        after = dict(first.adapter.kernels)
    result = {"probe": "tilelang_fp8_launch_handle_isolation", "mode": "isolated" if args.isolate else "class_table",
              "reference": manifest, "tilelang_version": tilelang.__version__, "torch_version": torch.__version__,
              "adapter": type(first.adapter).__name__, "compiled_shapes": [[2304, 5120], [5120, 2304]],
              "tables_shared_between_shapes": first.adapter.kernels is second.adapter.kernels,
              "first_launch_handle_replaced": any(after.get(n) != h for n, h in before.items()),
              "first_adapter_owns_instance_table": "kernels" in vars(first.adapter), "kernel_launches": 0,
              "scope": "Compiler/adapter metadata inspection only; no mismatched kernel is launched"}
    passed = (result["first_launch_handle_replaced"] is (not args.isolate)
              and result["tables_shared_between_shapes"] is (not args.isolate))
    result["status"] = "pass" if passed else "fail"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(result, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(result))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
