"""Narrow optional-reference runtime fixes; no arithmetic/kernel source changes."""

from functools import wraps


def isolate_nvrtc_handles(factory):
    """Snapshot a TileLang 0.1.12 NVRTC adapter's shared class-level handles.

    Different N/K specializations use the same generated function name. Keeping
    the class dictionary lets a later specialization replace an earlier one's
    launch handle. Isolate immediately after factory creation, before launch.
    """
    @wraps(factory)
    def create(*args, **kwargs):
        kernel = factory(*args, **kwargs)
        adapter = kernel.adapter
        if type(adapter).__name__ == "NVRTCKernelAdapter" and "kernels" not in vars(adapter):
            adapter.kernels = dict(adapter.kernels)
        return kernel
    return create


def configure_reference_runtime(module, tilelang_version: str) -> list[str]:
    if tilelang_version != "0.1.12":
        return []
    for name in ("act_quant_kernel", "fp4_gemm_kernel"):
        setattr(module, name, isolate_nvrtc_handles(getattr(module, name)))
    return ["TileLang 0.1.12 NVRTC: isolate launch handles per compiled specialization; arithmetic unchanged"]
