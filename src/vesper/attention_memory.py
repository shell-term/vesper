"""Pinned CUDA provenance and bounded data-movement qualification source.

This is a storage-adapted oracle, not the unmodified official kernel. Generated
CUDA/source and compiler versions are pinned separately from model weights.
"""

from __future__ import annotations

import hashlib
import re

from .checkpoint import MODEL, REVISION, REFERENCE_HASHES

CODEGEN_PROFILES = {
    "7a5cb385b43e66d5a15c04be4d7d5572494540f82b40a7d16ade25ae24ed97ab": (64, 256, False, 75776),
    "b3695dd24e118603abfd7b8c21823c9c2f0365e26d06bf0ac9d72d44715fc1b0": (64, 256, True, 75776),
    "3809b2ae5b01441053f3a7b1bfde90a20986f8a43f127a7d9463b30365cda3f7": (16, 512, False, 86016),
    "3fe1434c4ee48891fa23e500c5b6648d04fd25b46452b5a669e3f5838766c00f": (16, 512, True, 86016),
    "d8f6dbef7f2ddea62ce38d2956f3b9699f71f607425a8bd4d77aff9318a4d1ba": (64, 512, False, 141312),
    "920c550fdb4215687f393650a60564600a1ed2dfe2dc236473bd7939d1467195": (64, 512, True, 141312),
}
FUNCTION = "sparse_attn_kernel__kernel"
DECLARATION = re.compile(r"  void\* (\w+) = \(\(void\*\)\(\(char\*\)buf_dyn_shmem \+ (\d+)\)\);")


def validate_qualification(report):
    """Reject incomplete/failed adapter evidence before accepting real inputs."""
    from .attention_storage import STAGES
    required = {"probe": "official_attention_PTX_memory_adapter", "status": "pass", "model": MODEL, "revision": REVISION,
                "official_arithmetic_memory_adapter_gate": "pass", "negative_control_detected": True, "tilelang_version": "0.1.12"}
    if (not isinstance(report, dict) or any(report.get(k) != v for k, v in required.items())
            or tuple(report.get("nvrtc_version", ())) != (13, 4)
            or report.get("reference", {}).get("files", {}).get("kernel.py") != REFERENCE_HASHES["kernel.py"]
            or report.get("device", {}).get("compute_capability") != "8.9"):
        raise ValueError("Missing passing qualified Attention memory provenance")
    profiles = report.get("profiles", [])
    if len(profiles) != 6 or {p.get("original_generated_cuda_sha256") for p in profiles} != set(CODEGEN_PROFILES):
        raise ValueError("Attention qualification does not cover all six code-generation profiles")
    for p in profiles:
        audit = p.get("ptx_audit", {})
        expected_profile = CODEGEN_PROFILES[p["original_generated_cuda_sha256"]]
        if (tuple(p.get(k) for k in ("heads", "dimension", "diagnostics", "original_shared_bytes")) != expected_profile
                or p.get("original_binary_reproduction") is not True or p.get("query_slab_inspection") != "pass"
                or audit.get("inverse_ptx_recovery") != "pass" or audit.get("floating_point_instructions_unchanged") is not True
                or audit.get("control_instructions_unchanged") is not True):
            raise ValueError("Attention binary or arithmetic-retention qualification failed")
    def passing(check):
        return (check.get("status") == "pass" and check.get("mismatches") == 0
                and type(check.get("elements")) is int and check["elements"] > 0
                and re.fullmatch(r"[0-9a-f]{64}", check.get("expected_sha256", "")) is not None
                and check["expected_sha256"] == check.get("actual_sha256"))
    for key in ("load_checks", "store_checks"):
        checks = report.get(key, [])
        if len(checks) != 4 or {c.get("pattern") for c in checks} != set(range(4)) or not all(passing(c) for c in checks):
            raise ValueError("Attention bit-transport qualification is incomplete or failed")
    negative = report.get("negative_control", {})
    if negative.get("status") != "fail" or not isinstance(negative.get("mismatches"), int) or negative["mismatches"] <= 0:
        raise ValueError("Attention qualification lacks a detected negative control")
    rows = report.get("cases", [])
    keys = {(r.get("heads"), r.get("dimension"), r.get("case")) for r in rows}
    if len(rows) != 36 or keys != {(h, d, i) for h, d in ((64, 256), (16, 512), (64, 512)) for i in range(12)}:
        raise ValueError("Attention qualification input coverage differs")
    common = {"PTX_instrumentation_BF16", "register_instrumentation_BF16", "PTX_original_arithmetic_vs_register_BF16"}
    common |= {"PTX_vs_register_" + name for name in STAGES}
    original = {"unmodified_instrumentation_BF16", "unmodified_vs_PTX_adapter_BF16"} | {"unmodified_vs_PTX_" + name for name in STAGES}
    for row in rows:
        checks = row.get("checks", [])
        expected = common if (row["heads"], row["dimension"]) == (64, 512) else common | original
        if len(checks) != len(expected) or {c.get("check") for c in checks} != expected or not all(passing(c) for c in checks):
            raise ValueError("Attention intermediate qualification is incomplete or failed")
    return report


def _profile(source):
    if not isinstance(source, str) or len(source) > 128 * 1024:
        raise ValueError("Excessive or invalid generated Attention CUDA")
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest not in CODEGEN_PROFILES:
        raise ValueError("Unqualified generated CUDA profile; refusing memory edits")
    return digest, CODEGEN_PROFILES[digest]


def _edit(source, edits):
    """Apply nonoverlapping specified edits, then recover every original byte."""
    edits = sorted(edits)
    previous, delta = 0, 0
    reverse = []
    for start, end, replacement, kind in edits:
        if start < previous or not start < end <= len(source):
            raise ValueError("Invalid or overlapping Attention memory edit")
        reverse.append((start + delta, start + delta + len(replacement), source[start:end], kind))
        previous = end
        delta += len(replacement) - (end - start)
    adapted = source
    for start, end, replacement, _ in reversed(edits):
        adapted = adapted[:start] + replacement + adapted[end:]
    recovered = adapted
    for start, end, original, _ in reversed(reverse):
        recovered = recovered[:start] + original + recovered[end:]
    if recovered != source:
        raise RuntimeError("Memory edit failed exact source recovery")
    return adapted


def inspect_codegen(source):
    digest, (heads, dimension, diagnostics, required) = _profile(source)
    q_bytes = heads * dimension * 2
    declarations = list(DECLARATION.finditer(source))
    expected_names = {"q_shared", "kv_shared", "o_shared", "acc_s_cast", "workspace", "workspace_1"}
    if (len(declarations) != 6 or {m[1] for m in declarations} != expected_names
            or source.count("buf_dyn_shmem") != 7 or source.count("q_shared") != 3):
        raise ValueError("Unexpected Query/shared memory uses in the generated profile")
    for match in declarations:
        name, offset = match[1], int(match[2])
        if name == "q_shared":
            if offset != 0:
                raise ValueError("Query must occupy the first shared-memory slab")
        else:
            if offset < q_bytes or offset % 1024:
                raise ValueError("A live allocation overlaps the removed Query slab")
    signatures = list(re.finditer(rf"\b{FUNCTION}\(([^\n]*)\)", source))
    if len(signatures) != 2 or signatures[0][1] != signatures[1][1]:
        raise ValueError("Unexpected generated Attention signature")
    parameters = [re.search(r"\b(\w+)$", s.strip())[1] for s in signatures[0][1].split(",")]
    if not {"b", "m", "n", "topk", "q", "kv", "o", "attn_sink", "topk_idxs"}.issubset(parameters):
        raise ValueError("Missing original Attention arguments")
    query_loads = [m for m in re.finditer(r"tl::ptx_ldmatrix_x4\([^\n]+", source) if "q_shared" in m[0]]
    if len(query_loads) != 1:
        raise ValueError("Expected one shared Query ldmatrix instruction site")
    return {"original_generated_cuda_sha256": digest, "query_slab_inspection": "pass",
                      "heads": heads, "dimension": dimension, "diagnostics": diagnostics,
                      "original_shared_bytes": required, "adapted_shared_bytes": required-q_bytes,
                      "workspace_bytes_per_CTA": q_bytes, "kernel_argument_names": parameters + ["q_workspace"],
                      "scope": "Official generated arithmetic with explicit Query memory/load adaptation; not an unmodified kernel"}


def loader_check_cuda(source):
    from .attention_ptx import qualification_helper
    digest, (heads, dimension, diagnostics, _) = _profile(source)
    if (heads, dimension, diagnostics) != (64, 512, False):
        raise ValueError("The load qualification probe requires the full uninstrumented profile")
    header = source[:source.index('extern "C"')]
    store = source.index("*(uint4*)(((bfloat16_t*)q_shared)")
    copy_start = source.rindex("  #pragma unroll", 0, store)
    copy_end = source.index("  for (int t =", store)
    copy_code = source[copy_start:copy_end]
    query_load = next(line for line in source.splitlines() if "ptx_ldmatrix_x4" in line and "q_shared" in line)
    global_load = query_load.replace("tl::ptx_ldmatrix_x4", "vesper_global_ldmatrix_x4").replace("q_shared", "q_global").replace("A_local", "A_global")
    global_load = global_load.rsplit(");", 1)[0] + ", fault);"
    helper = qualification_helper()
    code = header + helper + '''
extern "C" __global__ __launch_bounds__(256) void check_query_load(
 const bfloat16_t* q,bfloat16_t* q_workspace,unsigned* expected,unsigned* actual,int m,int fault){
extern __shared__ __align__(1024) unsigned char buffer[];
void* q_shared=buffer;
void* q_global=q_workspace+((long long)blockIdx.y*m+blockIdx.x)*32768;
''' + copy_code + copy_code.replace("q_shared", "q_global") + '''
__syncthreads();
bfloat16_t A_local[8];bfloat16_t A_global[8];
for(int ki=0;ki<32;++ki){
''' + query_load + "\n" + global_load + '''
for(int r=0;r<4;++r){
 const long long offset=((((long long)blockIdx.y*m+blockIdx.x)*32+ki)*256+threadIdx.x)*4+r;
 expected[offset]=((unsigned*)A_local)[r]; actual[offset]=((unsigned*)A_global)[r];
}
}
}
'''
    return code, {"original_generated_cuda_sha256": digest, "source_sha256": hashlib.sha256(code.encode()).hexdigest(),
                  "shared_bytes": 65536, "geometry": "32 K steps x 256 threads x 4 packed words; original Query copy/address expressions"}
