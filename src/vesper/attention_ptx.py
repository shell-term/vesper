"""Preserve original PTX arithmetic/control instructions; virtualize Query RAM.

The caller must establish pinned CUDA provenance and reproduce the original
binary with the same compiler before treating this as a reference adapter.
"""

from __future__ import annotations

import hashlib
import re

from .attention_memory import CODEGEN_PROFILES, FUNCTION, _edit


def arithmetic_lines(ptx):
    result = []
    for line in ptx.splitlines():
        match = re.match(r"\s*(?:@!?%\w+\s+)?([a-z][\w.:]+)\s+([^;]+);", line)
        if (match and not match[1].startswith(("ld.", "st."))
                and (".f32" in match[1] or ".bf16" in match[1] or match[1].startswith("mma.sync"))):
            result.append(line)
    return result


def control_lines(ptx):
    return [line for line in ptx.splitlines()
            if re.match(r"\s*(?:@!?%\w+\s+)?(?:bra(?:\.uni)?|bar\.sync|ret)\b", line)]


def global_matrix_load(destinations, pointer, r32, r64, *, predicate="", fault=None):
    """Integer-only PTX. Same emitter is exercised against hardware ldmatrix."""
    if len(destinations) != 4 or len(r32) < 9 or len(r64) < 3:
        raise ValueError("Invalid matrix-load register layout")
    lines = [f"mov.b64 {{{r32[0]}, {r32[1]}}}, {pointer};",
             f"mov.u32 {r32[2]}, %laneid;", f"shr.u32 {r32[3]}, {r32[2]}, 2;",
             f"and.b32 {r32[4]}, {r32[2]}, 3;"]
    if fault is not None:
        lines.append(f"xor.b32 {r32[4]}, {r32[4]}, {fault};")
    lines += [f"shl.b32 {r32[4]}, {r32[4]}, 2;", f"cvt.u64.u32 {r64[2]}, {r32[4]};"]
    for i, destination in enumerate(destinations):
        lines += [f"add.u32 {r32[5]}, {r32[3]}, {8*i};",
                  f"shfl.sync.idx.b32 {r32[6]}, {r32[0]}, {r32[5]}, 31, -1;",
                  f"shfl.sync.idx.b32 {r32[7]}, {r32[1]}, {r32[5]}, 31, -1;",
                  f"mov.b64 {r64[1]}, {{{r32[6]}, {r32[7]}}};",
                  f"add.u64 {r64[1]}, {r64[1]}, {r64[2]};",
                  f"ld.global.cg.u32 {destination}, [{r64[1]}];"]
    return "\n".join((predicate + " " if predicate else "") + line for line in lines)


def qualification_helper():
    """C++ wrapper for the exact production PTX load emitter, plus fault input."""
    assembly = ("{\n.reg .b32 q32_<9>;\n.reg .b64 q64_<3>;\n" + global_matrix_load(
        [f"%{i}" for i in range(4)], "%4", [f"q32_{i}" for i in range(9)], [f"q64_{i}" for i in range(3)], fault="%5") + "\n}")
    assembly = assembly.replace("%laneid", "%%laneid")
    quoted = "\n".join('"' + line.replace("\\", "\\\\").replace('"', '\\"') + '\\n"' for line in assembly.splitlines())
    return '''__device__ __forceinline__ void vesper_global_ldmatrix_x4(const void* pointer,void* output,int fault=0){
unsigned* words=(unsigned*)output;
asm volatile(
''' + quoted + '''
: "=r"(words[0]),"=r"(words[1]),"=r"(words[2]),"=r"(words[3])
: "l"((unsigned long long)pointer),"r"(fault):"memory");
}
'''


def virtualize_ptx(ptx, profile):
    return _virtualize_ptx_for(ptx, profile, CODEGEN_PROFILES, version="9.4", target="sm_89")


def _virtualize_ptx_for(ptx, profile, profiles, *, version, target):
    """Common integer transport transform, after the caller's platform pins."""
    if (not isinstance(ptx, str) or len(ptx) > 2 * 1024**2
            or profile.get("original_generated_cuda_sha256") not in profiles):
        raise ValueError("Unqualified or excessive Attention PTX")
    h, d, diagnostics, required = profiles[profile["original_generated_cuda_sha256"]]
    if (profile["heads"], profile["dimension"], profile["diagnostics"], profile["original_shared_bytes"]) != (h, d, diagnostics, required):
        raise ValueError("PTX profile metadata differs from pinned CUDA")
    if (re.findall(r"(?m)^\.version ([\d.]+)\s*$", ptx) != [version]
            or re.findall(r"(?m)^\.target ([\w]+)\s*$", ptx) != [target] or "%vesper" in ptx):
        raise ValueError("Unqualified PTX version, target, or register namespace")
    entry = re.search(rf"\.visible \.entry {FUNCTION}\((.*?)\n\)", ptx, re.S)
    if entry is None or ptx.count(".entry ") != 1:
        raise ValueError("Expected one original PTX entry")
    parameters = re.findall(r"\.param \.(u64|u32) (\w+)", entry[1])
    names = profile["kernel_argument_names"][:-1]
    if len(parameters) != len(names) or any(kind != ("u32" if name in ("b", "m", "n", "topk") else "u64")
                                           for (kind, _), name in zip(parameters, names)):
        raise ValueError("PTX and CUDA parameter layouts differ")
    q_bytes = h * d * 2
    params = entry[1] + ",\n\t.param .u64 vesper_qworkspace"
    edits = [(entry.start(1), entry.end(1), params, "workspace_parameter")]
    declarations = list(re.finditer(r"\t\.reg [^\n]+;", ptx))
    if len(declarations) != 5:
        raise ValueError("Unexpected PTX register declarations")
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
 mov.u32 %vesper32_15, buf_dyn_shmem;
'''
    # Replace a nonempty existing span so exact recovery also covers insertion.
    edits.append((insertion-1, insertion, ";" + prolog, "integer_registers_and_prolog"))
    sites = []
    for line in re.finditer(r"^[ \t]+[^\n]+;", ptx, re.M):
        text = line[0]
        if ".shared" not in text or text.lstrip().startswith(".extern"):
            continue
        instruction = re.match(r"\s*(ldmatrix\.[\w.]+|(?:ld|st)\.shared\.[\w.]+)\s+", text)
        address = re.search(r"\[(%r\d+)(?:\+(\d+))?\]", text)
        if not instruction or not address:
            raise ValueError("Unsupported or predicated PTX shared access")
        opcode = instruction[1]
        if opcode not in ("st.shared.v4.u32", "st.shared.u32", "st.shared.f32", "ld.shared.f32", "ld.shared.v4.u32",
                          "ldmatrix.sync.aligned.x4.m8n8.shared.b16", "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16",
                          "ldmatrix.sync.aligned.x2.m8n8.shared.b16"):
            raise ValueError(f"Unqualified PTX shared opcode: {opcode}")
        prefix = f"add.u32 %vesper32_0, {address[1]}, {address[2] or '0'};\n"
        prefix += f"sub.u32 %vesper32_2, %vesper32_0, {q_bytes};\n"
        relocated = text[:address.start()] + "[%vesper32_2]" + text[address.end():]
        route_query = opcode in ("st.shared.v4.u32", "ldmatrix.sync.aligned.x4.m8n8.shared.b16")
        if route_query:
            prefix += "sub.u32 %vesper32_1, %vesper32_0, %vesper32_15;\n"
            prefix += f"setp.lo.u32 %vesperp0, %vesper32_1, {q_bytes};\n"
            prefix += "cvt.u64.u32 %vesper64_0, %vesper32_1;\nadd.u64 %vesper64_0, %vesper64_7, %vesper64_0;\n"
            replacement = prefix + "@!%vesperp0 " + relocated.strip() + "\n"
            if opcode == "st.shared.v4.u32":
                global_store = text.replace("st.shared", "st.global")
                global_store = re.sub(r"\[[^]]+\]", "[%vesper64_0]", global_store)
                replacement += "@%vesperp0 " + global_store.strip()
            else:
                destinations = re.search(r"\{([^}]+)\}", text)[1].split(",")
                replacement += global_matrix_load([s.strip() for s in destinations], "%vesper64_0",
                                                   [f"%vesper32_{i}" for i in range(3, 12)],
                                                   [f"%vesper64_{i}" for i in range(1, 4)], predicate="@%vesperp0")
        else:
            replacement = prefix + relocated
        edits.append((line.start(), line.end(), replacement, "shared_memory_access"))
        sites.append(opcode)
    if not sites:
        raise ValueError("Missing PTX shared accesses")
    adapted = _edit(ptx, edits)
    if arithmetic_lines(adapted) != arithmetic_lines(ptx) or control_lines(adapted) != control_lines(ptx):
        raise ValueError("PTX arithmetic or control instructions changed")
    return adapted, {"original_ptx_sha256": hashlib.sha256(ptx.encode()).hexdigest(),
                     "adapted_ptx_sha256": hashlib.sha256(adapted.encode()).hexdigest(),
                     "inverse_ptx_recovery": "pass", "floating_point_instructions_unchanged": True,
                     "control_instructions_unchanged": True, "floating_point_instruction_count": len(arithmetic_lines(ptx)),
                     "control_instruction_count": len(control_lines(ptx)), "adapted_shared_access_sites": len(sites),
                     "arithmetic_instruction_sha256": hashlib.sha256("\n".join(arithmetic_lines(ptx)).encode()).hexdigest()}
