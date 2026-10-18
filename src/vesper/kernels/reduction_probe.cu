// SPDX-License-Identifier: MIT
// Appended to format_probe.cu only for bounded intermediate-arithmetic probes.
// Records [row, K/32 block, {dot, scaled_dot, running_sum}] in FP32.

template <bool NativeFP8>
__device__ void trace_fp4_blocks(const unsigned char* activation, const unsigned char* act_scale,
                               const unsigned char* weight, const unsigned char* weight_scale,
                               float* trace, int rows, int width) {
    int lane = threadIdx.x & 31, quad = lane & 3;
    int row_base = ((blockIdx.x * blockDim.x + threadIdx.x) >> 5) * 8;
    if (row_base >= rows) return;
    int b_row = row_base + (lane >> 2), groups = width / 32;
    float sum0 = 0.f, sum1 = 0.f;
    const unsigned char table[8] = {0, 48, 56, 60, 64, 68, 72, 76};
    for (int group = 0; group < groups; ++group) {
        float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
        if constexpr (NativeFP8) {
#if __CUDA_ARCH__ >= 890
            unsigned int a0 = 0, a1 = 0, b0 = 0, b1 = 0;
            for (int j = 0; j < 4; ++j) {
                int k0 = group * 32 + quad * 4 + j, k1 = k0 + 16;
                a0 |= (unsigned int)activation[k0] << (j * 8);
                a1 |= (unsigned int)activation[k1] << (j * 8);
                unsigned char c0 = 0, c1 = 0;
                if (b_row < rows) {
                    unsigned char p0 = weight[b_row * (width / 2) + k0 / 2];
                    unsigned char p1 = weight[b_row * (width / 2) + k1 / 2];
                    c0 = (p0 >> ((k0 & 1) * 4)) & 15;
                    c1 = (p1 >> ((k1 & 1) * 4)) & 15;
                }
                b0 |= (unsigned int)(table[c0 & 7] | ((c0 & 8) << 4)) << (j * 8);
                b1 |= (unsigned int)(table[c1 & 7] | ((c1 & 8) << 4)) << (j * 8);
            }
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
                : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                : "r"(a0), "r"(a0), "r"(a1), "r"(a1), "r"(b0), "r"(b1),
                  "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f));
#else
            d0 = d1 = __uint_as_float(0x7fc00000u);
#endif
        } else {
            for (int half = 0; half < 2; ++half) {
                unsigned int a0 = 0, a1 = 0, b0 = 0, b1 = 0;
                for (int j = 0; j < 2; ++j) {
                    int k0 = group * 32 + half * 16 + quad * 2 + j, k1 = k0 + 8;
                    unsigned short ac0 = __bfloat16_as_ushort(__float2bfloat16_rn(fp8(activation[k0])));
                    unsigned short ac1 = __bfloat16_as_ushort(__float2bfloat16_rn(fp8(activation[k1])));
                    a0 |= (unsigned int)ac0 << (j * 16);
                    a1 |= (unsigned int)ac1 << (j * 16);
                    unsigned char c0 = 0, c1 = 0;
                    if (b_row < rows) {
                        unsigned char p0 = weight[b_row * (width / 2) + k0 / 2];
                        unsigned char p1 = weight[b_row * (width / 2) + k1 / 2];
                        c0 = (p0 >> ((k0 & 1) * 4)) & 15;
                        c1 = (p1 >> ((k1 & 1) * 4)) & 15;
                    }
                    b0 |= (unsigned int)__bfloat16_as_ushort(__float2bfloat16_rn(fp4(c0))) << (j * 16);
                    b1 |= (unsigned int)__bfloat16_as_ushort(__float2bfloat16_rn(fp4(c1))) << (j * 16);
                }
                float e0, e1, e2, e3;
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
                    : "=f"(e0), "=f"(e1), "=f"(e2), "=f"(e3)
                    : "r"(a0), "r"(a0), "r"(a1), "r"(a1), "r"(b0), "r"(b1),
                      "f"(d0), "f"(d1), "f"(d2), "f"(d3));
                d0 = e0; d1 = e1; d2 = e2; d3 = e3;
            }
        }
        int out0 = row_base + quad * 2, out1 = out0 + 1;
        float scaled0 = 0.f, scaled1 = 0.f, sa = scale8(act_scale[group]);
        if (out0 < rows) {
            scaled0 = __fmul_rn(__fmul_rn(d0, sa), scale8(weight_scale[out0 * groups + group]));
            sum0 = __fadd_rn(sum0, scaled0);
        }
        if (out1 < rows) {
            scaled1 = __fmul_rn(__fmul_rn(d1, sa), scale8(weight_scale[out1 * groups + group]));
            sum1 = __fadd_rn(sum1, scaled1);
        }
        if (lane < 4) {
            if (out0 < rows) {
                int offset = (out0 * groups + group) * 3;
                trace[offset] = d0; trace[offset + 1] = scaled0; trace[offset + 2] = sum0;
            }
            if (out1 < rows) {
                int offset = (out1 * groups + group) * 3;
                trace[offset] = d1; trace[offset + 1] = scaled1; trace[offset + 2] = sum1;
            }
        }
    }
}

extern "C" __global__ void trace_fp4_native(const unsigned char* a, const unsigned char* sa,
                                           const unsigned char* w, const unsigned char* sw,
                                           float* trace, int rows, int width) {
    trace_fp4_blocks<true>(a, sa, w, sw, trace, rows, width);
}

extern "C" __global__ void trace_fp4_bf16(const unsigned char* a, const unsigned char* sa,
                                         const unsigned char* w, const unsigned char* sw,
                                         float* trace, int rows, int width) {
    trace_fp4_blocks<false>(a, sa, w, sw, trace, rows, width);
}
