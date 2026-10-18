// SPDX-License-Identifier: MIT
// Vesper format probes. The per-32 scale formula follows DeepSeek-AI's
// MIT-licensed reference kernel.py. See THIRD_PARTY_NOTICES.md.
#include <cuda_fp8.h>
#include <cuda_bf16.h>

__device__ float fp4(unsigned char code) {
    const float table[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    unsigned int bits = __float_as_uint(table[code & 7]);
    return __uint_as_float(bits | ((unsigned int)(code & 8) << 28));
}

__device__ float scale8(unsigned char code) {
    return __uint_as_float(code == 0 ? 0x00400000u : code == 255 ? 0x7fc00000u : ((unsigned int)code << 23));
}

__device__ float fp8(unsigned char code) {
    __nv_fp8_e4m3 value;
    value.__x = code;
    return (float)value;
}

extern "C" __global__ void decode_fp4(const unsigned char* packed, const unsigned char* scales,
                                      float* output, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        unsigned char code = (packed[i / 2] >> ((i & 1) * 4)) & 15;
        output[i] = __fmul_rn(fp4(code), scale8(scales[i / 32]));
    }
}

extern "C" __global__ void encode_fp8(const float* input, unsigned char* output, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) output[i] = __nv_cvt_float_to_fp8(input[i], __NV_SATFINITE, __NV_E4M3);
}

extern "C" __global__ void activation32(const float* input, unsigned char* output,
                                       unsigned char* scales, int blocks) {
    int group = blockIdx.x * blockDim.x + threadIdx.x;
    if (group >= blocks) return;
    float amax = 1e-4f;
    for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(input[group * 32 + j]));
    unsigned int bits = __float_as_uint(__fmul_rn(amax, 1.f / 448.f));
    int exponent = (int)((bits >> 23) & 255) - 127 + ((bits & 0x7fffff) != 0);
    unsigned char code = (unsigned char)(exponent + 127);
    scales[group] = code;
    float s = scale8(code);
    for (int j = 0; j < 32; ++j) {
        float x = fminf(448.f, fmaxf(-448.f, __fdiv_rn(input[group * 32 + j], s)));
        output[group * 32 + j] = __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
    }
}

extern "C" __global__ void gemv_fp4(const unsigned char* activation, const unsigned char* act_scale,
                                   const unsigned char* weight, const unsigned char* weight_scale,
                                   unsigned short* output, int rows, int width) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float sum = 0.f;
    for (int group = 0; group < width / 32; ++group) {
        float dot = 0.f;
        for (int j = 0; j < 32; ++j) {
            int column = group * 32 + j;
            unsigned char packed = weight[row * (width / 2) + column / 2];
            unsigned char code = (packed >> ((column & 1) * 4)) & 15;
            dot = __fadd_rn(dot, __fmul_rn(fp8(activation[column]), fp4(code)));
        }
        float scaled = __fmul_rn(__fmul_rn(dot, scale8(act_scale[group])),
                                scale8(weight_scale[row * (width / 32) + group]));
        sum = __fadd_rn(sum, scaled);
    }
    output[row] = __bfloat16_as_ushort(__float2bfloat16_rn(sum));
}

// A native FP8 MMA probe for Ada+. FP4 values embed exactly in E4M3.
// Each warp computes eight output rows, with identical activations in all
// sixteen MMA rows. Only row zero is retained. This favors clarity over speed.
extern "C" __global__ void gemv_fp4_mma(const unsigned char* activation, const unsigned char* act_scale,
                                       const unsigned char* weight, const unsigned char* weight_scale,
                                       unsigned short* output, int rows, int width) {
#if __CUDA_ARCH__ >= 890
    int lane = threadIdx.x & 31;
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int row_base = warp * 8;
    if (row_base >= rows) return;
    int quad = lane & 3;
    int b_row = row_base + (lane >> 2);
    float sum0 = 0.f, sum1 = 0.f;
    const unsigned char table[8] = {0, 48, 56, 60, 64, 68, 72, 76};
    for (int group = 0; group < width / 32; ++group) {
        unsigned int a0 = 0, a1 = 0, b0 = 0, b1 = 0;
        for (int j = 0; j < 4; ++j) {
            int k0 = group * 32 + quad * 4 + j;
            int k1 = k0 + 16;
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
        float d0, d1, d2, d3;
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
            : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
            : "r"(a0), "r"(a0), "r"(a1), "r"(a1), "r"(b0), "r"(b1),
              "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f));
        int out0 = row_base + quad * 2, out1 = out0 + 1;
        float sa = scale8(act_scale[group]);
        if (out0 < rows) sum0 = __fadd_rn(sum0, __fmul_rn(__fmul_rn(d0, sa), scale8(weight_scale[out0 * (width / 32) + group])));
        if (out1 < rows) sum1 = __fadd_rn(sum1, __fmul_rn(__fmul_rn(d1, sa), scale8(weight_scale[out1 * (width / 32) + group])));
    }
    if (lane < 4) {
        int out0 = row_base + lane * 2, out1 = out0 + 1;
        if (out0 < rows) output[out0] = __bfloat16_as_ushort(__float2bfloat16_rn(sum0));
        if (out1 < rows) output[out1] = __bfloat16_as_ushort(__float2bfloat16_rn(sum1));
    }
#else
    // Do not silently substitute a different arithmetic path on Ampere.
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) output[row] = 0x7fc0; // unsupported path: NaN
#endif
}

// Explicit Ampere candidate: exactly widen FP8/FP4 operands to BF16 and use
// two K=16 MMA instructions per native K=32 scaling block. This path must pass
// the official FP8 reference check; representability alone is not sufficient.
extern "C" __global__ void gemv_fp4_bf16_mma(const unsigned char* activation, const unsigned char* act_scale,
                                            const unsigned char* weight, const unsigned char* weight_scale,
                                            unsigned short* output, int rows, int width) {
    int lane = threadIdx.x & 31;
    int row_base = ((blockIdx.x * blockDim.x + threadIdx.x) >> 5) * 8;
    if (row_base >= rows) return;
    int quad = lane & 3, b_row = row_base + (lane >> 2);
    float sum0 = 0.f, sum1 = 0.f;
    for (int group = 0; group < width / 32; ++group) {
        float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
        for (int half = 0; half < 2; ++half) {
            unsigned int a0 = 0, a1 = 0, b0 = 0, b1 = 0;
            for (int j = 0; j < 2; ++j) {
                int k0 = group * 32 + half * 16 + quad * 2 + j;
                int k1 = k0 + 8;
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
        int out0 = row_base + quad * 2, out1 = out0 + 1;
        float sa = scale8(act_scale[group]);
        if (out0 < rows) sum0 = __fadd_rn(sum0, __fmul_rn(__fmul_rn(d0, sa), scale8(weight_scale[out0 * (width / 32) + group])));
        if (out1 < rows) sum1 = __fadd_rn(sum1, __fmul_rn(__fmul_rn(d1, sa), scale8(weight_scale[out1 * (width / 32) + group])));
    }
    if (lane < 4) {
        int out0 = row_base + lane * 2, out1 = out0 + 1;
        if (out0 < rows) output[out0] = __bfloat16_as_ushort(__float2bfloat16_rn(sum0));
        if (out1 < rows) output[out1] = __bfloat16_as_ushort(__float2bfloat16_rn(sum1));
    }
}
