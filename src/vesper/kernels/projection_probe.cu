// SPDX-License-Identifier: MIT
// DeepSeek's per-32 FP8 GEMM order; see third_party/DeepSeek-LICENSE.
// Each input row is independent; retain K=32 MMA, two multiplies and an add.
extern "C" __global__ void projection_input(const unsigned short* input, float* output, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) output[i] = __bfloat162float(__ushort_as_bfloat16(input[i]));
}

extern "C" __global__ void projection_fp8(const unsigned char* activation,
                                         const unsigned char* act_scale,
                                         const unsigned char* weight,
                                         const unsigned char* weight_scale,
                                         float* trace, unsigned short* output,
                                         int rows, int width, int tokens) {
#if __CUDA_ARCH__ >= 890
    int lane = threadIdx.x & 31;
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int token = warp / (rows / 8), row_base = (warp % (rows / 8)) * 8;
    if (token >= tokens) return;
    int quad = lane & 3, b_row = row_base + (lane >> 2);
    float sum0 = 0.f, sum1 = 0.f;
    for (int group = 0; group < width / 32; ++group) {
        unsigned int a0 = 0, a1 = 0, b0 = 0, b1 = 0;
        for (int j = 0; j < 4; ++j) {
            int k0 = group * 32 + quad * 4 + j, k1 = k0 + 16;
            a0 |= (unsigned int)activation[token * width + k0] << (j * 8);
            a1 |= (unsigned int)activation[token * width + k1] << (j * 8);
            b0 |= (unsigned int)weight[b_row * width + k0] << (j * 8);
            b1 |= (unsigned int)weight[b_row * width + k1] << (j * 8);
        }
        float d0, d1, d2, d3;
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
            : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
            : "r"(a0), "r"(a0), "r"(a1), "r"(a1), "r"(b0), "r"(b1),
              "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f));
        int out0 = row_base + quad * 2, out1 = out0 + 1;
        float sa = scale8(act_scale[token * (width / 32) + group]);
        sum0 = __fadd_rn(sum0, __fmul_rn(__fmul_rn(d0, sa), scale8(weight_scale[(out0 / 32) * (width / 32) + group])));
        sum1 = __fadd_rn(sum1, __fmul_rn(__fmul_rn(d1, sa), scale8(weight_scale[(out1 / 32) * (width / 32) + group])));
    }
    if (lane < 4) {
        int i = token * rows + row_base + lane * 2;
        trace[i] = sum0;
        trace[i + 1] = sum1;
        output[i] = __bfloat16_as_ushort(__float2bfloat16_rn(sum0));
        output[i + 1] = __bfloat16_as_ushort(__float2bfloat16_rn(sum1));
    }
#endif
}

// Preserve convert.py's FP32 E4M3 * E8M0, then BF16 storage boundary.
extern "C" __global__ void projection_dequant_wo_a(const unsigned char* weight,
                                                  const unsigned char* scales,
                                                  unsigned short* output, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        int row = i / 4096, column = i % 4096;
        float value = __fmul_rn(fp8(weight[i]), scale8(scales[(row / 32) * 128 + column / 32]));
        output[i] = __bfloat16_as_ushort(__float2bfloat16_rn(value));
    }
}
