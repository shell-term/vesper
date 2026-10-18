// SPDX-License-Identifier: MIT
// Original E4M3 weights with one E8M0 scale per 32 x 32 weight block.
// Retain the reference's K=32 MMA and two multiplies followed by an FP32 add.
extern "C" __global__ void gemv_fp8_shared(const unsigned char* activation,
                                          const unsigned char* act_scale,
                                          const unsigned char* weight,
                                          const unsigned char* weight_scale,
                                          unsigned short* output, int rows, int width) {
#if __CUDA_ARCH__ >= 890
    int lane = threadIdx.x & 31;
    int row_base = ((blockIdx.x * blockDim.x + threadIdx.x) >> 5) * 8;
    if (row_base >= rows) return;
    int quad = lane & 3, b_row = row_base + (lane >> 2);
    float sum0 = 0.f, sum1 = 0.f;
    for (int group = 0; group < width / 32; ++group) {
        unsigned int a0 = 0, a1 = 0, b0 = 0, b1 = 0;
        for (int j = 0; j < 4; ++j) {
            int k0 = group * 32 + quad * 4 + j, k1 = k0 + 16;
            a0 |= (unsigned int)activation[k0] << (j * 8);
            a1 |= (unsigned int)activation[k1] << (j * 8);
            if (b_row < rows) {
                b0 |= (unsigned int)weight[b_row * width + k0] << (j * 8);
                b1 |= (unsigned int)weight[b_row * width + k1] << (j * 8);
            }
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
        if (out0 < rows)
            sum0 = __fadd_rn(sum0, __fmul_rn(__fmul_rn(d0, sa), scale8(weight_scale[(out0 / 32) * (width / 32) + group])));
        if (out1 < rows)
            sum1 = __fadd_rn(sum1, __fmul_rn(__fmul_rn(d1, sa), scale8(weight_scale[(out1 / 32) * (width / 32) + group])));
    }
    if (lane < 4) {
        int out0 = row_base + lane * 2, out1 = out0 + 1;
        if (out0 < rows) output[out0] = __bfloat16_as_ushort(__float2bfloat16_rn(sum0));
        if (out1 < rows) output[out1] = __bfloat16_as_ushort(__float2bfloat16_rn(sum1));
    }
#else
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) output[row] = 0x7fc0;
#endif
}
