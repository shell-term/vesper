// SPDX-License-Identifier: MIT
// Match single-rank DeepSeek MoE.forward: ascending expert IDs, FP32 sums,
// shared expert last, then one BF16 conversion. Each token occurs once per ID.
extern "C" __global__ void moe_zero(float* sum, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) sum[i] = 0.f;
}

extern "C" __global__ void moe_add_expert(float* sum, const unsigned short* output,
                                          const int* token_rows, int count, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count * width) return;
    int target = token_rows[i / width] * width + i % width;
    sum[target] = __fadd_rn(sum[target], __bfloat162float(__ushort_as_bfloat16(output[i])));
}

extern "C" __global__ void moe_add_shared(float* sum, const unsigned short* shared, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) sum[i] = __fadd_rn(sum[i], __bfloat162float(__ushort_as_bfloat16(shared[i])));
}

extern "C" __global__ void moe_finish(const float* sum, unsigned short* output, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) output[i] = __bfloat16_as_ushort(__float2bfloat16_rn(sum[i]));
}
