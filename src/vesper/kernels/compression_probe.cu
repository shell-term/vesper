// SPDX-License-Identifier: MIT
// DeepSeek Compressor/FP4 rounding and pinned ATen spatial softmax/sum order.
// See THIRD_PARTY_NOTICES.md and the retained DeepSeek/PyTorch licenses.
extern "C" __global__ void compression_promote(const unsigned short* input, float* output, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) output[i] = __bfloat162float(__ushort_as_bfloat16(input[i]));
}

extern "C" __global__ void compression_pool2(const float* kv, const float* scores,
                                             float* maxima, float* exponents, float* sums,
                                             float* weights, float* products, float* pooled,
                                             unsigned short* rounded, int rows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * 512) return;
    int group = i / 512, column = i % 512;
    int first = group * 1024 + column, second = first + 512;
    float maximum = fmaxf(scores[first], scores[second]);
    float a = expf(__fadd_rn(scores[first], -maximum));
    float b = expf(__fadd_rn(scores[second], -maximum));
    float sum = __fadd_rn(__fadd_rn(0.f, a), b);
    float wa = __fdiv_rn(a, sum), wb = __fdiv_rn(b, sum);
    float pa = __fmul_rn(kv[first], wa), pb = __fmul_rn(kv[second], wb);
    float result = __fadd_rn(__fadd_rn(0.f, pa), pb);
    maxima[i] = maximum; sums[i] = sum;
    exponents[first] = a; exponents[second] = b;
    weights[first] = wa; weights[second] = wb;
    products[first] = pa; products[second] = pb;
    pooled[i] = result;
    rounded[i] = __bfloat16_as_ushort(__float2bfloat16_rn(result));
}

__device__ unsigned char compression_fp4_code(float value) {
    float v = fabsf(value);
    const float levels[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    int best = 0;
    float distance = v;
    for (int k = 1; k < 8; ++k) {
        float d = fabsf(__fadd_rn(v, -levels[k]));
        if (d < distance || (d == distance && !(k & 1))) { best = k; distance = d; }
    }
    return (unsigned char)(best | ((__float_as_uint(value) >> 28) & 8));
}

extern "C" __global__ void compression_round16(const unsigned short* input,
                                                unsigned short* rounded, unsigned char* packed,
                                                unsigned char* scales, int groups) {
    int lane = threadIdx.x & 15;
    int group = (blockIdx.x * blockDim.x + threadIdx.x) >> 4;
    if (group >= groups) return;
    int index = group * 16 + lane;
    float value = __bfloat162float(__ushort_as_bfloat16(input[index]));
    float maximum = fabsf(value);
    for (int offset = 8; offset; offset >>= 1)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, offset, 16));
    maximum = fmaxf(maximum, 6.f * 0x1p-9f);
    unsigned char scale_code = __nv_cvt_float_to_fp8(__fdiv_rn(maximum, 6.f), __NV_SATFINITE, __NV_E4M3);
    float scale = fp8(scale_code);
    float normalized = fminf(6.f, fmaxf(-6.f, __fdiv_rn(value, scale)));
    unsigned char code = compression_fp4_code(normalized);
    unsigned char neighbor = (unsigned char)__shfl_xor_sync(0xffffffffu, (int)code, 1, 16);
    if (!(lane & 1)) packed[index / 2] = code | (neighbor << 4);
    if (!lane) scales[group] = scale_code;
    rounded[index] = __bfloat16_as_ushort(__float2bfloat16_rn(__fmul_rn(fp4(code), scale)));
}
