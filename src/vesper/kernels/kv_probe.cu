// SPDX-License-Identifier: MIT
// Per-32 quant+dequant follows pinned DeepSeek act_quant_kernel(inplace=True).
// Ring slots follow Attention._window_kv; see third_party/DeepSeek-LICENSE.
extern "C" __global__ void kv_round32(const unsigned short* input,
                                      unsigned short* rounded, unsigned char* codes,
                                      unsigned char* scales, int groups) {
    int lane = threadIdx.x & 31;
    int group = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (group >= groups) return;
    int index = group * 32 + lane;
    float value = __bfloat162float(__ushort_as_bfloat16(input[index]));
    float maximum = fabsf(value);
    for (int offset = 16; offset; offset >>= 1)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, offset));
    maximum = fmaxf(maximum, 1e-4f);
    unsigned int bits = __float_as_uint(__fmul_rn(maximum, 1.f / 448.f));
    unsigned char scale_code = (unsigned char)(((bits >> 23) & 255) + ((bits & 0x7fffff) != 0));
    float scale = scale8(scale_code);
    float normalized = fminf(448.f, fmaxf(-448.f, __fdiv_rn(value, scale)));
    unsigned char code = __nv_cvt_float_to_fp8(normalized, __NV_SATFINITE, __NV_E4M3);
    codes[index] = code;
    if (lane == 0) scales[group] = scale_code;
    rounded[index] = __bfloat16_as_ushort(__float2bfloat16_rn(__fmul_rn(fp8(code), scale)));
}

extern "C" __global__ void kv_stage_window(const unsigned short* previous,
                                           const unsigned short* rounded,
                                           unsigned short* staged,
                                           int batch, int sequence, int position) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch * 128 * 512) return;
    int row = (i / 512) % 128;
    int b = i / (128 * 512);
    int column = i % 512;
    if (position == 0) {
        int token = sequence - 1 - ((sequence - 1 - row + 128) % 128);
        staged[i] = token >= 0 ? rounded[(b * sequence + token) * 512 + column] : 0;
    } else {
        staged[i] = row == position % 128 ? rounded[b * 512 + column] : previous[i];
    }
}

extern "C" __global__ void kv_copy_window(const unsigned short* source,
                                          unsigned short* target, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) target[i] = source[i];
}

extern "C" __global__ void kv_zero_window(unsigned short* target, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) target[i] = 0;
}
