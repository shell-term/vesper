// SPDX-License-Identifier: MIT
// DeepSeek Indexer arithmetic and pinned ATen BF16 reduction/top-k gather.
// See THIRD_PARTY_NOTICES.md and retained DeepSeek/PyTorch licenses.
__device__ unsigned char index_fp4(float value) {
    const float levels[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    float v = fabsf(value), distance = v;
    int best = 0;
    for (int k = 1; k < 8; ++k) {
        float d = fabsf(__fadd_rn(v, -levels[k]));
        if (d < distance || (d == distance && !(k & 1))) { best = k; distance = d; }
    }
    return (unsigned char)(best | ((__float_as_uint(value) >> 28) & 8));
}

extern "C" __global__ void index_round32(const unsigned short* input, unsigned short* output,
                                         unsigned char* packed, unsigned char* scales, int groups) {
    int lane = threadIdx.x & 31;
    int group = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (group >= groups) return;
    int i = group * 32 + lane;
    float value = __bfloat162float(__ushort_as_bfloat16(input[i]));
    float maximum = fabsf(value);
    for (int offset = 16; offset; offset >>= 1)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, offset));
    maximum = fmaxf(maximum, 6.f * 0x1p-126f);
    unsigned int bits = __float_as_uint(__fmul_rn(maximum, 1.f / 6.f));
    unsigned char scale_code = (unsigned char)(((bits >> 23) & 255) + ((bits & 0x7fffff) != 0));
    float scale = scale8(scale_code);
    float normalized = fminf(6.f, fmaxf(-6.f, __fdiv_rn(value, scale)));
    unsigned char code = index_fp4(normalized);
    unsigned char other = (unsigned char)__shfl_xor_sync(0xffffffffu, (int)code, 1);
    if (!(lane & 1)) packed[i / 2] = code | (other << 4);
    if (!lane) scales[group] = scale_code;
    output[i] = __bfloat16_as_ushort(__float2bfloat16_rn(__fmul_rn(fp4(code), scale)));
}

extern "C" __global__ void index_scale_weights(const unsigned short* raw, unsigned short* scaled, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) scaled[i] = __bfloat16_as_ushort(__float2bfloat16_rn(__fmul_rn(__bfloat162float(__ushort_as_bfloat16(raw[i])), 1.f / 64.f)));
}

extern "C" __global__ void index_products(const unsigned short* dot, const unsigned short* weights,
                                          unsigned short* relu, unsigned short* product, int positions, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float x = __bfloat162float(__ushort_as_bfloat16(dot[i]));
    float r = x <= 0.f ? 0.f : x;
    relu[i] = __bfloat16_as_ushort(__float2bfloat16_rn(r));
    float w = __bfloat162float(__ushort_as_bfloat16(weights[i / positions]));
    product[i] = __bfloat16_as_ushort(__float2bfloat16_rn(__fmul_rn(r, w)));
}

extern "C" __global__ void index_reduce_heads(const unsigned short* products, float* accumulated,
                                              unsigned short* scores, int positions, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int row = i / positions, column = i % positions;
    float sum;
    if (positions == 1) {
        float values[32];
        for (int h = 0; h < 32; ++h) values[h] = __fadd_rn(0.f, __bfloat162float(__ushort_as_bfloat16(products[row * 32 + h])));
        for (int offset = 16; offset; offset >>= 1)
            for (int h = 0; h < offset; ++h) values[h] = __fadd_rn(values[h], values[h + offset]);
        sum = values[0];
    } else {
        float values[4] = {};
        for (int h = 0; h < 32; ++h)
            values[h & 3] = __fadd_rn(values[h & 3], __bfloat162float(__ushort_as_bfloat16(products[(row * 32 + h) * positions + column])));
        sum = __fadd_rn(__fadd_rn(__fadd_rn(values[0], values[1]), values[2]), values[3]);
    }
    accumulated[i] = sum;
    scores[i] = __bfloat16_as_ushort(__float2bfloat16_rn(sum));
}

__device__ unsigned int index_ordered(unsigned short value) {
    return value ^ ((value & 0x8000u) ? 0xffffu : 0x8000u);
}

// Scalar radix threshold, then the pinned sbtopk selected set. Position sort
// removes gather ordering; equal-threshold membership still follows first ID.
__device__ unsigned int index_threshold(const unsigned short* scores, int count, int k) {
    unsigned int mask = 0, desired = 0;
    int rank = k;
    for (int shift = 14; shift >= 0; shift -= 2) {
        int bins[4] = {};
        for (int i = 0; i < count; ++i) {
            unsigned int key = index_ordered(scores[i]);
            if ((key & mask) == desired) ++bins[(key >> shift) & 3];
        }
        int chosen = 3;
        for (; chosen > 0 && rank > bins[chosen]; --chosen) rank -= bins[chosen];
        mask |= 3u << shift;
        desired |= (unsigned int)chosen << shift;
    }
    return desired;
}

extern "C" __global__ void index_causal(const unsigned short* input, unsigned short* masked,
                                        const int* lengths, int positions, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) masked[i] = i % positions < lengths[i / positions] ? input[i] : 0xff80;
}

extern "C" __global__ void index_block_scores(const unsigned short* scores, unsigned short* blocks,
                                               const int* lengths, int positions, int num_blocks, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int row = i / num_blocks, block = i % num_blocks;
    unsigned short maximum = 0xff80;
    for (int j = 0; j < 8; ++j) {
        int p = block * 8 + j;
        unsigned short v = p < positions ? scores[row * positions + p] : 0xff80;
        float a = __bfloat162float(__ushort_as_bfloat16(maximum)), b = __bfloat162float(__ushort_as_bfloat16(v));
        maximum = a > b ? maximum : v;
    }
    int last = lengths[row] ? (lengths[row] - 1) / 8 : -1;
    blocks[i] = block == last ? 0x7f80 : maximum;
}

extern "C" __global__ void index_candidates(const unsigned short* blocks, unsigned char* keep,
                                             int num_blocks, int positions, int rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const unsigned short* values = blocks + row * num_blocks;
    int k = num_blocks < 2048 ? num_blocks : 2048;
    unsigned int threshold = index_threshold(values, num_blocks, k);
    int remaining = k;
    for (int j = 0; j < num_blocks; ++j) remaining -= index_ordered(values[j]) > threshold;
    for (int j = 0; j < num_blocks; ++j) {
        unsigned int key = index_ordered(values[j]);
        bool selected = key > threshold;
        if (key == threshold && remaining > 0) { selected = true; --remaining; }
        selected = selected && values[j] != 0xff80;
        for (int p = j * 8; p < (j + 1) * 8 && p < positions; ++p) keep[row * positions + p] = selected;
    }
}

extern "C" __global__ void index_candidate_mask(const unsigned short* causal, const unsigned char* candidates,
                                                unsigned short* output, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) output[i] = candidates[i] ? causal[i] : 0xff80;
}

extern "C" __global__ void index_select(const unsigned short* scores, const int* lengths,
                                        int* selected, int positions, int k, int rows, int offset) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const unsigned short* values = scores + row * positions;
    unsigned int threshold = index_threshold(values, positions, k);
    int remaining = k;
    for (int j = 0; j < positions; ++j) remaining -= index_ordered(values[j]) > threshold;
    int slot = 0;
    for (int j = 0; j < positions; ++j) {
        unsigned int key = index_ordered(values[j]);
        bool chosen = key > threshold;
        if (key == threshold && remaining > 0) { chosen = true; --remaining; }
        if (chosen) selected[row * k + slot++] = j < lengths[row] ? j + offset : -1;
    }
}
