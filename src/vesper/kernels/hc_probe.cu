// SPDX-License-Identifier: MIT
// DeepSeek Block/HC formulas; ATen Reduce.cuh reduction order. See notices.
extern "C" __global__ void hc_input(const unsigned short* x, float* promoted, float* square, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        float v = __bfloat162float(__ushort_as_bfloat16(x[i]));
        promoted[i] = v;
        square[i] = __fmul_rn(v, v);
    }
}

// For width 20480 and 1..8 rows, pinned ATen splits inputs across block.y.
// Block.x reduction precedes block.y; flattening this tree changes bits.
extern "C" __global__ void hc_mean(const float* square, float* means, int block_x) {
    int t = threadIdx.x, x = t % block_x, y = t / block_x;
    int height = 512 / block_x;
    const float* row = square + blockIdx.x * 20480;
    float a = 0.f, b = 0.f, c = 0.f, d = 0.f;
    for (int v = t; v < 5120; v += 512) {
        a = __fadd_rn(a, row[v * 4]);
        b = __fadd_rn(b, row[v * 4 + 1]);
        c = __fadd_rn(c, row[v * 4 + 2]);
        d = __fadd_rn(d, row[v * 4 + 3]);
    }
    float value = __fadd_rn(__fadd_rn(__fadd_rn(a, b), c), d);
    __shared__ float shared[512];
    shared[t] = value;
    for (int offset = block_x / 2; offset >= 32; offset >>= 1) {
        __syncthreads();
        if (x < offset) {
            value = __fadd_rn(value, shared[t + offset]);
            shared[t] = value;
        }
    }
    __syncthreads();
    if (x < 32) {
        for (int offset = 16; offset; offset >>= 1)
            value = __fadd_rn(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    if (x == 0) shared[t] = value;
    for (int offset = height / 2; offset; offset >>= 1) {
        __syncthreads();
        if (x == 0 && y < offset) {
            value = __fadd_rn(value, shared[(y + offset) * block_x]);
            shared[t] = value;
        }
    }
    if (t == 0) means[blockIdx.x] = __fmul_rn(value, 1.f / 20480.f);
}

extern "C" __global__ void hc_scale_mixes(const float* raw, const float* rstd, float* mixes, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) mixes[i] = __fmul_rn(raw[i], rstd[i / 24]);
}

extern "C" __global__ void hc_sinkhorn(const float* mixes, const float* scale, const float* base,
    float* pre, float* post, float* comb, float* affine_trace, float* max_trace,
    float* exp_trace, float* reductions, float* iterations) {
    int token = blockIdx.x, lane = threadIdx.x, cell = lane & 15;
    const float eps = 1e-6f;
    if (lane < 4) {
        float a = fmaf(mixes[token * 24 + lane], scale[0], base[lane]);
        float b = fmaf(mixes[token * 24 + lane + 4], scale[1], base[lane + 4]);
        pre[token * 4 + lane] = __fadd_rn(__fdiv_rn(1.f, __fadd_rn(1.f, expf(-a))), eps);
        post[token * 4 + lane] = __fmul_rn(2.f, __fdiv_rn(1.f, __fadd_rn(1.f, expf(-b))));
    }
    float value = fmaf(mixes[token * 24 + cell + 8], scale[2], base[cell + 8]);
    if (lane < 16) affine_trace[token * 16 + cell] = value;
    float maximum = value;
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, 2));
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, 1));
    if (lane < 16 && (cell & 3) == 0) max_trace[token * 4 + cell / 4] = maximum;
    value = expf(__fadd_rn(value, -maximum));
    if (lane < 16) exp_trace[token * 16 + cell] = value;
    for (int stage = 0; stage < 40; ++stage) {
        float sum = __fadd_rn(0.f, value);
        int first = (stage & 1) ? 8 : 2, second = (stage & 1) ? 4 : 1;
        sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, first));
        sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, second));
        if (lane < 16 && ((stage & 1) ? cell / 4 == 0 : (cell & 3) == 0))
            reductions[(token * 40 + stage) * 4 + ((stage & 1) ? cell & 3 : cell / 4)] = sum;
        value = stage == 0 ? __fadd_rn(__fdiv_rn(value, sum), eps) : __fdiv_rn(value, __fadd_rn(sum, eps));
        if (lane < 16) iterations[(token * 40 + stage) * 16 + cell] = value;
    }
    if (lane < 16) comb[token * 16 + cell] = value;
}

extern "C" __global__ void hc_pre(const unsigned short* input, const float* pre, float* products,
    float* sums, unsigned short* output, int tokens) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= tokens * 5120) return;
    int token = i / 5120, col = i % 5120;
    float p[4];
    for (int j = 0; j < 4; ++j) {
        int at = (token * 4 + j) * 5120 + col;
        float v = __bfloat162float(__ushort_as_bfloat16(input[at]));
        p[j] = __fmul_rn(pre[token * 4 + j], v);
        products[at] = p[j];
        p[j] = __fadd_rn(0.f, p[j]);
    }
    float sum = __fadd_rn(__fadd_rn(__fadd_rn(p[0], p[1]), p[2]), p[3]);
    sums[i] = sum;
    output[i] = __bfloat16_as_ushort(__float2bfloat16_rn(sum));
}

extern "C" __global__ void hc_post(const unsigned short* input, const unsigned short* residual,
    const float* post, const float* comb, float* post_product, float* residual_products,
    float* residual_sum, float* combined, unsigned short* output, int tokens) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= tokens * 4 * 5120) return;
    int col = i % 5120, target = (i / 5120) % 4, token = i / 20480;
    float p[4];
    for (int j = 0; j < 4; ++j) {
        float r = __bfloat162float(__ushort_as_bfloat16(residual[(token * 4 + j) * 5120 + col]));
        // comb[..., source, target] multiplies residual[..., source, d].
        p[j] = __fmul_rn(comb[(token * 4 + j) * 4 + target], r);
        residual_products[((token * 4 + j) * 4 + target) * 5120 + col] = p[j];
        p[j] = __fadd_rn(0.f, p[j]);
    }
    float sum = __fadd_rn(__fadd_rn(__fadd_rn(p[0], p[1]), p[2]), p[3]);
    float v = __bfloat162float(__ushort_as_bfloat16(input[token * 5120 + col]));
    float scaled = __fmul_rn(post[token * 4 + target], v);
    float result = __fadd_rn(scaled, sum);
    post_product[i] = scaled;
    residual_sum[i] = sum;
    combined[i] = result;
    output[i] = __bfloat16_as_ushort(__float2bfloat16_rn(result));
}
