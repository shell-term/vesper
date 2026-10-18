// RMS formulas and rotary pairs follow DeepSeek's pinned MIT model.py.
// Mean reduction order follows PyTorch ATen Reduce.cuh / MeanOps at
// 5c4886908584029761b579af026dcfb627c84070. See both third-party licenses.
extern "C" __global__ void norm_square(const unsigned short* input, float* squares, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        float x = __bfloat162float(__ushort_as_bfloat16(input[i]));
        squares[i] = __fmul_rn(x, x);
    }
}

extern "C" __global__ void norm_mean(const float* squares, float* means, int width, float factor) {
    int lane = threadIdx.x;
    const float* row = squares + blockIdx.x * width;
    float v0 = 0.f, v1 = 0.f, v2 = 0.f, v3 = 0.f;
    for (int vector = lane; vector < width / 4; vector += blockDim.x) {
        v0 = __fadd_rn(v0, row[vector * 4]);
        v1 = __fadd_rn(v1, row[vector * 4 + 1]);
        v2 = __fadd_rn(v2, row[vector * 4 + 2]);
        v3 = __fadd_rn(v3, row[vector * 4 + 3]);
    }
    float value = __fadd_rn(__fadd_rn(__fadd_rn(v0, v1), v2), v3);
    __shared__ float shared[512];
    shared[lane] = value;
    for (int offset = blockDim.x / 2; offset >= 32; offset >>= 1) {
        __syncthreads();
        if (lane < offset) {
            value = __fadd_rn(value, shared[lane + offset]);
            shared[lane] = value;
        }
    }
    __syncthreads();
    if (lane < 32) {
        for (int offset = 16; offset; offset >>= 1)
            value = __fadd_rn(value, __shfl_down_sync(0xffffffffu, value, offset));
        if (lane == 0) means[blockIdx.x] = __fmul_rn(value, factor);
    }
}

extern "C" __global__ void norm_rstd(const float* means, float* shifted, float* rstd, float eps, int rows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows) {
        float value = __fadd_rn(means[i], eps);
        shifted[i] = value;
        rstd[i] = rsqrtf(value);
    }
}

extern "C" __global__ void norm_apply(const unsigned short* input, const unsigned short* weight,
                                      const float* rstd, float* normalized, float* weighted,
                                      unsigned short* output, int width, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        float value = __bfloat162float(__ushort_as_bfloat16(input[i]));
        float w = __bfloat162float(__ushort_as_bfloat16(weight[i % width]));
        float scaled = __fmul_rn(value, rstd[i / width]);
        float product = __fmul_rn(w, scaled);
        normalized[i] = scaled;
        weighted[i] = product;
        output[i] = __bfloat16_as_ushort(__float2bfloat16_rn(product));
    }
}

extern "C" __global__ void rope_frequencies(const int* positions, float* inverse_freq,
                                                    float* angles, float* pairs, int count,
                                                    float base, int low, int high, int yarn, float factor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count * 32) return;
    int j = i % 32;
    float freq = __fdiv_rn(1.f, powf(base, __fdiv_rn((float)(2 * j), 64.f)));
    if (yarn) {
        // Tensor/scalar division in the pinned ATen kernel multiplies by
        // the rounded reciprocal; a direct correctly rounded divide differs.
        float reciprocal = __fdiv_rn(1.f, fmaxf((float)(high - low), .001f));
        float ramp = fminf(1.f, fmaxf(0.f, __fmul_rn((float)(j - low), reciprocal)));
        float smooth = __fadd_rn(1.f, -ramp);
        freq = __fadd_rn(__fmul_rn(__fdiv_rn(freq, factor), __fadd_rn(1.f, -smooth)), __fmul_rn(freq, smooth));
    }
    if (i < 32) inverse_freq[j] = freq;
    float angle = __fmul_rn((float)positions[i / 32], freq);
    angles[i] = angle;
    pairs[i * 2] = cosf(angle);
    pairs[i * 2 + 1] = sinf(angle);
}

extern "C" __global__ void rope_rotate(const unsigned short* input, const float* pairs,
                                                 unsigned short* output, float* rotated,
                                                 int batch, int sequence, int heads, int width, int inverse) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int count = batch * sequence * heads * width;
    if (i >= count) return;
    int column = i % width;
    if (column < width - 64) { output[i] = input[i]; return; }
    int element = column - width + 64;
    if (element & 1) return;
    int row = i / width;
    int s = (row / heads) % sequence;
    int offset = (s * 32 + element / 2) * 2;
    float real = __bfloat162float(__ushort_as_bfloat16(input[i]));
    float imag = __bfloat162float(__ushort_as_bfloat16(input[i + 1]));
    float c = pairs[offset], d = inverse ? -pairs[offset + 1] : pairs[offset + 1];
    // Explicitly retain the pinned PyTorch CUDA complex multiplication's FMA.
    float r = fmaf(real, c, -__fmul_rn(imag, d));
    float im = fmaf(real, d, __fmul_rn(imag, c));
    rotated[row * 64 + element] = r;
    rotated[row * 64 + element + 1] = im;
    output[i] = __bfloat16_as_ushort(__float2bfloat16_rn(r));
    output[i + 1] = __bfloat16_as_ushort(__float2bfloat16_rn(im));
}
