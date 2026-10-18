// Routing postprocessing follows DeepSeek's MIT-licensed Gate.forward.
// Projection and general batch shapes are excluded.
// The top-k gather/network order is adapted from PyTorch ATen CUDA at
// 5c4886908584029761b579af026dcfb627c84070; see third_party/PyTorch-LICENSE.
extern "C" __global__ void router_scores(const float *logits, const float *bias, const float *bias_vl,
                                         float *scores, float *selection, int image_span, int zero_bias) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 384) return;
    // The pinned gate temperature is 1.0. Softplus beta=1, threshold=20.
    float z = logits[i];
    float value = sqrtf(z > 20.0f ? z : log1pf(expf(z)));
    scores[i] = value;
    float correction = zero_bias ? 0.0f : (image_span ? bias_vl[i] : bias[i]);
    selection[i] = value + correction;
}

extern "C" __global__ void router_top6(const float *scores, const float *selection,
                                       float *raw, float *denominator, float *weights,
                                       int *indices, int *tied) {
    if (blockIdx.x || threadIdx.x) return;
    int ranked[6];
    int any_tie = 0;
    unsigned int threshold = 0;
    for (int slot = 0; slot < 6; ++slot) {
        unsigned int best = 0;
        int best_id = -1, tie = 0;
        for (int i = 0; i < 384; ++i) {
            bool used = false;
            for (int j = 0; j < slot; ++j) used |= ranked[j] == i;
            if (used) continue;
            unsigned int bits = __float_as_uint(selection[i]);
            unsigned int value = bits ^ ((bits & 0x80000000u) ? 0xffffffffu : 0x80000000u);
            if (best_id < 0 || value > best) { best = value; best_id = i; tie = 0; }
            else if (value == best) tie = 1;
        }
        ranked[slot] = best_id;
        threshold = best;
        any_tie |= tie;
    }

    // sbtopk's prefix-scan gather has one 384-thread iteration for this shape:
    // first all keys above the threshold, then the first equal keys by ID.
    float keys[32] = {};
    int ids[32] = {};
    bool valid[32] = {};
    int count = 0;
    for (int phase = 0; phase < 2; ++phase) {
        for (int i = 0; i < 384 && count < 6; ++i) {
            unsigned int bits = __float_as_uint(selection[i]);
            unsigned int value = bits ^ ((bits & 0x80000000u) ? 0xffffffffu : 0x80000000u);
            if (phase ? value == threshold : value > threshold) {
                keys[count] = selection[i];
                ids[count] = i;
                valid[count++] = true;
            }
        }
    }
    // ATen SmallBitonicSort pads to 32, even when only six entries are valid.
    // Each stage's pairs are disjoint, so serial execution preserves swaps.
    for (int size = 2; size <= 32; size *= 2) {
        for (int stride = size / 2; stride; stride /= 2) {
            for (int thread = 0; thread < 16; ++thread) {
                int pos = 2 * thread - (thread & (stride - 1));
                int other = pos + stride;
                bool direction = size == 32 ? false : (thread & (size / 2)) != 0;
                bool swap = (keys[pos] > keys[other] && valid[pos]) || !valid[other];
                if (swap == direction) {
                    float key = keys[pos]; keys[pos] = keys[other]; keys[other] = key;
                    int id = ids[pos]; ids[pos] = ids[other]; ids[other] = id;
                    bool v = valid[pos]; valid[pos] = valid[other]; valid[other] = v;
                }
            }
        }
    }
    for (int i = 0; i < 6; ++i) {
        indices[i] = ids[i];
        raw[i] = scores[ids[i]];
    }
    // Single-row, six-element reduction tree; checked independently against
    // the installed official PyTorch runtime, not assumed for other shapes.
    float sum = ((raw[0] + raw[4]) + raw[2]) + ((raw[1] + raw[5]) + raw[3]);
    float divisor = sum + 1e-20f;
    denominator[0] = divisor;
    for (int i = 0; i < 6; ++i) weights[i] = (raw[i] / divisor) * 1.5f;
    tied[0] = any_tie;
}
