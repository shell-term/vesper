// Routing postprocessing follows DeepSeek's MIT-licensed Gate.forward.
// Projection, general batch shapes and an exact top-k tie policy are excluded.
// This diagnostic kernel cannot by itself qualify a complete model gate.
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
    int ids[6];
    int any_tie = 0;
    for (int slot = 0; slot < 6; ++slot) {
        float best = -__int_as_float(0x7f800000);
        int best_id = -1, tie = 0;
        for (int i = 0; i < 384; ++i) {
            bool used = false;
            for (int j = 0; j < slot; ++j) used |= ids[j] == i;
            if (used) continue;
            float value = selection[i];
            if (value > best) { best = value; best_id = i; tie = 0; }
            else if (value == best) tie = 1;
        }
        ids[slot] = best_id;
        indices[slot] = best_id;
        raw[slot] = scores[best_id];
        any_tie |= tie;
    }
    // Single-row, six-element reduction tree; checked independently against
    // the installed official PyTorch runtime, not assumed for other shapes.
    float sum = ((raw[0] + raw[4]) + raw[2]) + ((raw[1] + raw[5]) + raw[3]);
    float divisor = sum + 1e-20f;
    denominator[0] = divisor;
    for (int i = 0; i < 6; ++i) weights[i] = (raw[i] / divisor) * 1.5f;
    tied[0] = any_tie;
}
