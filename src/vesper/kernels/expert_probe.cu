// SPDX-License-Identifier: MIT
// The clamp/order follows the pinned MIT DeepSeek Expert.forward.
// Each FP32 operation precedes the same BF16 boundary as the reference.
extern "C" __global__ void expert_swiglu(const unsigned short* gate_bits,
                                         const unsigned short* up_bits,
                                         float* down_input, float* trace,
                                         int count, float limit, float routing_weight, int weighted) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float gate = __bfloat162float(__ushort_as_bfloat16(gate_bits[i]));
    float up = __bfloat162float(__ushort_as_bfloat16(up_bits[i]));
    if (limit > 0.f) {
        if (gate > limit) gate = limit;
        if (up > limit) up = limit;
        if (up < -limit) up = -limit;
    }
    float silu = __fdiv_rn(gate, __fadd_rn(1.f, expf(-gate)));
    float product = __fmul_rn(silu, up);
    float routed = weighted ? __fmul_rn(routing_weight, product) : product;
    down_input[i] = __bfloat162float(__float2bfloat16_rn(routed));
    trace[i * 5] = gate;
    trace[i * 5 + 1] = up;
    trace[i * 5 + 2] = silu;
    trace[i * 5 + 3] = product;
    trace[i * 5 + 4] = routed;
}
