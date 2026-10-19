// SPDX-License-Identifier: MIT
// Original Engram.forward / ParallelEngramEmbedding.forward, DeepSeek MIT.
// Reduction calls use the separately attributed PyTorch reduction tree.
extern "C" __global__ void eg_embed(const unsigned char* codes, const unsigned char* scales,
                                  float* scaled, unsigned short* output, int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<count) {
        float v=__fmul_rn(fp8(codes[i]),scale8(scales[i/32]));
        scaled[i]=v; output[i]=__bfloat16_as_ushort(__float2bfloat16_rn(v));
    }
}

extern "C" __global__ void eg_prepare(const unsigned short* hidden,const unsigned short* kv,
    const unsigned short* qw,const unsigned short* kw,float* hsquare,float* ksquare,
    float* weights,float* hweight,float* products,int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<count) {
        int token=i/20480, column=i%20480;
        float h=__bfloat162float(__ushort_as_bfloat16(hidden[i]));
        float k=__bfloat162float(__ushort_as_bfloat16(kv[token*25600+column]));
        float w=__fmul_rn(__bfloat162float(__ushort_as_bfloat16(qw[column])),__bfloat162float(__ushort_as_bfloat16(kw[column])));
        if (i<20480) weights[i]=w;
        hsquare[i]=__fmul_rn(h,h); ksquare[i]=__fmul_rn(k,k);
        float hw=__fmul_rn(h,w); hweight[i]=hw; products[i]=__fmul_rn(hw,k);
    }
}

extern "C" __global__ void eg_gate(const float* hm,const float* km,const float* sums,const unsigned char* mask,
    float* he,float* ke,float* hr,float* kr,float* rstd,float* normalized,float* dot,
    float* clamped,float* root,float* signed_root,float* sigmoid,float* gate,int rows) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<rows) {
        float h=__fadd_rn(hm[i],1.e-20f),k=__fadd_rn(km[i],1.e-20f);
        he[i]=h; ke[i]=k; float rh=rsqrtf(h),rk=rsqrtf(k); hr[i]=rh; kr[i]=rk;
        float r=__fmul_rn(rh,rk); rstd[i]=r;
        float scaled=__fmul_rn(sums[i],r); normalized[i]=scaled;
        float d=__fmul_rn(scaled,0.013975424692034721f); dot[i]=d;
        float c=fmaxf(fabsf(d),1.e-6f); clamped[i]=c; float sq=sqrtf(c); root[i]=sq;
        float signed_sq=copysignf(sq,d); signed_root[i]=signed_sq;
        float g=__fdiv_rn(1.f,__fadd_rn(1.f,expf(-signed_sq))); sigmoid[i]=g;
        gate[i]=mask[i/4]?g:0.f;
    }
}

extern "C" __global__ void eg_post(const unsigned short* hidden,const unsigned short* kv,const float* gate,
                                   float* delta,float* residual,unsigned short* output,int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<count) {
        float h=__bfloat162float(__ushort_as_bfloat16(hidden[i]));
        float v=__bfloat162float(__ushort_as_bfloat16(kv[(i/20480)*25600+20480+i%5120]));
        float d=__fmul_rn(gate[i/5120],v); delta[i]=d;
        float r=__fadd_rn(h,d); residual[i]=r; output[i]=__bfloat16_as_ushort(__float2bfloat16_rn(r));
    }
}
