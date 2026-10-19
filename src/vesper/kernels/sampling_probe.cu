// Vesper's bounded FP32 specialization of the original sampler's CUDA path.
// Reduction/distribution order follows PyTorch at commit
// 5c4886908584029761b579af026dcfb627c84070; see third_party/PyTorch-LICENSE.
// The CUDA Toolkit supplies cuRAND headers; no SDK implementation is vendored.
#include <curand_kernel.h>
#define FLT_MAX 0x1.fffffep+127f

extern "C" __global__ void sample_scale(const float* input, float* output, float reciprocal, int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) output[i]=__fmul_rn(input[i],reciprocal);
}

template<bool Maximum> __device__ float sample_reduce(float value,float* shared) {
    int lane=threadIdx.x&31, warp=threadIdx.x>>5;
    for(int distance=16;distance;distance>>=1) {
        float other=__shfl_down_sync(0xffffffffu,value,distance);
        value=Maximum?fmaxf(value,other):__fadd_rn(value,other);
    }
    __syncthreads();
    if(lane==0) shared[warp]=value;
    __syncthreads();
    value=threadIdx.x<32?shared[lane]:(Maximum?-FLT_MAX:0.f);
    if(warp==0) for(int distance=16;distance;distance>>=1) {
        float other=__shfl_down_sync(0xffffffffu,value,distance);
        value=Maximum?fmaxf(value,other):__fadd_rn(value,other);
    }
    if(threadIdx.x==0) shared[0]=value;
    __syncthreads();
    return shared[0];
}

extern "C" __global__ void sample_softmax(const float* values,float* probabilities) {
    constexpr int width=129280, vector_end=(width/4096)*4096;
    __shared__ float scratch[32];
    const float* row=values+blockIdx.x*width;
    float maximum=-FLT_MAX;
    for(int offset=threadIdx.x;offset*4<vector_end;offset+=1024) {
        float4 v=reinterpret_cast<const float4*>(row)[offset];
        maximum=fmaxf(maximum,v.x); maximum=fmaxf(maximum,v.y);
        maximum=fmaxf(maximum,v.z); maximum=fmaxf(maximum,v.w);
    }
    for(int i=vector_end+threadIdx.x;i<width;i+=1024) maximum=fmaxf(maximum,row[i]);
    maximum=sample_reduce<true>(maximum,scratch);
    float sum=0.f;
    for(int offset=threadIdx.x;offset*4<vector_end;offset+=1024) {
        float4 v=reinterpret_cast<const float4*>(row)[offset];
        sum=__fadd_rn(sum,expf(__fadd_rn(v.x,-maximum))); sum=__fadd_rn(sum,expf(__fadd_rn(v.y,-maximum)));
        sum=__fadd_rn(sum,expf(__fadd_rn(v.z,-maximum))); sum=__fadd_rn(sum,expf(__fadd_rn(v.w,-maximum)));
    }
    for(int i=vector_end+threadIdx.x;i<width;i+=1024) sum=__fadd_rn(sum,expf(__fadd_rn(row[i],-maximum)));
    sum=sample_reduce<false>(sum,scratch);
    for(int i=threadIdx.x;i<width;i+=1024)
        probabilities[blockIdx.x*width+i]=__fdiv_rn(expf(__fadd_rn(row[i],-maximum)),sum);
}

extern "C" __global__ void sample_exponential(float* noise,unsigned long long seed,unsigned long long offset,int count) {
    int thread=blockIdx.x*blockDim.x+threadIdx.x;
    curandStatePhilox4_32_10_t state;
    curand_init(seed,thread,offset,&state);
    int stride=blockDim.x*gridDim.x;
    int rounded=((count-1)/(stride*4)+1)*stride*4;
    for(int base=thread;base<rounded;base+=stride*4) {
        float4 values=curand_uniform4(&state);
        for(int j=0;j<4;j++) {
            int at=base+stride*j;
            if(at<count) {
                float u=(&values.x)[j];
                float logarithm=u>=1.f-0x1p-24f?-0x1p-24f:__logf(u);
                noise[at]=-logarithm;
            }
        }
    }
}

extern "C" __global__ void sample_scores(const float* probabilities,const float* noise,float* scores,int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) scores[i]=__fdiv_rn(probabilities[i],noise[i]);
}
