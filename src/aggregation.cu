#include "aggregation.cuh"

using namespace std;

template<size_t warp_size, int32_t neutral_value>
__inline__ __device__ int32_t aggregation<warp_size, neutral_value>::warpReduce(int32_t x){
    #pragma unroll
    for (int m = warp_size >> 1; m > 0; m >>= 1)
        x += __shfl_xor(x, m);
    return x;
}

template<size_t warp_size, int32_t neutral_value>
__host__ aggregation<warp_size, neutral_value>::aggregation(d_operator_t * parent, int shared_offset, int grid_size, int dev): 
        parent(parent), res(neutral_value), shared_offset(shared_offset){}


template<size_t warp_size, int32_t neutral_value>
__device__ void aggregation<warp_size, neutral_value>::consume_open(){
    extern __shared__ int32_t s[];

    const int32_t i = get_warpid() * warp_size + get_laneid();
    s[shared_offset + i] = neutral_value;
}

template<size_t warp_size, int32_t neutral_value>
__device__ void aggregation<warp_size, neutral_value>::consume_warp(const int32_t *src, unsigned int N){
    extern __shared__ int32_t s[];

    const int32_t laneid    = get_laneid();

    const int32_t i         = get_warpid() * warp_size + laneid;
    int32_t aggr            = neutral_value;

    #pragma unroll
    for (int k = 0 ; k < 4 ; ++k){
        if (k*warp_size + laneid < N) aggr += src[k*warp_size + laneid];
    }

    s[shared_offset + i] += aggr;
}

template<size_t warp_size, int32_t neutral_value>
__device__ void aggregation<warp_size, neutral_value>::consume_close(){
    extern __shared__ int32_t s[];

    const int32_t i = get_warpid() * warp_size + get_laneid();

    int32_t x = warpReduce(s[shared_offset + i]);
    if (get_laneid() == 0) atomicAdd(&res, x);
}

template<size_t warp_size, int32_t neutral_value>
__device__ void aggregation<warp_size, neutral_value>::open(){}

template<size_t warp_size, int32_t neutral_value>
__device__ void aggregation<warp_size, neutral_value>::close(){
    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0){
        parent->consume_open();
        __syncthreads();
        if (get_warpid() == 0) parent->consume_warp(&res, 1);
        __syncthreads();
        parent->consume_close();
    }
}

template class aggregation<>;
