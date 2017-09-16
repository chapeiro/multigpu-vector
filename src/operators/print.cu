#include "print.cuh"

using namespace std;

__host__ dprint::dprint(){}

__device__ void dprint::consume_warp(const int32_t * __restrict__ src, cnt_t N, vid_t vid, cid_t cid){
    assert(N <= vector_size);

    if (get_laneid() != 0) return;

    for (uint32_t i = 0 ; i < N ; ++i) printf("%d ", src[i]);
}

__device__ void dprint::consume_close(){}

__device__ void dprint::consume_open(){}

__device__ void dprint::at_open(){}

__device__ void dprint::at_close(){}

__host__ void dprint::before_open(){}

__host__ void dprint::after_close(){}

