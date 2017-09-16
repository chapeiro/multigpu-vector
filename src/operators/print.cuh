#ifndef PRINT_CUH_
#define PRINT_CUH_

#include "../common.cuh"

class dprint{
public:
    __host__ dprint();

    __host__   void before_open();
    __device__ void at_open();

    __device__ void consume_open();
    __device__ void consume_warp(const int32_t * __restrict__ src, cnt_t N, vid_t vid, cid_t cid);
    __device__ void consume_close();

    __device__ void at_close();
    __host__   void after_close();

    __host__ ~dprint(){}
};

#endif /* PRINT_CUH_ */