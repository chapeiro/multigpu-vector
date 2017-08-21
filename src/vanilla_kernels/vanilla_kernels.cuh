#ifndef VANILLA_KERNELS_CUH_
#define VANILLA_KERNELS_CUH_

#include "../common.cuh"

__global__ void sum_select_less_than(int32_t * __restrict__ a, int32_t * __restrict__ b, size_t N, int32_t thres, int32_t * __restrict__ result);

__global__ void sum_select_less_than2(int32_t * __restrict__ a, int32_t * __restrict__ b, size_t N, int32_t thres, int32_t * __restrict__ result);

#endif