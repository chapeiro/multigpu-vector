#ifndef SELECT_CUH_
#define SELECT_CUH_

#include <cstdint>
#include <iostream>
#include <cassert>

__global__ void unstable_select(int32_t *src, int32_t *dst, int N, int32_t pred, int32_t *buffer, uint32_t *output_size, uint32_t *buffer_size);

uint32_t unstable_select_gpu(int32_t *src, int32_t *dst, uint32_t N, int32_t pred, const dim3 &dimGrid, const dim3 &dimBlock, cudaEvent_t &start, cudaEvent_t &stop);

void stable_select_cpu(int32_t *a, int32_t *b, int N);

#endif /* SELECT_CUH_ */