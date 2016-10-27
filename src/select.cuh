#ifndef SELECT_CUH_
#define SELECT_CUH_

#include <cstdint>
#include <iostream>
#include <cassert>
#include "common.cuh"

__global__ void unstable_select(int32_t *src, int32_t *dst, int N, int32_t pred, int32_t *buffer, uint32_t *output_size, uint32_t *buffer_size, uint32_t *finished);

uint32_t unstable_select_gpu_caller(int32_t *src, int32_t *dst, uint32_t N, int32_t pred, const dim3 &dimGrid, const dim3 &dimBlock, cudaEvent_t &start, cudaEvent_t &stop);

void stable_select_cpu(int32_t *a, int32_t *b, int N);




template<size_t warp_size = WARPSIZE, typename T = int32_t>
class unstable_select_gpu{
    static_assert(sizeof(T)*8 == 32, "Operator is only implemented for data types of 32-bits size");
private:
    T                 *buffer       ;
    uint32_t          *counters     ;

    const size_t       grid_size    ;
    const size_t       shared_mem   ;
    const dim3         dimGrid      ;
    const dim3         dimBlock     ;
    const cudaStream_t stream       ;
    const int          device       ;
public:
    unstable_select_gpu(const dim3 &dimGrid, const dim3 &dimBlock, cudaStream_t stream=0, int dev=0):
        // buffer    (NULL),
        // counters  (NULL),
        grid_size (dimGrid.x * dimGrid.y * dimGrid.z),
        shared_mem((9 * dimBlock.x * dimBlock.y + BRDCSTMEM(dimBlock) + ((dimBlock.x * dimBlock.y) / warp_size))*sizeof(T)),
        dimGrid   (dimGrid),
        dimBlock  (dimBlock),
        stream    (stream),
        device    (dev){
            set_device_on_scope d(device);

            gpu(cudaMalloc((void**)&buffer, ((grid_size + 1) * 4 * warp_size + 3) * sizeof(T)));
            counters = (uint32_t *) (buffer + (grid_size + 1) * 4 * warp_size);

            gpu(cudaMemsetAsync(buffer + (grid_size + 1) * (4 * warp_size - 1), 0, (grid_size + 4) * sizeof(T), stream));
        }

    uint32_t next(T *dst, T *src = NULL, uint32_t N = 0);

    ~unstable_select_gpu(){
        set_device_on_scope d(device);
        gpu(cudaFree(buffer));
    }
};

#endif /* SELECT_CUH_ */