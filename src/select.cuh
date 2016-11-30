#ifndef SELECT_CUH_
#define SELECT_CUH_

#include <cstdint>
#include <iostream>
#include <cassert>
#include "common.cuh"
#include "buffer_pool.cuh"

__global__ void unstable_select(int32_t *src, int32_t *dst, int N, int32_t pred, int32_t *buffer, uint32_t *output_size, uint32_t *buffer_size, uint32_t *finished);

template<size_t warp_size, typename T>
class unstable_select_gpu;

template<size_t warp_size = WARPSIZE, typename T = int32_t>
uint32_t unstable_select_gpu_caller(int32_t *src, unstable_select_gpu<warp_size, T> *dst, uint32_t N, int32_t pred, const dim3 &dimGrid, const dim3 &dimBlock, cudaEvent_t &start, cudaEvent_t &stop);

void stable_select_cpu(int32_t *a, int32_t *b, int N);

__managed__ buffer_pool<int32_t> *buffpool;//(1024, 0);



template<size_t warp_size = WARPSIZE, typename T = int32_t>
class unstable_select_gpu_device{
public:
    buffer_pool<int32_t>::buffer_t *output_buffer;

public:
    __host__ unstable_select_gpu(int dev=0){
        set_device_on_scope d(dev);

        output_buffer = NULL;// buffpool->acquire_buffer_blocked();
    }

    __device__ __forceinline__ void push_results(volatile int32_t *src, uint32_t* elems);
};




template<size_t warp_size = WARPSIZE, typename T = int32_t>
class unstable_select_gpu{
    static_assert(sizeof(T)*8 == 32, "Operator is only implemented for data types of 32-bits size");
public:
    T                 *buffer       ;
    uint32_t          *counters     ;

    const size_t       grid_size    ;
    const size_t       shared_mem   ;
    const dim3         dimGrid      ;
    const dim3         dimBlock     ;
    const cudaStream_t stream       ;
    const int          device       ;

    unstable_select_gpu_device *dev_data;
public:
    unstable_select_gpu(const dim3 &dimGrid, const dim3 &dimBlock, cudaStream_t stream=0, int dev=0):
        // buffer    (NULL),
        // counters  (NULL),
        grid_size (dimGrid.x * dimGrid.y * dimGrid.z),
        shared_mem((9 * dimBlock.x * dimBlock.y + BRDCSTMEM(dimBlock) + ((dimBlock.x * dimBlock.y) / warp_size))*sizeof(T)),
        dimGrid   (dimGrid),
        dimBlock  (dimBlock),
        stream    (stream),
        device    (dev),
        {
            set_device_on_scope d(device);

            gpu(cudaMalloc((void**)&buffer, ((grid_size + 1) * 4 * warp_size + 3) * sizeof(T)));
            counters = (uint32_t *) (buffer + (grid_size + 1) * 4 * warp_size);

            gpu(cudaMemsetAsync(buffer + (grid_size + 1) * (4 * warp_size - 1), 0, (grid_size + 4) * sizeof(T), stream));
            
            unstable_select_gpu_device *tmp = new unstable_select_gpu_device(dev);
            gpu(cudaMalloc(dev_data, sizeof(unstable_select_gpu_device)));
            gpu(cudaMemcpy(dev_data, tmp, sizeof(unstable_select_gpu_device)));
            free(tmp);  //NOTE: bad practice ? we want to allocate tmp by new to
                        //      trigger initialization but we want to free the 
                        //      corresponding memory after moving to device 
                        //      without triggering the destructor

            // output_buffer = NULL;// buffpool->acquire_buffer_blocked();
        }

    uint32_t next(T *dst, T *src = NULL, uint32_t N = 0);

    ~unstable_select_gpu(){
        set_device_on_scope d(device);
        gpu(cudaFree(buffer));

        unstable_select_gpu_device *tmp = malloc(sizeof(unstable_select_gpu_device));
        gpu(cudaMemcpy(tmp, dev_data, sizeof(unstable_select_gpu_device)));
        gpu(cudaFree(dev_data));
        delete tmp;
    }
};

#endif /* SELECT_CUH_ */