#ifndef SELECT_CUH_
#define SELECT_CUH_

#include <cstdint>
#include <iostream>
#include <cassert>
#include "common.cuh"
#include "buffer_pool.cuh"

__global__ void unstable_select(int32_t *src, int32_t *dst, int N, int32_t pred, int32_t *buffer, uint32_t *output_size, uint32_t *buffer_size, uint32_t *finished);

template<size_t warp_size, typename T>
class unstable_select_gpu_device;

template<size_t warp_size = WARPSIZE, typename T = int32_t>
uint32_t unstable_select_gpu_caller(int32_t *src, unstable_select_gpu_device<warp_size, T> *dst, uint32_t N, int32_t pred, const dim3 &dimGrid, const dim3 &dimBlock, cudaEvent_t &start, cudaEvent_t &stop);

void stable_select_cpu(int32_t *a, int32_t *b, int N);

extern __managed__ buffer_pool<int32_t> * buffpool;//(1024, 0);



template<size_t warp_size = WARPSIZE, typename T = int32_t>
class unstable_select_gpu_device{
public:
    typedef buffer_pool<int32_t>::buffer_t buffer_t;
    volatile buffer_t * volatile        output_buffer;
    buffer_pool<int32_t>         * volatile outpool;
public:
    __host__ unstable_select_gpu_device(buffer_pool<int32_t> *outpool, int dev=0): outpool(outpool){
        set_device_on_scope d(dev);

    buffer_pool<int32_t>::buffer_t ** buff;
    buffer_pool<int32_t>::buffer_t ** buff_ret;
    gpu(cudaMalloc(&buff, sizeof(buffer_pool<int32_t>::buffer_t *)));
    cudaMallocHost(&buff_ret, sizeof(buffer_pool<int32_t>::buffer_t *));
    cudaStream_t strm;
    cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking);
        // output_buffer = NULL;// buffpool->acquire_buffer_blocked();
        // printf("%lld\n", buffpool);
        output_buffer = buffpool->h_acquire_buffer_blocked(buff, buff_ret, strm);


    cudaStreamDestroy(strm);
    cudaFree(buff);
    cudaFreeHost(buff_ret);
        // output_buffer = cuda_new<buffer_pool<int32_t>::buffer_t>(dev);
        printf("->%llx\n", output_buffer);
    }

    __device__ __forceinline__ void push_results(volatile int32_t *src, uint32_t* elems) volatile;
    __device__ void unstable_select_flush(int32_t *buffer, uint32_t *buffer_size);
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

    unstable_select_gpu_device<warp_size, T> *dev_data;
public:
    unstable_select_gpu(const dim3 &dimGrid, const dim3 &dimBlock, buffer_pool<int32_t> *outpool, cudaStream_t stream=0, int dev=0):
        // buffer    (NULL),
        // counters  (NULL),
        grid_size (dimGrid.x * dimGrid.y * dimGrid.z),
        shared_mem((9 * dimBlock.x * dimBlock.y + BRDCSTMEM(dimBlock) + ((dimBlock.x * dimBlock.y) / warp_size)*32)*sizeof(T)),
        dimGrid   (dimGrid),
        dimBlock  (dimBlock),
        stream    (stream),
        device    (dev)
        {
            set_device_on_scope d(device);

            gpu(cudaMalloc((void**)&buffer, ((grid_size + 1) * 4 * warp_size + 3) * sizeof(T)));
            counters = (uint32_t *) (buffer + (grid_size + 1) * 4 * warp_size);

            gpu(cudaMemsetAsync(buffer + (grid_size + 1) * (4 * warp_size - 1), 0, (grid_size + 4) * sizeof(T), stream));
            
            dev_data = cuda_new<unstable_select_gpu_device<warp_size, T>>(device, outpool, device);
            
            // output_buffer = NULL;// buffpool->acquire_buffer_blocked();
        }

    __host__ uint32_t next(T *src = NULL, uint32_t N = 0);

    ~unstable_select_gpu(){
        set_device_on_scope d(device);
        gpu(cudaFree(buffer));

        cuda_delete(dev_data, device);
    }
};

#endif /* SELECT_CUH_ */