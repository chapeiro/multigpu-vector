#include "buffer_manager.cuh"

__device__ __constant__ lockfree_stack<buffer<int32_t, DEFAULT_BUFF_CAP, vec4>  *, (buffer<int32_t, DEFAULT_BUFF_CAP, vec4>  *) NULL> * pool;
__device__ __constant__ int deviceId;



__global__ void release_buffer_host(buffer<int32_t, DEFAULT_BUFF_CAP> *buff){
    assert(blockDim.x * blockDim.y * blockDim.z == 1);
    assert( gridDim.x *  gridDim.y *  gridDim.z == 1);
    buffer_manager<int32_t>::release_buffer(buff);
}

__global__ void get_buffer_host(buffer<int32_t, DEFAULT_BUFF_CAP> **buff){
    assert(blockDim.x * blockDim.y * blockDim.z == 1);
    assert( gridDim.x *  gridDim.y *  gridDim.z == 1);
    *buff = buffer_manager<int32_t>::get_buffer();
}
