#include "buffer_manager.cuh"

__device__ __constant__ lockfree_stack<buffer<int32_t, DEFAULT_BUFF_CAP, vec4>  *, (buffer<int32_t, DEFAULT_BUFF_CAP, vec4>  *) NULL> * pool;
__device__ __constant__ int deviceId;

threadsafe_stack<buffer<int32_t, DEFAULT_BUFF_CAP, vec4>  *, (buffer<int32_t, DEFAULT_BUFF_CAP, vec4>  *) NULL> * h_pool;


__global__ void release_buffer_host(buffer<int32_t, DEFAULT_BUFF_CAP> **buff, int buffs){
    assert(blockDim.x * blockDim.y * blockDim.z == 1);
    assert( gridDim.x *  gridDim.y *  gridDim.z == 1);
    for (int i = 0 ; i < buffs ; ++i) buffer_manager<int32_t>::release_buffer(buff[i]);
}

__global__ void get_buffer_host(buffer<int32_t, DEFAULT_BUFF_CAP> **buff, int buffs){
    assert(blockDim.x * blockDim.y * blockDim.z == 1);
    assert( gridDim.x *  gridDim.y *  gridDim.z == 1);
    for (int i = 0 ; i < buffs ; ++i) buff[i] = buffer_manager<int32_t>::get_buffer();
}

mutex                                              *devive_buffs_mutex;
vector<buffer<int32_t, DEFAULT_BUFF_CAP, vec4>  *> *device_buffs_pool;
buffer<int32_t, DEFAULT_BUFF_CAP, vec4>          ***device_buff;
int                                                 device_buff_size;
int                                                 keep_threshold;

cudaStream_t                                       *release_streams;
