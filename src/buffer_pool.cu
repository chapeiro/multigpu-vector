#include "buffer_pool.cuh"

using namespace std;

// template<typename T, size_t buffer_capacity>
// __host__ buffer_pool<T, buffer_capacity>::buffer_pool(size_t size, int device){
//     vector<buffer_pool<T, buffer_capacity>::buffer_t *> buffs;
//     for (size_t i = 0 ; i < size ; ++i) buffs.push_back(new buffer_pool<T, buffer_capacity>::buffer_t(device));
//     pool = new optimistic_safe_device_stack<buffer_pool<T, buffer_capacity>::buffer_t *, (buffer_pool<T, buffer_capacity>::buffer_t *) NULL>(size, buffs, device);
//     printf("asdasdasf\n");
// }

// template<typename T, size_t buffer_capacity>
// __host__ __device__ buffer_pool<T, buffer_capacity>::~buffer_pool(){
//     assert(pool->full());
//     while (!pool->empty()) delete pool->pop_blocked();
// }

// template<typename T, size_t buffer_capacity>
// __device__ inline buffer_pool<T, buffer_capacity>::buffer_t * buffer_pool<T, buffer_capacity>::acquire_buffer_blocked(){
//     printf("asdasdasf\n");
//     buffer_pool<T, buffer_capacity>::buffer_t * buff = pool->pop_blocked();
//     buff->clean();
//     return buff;
// }

// template<typename T, size_t buffer_capacity>
// __device__ inline bool buffer_pool<T, buffer_capacity>::try_acquire_buffer(buffer_pool<T, buffer_capacity>::buffer_t ** ret){
//     return pool->try_pop(ret);
// }

// template<typename T, size_t buffer_capacity>
// __device__ inline void buffer_pool<T, buffer_capacity>::release_buffer(buffer_pool<T, buffer_capacity>::buffer_t * buff){
//     pool->push(buff);
// }



// template<typename T, size_t buffer_capacity>
// __host__ __device__ inline buffer_pool<T, buffer_capacity>::buffer_t * buffer_pool<T, buffer_capacity>::acquire_buffer_blocked()

// __global__ void acquire_buffer_for_host_blocked(buffer_pool<vec4, 1024> *self, buffer<vec4, 1024> ** ret, size_t maxretries){
//     assert(blockDim.x * blockDim.y * blockDim.z == 1);
//     assert( gridDim.x *  gridDim.y *  gridDim.z == 1);
//     *ret = self->acquire_buffer_blocked(maxretries);
//     printf("b:%llx\n", *ret);
// }

__global__ void register_producer_for_host(buffer_pool<int32_t, DEFAULT_BUFF_CAP> *self, void* prod){
    assert(blockDim.x * blockDim.y * blockDim.z == 1);
    assert( gridDim.x *  gridDim.y *  gridDim.z == 1);
    self->register_producer(prod);
}

__global__ void unregister_producer_for_host(buffer_pool<int32_t, DEFAULT_BUFF_CAP> *self, void* prod){
    assert(blockDim.x * blockDim.y * blockDim.z == 1);
    assert( gridDim.x *  gridDim.y *  gridDim.z == 1);
    self->unregister_producer(prod);
}


__global__ void release_buffer_host2(buffer_pool<int32_t, DEFAULT_BUFF_CAP> *self, buffer<int32_t, DEFAULT_BUFF_CAP> *buff){
    assert(blockDim.x * blockDim.y * blockDim.z == 1);
    assert( gridDim.x *  gridDim.y *  gridDim.z == 1);
    self->release_buffer(buff);
}



__global__ void acquire_buffer_blocked_unsafe_for_host(buffer_pool<int32_t, DEFAULT_BUFF_CAP> *self, buffer<int32_t, DEFAULT_BUFF_CAP> ** ret){
    assert(blockDim.x * blockDim.y * blockDim.z == 1);
    assert( gridDim.x *  gridDim.y *  gridDim.z == 1);
    *ret = self->acquire_buffer_blocked_unsafe();
}


