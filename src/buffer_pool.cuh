#ifndef BUFFER_POOL_CUH_
#define BUFFER_POOL_CUH_

#include "common.cuh"
#include "buffer.cuh"
#include "optimistic_safe_device_stack.cuh"
#include <vector>

using namespace std;

template<typename T, size_t buffer_capacity = DEFAULT_BUFF_CAP>
class buffer_pool {
public:
    typedef buffer<T, buffer_capacity> buffer_t;

private:
    optimistic_safe_device_stack<buffer_t *, (buffer_t *) NULL> *pool;

public:
    __host__ buffer_pool(size_t size, int device = 0){
        vector<buffer_t *> buffs;
        for (size_t i = 0 ; i < size ; ++i) buffs.push_back(new buffer_t(device));
        pool = new optimistic_safe_device_stack<buffer_t *, (buffer_t *) NULL>(size, buffs, device);
    }

    __host__ __device__ ~buffer_pool(){
        assert(pool->full());
        while (!pool->empty()) delete pool->pop_blocked();
    }

    __device__ buffer_t * acquire_buffer_blocked(){
        buffer_t * buff = pool->pop_blocked();
        buff->clean();
        return buff;
    }

    __device__ bool try_acquire_buffer(buffer_t ** ret){
        return pool->try_pop(ret);
    }


    __device__ void release_buffer(buffer_t * buff){
        pool->push(buff);
    }

};

#endif /* BUFFER_POOL_CUH_ */