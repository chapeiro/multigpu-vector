#ifndef OPTIMISTIC_SAFE_DEVICE_STACK_CUH_
#define OPTIMISTIC_SAFE_DEVICE_STACK_CUH_

#include "common.cuh"
#include <cstdint>
#include <limits>

using namespace std;

template<typename T, T invalid_value = numeric_limits<T>::max()>
class optimistic_safe_device_stack{
public:
    volatile uint32_t count;
    T                *data ;

    size_t            retries;
    uint32_t          cap;

public:
    __host__ optimistic_safe_device_stack(uint32_t capacity, vector<T> vals, int device = 0) : count(vals.size()), retries(0), cap(capacity){
        set_device_on_scope d(device);
        cudaMalloc(&data, (capacity+1) * sizeof(T));
        T tmp = invalid_value;

        //TODO: find a better way to initialize the buffer than multiple memcpys
        //(initialization is needed to safely push new elements in the stack)
        cudaMemcpy(data, &tmp, sizeof(T), cudaMemcpyDefault);

        for (uint32_t i = 0 ; i < capacity ; ++i){
            T *ptmp = (i < vals.size()) ? &vals[i] : &tmp;
            cudaMemcpy(data+i+1, ptmp, sizeof(T), cudaMemcpyDefault);
        }
    }

    __host__ optimistic_safe_device_stack(uint32_t capacity, int device = 0) : count(0), retries(0), cap(capacity){
        set_device_on_scope d(device);
        cudaMalloc(&data, (capacity+1) * sizeof(T));
        T tmp = invalid_value;

        //TODO: find a better way to initialize the buffer than multiple memcpys
        //(initialization is needed to safely push new elements in the stack)
        for (uint32_t i = 0 ; i <= capacity ; ++i){
            cudaMemcpy(data+i, &tmp, sizeof(T), cudaMemcpyDefault);
        }
    }

    __host__ __device__ bool full() const{
        return count == cap;
    }

    __host__ ~optimistic_safe_device_stack(){
        printf("optimistic_safe_device_stack retries : %zu\n", retries);
        // assert(full());
        cudaFree(data);
    }

    __host__ __device__ uint32_t size() const{
        return count;
    }

    __host__ __device__ uint32_t capacity() const{
        return cap;
    }


    __host__ __device__ bool empty() const{
        return count == 0;
    }

    __device__ bool try_pop(T *ret){
        // printf("asdasfgg+++ggg)|");
#ifdef __CUDA_ARCH__
        *ret = atomicExch(data+count, invalid_value);
        if (*ret == invalid_value) return false;
        atomicSub((uint32_t *) &count, 1);
#endif
        return true;
    }

    __host__ __device__ static bool is_valid(T &x){
        return x != invalid_value;
    }

    __host__ __device__ static T get_invalid(){
        return invalid_value;
    }

    __device__ T pop_blocked(size_t maxretries = 0){
        T ret;
        size_t local_retries = 0;
        while (!try_pop(&ret)) {
            if (++local_retries == maxretries) {
                atomicAdd(&retries, local_retries);
                return invalid_value;
            }
        }
        atomicAdd(&retries, local_retries);
        return ret;
    }

    __device__ void push(T v){
        while (true){
            uint32_t old_cnt  = atomicAdd((uint32_t *) &count, 1);
            T ret = atomicCAS(data + old_cnt + 1, invalid_value, v);
            if (old_cnt + 1 == count && ret == invalid_value) return;
            ret   = atomicCAS(data + old_cnt + 1, v, invalid_value);
            if (ret == invalid_value) return;
            atomicSub((uint32_t *) &count, 1);
        }
    }

};

#endif /* OPTIMISTIC_SAFE_DEVICE_STACK_CUH_ */