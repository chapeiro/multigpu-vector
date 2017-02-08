#ifndef THREADSAFE_DEVICE_STACK_CUH_
#define THREADSAFE_DEVICE_STACK_CUH_

#include "common.cuh"
#include <cstdint>
#include <limits>

using namespace std;



/**
 * Based on a combination of :
 * * https://github.com/memsql/lockfree-bench/blob/master/stack/lockfree.h
 * * (use with care) http://moodycamel.com/blog/2014/solving-the-aba-problem-for-lock-free-free-lists
 * * http://www.boost.org/doc/libs/1_60_0/boost/lockfree/detail/freelist.hpp
 * * http://blog.memsql.com/common-pitfalls-in-writing-lock-free-algorithms/
 */
template<typename T, T invalid_value = numeric_limits<T>::max()>
class threadsafe_device_stack{ //FIXME: must have a bug
private:
    volatile uint32_t cnt;
    const    uint32_t size;

    volatile int lock;
    volatile T * data;

public:
    __host__ threadsafe_device_stack(uint32_t size, vector<T> fill, int device):
                cnt(0), size(size), lock(0){
        set_device_on_scope d(device);
        gpu(cudaMalloc(&data, size*sizeof(T)));

        if (fill.size() > 0){
            gpu(cudaMemcpy((void *) data, fill.data(), fill.size()*sizeof(T), cudaMemcpyDefault));
            cnt = fill.size();
        }
    }

    __host__ ~threadsafe_device_stack(){
        gpu(cudaFree(data));
    }

public:
    __device__ void push(T v){
        assert(__popc(__ballot(1)) == 1);

        while (atomicCAS((int *) &lock, 0, 1) != 0);

        assert(cnt < size);
        data[cnt++] = v;

        lock = 0;
    }

    __device__ bool try_pop(T *ret){
        assert(__popc(__ballot(1)) == 1);

        if (atomicCAS((int *) &lock, 0, 1) != 0) return false;

        if (cnt == 0) {
            lock = 0;
            return false;
        }

        *ret = data[--cnt];

        lock = 0;

        return true;
    }

    __device__ T pop(){ //blocking
        assert(__popc(__ballot(1)) == 1);
        T ret;
        while (!try_pop(&ret));
        return ret;
    }

    __host__ __device__ static bool is_valid(T &x){
        return x != invalid_value;
    }

    __host__ __device__ static T get_invalid(){
        return invalid_value;
    }
};

#endif /* THREADSAFE_DEVICE_STACK_CUH_ */