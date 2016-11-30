#ifndef BUFFER_CUH_
#define BUFFER_CUH_

#include "common.cuh"
#include <vector>

using namespace std;

template<typename T, size_t size = DEFAULT_BUFF_CAP>
class buffer {
public:
    size_t N;
    T     *data;

public:
    // __host__ __device__ buffer(T *data): N(0), data(data){ }

    __host__ buffer(int device = 0): N(0){
        set_device_on_scope d(device);
        cudaMalloc(&data, size * sizeof(T));
    }

    __host__ __device__ ~buffer(){
        cudaFree(data);
    }

    __host__ __device__ size_t count() const{
        return N;
    }

    __host__ __device__ static constexpr size_t capacity(){
        return size;
    }

    __host__ __device__ size_t remaining_capacity() const{
        return size - N;
    }

    __host__ __device__ void clean(){
        N = 0;
    }

    __host__ __device__ bool full() const{
        return N == size;
    }

    __host__ __device__ bool empty() const{
        return !N;
    }

    T * begin(){
        return data;
    }

    const T * begin() const{
        return data;
    }

    const T * cbegin() const{
        return data;
    }

    T * end(){
        return data + N;
    }

    const T * end() const{
        return data + N;
    }

    const T * cend() const{
        return data + N;
    }
};

#endif /* BUFFER_CUH_ */