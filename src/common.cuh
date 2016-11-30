#ifndef COMMON_CUH_
#define COMMON_CUH_

#include <iostream>
#include <cassert>
#include <type_traits>

#ifndef DEFAULT_BUFF_CAP
#define DEFAULT_BUFF_CAP (1024)
#endif

#define gpu(ans) { gpuAssert((ans), __FILE__, __LINE__); }

__host__ __device__ inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
    if (code != cudaSuccess) {
#ifndef __CUDA_ARCH__
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
#else
        printf("GPUassert: %s %s %d\n", "error", file, line);
#endif
    }
}

#define WARPSIZE (32)

// #if __CUDA_ARCH__ < 300 || defined (NUSE_SHFL)
#define BRDCSTMEM(blockDim) ((blockDim.x * blockDim.y)/ WARPSIZE)
// #else
// #define BRDCSTMEM(blockDim) (0)
// #endif

class set_device_on_scope{
private:
    int device;
public:
    inline set_device_on_scope(int set){
        gpu(cudaGetDevice(&device));
        gpu(cudaSetDevice(set));
    }

    inline ~set_device_on_scope(){
        gpu(cudaSetDevice(device));
    }
};

template<typename T,
         typename std::enable_if<sizeof(T) == sizeof(unsigned long long int),
            int>::type = 0>
__device__ T atomicExch(T *address, T val){
    return (T) atomicExch((unsigned long long int*) address, (unsigned long long int) val);
}

template<typename T,
         typename std::enable_if<sizeof(T) == sizeof(unsigned int) && !std::is_signed<T>::value,
            int>::type = 0>
__device__ T atomicExch(T *address, T val){
    return (T) atomicExch((unsigned int*) address, (unsigned int) val);
}

template<typename T,
         typename std::enable_if<sizeof(T) == sizeof(int) && std::is_signed<T>::value,
            int>::type = 0>
__device__ T atomicExch(T *address, T val){
    return (T) atomicExch((int*) address, (int) val);
}


#endif /* COMMON_CUH_ */