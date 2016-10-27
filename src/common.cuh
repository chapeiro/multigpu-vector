#ifndef COMMON_CUH_
#define COMMON_CUH_

#include <iostream>
#include <cassert>

#define gpu(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define WARPSIZE (32)

#if __CUDA_ARCH__ < 300 || defined (NUSE_SHFL)
#define BRDCSTMEM(blockDim) ((blockDim.x * blockDim.y)/ WARPSIZE)
#else
#define BRDCSTMEM(blockDim) (0)
#endif

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



#endif /* COMMON_CUH_ */