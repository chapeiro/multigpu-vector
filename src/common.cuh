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

#endif /* COMMON_CUH_ */