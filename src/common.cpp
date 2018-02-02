#include "common.cuh"

int get_device(const void *p){
#ifndef NCUDA
    cudaPointerAttributes attrs;
    cudaError_t error = cudaPointerGetAttributes(&attrs, p);
    if (error == cudaErrorInvalidValue) return -1;
    gpu(error);
    return (attrs.memoryType == cudaMemoryTypeHost) ? -1 : attrs.device;
#else
    return -1;
#endif
}
