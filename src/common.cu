#include "common.cuh"

__host__ int get_device(const void *p){
    cudaPointerAttributes attrs;
    cudaError_t error = cudaPointerGetAttributes(&attrs, p);
    if (error == cudaErrorInvalidValue) return -1;
    gpu(error);
    return (attrs.memoryType == cudaMemoryTypeHost) ? -1 : attrs.device;
}