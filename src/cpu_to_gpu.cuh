#ifndef CPU_TO_GPU_CUH_
#define CPU_TO_GPU_CUH_

#include "buffer_pool.cuh"
#include "operator.cuh"

using namespace std;

template<size_t warp_size = WARPSIZE, typename T = int32_t>
class cpu_to_gpu{
private:
    Operator      * parent;
    const dim3      parent_dimGrid;
    const dim3      parent_dimBlock;
    cudaStream_t    strm;
    int             device;

public:
    cpu_to_gpu(Operator * parent, dim3 parent_dimGrid, dim3 parent_dimBlock, int device);

    __host__ __device__ void open();

    __host__ __device__ void consume(buffer_pool<T>::buffer_t * data);

    __host__ __device__ void close();

    ~cpu_to_gpu();
};


#endif /* CPU_TO_GPU_CUH_ */