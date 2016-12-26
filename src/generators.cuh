#ifndef DATA_GENERATORS_CUH_
#define DATA_GENERATORS_CUH_

#include "buffer_pool.cuh"
#include "operator.cuh"
#include <vector>

using namespace std;

class generator{
private:
    Operator                           *parent;
    cudaStream_t                        strm;
    buffer_pool<int32_t>::buffer_t    **buff_ret;
    int32_t                            *src;
    uint32_t                            N;
public:
    __host__ generator(Operator * parent, int32_t *src, uint32_t N);

    __host__ __device__ void consume(buffer_pool<int32_t>::buffer_t * data);

    __host__ __device__ void join();

    __host__ ~generator();
};

#endif /* DATA_GENERATORS_CUH_ */