#ifndef OUTPUT_COMPOSER_CUH
#define OUTPUT_COMPOSER_CUH

#include "buffer_pool.cuh"
#include "operator.cuh"

using namespace std;

template<size_t warp_size = WARPSIZE, typename T = int32_t>
class output_composer{
private:
    static_assert(sizeof(T)*8 == 32, "Operator is only implemented for data types of 32-bits size");
    Operator                      * parent;
    volatile buffer_t * volatile    output_buffer;
public:
    output_composer(Operator * parent, int dev = 0);

    __host__ __device__ void push(volatile T *x);
    __host__ __device__ void push_flush(T *buffer, uint32_t buffer_size);
    
    ~output_composer();
};

#endif /* OUTPUT_COMPOSER_CUH */