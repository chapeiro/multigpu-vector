#ifndef SELECT2_CUH_
#define SELECT2_CUH_

#include "buffer_pool.cuh"
#include "operator.cuh"
#include "output_composer.cuh"

using namespace std;

template<size_t warp_size = WARPSIZE, typename T = int32_t>
class unstable_select{
private:
    static_assert(sizeof(T)*8 == 32, "Operator is only implemented for data types of 32-bits size");
    Operator                      * parent;
    output_composer<warp_size, T> * output;
    volatile uint32_t               buffer_size;
    uint32_t                        finished;
    T                             * buffer;
public:
    unstable_select(Operator * parent, int grid_size = 0, int dev = 0);

    __host__ __device__ void consume(buffer_pool<int32_t>::buffer_t * data);
    __host__ __device__ void consume2(buffer_pool<int32_t>::buffer_t * data);

    __host__ __device__ void join();

    __host__ __device__ void consume_warp(const vec4 &x, unsigned int N);
    __host__ __device__ void consume_close();

    ~unstable_select(){}
};

#endif /* SELECT2_CUH_ */