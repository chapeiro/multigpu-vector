#ifndef SELECT3_CUH_
#define SELECT3_CUH_

#include "buffer_pool.cuh"
#include "operator.cuh"
#include <vector>
// #include "output_composer.cuh"

using namespace std;

template<size_t warp_size = WARPSIZE, typename T = int32_t>
class unstable_select{
private:
    static_assert(sizeof(T)*8 == 32, "Operator is only implemented for data types of 32-bits size");
    d_operator_t                  * parent;
    // output_composer<warp_size, T>   output;
    volatile uint32_t               buffer_size;
    uint32_t                        finished;
    T                             * buffer;
public:
    __host__ unstable_select(d_operator_t * parent, int grid_size = 0, int dev = 0);

    __host__   void before_open();
    __device__ void at_open();

    __device__ void consume_open();
    __device__ void consume_warp(const int32_t *src, unsigned int N);
    __device__ void consume_close();

    __device__ void at_close();
    __host__   void after_close();

    __host__ ~unstable_select(){}
};

#endif /* SELECT3_CUH_ */