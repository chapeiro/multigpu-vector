#ifndef AGGREGATION_CUH_
#define AGGREGATION_CUH_

#include "operator.cuh"

using namespace std;

template<size_t warp_size = WARPSIZE, int32_t neutral_value = 0>
class aggregation{
private:
    d_operator_t                  * parent;
    int32_t                         res;
    int32_t                         shared_offset;

    static __inline__ __device__ int32_t warpReduce(int32_t x);
public:
    __host__ aggregation(d_operator_t * parent, int32_t shared_offset, int grid_size = 0, int dev = 0);

    __host__   void before_open();
    __device__ void at_open();

    __device__ void consume_open();
    __device__ void consume_warp(const int32_t *src, unsigned int N);
    __device__ void consume_close();

    __device__ void at_close();
    __host__   void after_close();

    __host__ ~aggregation(){}
};

#endif /* AGGREGATION_CUH_ */