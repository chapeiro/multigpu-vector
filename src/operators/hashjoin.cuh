#ifndef HASHJOIN_CUH_
#define HASHJOIN_CUH_

#include "operator.cuh"

using namespace std;

template<size_t warp_size = WARPSIZE>
class hashjoin_builder{
private:
    int32_t                         shared_offset;
    int32_t                         index;
    int32_t                       * first;
    int32_t                       * next;
    int32_t                       * values;
    int32_t                         log_table_size;
public:
    __host__ hashjoin_builder(int32_t log_table_size, int32_t * first, int32_t * next, int32_t * values, int dev = 0);

    __host__   void before_open();
    __device__ void at_open();

    __device__ void consume_open();
    __device__ void consume_warp(const int32_t *src, unsigned int N);
    __device__ void consume_close();

    __device__ void at_close();
    __host__   void after_close();

    __host__ ~hashjoin_builder(){}
};

template<size_t warp_size = WARPSIZE>
class hashjoin{
private:
    d_operator_t                  * parent;
    int32_t                         log_table_size;
    int32_t                       * first;
    int32_t                       * next;
    int32_t                       * values;
    int32_t                         res;
public:
    d_operator_t                  * builder;

    __host__ hashjoin(d_operator_t * parent, int32_t log_table_size, int32_t max_size, const launch_conf &conf);

    __host__   void before_open();
    __device__ void at_open();

    __device__ void consume_open();
    __device__ void consume_warp(const int32_t *src, unsigned int N);
    __device__ void consume_close();

    __device__ void at_close();
    __host__   void after_close();

    __host__ ~hashjoin();
};

#endif /* HASHJOIN_CUH_ */