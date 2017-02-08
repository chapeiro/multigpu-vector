#ifndef SPLIT_CUH_
#define SPLIT_CUH_

#include "../common.cuh"
#include "operator.cuh"
#include <vector>

using namespace std;

template<size_t warp_size = WARPSIZE, typename T = int32_t>
class split{
private:
    static_assert(sizeof(T)*8 == 32, "Operator is only implemented for data types of 32-bits size");
    d_operator_t                 ** parents;
    const int                       num_of_parents;
    T                             * to_parents;
    int max_index;

    constexpr int parent_base(int parent_index, int warp_global_index) const;
public:
    __host__ split(vector<d_operator_t *> parents, launch_conf conf, int dev = 0);

    __host__   void before_open();
    __device__ void at_open();

    __device__ void consume_open();
    __device__ void consume_warp(const T * src, unsigned int N);
    __device__ void consume_close();

    __device__ void at_close();
    __host__   void after_close();

    __host__ ~split(){}
};

#endif /* SPLIT_CUH_ */