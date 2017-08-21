#ifndef SPLIT_CUH_
#define SPLIT_CUH_

#include "../common.cuh"
#include "d_operator.cuh"
#include <vector>

using namespace std;

template<typename... T>
class split{
private:
    d_operator<T..., sel_t>       * parents;
    const int                       num_of_parents;
    sel_t *                         to_parents;
    int max_index;

    __device__ constexpr __forceinline__ int parent_base(int parent_index, int warp_global_index) const;
public:
    __host__ split(vector<d_operator<T..., sel_t>> parents, launch_conf conf, int dev = 0);

    __host__   void before_open();
    __device__ void at_open();

    __device__ void consume_open();
    __device__ void consume_warp(const T * __restrict__ ... src, cnt_t N, vid_t vid, cid_t cid) __restrict__;
    __device__ void consume_close();

    __device__ void at_close();
    __host__   void after_close();

    __host__ ~split(){}
};

#endif /* SPLIT_CUH_ */