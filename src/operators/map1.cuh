#ifndef MAP1_CUH_
#define MAP1_CUH_

#include "d_operator.cuh"

using namespace std;

template<typename F, typename Tin, typename Tout>
class map1{
private:
    d_operator<Tout>                parent;
    Tout                           *out;
    const F                         trans;
public:
    __host__ map1(d_operator<Tout> parent, F f, const launch_conf &conf);

    __host__   void before_open();
    __device__ void at_open();

    __device__ void consume_open();
    __device__ void consume_warp(const Tin * __restrict__ src, cnt_t N, vid_t vid, cid_t cid) __restrict__;
    __device__ void consume_close();

    __device__ void at_close();
    __host__   void after_close();

    __host__ ~map1(){}
};

#endif /* MAP1_CUH_ */