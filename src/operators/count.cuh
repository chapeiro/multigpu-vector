#ifndef COUNT_CUH_
#define COUNT_CUH_

#include "d_operator.cuh"

using namespace std;

template<typename Tin, typename Tout>
class dcount{
private:
    Tout                            out[vector_size];
    d_operator<Tout>                parent;
    
public:
    __host__ dcount(d_operator<Tout> parent, const launch_conf &conf);

    __host__   void before_open();
    __device__ void at_open();

    __device__ void consume_open();
    __device__ void consume_warp(const Tin * __restrict__ src, cnt_t N, vid_t vid, cid_t cid);
    __device__ void consume_close();

    __device__ void at_close();
    __host__   void after_close();

    __host__ ~dcount(){}
};

#endif /* COUNT_CUH_ */