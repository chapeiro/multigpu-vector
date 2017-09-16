#ifndef MAP_CUH_
#define MAP_CUH_

#include "d_operator.cuh"

using namespace std;

template<typename F, typename TL, typename TR, typename Tout>
class map2{
private:
    d_operator<Tout>                parent;
    Tout                           *out;
    const F                         trans;
public:
    __host__ map2(d_operator<Tout> parent, F f, const launch_conf &conf);

    __host__   void before_open();
    __device__ void at_open();

    __device__ void consume_open();
    __device__ void consume_warp(const TL * __restrict__ L, const TR * __restrict__ R, cnt_t N, vid_t vid, cid_t cid);
    __device__ void consume_close();

    __device__ void at_close();
    __host__   void after_close();

    __host__ ~map2(){}
};

#endif /* MAP_CUH_ */