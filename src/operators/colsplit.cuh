#ifndef COLSPLIT_CUH_
#define COLSPLIT_CUH_

#include "d_operator.cuh"

using namespace std;

template<typename TL, typename TR, typename... Tload>
class colsplit{
private:
    d_operator<TL, Tload...>       parentL;
    d_operator<TR, Tload...>       parentR;
    cid_t                       cidL;
    cid_t                       cidR;
public:
    __host__ colsplit(d_operator<TL, Tload...> parentL, d_operator<TR, Tload...> parentR, cid_t cidL, cid_t cidR, const launch_conf &conf);

    __host__   void before_open();
    __device__ void at_open();

    __device__ void consume_open();
    __device__ void consume_warp(const TL * __restrict__ srcL, const TR * __restrict__ srcR, const Tload * __restrict__ ... sel, cnt_t N, vid_t vid, cid_t cid);
    __device__ void consume_close();

    __device__ void at_close();
    __host__   void after_close();
};

#endif /* COLSPLIT_CUH_ */