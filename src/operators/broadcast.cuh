#ifndef BROADCAST_CUH_
#define BROADCAST_CUH_

#include "d_operator.cuh"
#include <vector>

using namespace std;

template<typename... T>
class broadcast{
private:
    d_operator<T...>            *parents;
    const int32_t                parent_cnt;
    
public:
    __host__ broadcast(vector<d_operator<T...>> parents, const launch_conf &conf);

    __host__   void before_open();
    __device__ void at_open();

    __device__ void consume_open();
    __device__ void consume_warp(const T * __restrict__ ... src, cnt_t N, vid_t vid, cid_t cid);
    __device__ void consume_close();

    __device__ void at_close();
    __host__   void after_close();
};

#endif /* BROADCAST_CUH_ */