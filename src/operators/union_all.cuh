#ifndef UNION_ALL_CUH_
#define UNION_ALL_CUH_

#include "../common.cuh"
#include "d_operator.cuh"
#include <vector>
#include <mutex>

using namespace std;

template<typename... T>
class union_all{
private:
    // static_assert(sizeof(T)*8 == 32, "Operator is only implemented for data types of 32-bits size");
    d_operator<T...>                parent;
    const int                       num_of_children;
    int                             num_of_active_children;
    int                             num_of_closed_children;
    int                           * num_of_consopened_warps;
    mutex                         * host_lock;

public:
    __host__ union_all(d_operator<T...> parent, int num_of_children, launch_conf conf);

    __host__   void before_open();
    __device__ void at_open();

    __device__ void consume_open();
    __device__ void consume_warp(const T * __restrict__ ... src, cnt_t N, vid_t vid, cid_t cid);
    __device__ void consume_close();

    __device__ void at_close();
    __host__   void after_close();

    __host__ ~union_all();
};

#endif /* UNION_ALL_CUH_ */