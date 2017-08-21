#ifndef APPLY_SELECTION_CUH
#define APPLY_SELECTION_CUH

#include "d_operator.cuh"

using namespace std;

template<typename T>
class apply_selection{
private:
    // static_assert(sizeof(T)*8 == 32, "Operator is only implemented for data types of 32-bits size");
    cid_t                           cid;
    d_operator<T>                   parent;
    uint32_t                       *cnts;
    T                              *output;
public:
    __host__ apply_selection(d_operator<T> parent, const launch_conf &conf, cid_t cid = 0);

    __host__   void before_open();
    __device__ void at_open();

    __device__ void consume_open();
    __device__ void consume_warp(const T * __restrict__ src, const sel_t *__restrict__ pred, cnt_t N, vid_t vid, cid_t cid);
    __device__ void consume_close();

    __device__ void at_close();
    __host__   void after_close();

    __host__ ~apply_selection(){}
};

#endif /* APPLY_SELECTION_CUH */