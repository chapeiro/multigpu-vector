#ifndef SYNC_CUH
#define SYNC_CUH

#include "d_operator.cuh"
#include <atomic>

using namespace std;

template<typename TL, typename TR>
class synchro{
private:
    // static_assert(sizeof(T)*8 == 32, "Operator is only implemented for data types of 32-bits size");
    d_operator<TL, TR>              parent;
    uint32_t                        slots;
    cid_t                           lcid;
    cid_t                           ocid;
    volatile uint32_t                       *lcnts;
    volatile uint32_t                       *rcnts;
    TL                             *buffer;
    atomic<uint32_t>               *opened;
    atomic<uint32_t>               *closed;
public:
    __host__ synchro(d_operator<TL, TR> parent, uint32_t slots, cid_t lcid, cid_t ocid, const launch_conf &conf);

    __host__   void before_open();
    __device__ void at_open();

    __device__ void consume_open();

    __device__ void consume_warp(const void * __restrict__ src, cnt_t N, vid_t vid, cid_t cid);

    __device__ void consume_close();

    __device__ void at_close();
    __host__   void after_close();

    __host__ ~synchro(){}
};

#endif /* SYNC_CUH */