#include "count.cuh"

using namespace std;

template<typename Tin, typename Tout>
__host__ dcount<Tin, Tout>::dcount(d_operator<Tout> parent, const launch_conf &conf): 
        parent(parent){
    assert(conf.device >= 0);
    out[0] = 0;
}

template<typename Tin, typename Tout>
__device__ void dcount<Tin, Tout>::consume_warp(const Tin * __restrict__ src, cnt_t N, vid_t vid, cid_t cid){
    const int32_t laneid            = get_laneid();

    if (laneid == 0) atomicAdd(out, N);
}

template<typename Tin, typename Tout>
__device__ void dcount<Tin, Tout>::consume_close(){}

template<typename Tin, typename Tout>
__device__ void dcount<Tin, Tout>::consume_open(){}

template<typename Tin, typename Tout>
__device__ void dcount<Tin, Tout>::at_open(){}

template<typename Tin, typename Tout>
__device__ void dcount<Tin, Tout>::at_close(){
    if (get_blockid() == 0){
        parent.consume_open();
        if (get_warpid() == 0) parent.consume_warp(out, 1, 0, 0);
        parent.consume_close();
    }
}

template<typename Tin, typename Tout>
__host__ void dcount<Tin, Tout>::before_open(){
    parent.open();
}

template<typename Tin, typename Tout>
__host__ void dcount<Tin, Tout>::after_close(){
    parent.close();
}

template class dcount<int32_t, int32_t>;
