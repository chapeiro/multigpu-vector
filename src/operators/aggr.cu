#include "aggr.cuh"

using namespace std;

template<typename Tin, typename Tout>
__host__ aggr<Tin, Tout>::aggr(d_operator<Tout> parent, const launch_conf &conf): 
        parent(parent){
    assert(conf.device >= 0);
    out[0] = 0;
}

template<typename Tin, typename Tout>
__device__ void aggr<Tin, Tout>::consume_warp(const Tin * __restrict__ src, cnt_t N, vid_t vid, cid_t cid){
    const int32_t laneid            = get_laneid();

    Tout tmp = 0;

    for (uint32_t i = 0 ; i < vector_size ; i += 4*warp_size){
        vec4 s = reinterpret_cast<const vec4 *>(src)[i/4 + laneid];
        
        #pragma unroll
        for (uint32_t k = 0 ; k < 4 ; ++k) if (i + laneid*4 + k < N) tmp += s.i[k];
        
        // tmp += src[i + laneid];
    }

    #pragma unroll
    for (int m = warp_size >> 1; m > 0; m >>= 1)
        tmp += __shfl_xor(tmp, m);
    
    if (laneid == 0) atomicAdd(out, tmp);
}

template<typename Tin, typename Tout>
__device__ void aggr<Tin, Tout>::consume_close(){
    // parent.consume_close();
}

template<typename Tin, typename Tout>
__device__ void aggr<Tin, Tout>::consume_open(){
    // parent.consume_open();
}

template<typename Tin, typename Tout>
__device__ void aggr<Tin, Tout>::at_open(){}

template<typename Tin, typename Tout>
__device__ void aggr<Tin, Tout>::at_close(){
    if (get_blockid() == 0){
        parent.consume_open();
        if (get_warpid() == 0) parent.consume_warp(out, 1, 0, 0);
        parent.consume_close();
    }
}

template<typename Tin, typename Tout>
__host__ void aggr<Tin, Tout>::before_open(){
    parent.open();
}

template<typename Tin, typename Tout>
__host__ void aggr<Tin, Tout>::after_close(){
    parent.close();
}

template class aggr<int32_t, int32_t>;
