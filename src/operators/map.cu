#include "map.cuh"
#include "../dfunctional.cuh"

using namespace std;

template<typename F, typename TL, typename TR, typename Tout>
__host__ map2<F, TL, TR, Tout>::map2(d_operator<Tout> parent, F f, const launch_conf &conf): 
        parent(parent), trans(f){
    assert(conf.device >= 0);
    set_device_on_scope d(conf.device);

    gpu(cudaMalloc((void **) &out, vector_size * conf.total_num_of_warps() * sizeof(Tout)));
}

template<typename F, typename TL, typename TR, typename Tout>
__device__ void map2<F, TL, TR, Tout>::consume_warp(const TL * __restrict__ L, const TR * __restrict__ R, cnt_t N, vid_t vid, cid_t cid){
    const int32_t laneid            = get_laneid();

    Tout * dst = out + vector_size * get_global_warpid();

    F f(trans);

    #pragma unroll
    for (uint32_t i = 0 ; i < vector_size ; i += 4*warp_size){
        vec4 l = reinterpret_cast<const vec4 *>(L)[i/4 + laneid];
        vec4 r = reinterpret_cast<const vec4 *>(R)[i/4 + laneid];
        vec4 o;
        
        #pragma unroll
        for (uint32_t k = 0 ; k < 4 ; ++k) if (i + 4*laneid + k < N) o.i[k] = f(l.i[k], r.i[k]);

        reinterpret_cast<int4 *>(dst)[i/4 + laneid] = o.vec;
    }

    parent.consume_warp(dst, N, vid, cid);
}

template<typename F, typename TL, typename TR, typename Tout>
__device__ void map2<F, TL, TR, Tout>::consume_close(){
    parent.consume_close();
}

template<typename F, typename TL, typename TR, typename Tout>
__device__ void map2<F, TL, TR, Tout>::consume_open(){
    parent.consume_open();
}

template<typename F, typename TL, typename TR, typename Tout>
__device__ void map2<F, TL, TR, Tout>::at_open(){}

template<typename F, typename TL, typename TR, typename Tout>
__device__ void map2<F, TL, TR, Tout>::at_close(){}

template<typename F, typename TL, typename TR, typename Tout>
__host__ void map2<F, TL, TR, Tout>::before_open(){
    parent.open();
}

template<typename F, typename TL, typename TR, typename Tout>
__host__ void map2<F, TL, TR, Tout>::after_close(){
    parent.close();
}

template class map2<product, int32_t, int32_t, int32_t>;
template class map2<log_and<sel_t>, sel_t, sel_t, sel_t>;