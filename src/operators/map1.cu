#include "map1.cuh"
#include "../dfunctional.cuh"

using namespace std;

template<typename F, typename Tin, typename Tout>
__host__ map1<F, Tin, Tout>::map1(d_operator<Tout> parent, F f, const launch_conf &conf): 
        parent(parent), trans(f){
    assert(conf.device >= 0);
    set_device_on_scope d(conf.device);

    gpu(cudaMalloc((void **) &out, vector_size * conf.total_num_of_warps() * sizeof(Tout)));
}

template<typename F, typename Tin, typename Tout>
__device__ void map1<F, Tin, Tout>::consume_warp(const Tin * __restrict__ src, cnt_t N, vid_t vid, cid_t cid) __restrict__{
    const int32_t laneid            = get_laneid();

    Tout * dst = out + vector_size * get_global_warpid();

    F f(trans);

    #pragma unroll
    for (uint32_t i = 0 ; i < vector_size ; i += 4*warp_size){
        vec4 s = reinterpret_cast<const vec4 *>(src)[i/4 + laneid];

        vec4 o;
        
        #pragma unroll
        for (uint32_t k = 0 ; k < 4 ; ++k) o.i[k] = f(s.i[k]);

        reinterpret_cast<int4 *>(dst)[i/4 + laneid] = o.vec;
    }

    parent.consume_warp(dst, N, vid, cid);
}

template<typename F, typename Tin, typename Tout>
__device__ void map1<F, Tin, Tout>::consume_close(){
    parent.consume_close();
}

template<typename F, typename Tin, typename Tout>
__device__ void map1<F, Tin, Tout>::consume_open(){
    parent.consume_open();
}

template<typename F, typename Tin, typename Tout>
__device__ void map1<F, Tin, Tout>::at_open(){}

template<typename F, typename Tin, typename Tout>
__device__ void map1<F, Tin, Tout>::at_close(){}

template<typename F, typename Tin, typename Tout>
__host__ void map1<F, Tin, Tout>::before_open(){
    parent.open();
}

template<typename F, typename Tin, typename Tout>
__host__ void map1<F, Tin, Tout>::after_close(){
    parent.close();
}

template class map1<equal_tov<int32_t>, int32_t, sel_t>;
template class map1<less_than<int32_t>, int32_t, sel_t>;
template class map1<in_range<int32_t>, int32_t, sel_t>;
