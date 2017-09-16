#include "sync.cuh"
#include <type_traits>

using namespace std;

template<typename TL, typename TR>
__host__ synchro<TL, TR>::synchro(d_operator<TL, TR> parent, uint32_t slots, cid_t lcid, cid_t ocid, const launch_conf &conf):
        parent(parent), slots(slots), lcid(lcid), ocid(ocid){
    assert(conf.device >= 0);
    set_device_on_scope d(conf.device);

    opened = new atomic<uint32_t>(0);
    closed = new atomic<uint32_t>(0);

    gpu(cudaMalloc((void **) &buffer, ((size_t) vector_size) * conf.total_num_of_warps() * slots * sizeof(TL)));

    gpu(cudaMalloc((void **) &lcnts, conf.total_num_of_warps() * sizeof(uint32_t)));
    gpu(cudaMalloc((void **) &rcnts, conf.total_num_of_warps() * sizeof(uint32_t)));

    gpu(cudaMemset((uint32_t *) lcnts, 0, conf.total_num_of_warps() * sizeof(uint32_t)));
    gpu(cudaMemset((uint32_t *) rcnts, 0, conf.total_num_of_warps() * sizeof(uint32_t)));
}

// template<typename TL, typename TR>
// template<typename std::enable_if<!std::is_same<TL, TR>::type>>
// __device__ void synchro<TL, TR>::consume_warp(const TL * __restrict__ src, vid_t vid, cid_t cid) __restrict__{
//     const int32_t warpid            = get_global_warpid();
//     const int32_t laneid            = get_laneid();
    
//     TL * dst = buffer + vector_size * (slots + (lcnts[warpid] % slots)) * warpid;

//     if (laneid == 0) ++lcnts[warpid];

//     #pragma unroll
//     for (uint32_t i = 0 ; i < vector_size ; i += 4*warp_size){
//         reinterpret_cast<int4 *>(dst)[i/4 + laneid] = reinterpret_cast<const vec4 *>(src)[i/4 + laneid].vec;
//     }
// }


// template<typename TL, typename TR>
// __device__ void synchro<TL, TR>::consume_warp(const void * __restrict__ src, cnt_t N, vid_t vid, cid_t cid) __restrict__{
//     const int32_t warpid            = get_global_warpid();
//     const int32_t laneid            = get_laneid();
    
//     assert(opened > 0);
//     assert(closed < 2);
//     if (cid == lcid){
//         TL * dst  = buffer + vector_size * (slots * warpid + (lcnts[warpid] % slots));

//         if (laneid == 0) {
//             ++lcnts[warpid];
//             assert(((int32_t) (lcnts[warpid] - rcnts[warpid])) <= slots);
//         }

//         #pragma unroll
//         for (uint32_t i = 0 ; i < vector_size ; i += 4*warp_size){
//             reinterpret_cast<int4 *>(dst)[i/4 + laneid] = reinterpret_cast<const vec4 *>(src)[i/4 + laneid].vec;
//         }

//     } else {
//         TL * lsrc = buffer + vector_size * (slots * warpid + (rcnts[warpid] % slots));
        
//         if (laneid == 0) {
//             assert(lcnts[warpid] >= rcnts[warpid]);
//             ++rcnts[warpid];
//         }

//         parent.consume_warp((TL *) lsrc, (TR *) src, N, vid, cid);
//     }
// }

template<typename TL, typename TR>
__device__ void synchro<TL, TR>::consume_warp(const void * __restrict__ src, cnt_t N, vid_t vid, cid_t cid){
    static_assert(sizeof(TL) == sizeof(TR), "Unimplemented for different sizes");
    const int32_t warpid            = get_global_warpid();
    const int32_t laneid            = get_laneid();
    
    if (cid == lcid){
        void * dst  = buffer + vector_size * (slots * warpid + (lcnts[warpid] % slots));
        if (lcnts[warpid] >= rcnts[warpid]){
            #pragma unroll
            for (uint32_t i = 0 ; i < vector_size ; i += 4*warp_size){
                reinterpret_cast<int4 *>(dst)[i/4 + laneid] = reinterpret_cast<const vec4 *>(src)[i/4 + laneid].vec;
            }
        } else {
            parent.consume_warp((TL *) src, (TR *) dst, N, vid, ocid);
        }
        if (laneid == 0) {
            assert(lcnts[warpid] < rcnts[warpid] + slots);
            ++lcnts[warpid];
        }
    } else {
        void * dst  = buffer + vector_size * (slots * warpid + (rcnts[warpid] % slots));
        if (lcnts[warpid] <= rcnts[warpid]){
            #pragma unroll
            for (uint32_t i = 0 ; i < vector_size ; i += 4*warp_size){
                reinterpret_cast<int4 *>(dst)[i/4 + laneid] = reinterpret_cast<const vec4 *>(src)[i/4 + laneid].vec;
            }
        } else {
            parent.consume_warp((TL *) dst, (TR *) src, N, vid, ocid);
        }

        if (laneid == 0) {
            assert(rcnts[warpid] < lcnts[warpid] + slots);
            ++rcnts[warpid];
        }
    }
}

template<typename TL, typename TR>
__device__ void synchro<TL, TR>::consume_close(){
    parent.consume_close();
}

template<typename TL, typename TR>
__device__ void synchro<TL, TR>::consume_open(){
    parent.consume_open();
}

template<typename TL, typename TR>
__device__ void synchro<TL, TR>::at_open(){}

template<typename TL, typename TR>
__device__ void synchro<TL, TR>::at_close(){
// #ifndef NDEBUG
//     if (closed == 1 && get_laneid() == 0 && lcnts[get_global_warpid()] != rcnts[get_global_warpid()]){
//         printf("%d %d %d %d\n", lcnts[get_global_warpid()], rcnts[get_global_warpid()], lcid, ocid);
//         assert(false);
//     }
// #endif
}

template<typename TL, typename TR>
__host__ void synchro<TL, TR>::before_open(){
    atomic<uint32_t> *opened;
    gpu(cudaMemcpy(&opened, &(this->opened), sizeof(atomic<uint32_t> *), cudaMemcpyDefault));
    if (opened->fetch_add(1, std::memory_order_relaxed) == 0) parent.open();
}

template<typename TL, typename TR>
__host__ void synchro<TL, TR>::after_close(){
    atomic<uint32_t> *closed;
    gpu(cudaMemcpy(&closed, &(this->closed), sizeof(atomic<uint32_t> *), cudaMemcpyDefault));
    if (closed->fetch_add(1, std::memory_order_relaxed) == 1) parent.close();
}

template class synchro<int32_t, sel_t>;
template class synchro<sel_t, sel_t>;
template class synchro<int32_t, int32_t>;
