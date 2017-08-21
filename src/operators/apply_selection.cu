#include "apply_selection.cuh"

using namespace std;

template<typename T>
__host__ apply_selection<T>::apply_selection(d_operator<T> parent, const launch_conf &conf, cid_t cid): 
        parent(parent), cid(cid){
    assert(conf.device >= 0);
    set_device_on_scope d(conf.device);

    gpu(cudaMalloc((void **) &output, (vector_size + warp_size) * conf.total_num_of_warps() * sizeof(T)));//FIXME: corrent type

    gpu(cudaMalloc((void **) &cnts  , conf.total_num_of_warps() * sizeof(int32_t)));
    gpu(cudaMemset(cnts,           0, conf.total_num_of_warps() * sizeof(int32_t)));
}

template<typename T>
__device__ void apply_selection<T>::consume_warp(const T * __restrict__ src, const sel_t *__restrict__ pred, cnt_t N, vid_t vid, cid_t cid){
    const int32_t laneid            = get_laneid();
    const int32_t warpid            = get_global_warpid();
    const int32_t prevwrapmask      = (1 << laneid) - 1;

    T * dst = output + (vector_size + warp_size) * warpid;

    assert(warpid < 1024);
    int32_t filterout = cnts[warpid];

    for (uint32_t i = 0 ; i < vector_size ; i += warp_size){
        sel_t predicate = ((i + laneid) < N) ? pred[i + laneid] : false;

        //aggregate predicate results
        int32_t filter = __ballot(predicate);

        int32_t newpop = __popc(filter);

        if (predicate){
            int32_t offset = filterout + __popc(filter & prevwrapmask);

            dst[offset]    =  src[i + laneid];
        }

        filterout += newpop;

        if (filterout >= vector_size){
            parent.consume_warp(dst, vector_size, vid, this->cid);

            dst[laneid]             = dst[laneid + vector_size];
            filterout              -= vector_size;
        }
    }

    if (laneid == 0) cnts[warpid] = filterout;
}

template<typename T>
__device__ void apply_selection<T>::consume_close(){
    parent.consume_close();
}

template<typename T>
__device__ void apply_selection<T>::consume_open(){
    parent.consume_open();
}

template<typename T>
__device__ void apply_selection<T>::at_open(){}

template<typename T>
__device__ void apply_selection<T>::at_close(){
    parent.consume_open();
    const int32_t laneid            = get_laneid();
    const int32_t warpid            = get_global_warpid();

    T * dst = output + (vector_size + warp_size) * warpid;

    int32_t filterout = cnts[warpid];

    if (filterout > 0) parent.consume_warp(dst, filterout, 0, cid);
    
    parent.consume_close();
}

template<typename T>
__host__ void apply_selection<T>::before_open(){
    parent.open();
}

template<typename T>
__host__ void apply_selection<T>::after_close(){
    parent.close();
}

template class apply_selection<int32_t>;
