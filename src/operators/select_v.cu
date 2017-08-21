#include "select_v.cuh"

using namespace std;

template<size_t warp_size, typename F, typename T>
__host__ select_v<warp_size, F, T>::select_v(d_operator_t * parent, F f, const launch_conf &conf): 
        parent(parent), filt(f){
    assert(conf.device >= 0);
    set_device_on_scope d(conf.device);

    gpu(cudaMalloc((void **) &sel_vec_buffer, vector_size * conf.total_num_of_warps() * sizeof(uint32_t)));//FIXME: corrent type
}

template<size_t warp_size, typename F, typename T>
__device__ void select_v<warp_size, F, T>::consume_warp(const T * __restrict__ src, vid_t vid, cid_t cid){
    const int32_t laneid            = get_laneid();

    uint32_t * sel_vec = sel_vec_buffer + vector_size * get_global_warpid();

    F filter(filt);

    for (uint32_t i = 0 ; i < vector_size ; i += warp_size){
        sel_vec[i + laneid] = filter(src[i + laneid]);
    }

    parent->consume_warp((int32_t *) sel_vec, vector_size);//vid, cid);
}

template<size_t warp_size, typename F, typename T>
__device__ void select_v<warp_size, F, T>::consume_close(){
    parent->consume_close();
}

template<size_t warp_size, typename F, typename T>
__device__ void select_v<warp_size, F, T>::consume_open(){
    parent->consume_open();
}

template<size_t warp_size, typename F, typename T>
__device__ void select_v<warp_size, F, T>::at_open(){}

template<size_t warp_size, typename F, typename T>
__device__ void select_v<warp_size, F, T>::at_close(){}

template<size_t warp_size, typename F, typename T>
__host__ void select_v<warp_size, F, T>::before_open(){
    decltype(this->parent) p;
    gpu(cudaMemcpy(&p, &(this->parent), sizeof(decltype(this->parent)), cudaMemcpyDefault));
    p->open();
}

template<size_t warp_size, typename F, typename T>
__host__ void select_v<warp_size, F, T>::after_close(){
    decltype(this->parent) p;
    gpu(cudaMemcpy(&p, &(this->parent), sizeof(decltype(this->parent)), cudaMemcpyDefault));
    p->close();
}

template class select_v<WARPSIZE, less_eq_than_v<int32_t>, int32_t>;
