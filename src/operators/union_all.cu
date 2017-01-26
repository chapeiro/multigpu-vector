#include "union_all.cuh"
#include <iostream>

using namespace std;

template<size_t warp_size, typename T>
__host__ union_all<warp_size, T>::union_all(d_operator_t * parent, int num_of_children, launch_conf conf, int dev):
        parent(parent), num_of_children(num_of_children), num_of_active_children(0), num_of_closed_children(0){
    assert(dev == conf.device);
    host_lock = new mutex;
    
    set_device_on_scope d(conf.device);

    gpu(cudaMalloc(&num_of_consopened_warps,    sizeof(uint32_t)*conf.total_num_of_warps()));
    gpu(cudaMemset( num_of_consopened_warps, 0, sizeof(uint32_t)*conf.total_num_of_warps()));
}

template<size_t warp_size, typename T>
__host__ void union_all<warp_size, T>::before_open(){ //FIXME: review, does it break execution guarantee for the parent for the whole block ?
    union_all<warp_size, T> *s = (union_all<warp_size, T> *) malloc(sizeof(union_all<warp_size, T>));
    gpu(cudaMemcpy(s, this, sizeof(union_all<warp_size, T>), cudaMemcpyDefault));

    s->host_lock->lock();
    s->num_of_active_children++;
    if (s->num_of_active_children == 1) s->parent->open();
    gpu(cudaMemcpy(this, s, sizeof(union_all<warp_size, T>), cudaMemcpyDefault));
    s->host_lock->unlock();

    free(s);
}

template<size_t warp_size, typename T>
__device__ void union_all<warp_size, T>::at_open(){}

template<size_t warp_size, typename T>
__device__ void union_all<warp_size, T>::consume_open(){
    int consop;
    if (get_laneid() == 0) consop = atomicAdd(num_of_consopened_warps+get_global_warpid(), 1);
    consop = brdcst(consop, 0);
    if (consop == 0) parent->consume_open();
    __syncthreads();
}

template<size_t warp_size, typename T>
__device__ void union_all<warp_size, T>::consume_warp(const T * src, unsigned int N){
    parent->consume_warp(src, N);
}

template<size_t warp_size, typename T>
__device__ void union_all<warp_size, T>::consume_close(){
    __syncthreads();

    int consop;
    if (get_laneid() == 0) consop = atomicSub(num_of_consopened_warps+get_global_warpid(), 1);
    consop = brdcst(consop, 0);
    if (consop == 1) parent->consume_close();
}

template<size_t warp_size, typename T>
__device__ void union_all<warp_size, T>::at_close(){}

template<size_t warp_size, typename T>
__host__ void union_all<warp_size, T>::after_close(){ //FIXME: review
    union_all<warp_size, T> *s = (union_all<warp_size, T> *) malloc(sizeof(union_all<warp_size, T>));
    gpu(cudaMemcpy(s, this, sizeof(union_all<warp_size, T>), cudaMemcpyDefault));

    s->host_lock->lock();
    s->num_of_closed_children++;
    if (s->num_of_closed_children == s->num_of_children) s->parent->close();
    gpu(cudaMemcpy(this, s, sizeof(union_all<warp_size, T>), cudaMemcpyDefault));
    s->host_lock->unlock();

    free(s);
}

template<size_t warp_size, typename T>
__host__ union_all<warp_size, T>::~union_all(){
    delete host_lock;
}


template class union_all<WARPSIZE, int32_t>;
