#include "union_all.cuh"
#include <iostream>

using namespace std;

template<typename... T>
__host__ union_all<T...>::union_all(d_operator<T...> parent, int num_of_children, launch_conf conf):
        parent(parent), num_of_children(num_of_children), num_of_active_children(0), num_of_closed_children(0){
    host_lock = new mutex;
    
    set_device_on_scope d(conf.device);

    gpu(cudaMalloc(&num_of_consopened_warps,    sizeof(uint32_t)*conf.total_num_of_warps()));
    gpu(cudaMemset( num_of_consopened_warps, 0, sizeof(uint32_t)*conf.total_num_of_warps()));
}

template<typename... T>
__host__ void union_all<T...>::before_open(){ //FIXME: review, does it break execution guarantee for the parent for the whole block ?
    union_all<T...> *s = (union_all<T...> *) malloc(sizeof(union_all<T...>));
    mutex *          m;
    gpu(cudaMemcpy(&m, &(this->host_lock), sizeof(mutex *), cudaMemcpyDefault));

    m->lock();
    gpu(cudaMemcpy(s, this, sizeof(union_all<T...>), cudaMemcpyDefault));
    s->num_of_active_children++;
    if (s->num_of_active_children == 1) this->parent.open();
    gpu(cudaMemcpy(this, s, sizeof(union_all<T...>), cudaMemcpyDefault));
    m->unlock();

    free(s);
}

template<typename... T>
__device__ void union_all<T...>::at_open(){}

template<typename... T>
__device__ void union_all<T...>::consume_open(){
    int consop;
    if (get_laneid() == 0) consop = atomicAdd(num_of_consopened_warps+get_global_warpid(), 1);
    consop = brdcst(consop, 0);
    if (consop == 0) parent.consume_open();
}

template<typename... T>
__device__ void union_all<T...>::consume_warp(const T * __restrict__ ... src, cnt_t N, vid_t vid, cid_t cid){
    parent.consume_warp(src..., N, vid, cid);
}

template<typename... T>
__device__ void union_all<T...>::consume_close(){
    int consop;
    if (get_laneid() == 0) consop = atomicSub(num_of_consopened_warps+get_global_warpid(), 1);
    consop = brdcst(consop, 0);
    if (consop == 1) parent.consume_close();
}

template<typename... T>
__device__ void union_all<T...>::at_close(){}

template<typename... T>
__host__ void union_all<T...>::after_close(){ //FIXME: review
    union_all<T...> *s = (union_all<T...> *) malloc(sizeof(union_all<T...>));
    mutex *          m;
    gpu(cudaMemcpy(&m, &(this->host_lock), sizeof(mutex *), cudaMemcpyDefault));

    m->lock();
    gpu(cudaMemcpy(s, this, sizeof(union_all<T...>), cudaMemcpyDefault));
    s->num_of_closed_children++;
    if (s->num_of_closed_children == s->num_of_children) this->parent.close();
    gpu(cudaMemcpy(this, s, sizeof(union_all<T...>), cudaMemcpyDefault));
    m->unlock();

    free(s);
}

template<typename... T>
__host__ union_all<T...>::~union_all(){
    delete host_lock;
}


template class union_all<int32_t>;
template class union_all<int32_t, int32_t>;
