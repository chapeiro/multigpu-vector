#include "broadcast.cuh"

using namespace std;

template<typename... T>
__host__ broadcast<T...>::broadcast(vector<d_operator<T...>> parents, const launch_conf &conf): 
                                                    parent_cnt(parents.size()){
    assert(conf.device >= 0);

    set_device_on_scope d(conf.device);
    
    gpu(cudaMalloc(&(this->parents), parent_cnt*sizeof(d_operator<T...>)));

    gpu(cudaMemcpy(this->parents, parents.data(), parent_cnt*sizeof(d_operator<T...>), cudaMemcpyDefault));
}

template<typename... T>
__device__ void broadcast<T...>::consume_warp(const T * __restrict__ ... src, cnt_t N, vid_t vid, cid_t cid) {
    for (int32_t i = 0 ; i < parent_cnt ; ++i) parents[i].consume_warp(src..., N, vid, cid);
}

template<typename... T>
__device__ void broadcast<T...>::consume_close(){
    for (int32_t i = 0 ; i < parent_cnt ; ++i) parents[i].consume_close();
}

template<typename... T>
__device__ void broadcast<T...>::consume_open(){
    for (int32_t i = 0 ; i < parent_cnt ; ++i) parents[i].consume_open();
}

template<typename... T>
__device__ void broadcast<T...>::at_open(){}

template<typename... T>
__device__ void broadcast<T...>::at_close(){}

template<typename... T>
__host__ void broadcast<T...>::before_open(){
    broadcast<T...> * tmp = (broadcast<T...> *) malloc(sizeof(broadcast<T...>));
    gpu(cudaMemcpy(tmp, this, sizeof(broadcast<T...>), cudaMemcpyDefault));
    for (int32_t i = 0 ; i < tmp->parent_cnt ; ++i) tmp->parents[i].open();
    free(tmp);
}

template<typename... T>
__host__ void broadcast<T...>::after_close(){
    broadcast<T...> * tmp = (broadcast<T...> *) malloc(sizeof(broadcast<T...>));
    gpu(cudaMemcpy(tmp, this, sizeof(broadcast<T...>), cudaMemcpyDefault));
    for (int32_t i = 0 ; i < tmp->parent_cnt ; ++i) tmp->parents[i].close();
    free(tmp);
}

template class broadcast<int32_t>;
template class broadcast<sel_t>;

