#include "split.cuh"

using namespace std;

template<typename... T>
__host__ split<T...>::split(vector<d_operator<T..., sel_t>> parents, launch_conf conf, int dev):
                num_of_parents(parents.size()){
    assert(dev == conf.device);
    set_device_on_scope d(dev);

    max_index = parents.size() * vector_size * conf.total_num_of_warps();

    gpu(cudaMalloc(&(this->parents), sizeof(d_operator<T..., sel_t>) * parents.size()));
    gpu(cudaMalloc(&to_parents     , sizeof(sel_t) * max_index));
    // gpu(cudaMemset(to_parents   , 0, sizeof(T) * max_index));
    // gpu(cudaMalloc(&to_parents     , sizeof(T) * parents.size() * 5 * warp_size * conf.total_num_of_warps()));

    gpu(cudaMemcpy(this->parents, parents.data(), sizeof(d_operator<T..., sel_t>) * parents.size(), cudaMemcpyDefault));
}

template<typename... T>
__host__ void split<T...>::before_open(){
    split<T...> * s = (split<T...> *) malloc(sizeof(split<T...>));
    gpu(cudaMemcpy(s, this, sizeof(split<T...>), cudaMemcpyDefault));
    
    for (int i = 0 ; i < s->num_of_parents ; ++i) s->parents[i].open();

    free(s);
}

template<typename... T>
__device__ constexpr int split<T...>::parent_base(int parent_index, int warp_global_index) const{
    return vector_size * num_of_parents * warp_global_index + vector_size * parent_index;
}

template<typename... T>
__device__ void split<T...>::at_open(){}

template<typename... T>
__device__ void split<T...>::consume_open(){
    for (int i = 0 ; i < num_of_parents ; ++i) parents[i].consume_open();
}

__device__ int hash3(int32_t x){
    return (x & 1);
}

__device__ int hash3(int32_t x, int32_t y){
    return (x & 1);
}

template<typename... T>
__device__ void split<T...>::consume_warp(const T * __restrict__ ... src, cnt_t N, vid_t vid, cid_t cid) __restrict__{
    // const int32_t warpid            = get_warpid();
    const int32_t laneid            = get_laneid();

    // const int32_t prevwrapmask      = (1 << laneid) - 1;

    const uint32_t warp_global_index = get_global_warpid();

    #pragma unroll
    for (int i = 0 ; i < vector_size ; i += warp_size){
        int dest = -1;
        if (i + laneid < N){
            //compute destination
            dest = hash3(src[i + laneid]...);
        }

        for (int j = 0 ; j < num_of_parents ; ++j){
            int base                        = parent_base(j, warp_global_index);
            to_parents[base + i + laneid]   = (dest == j);
        }
    }

    for (int j = 0 ; j < num_of_parents ; ++j){
        parents[j].consume_warp(src..., to_parents + parent_base(j, warp_global_index), N, vid, cid);
    }
}




template<typename... T>
__device__ void split<T...>::consume_close(){
    for (int i = 0 ; i < num_of_parents ; ++i) parents[i].consume_close();
}

template<typename... T>
__device__ void split<T...>::at_close(){}

template<typename... T>
__host__ void split<T...>::after_close(){
    split<T...> * s = (split<T...> *) malloc(sizeof(split<T...>));
    gpu(cudaMemcpy(s, this, sizeof(split<T...>), cudaMemcpyDefault));

    for (int i = 0 ; i < s->num_of_parents ; ++i) s->parents[i].close();

    free(s);
}

template class split<int32_t>;
template class split<int32_t, int32_t>;
