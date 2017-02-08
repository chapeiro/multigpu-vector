#include "split.cuh"

using namespace std;

template<size_t warp_size, typename T>
__host__ split<warp_size, T>::split(vector<d_operator_t *> parents, launch_conf conf, int dev):
                num_of_parents(parents.size()){
    assert(dev == conf.device);
    set_device_on_scope d(dev);

    max_index = parents.size() * 5 * warp_size * conf.total_num_of_warps();

    gpu(cudaMalloc(&(this->parents), sizeof(d_operator_t *)*parents.size()));
    gpu(cudaMalloc(&to_parents     , sizeof(T) * max_index));
    gpu(cudaMemset(to_parents   , 0, sizeof(T) * max_index));
    // gpu(cudaMalloc(&to_parents     , sizeof(T) * parents.size() * 5 * warp_size * conf.total_num_of_warps()));

    gpu(cudaMemcpy(this->parents, parents.data(), sizeof(d_operator_t *)*parents.size(), cudaMemcpyDefault));
}

template<size_t warp_size, typename T>
__host__ void split<warp_size, T>::before_open(){
    split<warp_size, T> *s = (split<warp_size, T> *) malloc(sizeof(split<warp_size, T>));
    gpu(cudaMemcpy(s, this, sizeof(split<warp_size, T>), cudaMemcpyDefault));

    d_operator_t *ps[s->num_of_parents];
    gpu(cudaMemcpy(&ps, s->parents, sizeof(d_operator_t *)*s->num_of_parents, cudaMemcpyDefault));
    printf("+%p\n", s->to_parents);

    for (int i = 0 ; i < s->num_of_parents ; ++i) ps[i]->open();

    free(s);
}

template<size_t warp_size, typename T>
__device__ constexpr int split<warp_size, T>::parent_base(int parent_index, int warp_global_index) const{
    return 5 * warp_size * num_of_parents * warp_global_index + 5 * warp_size * parent_index;
}

template<size_t warp_size, typename T>
__device__ void split<warp_size, T>::at_open(){}

template<size_t warp_size, typename T>
__device__ void split<warp_size, T>::consume_open(){
    for (int i = 0 ; i < num_of_parents ; ++i) parents[i]->consume_open();
    __syncthreads();
}

__device__ int hash3(int32_t x){
    return (x & 1);
}

template<size_t warp_size, typename T>
__device__ void split<warp_size, T>::consume_warp(const T * src, unsigned int N){
    assert(N <= 4*warp_size);

    const int32_t warpid            = get_warpid();
    const int32_t laneid            = get_laneid();

    const int32_t prevwrapmask      = (1 << laneid) - 1;

    const uint32_t warp_global_index = get_global_warpid();

    #pragma unroll
    for (int k = 0 ; k < 4 ; ++k){
        int dest = -1;
        if (k*warpSize + laneid < N){
            //compute destination
            dest = hash3(src[k*warpSize + laneid]);
        }

        for (int i = 0 ; i < num_of_parents ; ++i){
            int32_t filter    = __ballot(dest == i);

            int32_t newpop    = __popc(filter);

            int base          = parent_base(i, warp_global_index);
            int32_t filterout;
            if (laneid == 0) assert(base + 5 * warp_size - 1 < max_index);
            if (laneid == 0) filterout = to_parents[base + 5 * warp_size - 1];
            filterout = brdcst(filterout, 0);

            //compute position of result
            if (dest == i){
                assert(filterout >= 0);
                assert(filterout <  5*warpSize);
                int32_t offset = filterout + __popc(filter & prevwrapmask);
                assert(offset >= 0);
                assert(offset <  5*warpSize);
                assert(base + offset < max_index);
                assert(base + offset >= 0);
                to_parents[base + offset] = src[k*warpSize + laneid];
            }

            filterout += newpop;

            if (filterout >= 4*warpSize){
                // output.push(wrapoutput);
                parents[i]->consume_warp(to_parents + base, 4*warpSize);

                assert(base + laneid < max_index);
                assert(base + laneid >= 0);
                assert(base + laneid +4*warpSize < max_index);
                assert(base + laneid +4*warpSize >= 0);
                to_parents[base + laneid]      = to_parents[base + laneid + 4*warpSize];
                filterout                     -= 4*warpSize;
            }

            if (laneid == 0) {
                assert(filterout >= 0);
                assert(filterout <  4*warpSize);
                assert(base + 5 * warpSize - 1 < max_index);
                assert(base + 5 * warpSize - 1 >= 0);
                to_parents[base + 5 * warp_size - 1] = filterout;
            }
        }
    }
}




template<size_t warp_size, typename T>
__device__ void split<warp_size, T>::consume_close(){
    __syncthreads();

    for (int i = 0 ; i < num_of_parents ; ++i) parents[i]->consume_close();
}

template<size_t warp_size, typename T>
__device__ void split<warp_size, T>::at_close(){ //FIXME: optimize
    const uint32_t laneid = get_laneid();
    const uint32_t warp_global_index = get_global_warpid();
    for (int i = 0 ; i < num_of_parents ; ++i){
        parents[i]->consume_open();
        __syncthreads();
        int base = parent_base(i, warp_global_index);
        int32_t cnt;
        if (laneid == 0) assert(base + 5 * warp_size - 1 < max_index);
        if (laneid == 0) cnt = to_parents[base + 5 * warp_size - 1];
        cnt = brdcst(cnt, 0);
        if (cnt > 0){
            assert(cnt > 0 && cnt <= 4*warp_size);
            parents[i]->consume_warp(to_parents + base, cnt);
        }
        __syncthreads();
        parents[i]->consume_close();
    }
}

template<size_t warp_size, typename T>
__host__ void split<warp_size, T>::after_close(){
    split<warp_size, T> *s = (split<warp_size, T> *) malloc(sizeof(split<warp_size, T>));
    gpu(cudaMemcpy(s, this, sizeof(split<warp_size, T>), cudaMemcpyDefault));

    d_operator_t *ps[s->num_of_parents];
    gpu(cudaMemcpy(&ps, s->parents, sizeof(d_operator_t *)*s->num_of_parents, cudaMemcpyDefault));
    printf("%p\n", s->to_parents);

    for (int i = 0 ; i < s->num_of_parents ; ++i) ps[i]->close();

    free(s);
}

template class split<WARPSIZE, int32_t>;
