#include "common.cuh"
#include "hashjoin.cuh"

__device__ uint32_t hashMurmur(uint32_t x){
    x ^= x >> 16;
    x *= 0x85ebca6b;
    x ^= x >> 13;
    x *= 0xc2b2ae35;
    x ^= x >> 16;
    return x;
}

// NOTE: do not use this as atomicExch is loopless and has better guarantees when val is the same for multiple threads
__device__ int32_t atomicSet(int32_t* address, int32_t val) {
    int32_t old = *address;
    int32_t expected;
    do {
        expected = old;
        old = atomicCAS(address, expected, val);
    } while (expected != old);
    return old;
}

template<size_t warp_size>
__device__ void hashjoin_builder<warp_size>::at_open(){}

template<size_t warp_size>
__host__ hashjoin_builder<warp_size>::hashjoin_builder(int32_t log_table_size, int32_t * first, int32_t * next, int32_t * values, int dev):
        index(0), first(first), next(next), values(values), log_table_size(log_table_size){}

template<size_t warp_size>
__device__ void hashjoin_builder<warp_size>::at_close(){}

template<size_t warp_size>
__device__ void hashjoin_builder<warp_size>::consume_open(){}

template<size_t warp_size>
__device__ void hashjoin_builder<warp_size>::consume_close(){}

template<size_t warp_size>
__device__ void hashjoin_builder<warp_size>::consume_warp(const int32_t *src, unsigned int N){
    const int32_t laneid = get_laneid();

    int32_t offset;
    if (laneid == 0) offset = atomicAdd(&index, N);
    offset = brdcst(offset, 0);

    #pragma unroll
    for (int k = 0 ; k < 4 ; ++k){
        const uint32_t ind = k*warp_size + laneid;
        if (ind < N) {
            uint32_t bucket      = hashMurmur(src[ind]) & ((1 << log_table_size) - 1);

            next  [offset + ind] = atomicExch(first + bucket, offset + ind);

            values[offset + ind] = src[ind];
        }
    }
}

template<size_t warp_size>
__host__ hashjoin<warp_size>::hashjoin(d_operator_t * parent, int32_t log_table_size, int32_t max_size, const launch_conf &conf):
        parent(parent), log_table_size(log_table_size), res(0){
    set_device_on_scope d(conf.device);

    gpu(cudaMalloc(&first , (1 << log_table_size) * sizeof(int32_t)));
    gpu(cudaMalloc(&next  ,             max_size  * sizeof(int32_t)));
    gpu(cudaMalloc(&values,             max_size  * sizeof(int32_t)));

    //carefull, this works because all bytes of -1 are the same
    gpu(cudaMemset(first, -1, (1 << log_table_size) * sizeof(int32_t)));

    builder = d_operator_t::create<hashjoin_builder<warp_size>>(conf, log_table_size, first, next, values, conf.device);
}

template<size_t warp_size>
__host__ hashjoin<warp_size>::~hashjoin(){
    gpu(cudaFree(first ));
    gpu(cudaFree(next  ));
    gpu(cudaFree(values));
}

template<size_t warp_size>
__device__ void hashjoin<warp_size>::at_open(){}

template<size_t warp_size>
__device__ void hashjoin<warp_size>::at_close(){
    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0){
        parent->consume_open();
        __syncthreads();
        if (get_warpid() == 0) parent->consume_warp(&res, 1);
        __syncthreads();
        parent->consume_close();
    }
}

template<size_t warp_size>
__device__ void hashjoin<warp_size>::consume_open(){}

template<size_t warp_size>
__device__ void hashjoin<warp_size>::consume_close(){}

template<size_t warp_size>
__device__ void hashjoin<warp_size>::consume_warp(const int32_t *src, unsigned int N){
    const int32_t laneid = get_laneid();

    #pragma unroll
    for (int k = 0 ; k < 4 ; ++k){
        const uint32_t ind = k*warp_size + laneid;
        int32_t current = -1;
        int32_t probe;
        uint32_t bucket;
        if (ind < N) {
            probe   = src[ind];
            bucket  = hashMurmur(probe) & ((1 << log_table_size) - 1);

            current = first[bucket];
            while (current >= 0){
                if (values[current] == probe) atomicAdd(&res, 1);
                current = next[current];
            }
        }
    }
}

template<size_t warp_size>
__host__ void hashjoin<warp_size>::before_open(){
    decltype(this->parent) p;
    gpu(cudaMemcpy(&p, &(this->parent), sizeof(decltype(this->parent)), cudaMemcpyDefault));
    p->open();
}

template<size_t warp_size>
__host__ void hashjoin<warp_size>::after_close(){
    decltype(this->parent) p;
    gpu(cudaMemcpy(&p, &(this->parent), sizeof(decltype(this->parent)), cudaMemcpyDefault));
    p->close();
}

template<size_t warp_size>
__host__ void hashjoin_builder<warp_size>::before_open(){}

template<size_t warp_size>
__host__ void hashjoin_builder<warp_size>::after_close(){}


template class hashjoin_builder<>;
template class hashjoin<>;
