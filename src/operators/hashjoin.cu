#include "../common.cuh"
#include "hashjoin.cuh"

__device__ uint32_t hashMurmur(uint32_t x){
    x ^= x >> 16;
    x *= 0x85ebca6b;
    x ^= x >> 13;
    x *= 0xc2b2ae35;
    x ^= x >> 16;
    return x;
}

template<typename T>
__device__ void hashjoin_builder<T>::at_open(){}

template<typename T>
__host__ hashjoin_builder<T>::hashjoin_builder(int32_t log_table_size, volatile int32_t * first, volatile node<T> * next
#ifndef NDEBUG
                                , int32_t max_size
#endif
    ):
        index(0), first(first), next(next), log_table_size(log_table_size)
#ifndef NDEBUG
                                , max_size(max_size)
#endif
        {}

template<typename T>
__device__ void hashjoin_builder<T>::at_close(){
    if (get_global_warpid() == 0 && get_laneid() == 0) printf("%d\n", index);
}

template<typename T>
__device__ void hashjoin_builder<T>::consume_open(){}

template<typename T>
__device__ void hashjoin_builder<T>::consume_close(){}

// template<typename T>
// __device__ void hashjoin_builder<T>::consume_warp(const T * __restrict__ src, cnt_t N, vid_t vid, cid_t cid) __restrict__ {
//     const  int32_t laneid = get_laneid();
//     const uint32_t mask   = ((1 << log_table_size) - 1);

//     // const int32_t * first;
//     // const node<T> * next;

//     vid_t offset;
//     if (laneid == 0) offset = atomicAdd(&index, (vid_t) N);
//     offset = brdcst(offset, 0);

//     for (uint32_t i = 0 ; i < vector_size ; i += 4*warp_size){
//         vec4 s = reinterpret_cast<const vec4 *>(src)[i/4 + laneid];
        
//         #pragma unroll
//         for (uint32_t k = 0 ; k < 4 ; ++k) {
//             const uint32_t ind = i + 4*laneid + k;
//             if (ind < N){
//                 uint32_t bucket = hashMurmur(s.i[k]) & mask;

//                 node<T> tmp;
//                 tmp.next           = atomicExch(first + bucket, offset + ind);
//                 tmp.data           = s.i[k];
//                 tmp.oid            = ind + vid;

//                 next[offset + ind] = tmp;
//                 // assert(offset + ind < max_size);
//             }
//         }
//     }
// }

template<typename T>
__device__ void hashjoin_builder<T>::consume_warp(const T * __restrict__ src, cnt_t N, vid_t vid, cid_t cid) __restrict__ {
    const  int32_t laneid = get_laneid();
    const uint32_t mask   = ((1 << log_table_size) - 1);

    volatile int32_t * first = this->first;
    node<T> * next  = (node<T> *) this->next ;

    vid_t offset;
    if (laneid == 0) offset = atomicAdd(&index, (vid_t) N);
    offset = brdcst(offset, 0);

    for (uint32_t i = 0 ; i < vector_size ; i += warp_size){
        // vec4 s = reinterpret_cast<const vec4 *>(src)[i/4 + laneid];

        const uint32_t ind = i + laneid;
        if (ind < N){
            uint32_t bucket = hashMurmur(src[ind]) & mask;

            node<T> tmp;
            tmp.next           = atomicExch((int32_t *) (first + bucket), offset + ind);
            tmp.data           = src[ind];
            tmp.oid            = ind + vid;

            next[offset + ind] = tmp;
            assert(offset + ind < max_size);
        }
    }
}

template<typename T>
__host__ void hashjoin_builder<T>::before_open(){}

template<typename T>
__host__ void hashjoin_builder<T>::after_close(){}









template<typename Teq, typename Tpayload>
__host__ hashjoin<Teq, Tpayload>::hashjoin(d_operator<int32_t> parent, int32_t log_table_size, int32_t max_size, const launch_conf &conf):
        parent(parent), log_table_size(log_table_size){
    set_device_on_scope d(conf.device);

    gpu(cudaMalloc(&first ,    (1 << log_table_size) * sizeof(  int32_t)));
    gpu(cudaMalloc(&next  ,      ((size_t) max_size) * sizeof(node<Teq>)));

    //carefull, this works because all bytes of -1 are the same
    gpu(cudaMemset((int32_t *) first, -1,  (1 << log_table_size) * sizeof(int32_t)));

    out[0]  = 0;
    out[1]  = 0;
    builder = d_operator<Teq>(conf, cuda_new<hashjoin_builder<Teq>>(conf.device, log_table_size, first, next
#ifndef NDEBUG
                                , max_size
#endif
                                ));
}

template<typename Teq, typename Tpayload>
__host__ hashjoin<Teq, Tpayload>::~hashjoin(){
    gpu(cudaFree((int32_t   *) first ));
    gpu(cudaFree((node<Teq> *) next  ));
}

template<typename Teq, typename Tpayload>
__device__ void hashjoin<Teq, Tpayload>::at_open(){}

template<typename Teq, typename Tpayload>
__device__ void hashjoin<Teq, Tpayload>::at_close(){
    if (get_blockid() == 0){
        parent.consume_open();
        if (get_warpid() == 0) {
            if (get_laneid() == 0) printf("%d %d %d\n", out[0], out[1], out[2]);
            parent.consume_warp(out, 3, 0, 0);
        }
        parent.consume_close();
    }
}

template<typename Teq, typename Tpayload>
__device__ void hashjoin<Teq, Tpayload>::consume_open(){}

template<typename Teq, typename Tpayload>
__device__ void hashjoin<Teq, Tpayload>::consume_close(){}

// template<size_t warp_size>
// __device__ void hashjoin<warp_size>::consume_warp(const int32_t *src, unsigned int N){
//     const int32_t laneid = get_laneid();

//     int32_t r = 0;

//     #pragma unroll
//     for (int k = 0 ; k < 4 ; ++k){
//         const uint32_t ind = k*warp_size + laneid;
//         int32_t current = -1;
//         int32_t probe;
//         uint32_t bucket;
//         if (ind < N) {
//             probe   = src[ind];
//             bucket  = hashMurmur(probe) & ((1 << log_table_size) - 1);

//             current = first[bucket];
//             while (current >= 0){
//                 if (next[(current << 1) + 1] == probe) ++r;//atomicAdd(&res, 1);
//                 // ++r;
//                 current = next[(current << 1) + 0];
//                 // if (values[current] == probe) ++r;
//                 // current = next[current];
//             }
//         }
//     }

//     atomicAdd(&res, r);
// }


template<typename Teq, typename Tpayload>
__host__ d_operator<Teq> hashjoin<Teq, Tpayload>::get_builder(){
    d_operator<Teq> hjbuilder;
    gpu(cudaMemcpy(&hjbuilder, &builder, sizeof(d_operator<Teq>), cudaMemcpyDefault));
    return hjbuilder;
}

template<typename Teq, typename Tpayload>
template<typename Tt, typename>
__device__ void hashjoin<Teq, Tpayload>::consume_warp(const Teq * __restrict__ src, cnt_t N, vid_t vid, cid_t cid) __restrict__{
    const int32_t laneid = get_laneid();
    const uint32_t mask  = ((1 << log_table_size) - 1);

    int32_t r = 0;
    int32_t r2 = 0;

    for (uint32_t i = 0 ; i < vector_size ; i += 4*warp_size){
        vec4 x = reinterpret_cast<const vec4 *>(src)[i/4 + laneid];

        vec4 curr;
        #pragma unroll
        for (int k = 0 ; k < 4 ; ++k) curr.i[k] = (i + 4*laneid + k < N) ? first[hashMurmur(x.i[k]) & mask] : -1;

        while (curr.i[0] >= 0 || curr.i[1] >= 0 || curr.i[2] >= 0 || curr.i[3] >= 0){
            #pragma unroll
            for (int k = 0 ; k < 4 ; ++k){
                if (curr.i[k] >= 0){
                    node<Teq> tmp(next[curr.i[k]]);

                    if (tmp.data == x.i[k]) {
                        r += x.i[k];
                        ++r2;
                    }

                    assert(curr.i[k] != tmp.next);
                    curr.i[k] = tmp.next;
                }
            }
        }
    }
    // r = N;
    // if (laneid == 0) atomicAdd(out, r);

    atomicAdd(out, r);
    if (laneid == 0) atomicAdd(out+1, N);
    atomicAdd(out+2, r2);
    // if (laneid == 0) printf("                                               %d %d %d\n", out[0], out[1], out[2]);

    // #pragma unroll
    // for (int k = 0 ; k < 4 ; ++k){
    //     const uint32_t ind = k*warp_size + laneid;
    //     int32_t current = -1;
    //     int32_t probe;
    //     uint32_t bucket;
    //     if (ind < N) {
    //         probe   = src[ind];
    //         bucket  = hashMurmur(probe) & ((1 << log_table_size) - 1);

    //         current = first[bucket];
    //         while (current >= 0){
    //             if (values[current] == probe) atomicAdd(&res, 1);
    //             current = next[current];
    //         }
    //     }
    // }
}

template<typename Teq, typename Tpayload>
template<typename Tt, typename>
__device__ void hashjoin<Teq, Tpayload>::consume_warp(const Teq * __restrict__ src, const Tpayload * __restrict__ payload, cnt_t N, vid_t vid, cid_t cid) __restrict__{
    static_assert(is_same<Tpayload, int32_t>::value, "not implemented");
    const int32_t laneid = get_laneid();
    const uint32_t mask  = ((1 << log_table_size) - 1);

    int32_t r = 0;
    int32_t r2 = 0;

    for (uint32_t i = 0 ; i < vector_size ; i += 4*warp_size){
        vec4 x = reinterpret_cast<const vec4 *>(src    )[i/4 + laneid];
        vec4 p = reinterpret_cast<const vec4 *>(payload)[i/4 + laneid];

        // if (i == 0 && laneid == 0 && src[0] >= 10000000) printf("%p %d %d %d\n", this, src[0], payload[0], cid);

        vec4 curr;
        #pragma unroll
        for (int k = 0 ; k < 4 ; ++k) curr.i[k] = (i + 4*laneid + k < N) ? first[hashMurmur(x.i[k]) & mask] : -1;

        while (curr.i[0] >= 0 || curr.i[1] >= 0 || curr.i[2] >= 0 || curr.i[3] >= 0){
            #pragma unroll
            for (int k = 0 ; k < 4 ; ++k){
                if (curr.i[k] >= 0){
                    node<Teq> tmp(next[curr.i[k]]);

                    if (tmp.data == x.i[k]) {
                        r += p.i[k];
                        ++r2;
                    }

                    assert(curr.i[k] != tmp.next);
                    curr.i[k] = tmp.next;
                }
            }
        }
    }
    // r = N;
    // if (laneid == 0) atomicAdd(out, r);

    atomicAdd(out, r);
    if (laneid == 0) atomicAdd(out+1, N);
    atomicAdd(out+2, r2);
    // if (laneid == 0) printf("                                               %d %d %d\n", out[0], out[1], out[2]);

    // #pragma unroll
    // for (int k = 0 ; k < 4 ; ++k){
    //     const uint32_t ind = k*warp_size + laneid;
    //     int32_t current = -1;
    //     int32_t probe;
    //     uint32_t bucket;
    //     if (ind < N) {
    //         probe   = src[ind];
    //         bucket  = hashMurmur(probe) & ((1 << log_table_size) - 1);

    //         current = first[bucket];
    //         while (current >= 0){
    //             if (values[current] == probe) atomicAdd(&res, 1);
    //             current = next[current];
    //         }
    //     }
    // }
}

template<typename Teq, typename Tpayload>
__host__ void hashjoin<Teq, Tpayload>::before_open(){
    parent.open();
}

template<typename Teq, typename Tpayload>
__host__ void hashjoin<Teq, Tpayload>::after_close(){
    parent.close();
}


template class hashjoin_builder<int32_t>;
template class hashjoin<int32_t>;
template __device__ void hashjoin<int32_t>::consume_warp<>(const int32_t * __restrict__, cnt_t, vid_t, cid_t) __restrict__;
template class hashjoin<int32_t, int32_t>;
template __device__ void hashjoin<int32_t, int32_t>::consume_warp<>(const int32_t * __restrict__, const int32_t * __restrict__, cnt_t, vid_t, cid_t) __restrict__;
