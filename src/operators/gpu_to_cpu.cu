#include "gpu_to_cpu.cuh"
#include "../common.cuh"
#include "../buffer_manager.cuh"
// #include <nvToolsExt.h>
#include "../numa_utils.cuh"

template<size_t size, typename... T>
gpu_to_cpu<size, T...>::gpu_to_cpu(h_operator<T...> parent, cid_t ocid, int device): 
            packet_used(0), lock(0), end(0){
    store      = (packet<T...> *) cudaMallocHost_local_to_gpu((sizeof(packet<T...>) + sizeof(int))*size + sizeof(int), device);

    flags      = (volatile int *) (store + size);

    for (int i = 0 ; i < size ; ++i) flags[i] = 0;
    // printf("Init %llu %d\n", flags, flags[0]);

    eof        = flags + size;

    *eof       = 0;

    teleporter_catcher_obj = new gpu_to_cpu_host(parent, store, flags, eof);

    teleporter_catcher = new thread(&gpu_to_cpu_host::catcher, teleporter_catcher_obj, ocid, device);
}

template<size_t size, typename... T>
gpu_to_cpu<size, T...>::gpu_to_cpu_host::gpu_to_cpu_host(h_operator<T...> parent, volatile packet<T...> *store, volatile int *flags, volatile int *eof): 
            parent(parent), front(0), store(store), flags(flags), eof(eof){
}

template<size_t size, typename... T>
__device__ void gpu_to_cpu<size, T...>::throw2cpu(){
    if (get_laneid() == 0){
        while (atomicCAS((int *) &lock, 0, 1));

        while (*(flags+end) != 0);

        store[end] = current_packet;
        __threadfence_system();
        flags[end] = 1;
        __threadfence_system();

        end = (end + 1) % size;

        assert(lock == 1);
        lock = 0;
    }
}

template<size_t size, typename... T>
__device__ void gpu_to_cpu<size, T...>::replace_buffers(){
    if (get_laneid() == 0) packet_handler<sizeof...(T), 0, gpu_to_cpu<size, T...>, T...>::replace(*this);
}

template<size_t size, typename... T>
__device__ void gpu_to_cpu<size, T...>::consume_warp(const T * __restrict__ ... src, cnt_t N, vid_t vid, cid_t cid){
    const int laneid = get_laneid();
    cnt_t consumed = 0;

    while(N > 0){
        
        cnt_t base;
        if (laneid == 0) base = atomicAdd(&packet_used, N);
        base = brdcst(base, 0);

        if (base < h_vector_size){
            cnt_t wcnt = min(N, h_vector_size - base);
            
            packet_handler<sizeof...(T), 0, gpu_to_cpu<size, T...>, T...>::write((src + consumed)..., base, wcnt, *this);

            N         -= wcnt;
            consumed  += wcnt;
        } else {
            if (laneid == 0) atomicMin(&packet_used, h_vector_size);
        }
    }
}



//     const uint32_t laneid = get_laneid();
//     if (N == 4 * warp_size){
//         // output.push(x);
//         vec4 tmp_out;// = reinterpret_cast<const vec4 *>(src)[laneid];
//         #pragma unroll
//         for (int k = 0 ; k < 4 ; ++k) tmp_out.i[k] = src[k*warpSize + laneid];

//         __threadfence();
//         const uint32_t laneid = get_laneid();

//         buffer_t * outbuff = (buffer_t *) output_buffer;
//         while (!outbuff->try_write(tmp_out)){
//             if (laneid == 0){
//                 buffer_t * repl = NULL;
//     #ifndef NDEBUG
//                 bool n_endofbuffers = 
//     #endif
//                 buffer_manager<int32_t>::acquire_buffer_blocked_to((buffer_t **) &output_buffer, &repl);
//                 assert(n_endofbuffers);

//                 if (repl) throw2cpu(repl);
//             }
//             outbuff = (buffer_t *) output_buffer;
//         }
//     } else {
//         // output.push_flush(x, N);
//         buffer_t * outbuff = (buffer_t *) output_buffer;

//         // if (laneid == 0) printf("========================%d %d\n", outbuff->count(), N);
//         while (!outbuff->try_partial_final_write(src, N)){
//             if (laneid == 0){
//                 buffer_t * repl = NULL;
//     #ifndef NDEBUG
//                 bool n_endofbuffers = 
//     #endif
//                 buffer_manager<int32_t>::acquire_buffer_blocked_to((buffer_t **) &output_buffer, &repl);
//                 assert(n_endofbuffers);

//                 if (repl) throw2cpu(repl);
//             }
//             outbuff = (buffer_t *) output_buffer;
//         }
//     }
// }

template<size_t size, typename... T>
__device__ void gpu_to_cpu<size, T...>::at_close(){
    const int32_t blocki  = blockIdx.x  +  blockIdx.y *  gridDim.x;
    const int32_t warpid  = get_warpid();

    bool ellig = (blocki == 0) && (warpid == 0);

    if (!ellig) return;

    const uint32_t laneid = get_laneid();

    if (laneid == 0) throw2cpu();

    *eof = 1;
    __threadfence_system();
}


template<size_t size, typename... T>
__device__ void gpu_to_cpu<size, T...>::consume_open(){
}

template<size_t size, typename... T>
__device__ void gpu_to_cpu<size, T...>::consume_close(){
}

template<size_t size, typename... T>
__device__ void gpu_to_cpu<size, T...>::at_open(){
    if (get_global_warpid() == 0 && get_laneid() == 0) {
        replace_buffers();
        current_packet.vid = 0;
    }
}

template<size_t size, typename... T>
gpu_to_cpu<size, T...>::~gpu_to_cpu(){
    teleporter_catcher->join();

    delete teleporter_catcher;

    cudaFreeHost_local_to_gpu((void *) store, (sizeof(packet<T...>) + sizeof(int))*size + sizeof(int));
}


namespace g2ctuple_expansion{
    template <typename F, typename T, typename Tuple, bool Done, int Total, int... N>
    struct call_impl_obj{
        __host__ __device__ static void call(F f, T * obj, Tuple &&t, cnt_t cnt, vid_t vid, cid_t cid){
            call_impl_obj<F, T, Tuple, Total == 1 + sizeof...(N), Total, N..., sizeof...(N)>::call(f, obj, std::forward<Tuple>(t), cnt, vid, cid);
        }
    };

    template <typename F, typename T, typename Tuple, int Total, int... N>
    struct call_impl_obj<F, T, Tuple, true, Total, N...>{
        __host__ __device__ static void call(F f, T * obj, Tuple && t, cnt_t cnt, vid_t vid, cid_t cid){
            (obj->*f)(get<N>(std::forward<Tuple>(t))..., cnt, vid, cid);
        }
    };
}

template <typename... Ttuple, typename F, typename T, typename Tuple>
__host__ __device__ void call2(F f, T * obj, Tuple && t, cnt_t cnt, vid_t vid, cid_t cid){
    typedef typename std::decay<Tuple>::type ttype;
    g2ctuple_expansion::call_impl_obj<F, T, Tuple, 0 == sizeof...(Ttuple), sizeof...(Ttuple)>::call(f, obj, std::forward<Tuple>(t), cnt, vid, cid);
}

template<size_t size, typename... T>
void gpu_to_cpu<size, T...>::gpu_to_cpu_host::catcher(cid_t ocid, int device){
    set_affinity_local_to_gpu(device);

    while (true){
        while (flags[front] != 1) {
            if (*eof == 1){
                // parent->close();
                return;
            }
            this_thread::yield();
        }
        
        // nvtxMarkA("pop");
        packet<T...> tmp = store[front];

        flags[front] = 0;
        front = (front + 1) % size;

        call2<T...>(static_cast<void (h_operator<T...>::*)(const T * ..., cnt_t, vid_t, cid_t)>(&h_operator<T...>::consume), &parent, tmp.vectors, tmp.N, tmp.vid, ocid);
    }
}


template<size_t size, typename... T>
__host__ void gpu_to_cpu<size, T...>::before_open(){
    decltype(this->teleporter_catcher_obj) p;
    gpu(cudaMemcpy(&p, &(this->teleporter_catcher_obj), sizeof(decltype(this->teleporter_catcher_obj)), cudaMemcpyDefault));
    p->parent.open();
}

template<size_t size, typename... T>
__host__ void gpu_to_cpu<size, T...>::after_close(){
    decltype(this->teleporter_catcher_obj) p;
    gpu(cudaMemcpy(&p, &(this->teleporter_catcher_obj), sizeof(decltype(this->teleporter_catcher_obj)), cudaMemcpyDefault));
    
    thread         *t;
    gpu(cudaMemcpy(&t, &(this->teleporter_catcher)    , sizeof(decltype(this->teleporter_catcher))    , cudaMemcpyDefault));
    t->join();
    p->parent.close();
}

// template class gpu_to_cpu<WARPSIZE, 64, buffer_t *>;
// template class gpu_to_cpu_host<WARPSIZE, 64, buffer_t *>;

template class gpu_to_cpu<64, int32_t>;
template class gpu_to_cpu<64, int32_t, sel_t>;
template class gpu_to_cpu<64, int32_t, vid_t>;
template class gpu_to_cpu<64, int32_t, int32_t>;