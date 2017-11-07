#include "mem_move.cuh"
#include "../buffer_manager.cuh"
#include "../numa_utils.cuh"


template<typename T>
void make_mem_move(const T * __restrict__ &src, int target_device, cnt_t N, cudaStream_t strm){
    int dev = get_device(src);

    if (dev != target_device){
        set_device_on_scope d(dev);

        if (dev >= 0) set_affinity_local_to_gpu(dev);

        T * buff = buffer_manager<int32_t>::h_get_buffer(target_device);

        buffer_manager<int32_t>::overwrite     (buff, src, N, strm, true);
        // buffer_manager<int32_t>::release_buffer(src                     );

        src = buff;
    }
}

template<typename T>
__host__ mem_move<T>::mem_move(h_operator<T> parent, int target_device): parent(parent), target_device(target_device), t(new thread([]{})){
    gpu(cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking));
}

template<typename T>
__host__ void mem_move<T>::consume(const T * __restrict__ src, cnt_t N, vid_t vid, cid_t cid){
    make_mem_move(src, target_device, N, strm);

    t->join();
    delete t;

    t = new thread([src, N, vid, cid, this](){ //what about src ? if no memcpy ? who will free it ?
        parent.consume(src, N, vid, cid);
    });
}

template<typename T>
template<typename Tw>
__host__ void mem_move<T>::serialize_with(mem_move<Tw> *t){
    gpu(cudaStreamSynchronize(strm));
    gpu(cudaStreamDestroy(strm));
    strm  = t->strm;
}

template<typename T>
__host__ void mem_move<T>::close(){
    t->join();
    delete t;
    parent.close();
}

template<typename T>
__host__ void mem_move<T>::open(){
    parent.open();
}

template class mem_move<int32_t>;
template __host__ void mem_move<int32_t>::serialize_with(mem_move<int32_t> *);
template __host__ void mem_move<int32_t>::serialize_with(mem_move<sel_t  > *);



template<int index, typename... Ts>
struct mem_g_move_tuple {
    void operator() (tuple<Ts...>& t, int target_device, cnt_t N, cudaStream_t strm) {
        make_mem_move(get<index>(t), target_device, N, strm);

        mem_g_move_tuple<index - 1, Ts...>{}(t, target_device, N, strm);
    }
};

template<typename... Ts>
struct mem_g_move_tuple<-1, Ts...> {
    void operator() (tuple<Ts...>& t, int target_device, cnt_t N, cudaStream_t strm) {}
};

template<typename... Ts>
void mem_g_move(tuple<Ts...>& t, int target_device, cnt_t N, cudaStream_t strm) {
    const auto size = tuple_size<tuple<Ts...>>::value;
    mem_g_move_tuple<size - 1, Ts...>{}(t, target_device, N, strm);
}


template<typename... T>
__host__ mem_multimove<T...>::mem_multimove(h_operator<T...> parent, int target_device): parent(parent), target_device(target_device), t(new thread([]{})){
    gpu(cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking));
}



template<typename... T>
__host__ void mem_multimove<T...>::consume(const T * __restrict__ ... src, cnt_t N, vid_t vid, cid_t cid){
    std::tuple<const T *...> tsrc(make_tuple(src...));
    mem_g_move(tsrc, target_device, N, strm);

    CUcontext ctx;
    gpu(cuCtxGetCurrent(&ctx));

    t->join();
    delete t;

    t = new thread([tsrc, N, vid, cid, ctx, this](){ //what about src ? if no memcpy ? who will free it ?
        gpu(cuCtxSetCurrent(ctx));
        parent.consume(tsrc, N, vid, cid);
    });
}

template<typename... T>
template<typename... Tw>
__host__ void mem_multimove<T...>::serialize_with(mem_multimove<Tw...> *t){
    gpu(cudaStreamSynchronize(strm));
    gpu(cudaStreamDestroy(strm));
    strm  = t->strm;
}

template<typename... T>
__host__ void mem_multimove<T...>::close(){
    t->join();
    delete t;
    parent.close();
}

template<typename... T>
__host__ void mem_multimove<T...>::open(){
    parent.open();
}


template class mem_multimove<int32_t, int32_t>;
template __host__ void mem_multimove<int32_t, int32_t>::serialize_with(mem_multimove<int32_t, int32_t> *);

template class mem_multimove<int32_t, int32_t, int32_t, int32_t>;
template __host__ void mem_multimove<int32_t, int32_t, int32_t, int32_t>::serialize_with(mem_multimove<int32_t, int32_t, int32_t, int32_t> *);