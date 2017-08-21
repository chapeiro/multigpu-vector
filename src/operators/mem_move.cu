#include "mem_move.cuh"
#include "../buffer_manager.cuh"
#include "../numa_utils.cuh"

template<typename T>
__host__ mem_move<T>::mem_move(h_operator<T> parent, int target_device): parent(parent), target_device(target_device), t(new thread([]{})){
    gpu(cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking));
}



template<typename T>
__host__ void mem_move<T>::consume(const T * __restrict__ src, cnt_t N, vid_t vid, cid_t cid){
    int dev = get_device(src);

    if (dev != target_device){
        set_device_on_scope d(dev);

        if (dev >= 0) set_affinity_local_to_gpu(dev);

        T * buff = buffer_manager<int32_t>::h_get_buffer(target_device);

        buffer_manager<T>::overwrite(buff, src, N, strm, true);

        src = buff;
    }

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