#include "mem_move_local_to.cuh"
#include "../buffer_manager.cuh"
#include "../numa_utils.cuh"

template<typename T>
__host__ mem_move_local_to<T>::mem_move_local_to(h_operator<T> parent, int target_device): parent(parent), target_device(target_device), t(new thread([]{})){
    gpu(cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking));
}

template<typename T>
__host__ void mem_move_local_to<T>::consume(const T * __restrict__ src, cnt_t N, vid_t vid, cid_t cid){
    bool release = false;
    
    if (get_device(src) < 0){

        int dev = get_numa_addressed(src);

        if (dev != target_device){
            set_affinity(cpu_numa_affinity+target_device);

            T * buff = buffer_manager<int32_t>::get_buffer_numa(target_device);


            auto start = chrono::system_clock::now();
            thread t1([buff, src, N](cpu_set_t * aff){
                set_affinity(aff);
                memcpy(buff, src, (N/2)*sizeof(T));
            }, cpu_numa_affinity + target_device);
            memcpy(buff + (N/2), src + (N/2), (N-(N/2))*sizeof(T));
            t1.join();
            auto end   = chrono::system_clock::now();
            cout << "T: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " BW: " << (N*sizeof(T)*1000.0/1024/1024/1024)/chrono::duration_cast<chrono::milliseconds>(end - start).count() << "GBps" << endl;
            // buffer_manager<T>::overwrite(buff, src, N, strm, true);

            src = buff;
            release = true;
        }

    }
    t->join();
    delete t;

    t = new thread([src, N, vid, cid, this, release](){ //what about src ? if no memcpy ? who will free it ?
        parent.consume(src, N, vid, cid);
        if (release) buffer_manager<int32_t>::release_buffer((int32_t *) src);
    });
}

template<typename T>
template<typename Tw>
__host__ void mem_move_local_to<T>::serialize_with(mem_move_local_to<Tw> *t){
    gpu(cudaStreamSynchronize(strm));
    gpu(cudaStreamDestroy(strm));
    strm  = t->strm;
}

template<typename T>
__host__ void mem_move_local_to<T>::close(){
    t->join();
    delete t;
    parent.close();
}

template<typename T>
__host__ void mem_move_local_to<T>::open(){
    parent.open();
}

template class mem_move_local_to<int32_t>;
template __host__ void mem_move_local_to<int32_t>::serialize_with(mem_move_local_to<int32_t> *);
template __host__ void mem_move_local_to<int32_t>::serialize_with(mem_move_local_to<sel_t  > *);



template<int index, typename... Ts>
struct mem_g_move_local_to_tuple {
    void operator() (tuple<Ts...>& t, int target_device, cnt_t N) {
        auto src = get<index>(t);

        if (get_device(src) < 0){

            int dev = get_numa_addressed(src);

            if (dev != target_device){
                set_affinity(cpu_numa_affinity+target_device);

                auto buff = buffer_manager<int32_t>::get_buffer_numa(target_device);


                auto start = chrono::system_clock::now();
                thread t1([buff, src, N](cpu_set_t * aff){
                    set_affinity(aff);
                    memcpy(buff, src, (N/2)*sizeof(*src));
                }, cpu_numa_affinity + target_device);
                memcpy(buff + (N/2), src + (N/2), (N-(N/2))*sizeof(*src));
                t1.join();
                auto end   = chrono::system_clock::now();
                cout << "T: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " BW: " << (N*sizeof(*src)*1000.0/1024/1024/1024)/chrono::duration_cast<chrono::milliseconds>(end - start).count() << "GBps" << endl;
                // buffer_manager<T>::overwrite(buff, src, N, strm, true);

                get<index>(t) = buff;
                // release = true;
            }

        }

        mem_g_move_local_to_tuple<index - 1, Ts...>{}(t, target_device, N);
    }
};

template<typename... Ts>
struct mem_g_move_local_to_tuple<0, Ts...> {
    void operator() (tuple<Ts...>& t, int target_device, cnt_t N) {
        auto src = get<0>(t);

        if (get_device(src) < 0){

            int dev = get_numa_addressed(src);

            if (dev != target_device){
                set_affinity(cpu_numa_affinity+target_device);

                auto buff = buffer_manager<int32_t>::get_buffer_numa(target_device);


                auto start = chrono::system_clock::now();
                thread t1([buff, src, N](cpu_set_t * aff){
                    set_affinity(aff);
                    memcpy(buff, src, (N/2)*sizeof(*src));
                }, cpu_numa_affinity + target_device);
                memcpy(buff + (N/2), src + (N/2), (N-(N/2))*sizeof(*src));
                t1.join();
                auto end   = chrono::system_clock::now();
                cout << "T: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " BW: " << (N*sizeof(*src)*1000.0/1024/1024/1024)/chrono::duration_cast<chrono::milliseconds>(end - start).count() << "GBps" << endl;
                // buffer_manager<T>::overwrite(buff, src, N, strm, true);

                get<0>(t) = buff;
                // release = true;
            }

        }
    }
};

template<typename... Ts>
void mem_g_move_local_to(tuple<Ts...>& t, int target_device, cnt_t N) {
    const auto size = tuple_size<tuple<Ts...>>::value;
    mem_g_move_local_to_tuple<size - 1, Ts...>{}(t, target_device, N);
}


template<typename... T>
__host__ mem_multimove_local_to<T...>::mem_multimove_local_to(h_operator<T...> parent, int target_device): parent(parent), target_device(target_device), t(new thread([]{})){
    gpu(cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking));
}

template<typename... T>
__host__ void mem_multimove_local_to<T...>::consume(const T * __restrict__ ... src, cnt_t N, vid_t vid, cid_t cid){
    // bool release = false;
    
    std::tuple<const T *...> tsrc_old(make_tuple(src...));
    std::tuple<const T *...> tsrc    (make_tuple(src...));
    mem_g_move_local_to(tsrc, target_device, N);

    t->join();
    delete t;

    t = new thread([tsrc, tsrc_old, N, vid, cid, this](){//, release](){ //what about src ? if no memcpy ? who will free it ?
        parent.consume(tsrc, N, vid, cid);
        //FIXME: release every tsrc_old : tsrc_old != tsrc
        // if (release) buffer_manager<int32_t>::release_buffer((int32_t *) src);
    });
}

template<typename... T>
template<typename... Tw>
__host__ void mem_multimove_local_to<T...>::serialize_with(mem_multimove_local_to<Tw...> *t){
    gpu(cudaStreamSynchronize(strm));
    gpu(cudaStreamDestroy(strm));
    strm  = t->strm;
}

template<typename... T>
__host__ void mem_multimove_local_to<T...>::close(){
    t->join();
    delete t;
    parent.close();
}

template<typename... T>
__host__ void mem_multimove_local_to<T...>::open(){
    parent.open();
}

template class mem_multimove_local_to<int32_t, int32_t>;
template __host__ void mem_multimove_local_to<int32_t, int32_t>::serialize_with(mem_multimove_local_to<int32_t, int32_t> *);
template __host__ void mem_multimove_local_to<int32_t, int32_t>::serialize_with(mem_multimove_local_to<sel_t  > *);

template class mem_multimove_local_to<int32_t, int32_t, int32_t, int32_t>;
template __host__ void mem_multimove_local_to<int32_t, int32_t, int32_t, int32_t>::serialize_with(mem_multimove_local_to<int32_t, int32_t, int32_t, int32_t> *);
template __host__ void mem_multimove_local_to<int32_t, int32_t, int32_t, int32_t>::serialize_with(mem_multimove_local_to<sel_t  > *);