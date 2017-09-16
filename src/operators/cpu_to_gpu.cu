#include "cpu_to_gpu.cuh"
#include "../numa_utils.cuh"
#include "../buffer_manager.cuh"

using namespace std;

template<typename... T>
__host__ cpu_to_gpu<T...>::cpu_to_gpu(d_operator<T...> *parent, launch_conf conf): parent(parent), conf(conf), old_buffs(NULL){
    assert(conf.device >= 0);

    set_device_on_scope d(conf.device);
    
    cudaStreamCreateWithFlags(&strm_op , cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&strm_mem, cudaStreamNonBlocking);
}

template<typename... T>
template<typename... Tw>
__host__ void cpu_to_gpu<T...>::serialize_with(cpu_to_gpu<Tw...> *t){
    gpu(cudaStreamSynchronize(strm_op ));
    gpu(cudaStreamSynchronize(strm_mem));
    gpu(cudaStreamDestroy(strm_op));
    gpu(cudaStreamDestroy(strm_mem));
    strm_op  = t->strm_op ;
    strm_mem = t->strm_mem;
}

template<typename... T>
__host__ void cpu_to_gpu<T...>::open(){
    set_affinity_local_to_gpu(conf.device);

    parent->open();
}

template<typename... T>
__host__ void cpu_to_gpu<T...>::close(){
    gpu(cudaStreamSynchronize(strm_op ));
    gpu(cudaStreamSynchronize(strm_mem));
    if (old_buffs) buffer_manager<int32_t>::release_buffer((int32_t *) old_buffs); //FIXME: generalize
    parent->close();
}

template<typename... Tin>
__global__ __launch_bounds__(65536, 4) void launch_consume_pipeline(d_operator<Tin...> * op, const Tin * ... src, cnt_t N, vid_t vid, cid_t cid, const Tin * ... old_src){//, buffer_t * prev_buff = NULL){
    // assert(N % (vector_size * get_total_num_of_warps()) == 0);

    if (get_global_warpid() == 0 && get_laneid() == 0) buffer_manager<int32_t>::release_buffer(((int32_t *) old_src)...); //FIXME: generalize

    op->consume_open();

    for (int j = 0 ; j < N ; j += vector_size * get_total_num_of_warps()){
        cnt_t base = j + vector_size * get_global_warpid();
        if (N >= base){
            op->consume_warp((src + base)..., min(vector_size, N - base), vid + j, cid);
        }
    }

    op->consume_close();
}

template<typename... T>
__host__ void cpu_to_gpu<T...>::consume(const T * ... src, cnt_t N, vid_t vid, cid_t cid){
    set_device_on_scope d(conf.device);

    launch_consume_pipeline<T...><<<conf.gridDim, conf.blockDim, conf.shared_mem, strm_op>>>(parent, src..., N, vid, cid, old_buffs);//, prev_buff);
    
    tie(old_buffs) = make_tuple(((T *) src)...); //FIXME : not owner... someone else should release...

    //launch on buffer
#ifndef NQUEUE_WORK
    gpu(cudaStreamSynchronize(strm_op));
#endif
}

template class cpu_to_gpu<int32_t>;
template void  cpu_to_gpu<int32_t>::serialize_with<int32_t>(cpu_to_gpu<int32_t> *);