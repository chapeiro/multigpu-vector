#include "mem_move.cuh"
#include "../buffer_manager.cuh"
#include <thread>
#include <numaif.h>

__host__ mem_move::mem_move(h_operator_t * parent): parent(parent){
    gpu(cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking));
}

__host__ void mem_move::consume(buffer_pool<int32_t>::buffer_t * data){
    int dev = get_device(data);

    if (dev >= 0){
        set_device_on_scope d(dev);
#ifndef NDEBUG
        int rc =
#endif
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &gpu_affinity[dev]);
        assert(rc == 0);
        // this_thread::yield();



        buffer_t * buff = buffer_manager<int32_t>::get_buffer();

            int status[1];
            int ret_code;
            status[0]=-1;
            ret_code=move_pages(0 /*self memory */, 1, (void **) &buff->data, NULL, status, 0);
            printf("mem_move:Memory at %p is at %d node (retcode %d) cpu: %d\n", buff->data, status[0], ret_code, sched_getcpu());

        buffer_pool_t::buffer_t::inspector_t from(strm);
        // buffer_pool_t::buffer_t::inspector_t to  (strm);

        from.load(data, true);
        // to.load  (buff, true );

        // to.overwrite(from.data(), from.count(), false);

        // to.save(buff, true);

        buffer_manager<int32_t>::overwrite(buff, from.data(), from.count(), strm, true);


        buffer_manager<int32_t>::release_buffer(data);

        data = buff;
    }
    parent->consume(data);
}

__host__ void mem_move::close(){
    parent->close();
}

__host__ void mem_move::open(){
    parent->open();
}