#ifndef NUMA_UTILS_CUH_
#define NUMA_UTILS_CUH_

// #include "buffer_manager.cuh"
#include "common/gpu/gpu-common.hpp"
#include <thread>
#include <numaif.h>
#include <numa.h>

// #define NNUMA

#ifndef NNUMA
#define numa_assert(x) assert(x)
#else
#define numa_assert(x) ((void)0)
#endif


void inline set_affinity(cpu_set_t *aff){
#ifndef NNUMA
#ifndef NDEBUG
    int rc =
#endif
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), aff);
    assert(rc == 0);
    // this_thread::yield();
#endif
}

void inline set_affinity_local_to_gpu(int device){
    set_affinity(&gpu_affinity[device]);
}

inline int numa_node_of_gpu(int device){
    return gpu_numa_node[device];
}

inline int calc_numa_node_of_gpu(int device){ // a portable but slow way...
    cpu_set_t cpus = gpu_affinity[device];
    for (int i = 0 ; i < cpu_cnt ; ++i) if (CPU_ISSET(i, &cpus)) return numa_node_of_cpu(i);
    assert(false);
    return -1;
}

template<typename T = char>
T * malloc_host_local_to_gpu(size_t size, int device){
    assert(device >= 0);

    T      *mem = (T *) numa_alloc_onnode(sizeof(T)*size, numa_node_of_gpu(device));
    assert(mem);
    gpu_run(cudaHostRegister(mem, sizeof(T)*size, 0));

    // T * mem;
    // gpu_run(cudaMallocHost(&mem, sizeof(T)*size));

    return mem;
}

inline void * cudaMallocHost_local_to_gpu(size_t size, int device){
    assert(device >= 0);

    void      *mem = numa_alloc_onnode(size, numa_node_of_gpu(device));
    assert(mem);
    gpu_run(cudaHostRegister(mem, size, 0));

    // T * mem;
    // gpu_run(cudaMallocHost(&mem, sizeof(T)*size));

    return mem;
}

inline void * cudaMallocHost_local_to_gpu(size_t size){
    return cudaMallocHost_local_to_gpu(size, get_device());
}

inline void cudaFreeHost_local_to_gpu(void * mem, size_t size){
    gpu_run(cudaHostUnregister(mem));

    numa_free(mem, size);
}


#endif /* NUMA_UTILS_CUH_ */