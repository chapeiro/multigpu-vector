#ifndef NUMA_UTILS_CUH_
#define NUMA_UTILS_CUH_

// #include "buffer_manager.cuh"
#include "common/gpu/gpu-common.hpp"
#include <thread>

#ifndef NNUMA
#include <numaif.h>
#include <numa.h>
#endif

// #define NNUMA

#ifndef NNUMA
#define numa_assert(x) assert(x)
#else
#define numa_assert(x) ((void)0)

inline constexpr int numa_node_of_cpu(int cpu){return 0;}
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
#ifndef NCUDA
    set_affinity(&gpu_affinity[device]);
#endif
}

inline int numa_node_of_gpu(int device){
#if (!defined NCUDA) || (!defined NNUMA)
    return gpu_numa_node[device];
#else
    return 0;
#endif
}

inline int calc_numa_node_of_gpu(int device){ // a portable but slow way...
#if (!defined NCUDA) || (!defined NNUMA)
    cpu_set_t cpus = gpu_affinity[device];
    for (int i = 0 ; i < cpu_cnt ; ++i) if (CPU_ISSET(i, &cpus)) return numa_node_of_cpu(i);
    assert(false);
    return -1;
#else
    return 0;
#endif
}


inline void * cudaMallocHost_local_to_cpu(size_t size, int device){
    assert(device >= 0);

#ifndef NNUMA
    void      *mem = numa_alloc_onnode(size, device);
#else
    void      *mem = malloc(size);
#endif
    assert(mem);
    gpu_run(cudaHostRegister(mem, size, 0));

    // T * mem;
    // gpu_run(cudaMallocHost(&mem, sizeof(T)*size));

    return mem;
}

template<typename T = char>
T * malloc_host_local_to_gpu(size_t size, int device){
    assert(device >= 0);

    T      *mem = (T *) cudaMallocHost_local_to_cpu(sizeof(T)*size, numa_node_of_gpu(device));
    assert(mem);
    gpu_run(cudaHostRegister(mem, sizeof(T)*size, 0));

    // T * mem;
    // gpu_run(cudaMallocHost(&mem, sizeof(T)*size));

    return mem;
}

inline void * cudaMallocHost_local_to_gpu(size_t size, int device){
    return cudaMallocHost_local_to_cpu(size, numa_node_of_gpu(device));
}

inline void * cudaMallocHost_local_to_gpu(size_t size){
    return cudaMallocHost_local_to_gpu(size, get_device());
}

inline void * cudaMallocHost_local_to_cpu(size_t size){
    return cudaMallocHost_local_to_cpu(size, numa_node_of_cpu(sched_getcpu()));
}


inline void cudaFreeHost_local_to_gpu(void * mem, size_t size){
    gpu_run(cudaHostUnregister(mem));

#ifndef NNUMA
    numa_free(mem, size);
#else
    free(mem);
#endif
}

inline int numa_num_task_nodes(){
    return 1;
}

inline void cudaFreeHost_local_to_cpu(void * mem, size_t size){
    cudaFreeHost_local_to_gpu(mem, size);
}


#endif /* NUMA_UTILS_CUH_ */