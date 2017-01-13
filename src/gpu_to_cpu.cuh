#ifndef GPU_TO_CPU_CUH_
#define GPU_TO_CPU_CUH_

#include "buffer_pool.cuh"
#include "operator.cuh"
#include <thread>
#include <atomic>

using namespace std;

template<size_t warp_size = WARPSIZE, size_t size = 64, typename T = int32_t>
class gpu_to_cpu_host{
private:
    h_operator_t                           *parent;
    volatile T                             *store;
    volatile int                           *flags;
    volatile int                           *eof;
    size_t                                  front;

private:
public:
    void catcher();

public:
    gpu_to_cpu_host(h_operator_t *parent, volatile T *store, volatile int *flags, volatile int *eof);

    ~gpu_to_cpu_host();
};

template<size_t warp_size = WARPSIZE, size_t size = 64, typename T = int32_t>
class gpu_to_cpu{
private:
    volatile int    lock;
    volatile int    end;

    volatile T     *store;
    volatile int   *flags;
    volatile int   *eof;
    thread         *teleporter_catcher;
    gpu_to_cpu_host<warp_size, size, T> *teleporter_catcher_obj;
    volatile buffer_t * volatile    output_buffer;

    __device__ void throw2cpu(T data);

public:
    __host__ gpu_to_cpu(h_operator_t * parent, int device);

    __device__ void open();

    __device__ void consume_open();
    __device__ void consume_warp(const int32_t *x, unsigned int N);
    __device__ void consume_close();

    __device__ void close();

    __host__ ~gpu_to_cpu();
};


#endif /* GPU_TO_CPU_CUH_ */