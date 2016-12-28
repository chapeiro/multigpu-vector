#ifndef GPU_TO_CPU_CUH_
#define GPU_TO_CPU_CUH_

#include "buffer_pool.cuh"
#include "operator.cuh"
#include "output_composer.cuh"
#include <thread>
#include <atomic>

using namespace std;

template<size_t warp_size = WARPSIZE, size_t size = 64, typename T = int32_t>
class gpu_to_cpu_dev{
private:
    volatile int    lock;
    volatile int    end;

    volatile T     *store;
    volatile int   *flags;
    volatile int   *eof;
public:
    gpu_to_cpu_dev(volatile T *store, volatile int *flags, volatile int *eof);

    __host__ __device__ void consume(T data);

    __host__ __device__ void join();
};

template<size_t warp_size = WARPSIZE, size_t size = 64, typename T = int32_t>
class gpu_to_cpu{
private:
    Operator                               *parent;
    volatile T                             *store;
    volatile int                           *flags;
    volatile int                           *eof;
    size_t                                  front;

    thread                                 *teleporter_catcher;

public:
    gpu_to_cpu_dev<warp_size, size, T>     *teleporter_thrower;

private:
    void catcher();

public:
    gpu_to_cpu(Operator * parent, int device);
    ~gpu_to_cpu();
};


#endif /* GPU_TO_CPU_CUH_ */