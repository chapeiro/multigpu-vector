#include "gpu_to_cpu.cuh"
#include "common.cuh"

template<size_t warp_size, size_t size, typename T>
gpu_to_cpu_dev<warp_size, size, T>::gpu_to_cpu_dev(volatile T *store, volatile int *flags, volatile int *eof): 
        lock(0), end(0), store(store), flags(flags), eof(eof){}

template<size_t warp_size, size_t size, typename T>
__host__ __device__ void gpu_to_cpu_dev<warp_size, size, T>::consume(T data){
#ifdef __CUDA_ARCH__
    if (get_laneid() == 0){
        while (atomicCAS((int *) &lock, 0, 1));

        while (*(flags+end) != 0);

        store[end] = data;
        __threadfence_system();
        flags[end] = 1;
        __threadfence_system();

        end = (end + 1) % size;

        assert(lock == 1);
        lock = 0;
    }
#else
    assert(false);
#endif
}

template<size_t warp_size, size_t size, typename T>
__host__ __device__ void gpu_to_cpu_dev<warp_size, size, T>::join(){
#ifdef __CUDA_ARCH__
    *eof = 1;
    __threadfence_system();
#else
    assert(false);
#endif
}

template<size_t warp_size, size_t size, typename T>
gpu_to_cpu<warp_size, size, T>::gpu_to_cpu(Operator * parent, int device): parent(parent), front(0){
    gpu(cudaMallocHost(&store, (sizeof(T)+sizeof(int))*size + sizeof(int)));
    flags      = (volatile int *) (store + size);

    for (int i = 0 ; i < size ; ++i) flags[i] = 0;
    printf("Init %llu %d\n", flags, flags[0]);

    eof        = flags + size;

    *eof       = 0;

    teleporter_thrower = cuda_new<gpu_to_cpu_dev<warp_size, size, T>>(device, store, flags, eof);

    teleporter_catcher = new thread(&gpu_to_cpu<warp_size, size, T>::catcher, this);
}

template<size_t warp_size, size_t size, typename T>
gpu_to_cpu<warp_size, size, T>::~gpu_to_cpu(){
    teleporter_catcher->join();

    delete teleporter_catcher;
    
    cuda_delete(teleporter_thrower);
    gpu(cudaFreeHost((void *) store));
}

template<size_t warp_size, size_t size, typename T>
void gpu_to_cpu<warp_size, size, T>::catcher(){
    while (true){
        while (flags[front] != 1) {
            if (*eof == 1){
                parent->close();
                return;
            }
            this_thread::yield();
        }
        
        parent->consume(store[front]);

        flags[front] = 0;
        front = (front + 1) % size;
    }
}

template class gpu_to_cpu<WARPSIZE, 64, buffer_t *>;
template class gpu_to_cpu_dev<WARPSIZE, 64, buffer_t *>;