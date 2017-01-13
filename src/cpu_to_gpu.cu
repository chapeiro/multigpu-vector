#include "cpu_to_gpu.cuh"
#include "common.cuh"


template<size_t warp_size, typename T>
cpu_to_gpu::cpu_to_gpu(Operator * parent, dim3 parent_dimGrid, dim3 parent_dimBlock, int device):
        parent(parent), parent_dimGrid(parent_dimGrid), parent_dimBlock(parent_dimBlock), device(device){
    set_device_on_scope d(device);

    gpu(cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking));
}


template<size_t warp_size, typename T>
__host__ __device__ void cpu_to_gpu::open(){}

template<size_t warp_size, typename T>
__host__ __device__ void cpu_to_gpu::consume(buffer_pool<int32_t>::buffer_t * data){

}

    __host__ __device__ void close();

    ~cpu_to_gpu();
};





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

template class cpu_to_gpu<>;