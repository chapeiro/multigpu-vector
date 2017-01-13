#include "gpu_to_cpu.cuh"
#include "common.cuh"
#include "buffer_manager.cuh"

template<size_t warp_size, size_t size, typename T>
gpu_to_cpu_dev<warp_size, size, T>::gpu_to_cpu_dev(volatile T *store, volatile int *flags, volatile int *eof, int dev): 
        lock(0), end(0), store(store), flags(flags), eof(eof){
    set_device_on_scope d(dev);

    // buffer_pool<int32_t>::buffer_t ** buff;
    // gpu(cudaMallocHost(&buff, sizeof(buffer_pool<int32_t>::buffer_t *)));
    // cudaStream_t strm;
    // cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking);

    output_buffer = buffer_manager<int32_t>::h_get_buffer(dev);
}

template<size_t warp_size, size_t size, typename T>
__host__ __device__ void gpu_to_cpu_dev<warp_size, size, T>::throw2cpu(T data){
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
__host__ __device__ void gpu_to_cpu_dev<warp_size, size, T>::consume_warp(const int32_t *src, unsigned int N){
#ifdef __CUDA_ARCH__
    const uint32_t laneid = get_laneid();
    if (N == 4 * warp_size){
        // output.push(x);
        vec4 tmp_out;
        #pragma unroll
        for (int k = 0 ; k < 4 ; ++k) tmp_out.i[k] = src[k*warpSize + laneid];

        __threadfence();
        const uint32_t laneid = get_laneid();

        buffer_t * outbuff = (buffer_t *) output_buffer;
        while (!outbuff->try_write(tmp_out)){
            if (laneid == 0){
                buffer_t * repl = NULL;
    #ifndef NDEBUG
                bool n_endofbuffers = 
    #endif
                buffer_manager<int32_t>::acquire_buffer_blocked_to((buffer_t **) &output_buffer, &repl);
                assert(n_endofbuffers);

                if (repl) throw2cpu(repl);
            }
            outbuff = (buffer_t *) output_buffer;
        }
    } else {
        // output.push_flush(x, N);
        buffer_t * outbuff = (buffer_t *) output_buffer;

        if (laneid == 0) printf("========================%d %d\n", outbuff->count(), N);
        while (!outbuff->try_partial_final_write(src, N)){
            if (laneid == 0){
                buffer_t * repl = NULL;
    #ifndef NDEBUG
                bool n_endofbuffers = 
    #endif
                buffer_manager<int32_t>::acquire_buffer_blocked_to((buffer_t **) &output_buffer, &repl);
                assert(n_endofbuffers);

                if (repl) throw2cpu(repl);
            }
            outbuff = (buffer_t *) output_buffer;
        }
    }
#else
    assert(false);
#endif
}


template<size_t warp_size, size_t size, typename T>
__host__ __device__ void gpu_to_cpu_dev<warp_size, size, T>::close(){
#ifdef __CUDA_ARCH__
    const int32_t blocki  = blockIdx.x  +  blockIdx.y *  gridDim.x;
    const int32_t warpid  = get_warpid();

    bool ellig = (blocki == 0) && (warpid == 0);

    if (!ellig) return;

    const uint32_t laneid = get_laneid();

    if (laneid == 0) {
        throw2cpu((buffer_t *) output_buffer);
    }

    *eof = 1;
    __threadfence_system();
#else
    assert(false);
#endif
}

template<size_t warp_size, size_t size, typename T>
__host__ __device__ void gpu_to_cpu_dev<warp_size, size, T>::consume_open(){
#ifdef __CUDA_ARCH__
#else
    assert(false);
#endif
}

template<size_t warp_size, size_t size, typename T>
__host__ __device__ void gpu_to_cpu_dev<warp_size, size, T>::consume_close(){
#ifdef __CUDA_ARCH__
#else
    assert(false);
#endif
}

template<size_t warp_size, size_t size, typename T>
__host__ __device__ void gpu_to_cpu_dev<warp_size, size, T>::open(){
#ifdef __CUDA_ARCH__
#else
    assert(false);
#endif
}

template<size_t warp_size, size_t size, typename T>
gpu_to_cpu<warp_size, size, T>::gpu_to_cpu(h_operator_t * parent, int device): parent(parent), front(0){
    gpu(cudaMallocHost(&store, (sizeof(T)+sizeof(int))*size + sizeof(int)));
    flags      = (volatile int *) (store + size);

    for (int i = 0 ; i < size ; ++i) flags[i] = 0;
    // printf("Init %llu %d\n", flags, flags[0]);

    eof        = flags + size;

    *eof       = 0;

    teleporter_thrower = cuda_new<gpu_to_cpu_dev<warp_size, size, T>>(device, store, flags, eof, device);

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
                // parent->close();
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