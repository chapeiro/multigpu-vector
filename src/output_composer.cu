#include "output_composer.cuh"
#include "buffer_manager.cuh"
#include "common.cuh"

template<size_t warp_size, typename T>
output_composer<warp_size, T>::output_composer(Operator * parent, int dev): parent(parent), elems(0){
    set_device_on_scope d(dev);

    buffer_pool<int32_t>::buffer_t ** buff;
    gpu(cudaMallocHost(&buff, sizeof(buffer_pool<int32_t>::buffer_t *)));
    cudaStream_t strm;
    cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking);

    buffer_manager<int32_t>::h_get_buffer(buff, strm, dev);

    cudaStreamSynchronize(strm);
    output_buffer = *buff;
    cudaStreamDestroy(strm);
    cudaFreeHost(buff);
}

__global__ void launch(Operator *op, buffer_t * buff){
    op->consume(buff);
}

template<size_t warp_size, typename T>
__host__ __device__ void output_composer<warp_size, T>::push(volatile T *src){
#ifndef __CUDA_ARCH__
    assert(false);
#else
    uint32_t laneid;
    asm("mov.u32 %0, %%laneid;" : "=r"(laneid));

    uint32_t elems_old;
    if (laneid == 0) elems_old = atomicAdd(&elems, 4*warpSize);
    elems_old = brdcst(elems_old, 0);

    __threadfence();
    vec4 tmp_out;
    #pragma unroll
    for (int k = 0 ; k < 4 ; ++k) tmp_out.i[k] = src[k*warpSize + laneid];

    buffer_t * outbuff = (buffer_t *) output_buffer;
    while (!outbuff->try_write(tmp_out)){
        if (laneid == 0){
            buffer_t * repl = NULL;
#ifndef NDEBUG
            bool n_endofbuffers = 
#endif
            buffer_manager<int32_t>::acquire_buffer_blocked_to((buffer_t **) &output_buffer, &repl);
            assert(n_endofbuffers);

            if (repl){
                printf("releasing filled buffer %llx\n", repl);
                        cudaStream_t s;
                        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
                        launch<<<1, 1, 0, s>>>(parent, repl);
                        cudaStreamDestroy(s);
                // parent->consume(repl);
                // parent->consume(repl); //FIXME: check for overflows
            }
        }
        outbuff = (buffer_t *) output_buffer;
    }
#endif
}

template<size_t warp_size, typename T>
__host__ __device__ void output_composer<warp_size, T>::push_flush(T *buffer, uint32_t buffer_size){
#ifndef __CUDA_ARCH__
    assert(false);
#else
    const int32_t blocki  = blockIdx.x  +  blockIdx.y *  gridDim.x;
    const int32_t i       = threadIdx.x + threadIdx.y * blockDim.x;
    const int32_t warpid  = i / warpSize;

    bool ellig = (blocki == 0) && (warpid == 0);

    uint32_t laneid;
    asm("mov.u32 %0, %%laneid;" : "=r"(laneid));
    
    buffer_t * outbuff = (buffer_t *) output_buffer;
    if (ellig){
        uint32_t N = buffer_size;

        if (laneid == 0) printf("========================%d %d\n", outbuff->count(), N);
        while (!outbuff->try_partial_final_write(buffer, N)){
            if (laneid == 0){
                buffer_t * repl = NULL;
    #ifndef NDEBUG
                bool n_endofbuffers = 
    #endif
                buffer_manager<int32_t>::acquire_buffer_blocked_to((buffer_t **) &output_buffer, &repl);
                assert(n_endofbuffers);

                if (repl){
                    printf("releasing filled buffer %llx\n", repl);
                    cudaStream_t s;
                    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
                    launch<<<1, 1, 0, s>>>(parent, repl);
                    cudaStreamDestroy(s);
                    // printf("releasi----------ng filled buffer %llx\n", repl);
                    // outpool->release_buffer(repl); //FIXME: check for overflows
                }
            }
            outbuff = (buffer_t *) output_buffer;
        }
    }
    
    __syncthreads();

    if (!ellig) return;

    if (laneid == 0) {
        cudaStream_t s;
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        launch<<<1, 1, 0, s>>>(parent, outbuff);
        cudaStreamDestroy(s);
    }//outpool->release_buffer(outbuff); //FIXME: check for overflows
#endif
}

template class output_composer<>;