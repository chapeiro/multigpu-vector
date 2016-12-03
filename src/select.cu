#include <cstdio>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <cstring>
#include "select.cuh"
#include "common.cuh"
#include "buffer_manager.cuh"

using namespace std;



__device__ __forceinline__ void push_results(volatile int32_t *src, int32_t *dst, uint32_t* elems){
    uint32_t laneid;
    asm("mov.u32 %0, %%laneid;" : "=r"(laneid));

    uint32_t elems_old;
    if (laneid == 0) elems_old = atomicAdd(elems, 4*warpSize);
    elems_old = broadcast(elems_old, 0);

    vec4 tmp_out;
    #pragma unroll
    for (int k = 0 ; k < 4 ; ++k) tmp_out.i[k] = src[k*warpSize + laneid];
    reinterpret_cast<vec4*>(dst)[elems_old/4 + laneid] = tmp_out;
}


template<size_t warp_size, typename T>
__device__ __forceinline__ void unstable_select_gpu_device<warp_size, T>::push_results(volatile int32_t *src, uint32_t* elems) volatile{
    uint32_t laneid;
    asm("mov.u32 %0, %%laneid;" : "=r"(laneid));

    uint32_t elems_old;
    if (laneid == 0) elems_old = atomicAdd(elems, 4*warpSize);
    elems_old = broadcast(elems_old, 0);

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
                outpool->release_buffer(repl); //FIXME: check for overflows
            }
        }
        outbuff = (buffer_t *) output_buffer;
    }
}

template<size_t warp_size, typename T>
__global__ __launch_bounds__(65536, 4) void unstable_select(int32_t *src, unstable_select_gpu_device<warp_size, T> *dst, int N, int32_t pred, int32_t *buffer, uint32_t *output_size, uint32_t *buffer_size, uint32_t *finished){
    extern __shared__ int32_t s[];

    // int32_t *input  = (int32_t *) (s               );
    const int32_t width = blockDim.x * blockDim.y;
    const int32_t gridwidth = gridDim.x * gridDim.y;
    const int32_t bigwidth = width * gridwidth;
    volatile int32_t *output = (int32_t *) (s + 4*width);
    volatile int32_t *fcount = (int32_t *) (s + 9*width+BRDCSTMEM(blockDim));
    uint32_t *elems  = output_size;//(int32_t *) (s + 9*width+BRDCSTMEM(blockDim)+((blockDim.x * blockDim.y)/ WARPSIZE));

    const int32_t i       = threadIdx.x + threadIdx.y * blockDim.x;
    const int32_t blocki  = blockIdx.x  +  blockIdx.y *  gridDim.x;
    const int32_t laneid  = i % warpSize;

// #if __CUDA_ARCH__ < 300 || defined (NUSE_SHFL)
    const int32_t warpid  = i / warpSize;

// #endif
    const int32_t prevwrapmask = (1 << laneid) - 1;

    int32_t filterout = 0;

    volatile int32_t *wrapoutput = output + 5 * warpSize * warpid;

    //read from global memory
    for (int j = 0 ; j < N/4 ; j += bigwidth){
        bool predicate[4] = {false, false, false, false};
        vec4 tmp = reinterpret_cast<vec4*>(src)[i+j+blocki*width];

        #pragma unroll
        for (int k = 0 ; k < 4 ; ++k){
            if (4*(i+j+blocki*width)+k < N){
                // input[blockDim.x*k + i] = tmp.i[k];

                //compute predicate
                predicate[k] = tmp.i[k] <= pred;
            }
        }
        
        #pragma unroll
        for (int k = 0 ; k < 4 ; ++k){
            //aggreagate predicate results
            int32_t filter = __ballot(predicate[k]); //filter now contains set bits only for threads

            int32_t newpop = __popc(filter);

            assert(filterout < 4*warpSize);

            //compute position of result
            if (predicate[k]){
                int32_t offset = filterout + __popc(filter & prevwrapmask);
                assert(offset >= 0);
                assert(offset <  5*warpSize);
                wrapoutput[offset] = tmp.i[k];//input[blockDim.x*k + i];
            }

            filterout += newpop;

            if (filterout >= 4*warpSize){
                dst->push_results(wrapoutput, elems);

                wrapoutput[laneid]             = wrapoutput[laneid + 4*warpSize];
                filterout                     -= 4*warpSize;
            }

            // __syncthreads(); //FIXME: this should not be needed, but racecheck produces errors without it
        }
    }

    if (laneid == 0) fcount[warpid] = filterout;

    __syncthreads(); //this is needed to guarantee that all previous writes to dst are aligned

    for (int32_t m = 1 ; m <= 5 ; ++m){ //fixme: not until 5, but until ceil(log(max warpid)) ? also ceil on target_filter_out condition
        int32_t mask = (1 << m) - 1;

        if (!(warpid & mask)){
            int32_t target_wrapid               = warpid + (1 << (m - 1));
            int32_t target_filter_out           = (target_wrapid < blockDim.x * blockDim.y/warpSize) ? fcount[target_wrapid] : 0;
            int32_t target_filter_out_rem       = target_filter_out;

            volatile int32_t *target_wrapoutput = output + 5 * warpSize * target_wrapid;

            for (int32_t k = 0; k < (target_filter_out + warpSize - 1)/warpSize ; ++k){
                assert(k < 4);
                if (laneid + k * warpSize < target_filter_out) {
                    assert(filterout + laneid < 5*warpSize);
                    wrapoutput[filterout + laneid] = target_wrapoutput[laneid + k * warpSize];
                }
                int32_t delta = min(target_filter_out_rem, warpSize);
                target_filter_out_rem -= delta;
                filterout += delta;

                if (filterout >= 4*warpSize){
                    dst->push_results(wrapoutput, elems);
                    wrapoutput[laneid]             = wrapoutput[laneid + 4*warpSize];
                    filterout                     -= 4*warpSize;
                }
            }

            //no __synchthreads is needed here due to the pattern of accesses on fcount
            if (laneid == 0) fcount[warpid] = filterout;
        }
        __syncthreads();
    }
    if (warpid == 0 && filterout){
        int32_t elems_old;
        if (laneid == 0) elems_old = atomicAdd(buffer_size, filterout);
        elems_old = broadcast(elems_old, 0);

        volatile int32_t * buffoff = buffer  + elems_old;
        volatile int32_t * aligned = (int32_t *) round_up((uintptr_t) buffoff, warpSize * sizeof(int32_t));
        int32_t preamble  = min((int32_t) (aligned - buffoff), filterout);
        int32_t rem_elems = filterout - preamble;

        if (laneid < preamble){
            buffoff[laneid] = wrapoutput[laneid];
        }

        for (int32_t k = laneid; k < rem_elems ; k += warpSize){
            aligned[k] = wrapoutput[preamble + k];
        }

        int32_t * cnts = buffer + ((4*warpSize-1)*(gridwidth+1));

        int32_t bnum0  = elems_old/(4*warpSize);
        int32_t bnum1  = (elems_old + filterout)/(4*warpSize);

        int32_t nset0  = (bnum0 == bnum1) ? filterout : (bnum1 * 4 * warpSize - elems_old);
        int32_t nset1  = filterout - nset0;

        int32_t totcnt0;
        if (laneid == 0) totcnt0 = atomicAdd(cnts + bnum0, nset0);
        totcnt0 = broadcast(totcnt0, 0) + nset0;

        int32_t totcnt1 = -1;
        if (nset1){
            if (laneid == 0) totcnt1 = atomicAdd(cnts + bnum1, nset1);
            totcnt1 = broadcast(totcnt1, 0) + nset1;
        }

        if (totcnt0 >= 4*warpSize){
            assert(totcnt0 <= 4*warpSize);
            dst->push_results(buffer+bnum0*(4*warpSize), elems);
            if (laneid == 0) cnts[bnum0] = 0; //clean up for next round
        }
        if (totcnt1 >= 4*warpSize){
            assert(totcnt1 <= 4*warpSize);
            dst->push_results(buffer+bnum1*(4*warpSize), elems);
            if (laneid == 0) cnts[bnum1] = 0; //clean up for next round
        }
    }
    if (warpid == 0) {
        int32_t * cnts = buffer + ((4*warpSize-1)*(gridwidth+1));

        int32_t finished_old;
        if (laneid == 0) finished_old = atomicAdd(finished, 1);
        finished_old = broadcast(finished_old, 0);

        if (finished_old == gridwidth - 1){ //every other block has finished
            int32_t buffelems = *buffer_size;
            int32_t start     = round_down(buffelems, 4*warpSize);
            int32_t *buffstart= buffer + start;

            // vec4 tmp_out;
            #pragma unroll
            for (int k = 0 ; k < 4 ; ++k) buffer[k*warpSize + laneid] = buffstart[k*warpSize + laneid];
            // reinterpret_cast<vec4*>(buffer)[laneid] = tmp_out;

            if (laneid == 0) {
                *buffer_size                    = buffelems - start;
                *finished                       = 0;
                cnts[buffelems/(4*warpSize)]    = 0;
                cnts[0]                         = buffelems - start;
            }
        }
    }
}

void stable_select_cpu(int32_t *a, int32_t *b, int N){
    int i = 0;
    for (int j = 0 ; j < N ; ++j) if (a[j] <= 50) b[i++] = a[j];
    b[i] = -1;
}

template<size_t warp_size, typename T>
__device__ void unstable_select_gpu_device<warp_size, T>::unstable_select_flush(int32_t *buffer, uint32_t *buffer_size){
    uint32_t laneid;
    asm("mov.u32 %0, %%laneid;" : "=r"(laneid));
    
    buffer_t * outbuff = (buffer_t *) output_buffer;
    uint32_t N = *buffer_size;

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
                printf("releasi----------ng filled buffer %llx\n", repl);
                outpool->release_buffer(repl); //FIXME: check for overflows
            }
        }
        outbuff = (buffer_t *) output_buffer;
    }
    
    __syncthreads();
    if (laneid == 0) outpool->release_buffer(outbuff); //FIXME: check for overflows
}

template<size_t warp_size, typename T>
__global__ void unstable_select_flush(unstable_select_gpu_device<warp_size, T> *dst, int32_t *buffer, uint32_t *buffer_size){
    dst->unstable_select_flush(buffer, buffer_size);
}

template<size_t warp_size, typename T>
__host__ uint32_t unstable_select_gpu<warp_size, T>::next(T *src, uint32_t N){
    set_device_on_scope d(device);
    if (!src){
        // uint32_t h_counters[2];
        // gpu(cudaMemcpyAsync(h_counters, counters, 2 * sizeof(uint32_t), cudaMemcpyDefault, stream));
        // gpu(cudaStreamSynchronize(stream));
        // uint32_t h_output_size = h_counters[0];
        // uint32_t h_buffer_end  = h_counters[1];
        // assert(h_buffer_end < 4*WARPSIZE);

        // // combine results
        // if (h_buffer_end > 0){
        //     cudaMemcpyAsync(dst+h_output_size, buffer, h_buffer_end * sizeof(T), cudaMemcpyDefault, stream);
        //     gpu(cudaStreamSynchronize(stream));
        // }
        // printf("results: %d\n", h_output_size + h_buffer_end);

        unstable_select_flush<<<1, WARPSIZE, shared_mem, stream>>>(dev_data, buffer, counters+1);
        return 0;
    }
    unstable_select<<<dimGrid, dimBlock, shared_mem, stream>>>(src, dev_data, N, 50, buffer, counters, counters+1, counters+2);
// #ifndef NDEBUG
//     gpu(cudaPeekAtLastError()  );
//     gpu(cudaDeviceSynchronize());
// #endif
    return 0;
}


uint32_t unstable_select_gpu_caller(int32_t *src, int32_t *dst, uint32_t N, int32_t pred, const dim3 &dimGrid, const dim3 &dimBlock, cudaEvent_t &start, cudaEvent_t &stop){
    unstable_select_gpu<> filter(dimGrid, dimBlock, 0);
    cudaEventRecord(start, 0);
    // filter.next(dst, src, N/2);
    // filter.next(dst, src+N/2, N/2);
    filter.next(src, N);
    uint32_t res = filter.next();
    // filter.next(dst, src, N);
    // uint32_t res = filter.next(dst);
    cudaEventRecord(stop, 0);
    return res;
}
