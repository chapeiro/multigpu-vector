#include <cstdio>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <cstring>
#include "select.cuh"
#include "common.cuh"

using namespace std;

#define WARPSIZE (32)

#if __CUDA_ARCH__ < 300 || defined (NUSE_SHFL)
#define BRDCSTMEM(blockDim) ((blockDim.x * blockDim.y)/ WARPSIZE)
#else
#define BRDCSTMEM(blockDim) (0)
#endif

union vec4{
    int4 vec;
    int  i[4];
};

extern __shared__ int32_t s[];

template<typename T>
__device__ __host__ inline constexpr T round_up(T num, T mult){
    return ((num + mult - 1) / mult) * mult;
}

#if __CUDA_ARCH__ < 300 || defined (NUSE_SHFL)
template<typename T>
__device__ __forceinline__ T broadcast(T val, uint32_t src){
    uint32_t laneid;
    asm("mov.u32 %0, %%laneid;" : "=r"(laneid));

    volatile int32_t *bcount = (int32_t *) (s + 9 * blockDim.x * blockDim.y);
    uint32_t warpid;
    asm("mov.u32 %0, %%warpid;" : "=r"(warpid));

    if (laneid == src) bcount[warpid] = val;
    return bcount[warpid];
}
#else
   #define broadcast(v, l) (__shfl(v, l))
#endif

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

__global__ __launch_bounds__(65536, 4) void unstable_select(int32_t *src, int32_t *dst, int N, int32_t pred, int32_t *buffer, uint32_t *output_size, uint32_t *buffer_size){
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

            //compute position of result
            if (predicate[k]){
                int32_t offset = filterout + __popc(filter & prevwrapmask);
                wrapoutput[offset] = tmp.i[k];//input[blockDim.x*k + i];
            }

            filterout += newpop;

            if (filterout >= 4*warpSize){
                push_results(wrapoutput, dst, elems);

                wrapoutput[laneid]             = wrapoutput[laneid + 4*warpSize];
                filterout                     -= 4*warpSize;
            }
        }
    }
    if (laneid == 0) fcount[warpid] = filterout;
    __syncthreads(); //this is needed to guarantee that all previous writes to dst are aligned

    for (int32_t m = 1 ; m <= 5 ; ++m){
        int32_t mask = (1 << m) - 1;
        if (!(warpid & mask)){
            int32_t target_wrapid               = warpid + (1 << (m - 1));
            int32_t target_filter_out           = fcount[target_wrapid];
            int32_t target_filter_out_rem       = target_filter_out;

            volatile int32_t *target_wrapoutput = output + 5 * warpSize * target_wrapid;

            for (int32_t k = 0; k < (target_filter_out + warpSize - 1)/warpSize ; ++k){
                if (laneid + k * warpSize < target_filter_out) {
                    wrapoutput[filterout + laneid] = target_wrapoutput[laneid + k * warpSize];
                }
                int32_t delta = min(target_filter_out_rem, warpSize);
                target_filter_out_rem -= delta;
                filterout += delta;

                if (filterout >= 4*warpSize){
                    push_results(wrapoutput, dst, elems);

                    wrapoutput[laneid]             = wrapoutput[laneid + 4*warpSize];
                    filterout                     -= 4*warpSize;
                }
            }
            fcount[warpid] = filterout;
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

        int32_t * cnts = buffer + ((4*warpSize-1)*gridwidth);

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

        if (totcnt0 >= 4*warpSize) push_results(buffer+bnum0*(4*warpSize), dst, elems);
        if (totcnt1 >= 4*warpSize) push_results(buffer+bnum1*(4*warpSize), dst, elems);
    }
}

uint32_t unstable_select_gpu(int32_t *src, int32_t *dst, uint32_t N, int32_t pred, const dim3 &dimGrid, const dim3 &dimBlock, cudaEvent_t &start, cudaEvent_t &stop){
    int32_t *buffer;
    int32_t grid_size = dimGrid.x * dimGrid.y * dimGrid.z;

    gpu(cudaMalloc((void**)&buffer, (grid_size * 4 * WARPSIZE + 2)* sizeof(int32_t)));

    uint32_t* counters = (uint32_t *) (buffer + grid_size * 4 * WARPSIZE);
    
    // initialize global counters
    gpu(cudaMemset(buffer + grid_size * 4 * WARPSIZE - grid_size, 0, (grid_size + 2) * sizeof(int32_t)));

    cudaEventRecord(start);

    size_t shared_mem = (9 * dimBlock.x * dimBlock.y + BRDCSTMEM(dimBlock) + ((dimBlock.x * dimBlock.y) / WARPSIZE))*sizeof(int32_t);

    // run kernel
    unstable_select<<<dimGrid, dimBlock, shared_mem>>>(src, dst, N, pred, buffer, counters, counters+1);
    
#ifndef NDEBUG
    gpu(cudaPeekAtLastError()  );
    gpu(cudaDeviceSynchronize());
#endif

    // wait to read counters from device
    uint32_t h_counters[2];
    gpu(cudaMemcpy(h_counters, counters, 2 * sizeof(uint32_t), cudaMemcpyDefault));
    uint32_t h_output_size = h_counters[0];
    uint32_t h_buffer_end  = h_counters[1];
    uint32_t h_buffer_start= (h_counters[1]/(4*WARPSIZE))*(4*WARPSIZE);
    uint32_t h_buffer_size = h_buffer_end - h_buffer_start;
    assert(h_buffer_start % (4*WARPSIZE) == 0);
    assert(h_buffer_end >= h_buffer_start);
    assert(h_buffer_size < 4*WARPSIZE);

    // combine results
    if (h_buffer_size > 0) cudaMemcpy(dst+h_output_size, buffer+h_buffer_start, h_buffer_size * sizeof(int32_t), cudaMemcpyDefault);
    cudaEventRecord(stop);
    return h_output_size + h_buffer_size;
}

void stable_select_cpu(int32_t *a, int32_t *b, int N){
    int i = 0;
    for (int j = 0 ; j < N ; ++j) if (a[j] <= 50) b[i++] = a[j];
    b[i] = -1;
}
