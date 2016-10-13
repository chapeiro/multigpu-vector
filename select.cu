#include <cstdio>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cassert>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 10000*1024*4;

#define gpu(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

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
__device__ __host__ inline T round_up(T num, T mult){
    T rem = num % mult;
    if (rem == 0) return num;
    return num + mult - rem;
}


__global__ void unstable_select(int32_t *src, int32_t *dst, int N, int32_t *buffer, uint32_t *output_size, uint32_t *buffer_size){
    // int32_t *input  = (int32_t *) (s               );
    const int32_t width = blockDim.x * blockDim.y;
    const int32_t gridwidth = gridDim.x * gridDim.y;
    const int32_t bigwidth = width * gridwidth;
    volatile int32_t *output = (int32_t *) (s + 4*width);
#if __CUDA_ARCH__ < 300 || defined (NUSE_SHFL)
    volatile int32_t *bcount = (int32_t *) (s + 9*width);
#endif
    volatile int32_t *fcount = (int32_t *) (s + 9*width+BRDCSTMEM(blockDim));
    uint32_t *elems  = output_size;//(int32_t *) (s + 9*width+BRDCSTMEM(blockDim)+((blockDim.x * blockDim.y)/ WARPSIZE));

    const int32_t i       = threadIdx.x + threadIdx.y * blockDim.x;
    const int32_t blocki  = blockIdx.x  +  blockIdx.y *  gridDim.x;
    const int32_t laneid  = i % warpSize;

#if __CUDA_ARCH__ < 300 || defined (NUSE_SHFL)
    const int32_t warpid  = i / warpSize;
#endif
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
                predicate[k] = tmp.i[k] <= 50;
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
                // if (laneid == 0) bcount[warpid] = atomicAdd(elems, warpSize);
                // int32_t elems_old              = bcount[warpid];
#if __CUDA_ARCH__ < 300 || defined (NUSE_SHFL)
                if (laneid == 0) bcount[warpid] = atomicAdd(elems, 4*warpSize);
                int32_t elems_old = bcount[warpid];
#else
                int32_t tmp;
                if (laneid == 0) tmp = atomicAdd(elems, 4*warpSize);
                int32_t elems_old = __shfl(tmp, 0);
#endif
                vec4 tmp_out;
                #pragma unroll
                for (int m = 0 ; m < 4 ; ++m) tmp_out.i[m] = wrapoutput[m*warpSize + laneid];
                reinterpret_cast<vec4*>(dst)[elems_old/4 + laneid] = tmp_out;
                // dst[elems_old + laneid]      = output[i];
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
                // if (laneid == 0) bcount[warpid] = atomicAdd(elems, warpSize);
                // int32_t elems_old              = bcount[warpid];
#if __CUDA_ARCH__ < 300 || defined (NUSE_SHFL)
                    if (laneid == 0) bcount[warpid] = atomicAdd(elems, 4*warpSize);
                    int32_t elems_old = bcount[warpid];
#else
                    int32_t tmp;
                    if (laneid == 0) tmp = atomicAdd(elems, 4*warpSize);
                    int32_t elems_old = __shfl(tmp, 0);
#endif
                    vec4 tmp_out;
                    for (int k = 0 ; k < 4 ; ++k) tmp_out.i[k] = wrapoutput[k*warpSize + laneid];
                    reinterpret_cast<vec4*>(dst)[elems_old/4 + laneid] = tmp_out;
                    // dst[elems_old + laneid]      = output[i];
                    wrapoutput[laneid]             = wrapoutput[laneid + 4*warpSize];
                    filterout                     -= 4*warpSize;
                }
            }
            fcount[warpid] = filterout;
        }
        __syncthreads();
    }
    if (warpid == 0){
        // assert(filterout < 4*warpSize);
#if __CUDA_ARCH__ < 300 || defined (NUSE_SHFL)
        if (laneid == 0) bcount[warpid] = atomicAdd(buffer_size, filterout);
        int32_t elems_old = bcount[warpid];
#else
        int32_t tmp;
        if (laneid == 0) tmp = atomicAdd(buffer_size, filterout);
        int32_t elems_old = __shfl(tmp, 0);
#endif
        int32_t * buffoff = buffer  + elems_old;
        int32_t * aligned = (int32_t *) round_up((uintptr_t) buffoff, warpSize * sizeof(int32_t));
        int32_t preamble  = min((int32_t) (aligned - buffoff), filterout);
        int32_t rem_elems = filterout - preamble;

        if (laneid < preamble){
            buffoff[laneid] = wrapoutput[laneid];
        }

        for (int32_t k = laneid; k < rem_elems ; k += warpSize){
            aligned[k] = wrapoutput[preamble + k];
        }

        // // assert(filterout < 4*warpSize);
        // for (int32_t k = 0; k < (filterout + 1 + warpSize - 1) / warpSize ; ++k){ //+1 for writting the final "-1"
        //     b_dev[*elems + laneid + k * warpSize] = ((laneid + k * warpSize < filterout) ? wrapoutput[k*warpSize + laneid] : -1);
        // }
        // *elems += filterout;
        // if (laneid == 0) b_dev[*elems] = -1;

        // The next is wrong
        // vec4 tmp_out;
        // for (int k = 0 ; k < 4 ; ++k) tmp_out.i[k] = wrapoutput[k*warpSize + laneid];
        // reinterpret_cast<vec4*>(b_dev)[*elems/4 + laneid] = tmp_out;
        // *elems += filterout;
        // if (laneid == 0) b_dev[*elems] = -1;
    }
}

constexpr uint32_t zeros[2] = {0, 0};

uint32_t unstable_select_gpu(int32_t *src, int32_t *dst, uint32_t N, const dim3 &dimGrid, const dim3 &dimBlock, cudaEvent_t &start, cudaEvent_t &stop){
    int32_t *buffer;
    gpu(cudaMalloc((void**)&buffer, (dimGrid.x * dimGrid.y) * 4 * WARPSIZE * sizeof(int32_t)));

    uint32_t* counters;
    gpu(cudaMalloc((void**)&counters, 2*sizeof(uint32_t)));

    // initialize global counters
    gpu(cudaMemcpy(counters, (int32_t *) &zeros, 2*sizeof(int32_t), cudaMemcpyDefault));

    cudaEventRecord(start);

    size_t shared_mem = (9 * dimBlock.x * dimBlock.y + BRDCSTMEM(dimBlock) + ((dimBlock.x * dimBlock.y)/ WARPSIZE))*sizeof(int32_t);

    // run kernel
    unstable_select<<<dimGrid, dimBlock, shared_mem>>>(src, dst, N, buffer, counters, counters+1);
    
#ifndef NDEBUG
    gpu(cudaPeekAtLastError()  );
    gpu(cudaDeviceSynchronize());
#endif

    // wait to read counters from device
    uint32_t h_counters[2];
    gpu(cudaMemcpy(h_counters, counters, 2 * sizeof(int32_t), cudaMemcpyDefault));
    uint32_t h_output_size = h_counters[0];
    uint32_t h_buffer_size = h_counters[1];

    // combine results
    cudaMemcpy(dst+h_output_size, buffer, h_buffer_size * sizeof(int32_t), cudaMemcpyDefault);

    cudaEventRecord(stop);
    return h_output_size + h_buffer_size;
}

int32_t *a;
int32_t *b;

void stable_select_cpu(int32_t *a, int32_t *b, int N){
    int i = 0;
    for (int j = 0 ; j < N ; ++j) if (a[j] <= 50) b[i++] = a[j];
    b[i] = -1;
}

int main(){
    srand(time(0));

    a = (int32_t*) malloc(N*sizeof(int32_t));
    b = (int32_t*) malloc(N*sizeof(int32_t));

    for (int i = 0 ; i < N ; ++i) a[i] = rand() % 100 + 1;
    
    // char *ad;
    // int *bd;
    // const int csize = N*sizeof(char);
    // const int isize = N*sizeof(int);

    double millis = 0;
    {
        auto start = chrono::high_resolution_clock::now();
        stable_select_cpu(a, b, N);
        auto end   = chrono::high_resolution_clock::now();
        auto diff = end - start;

        millis = chrono::duration<double, milli>(diff).count();
        cout << millis << " ms" << endl;
    }
    int32_t *a_pinned;
    int32_t *b_pinned;
    cudaEvent_t start, stop, start1, stop1, start2, stop2;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    dim3 dimBlock(1024, 1 );
    dim3 dimGrid( 8, 1 );

    cudaMallocHost((void**)&a_pinned, N*sizeof(int32_t));
    cudaMallocHost((void**)&b_pinned, N*sizeof(int32_t));

    memcpy(a_pinned, a, N*sizeof(int32_t));

#ifndef NTESTUVA
    int results1 = unstable_select_gpu(a_pinned, b_pinned, N, dimGrid, dimBlock, start1, stop1);
#else
    int results1 = 0;
#endif

#ifndef NTESTMEMCPY
    int32_t *a_dev, *b_dev;

    gpu(cudaMalloc( (void**)&a_dev, N*sizeof(int32_t)));
    gpu(cudaMalloc( (void**)&b_dev, N*sizeof(int32_t)));

    cudaEventRecord(start);
    gpu(cudaMemcpy( a_dev, a_pinned, N*sizeof(int32_t), cudaMemcpyDefault));
    
    int results2 = unstable_select_gpu(a_dev, b_dev, N, dimGrid, dimBlock, start2, stop2);

    gpu(cudaMemcpy(a_pinned, b_dev, N*sizeof(int32_t), cudaMemcpyDefault));
    cudaEventRecord(stop);

    gpu(cudaFree(a_dev));
    gpu(cudaFree(b_dev));
#else
    int results2 = 0;
#endif

    cudaEventSynchronize(stop);

    cudaDeviceSynchronize();
#ifndef NDEBUG
    int results = N;
    for (int i = 0 ; i < N ; ++i) {
        if (b[i] == -1) {
            results = i;
            break;
        } else {
            assert(b[i] <= 50);
            assert(b[i] > 0);
        }
    }
#ifndef NTESTUVA
    for (int i = 0 ; i < results1 ; ++i) {
        if (b_pinned[i] <= 0 || b_pinned[i] > 50){
            cout << b_pinned[i] << " " << i << endl;
        }
        // assert(b_pinned[i] <= 50);
        // assert(b_pinned[i] > 0);
    }
#endif
#ifndef NTESTMEMCPY
    for (int i = 0 ; i < results2 ; ++i) {
        if (a_pinned[i] <= 0 || a_pinned[i] > 50){
            cout << a_pinned[i] << " " << i << endl;
        }
        // assert(b_pinned[i] <= 50);
        // assert(b_pinned[i] > 0);
    }
#endif
    cout << results << " " << results1 << " " << results2 << " " << a_pinned[4] << endl;

    // assert(results1 == results2);
    if (results != results1){
        cout << "Wrong results!!!!!!" << endl;
    } else {
        sort(b_pinned, b_pinned + results);
        sort(b       , b        + results);
        for (int i = 0 ; i < results ; ++i){
            if (b[i] != b_pinned[i]){
                cout << "Wrong result: " << b_pinned[i] << " (vs " << b[i] << ") @" << i << " !!!!!!" << endl;
                exit(-1);
            }
        }
    }
#endif
    gpu(cudaFreeHost(a_pinned));
    gpu(cudaFreeHost(b_pinned));

    float milliseconds1 = 0;
    cudaEventElapsedTime(&milliseconds1, start, stop);
    cout << milliseconds1 << endl;
    float milliseconds2 = 0;
    cudaEventElapsedTime(&milliseconds2, start1, stop1);
    cout << milliseconds2 << endl;
    float milliseconds3 = 0;
    cudaEventElapsedTime(&milliseconds3, start2, stop2);
    cout << milliseconds3 << endl;

    cout << endl;
    cout << millis/milliseconds1 << endl;
    cout << millis/milliseconds2 << endl;
    cout << millis/milliseconds3 << endl;
    
    cudaDeviceSynchronize();
    cudaDeviceReset();

    return EXIT_SUCCESS;
}
