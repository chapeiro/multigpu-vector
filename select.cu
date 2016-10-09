#include <cstdio>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cassert>

using namespace std;

const int N = 100000000;

#define gpu(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define WARPSIZE (32)

#if __CUDA_ARCH__ < 300
#define BRDCSTMEM(dimBlock) (dimBlock.x / WARPSIZE)
#else
#define BRDCSTMEM(dimBlock) (0)
#endif

extern __shared__ int32_t s[];

__global__ void unstable_select(int32_t *a_dev, int32_t *b_dev, int N){
    int32_t *input  = (int32_t *) (s               );
    int32_t *output = (int32_t *) (s +   blockDim.x);
#if __CUDA_ARCH__ < 300
    int32_t *bcount = (int32_t *) (s + 3*blockDim.x);
#endif
    int32_t *elems  = (int32_t *) (s + 3*blockDim.x+BRDCSTMEM(blockDim));

    int32_t i       = threadIdx.x;
    int32_t laneid  = i % warpSize;

#if __CUDA_ARCH__ < 300
    int32_t warpid  = i / warpSize;
#endif
    int32_t prevwrapmask = (1 << laneid) - 1;

    int32_t filterout = 0;

    if (i == 0) *elems = 0;
    __syncthreads();
    
    //read from global memory
    for (int j = 0 ; j < N ; j += blockDim.x){
        bool predicate = false;
        if (i+j < N){
            input[i] = a_dev[i+j];

            //compute predicate
            predicate = input[i] < 50;
        }
        
        //aggreagate predicate results
        int32_t filter = __ballot(predicate); //filter now contains set bits only for threads

        int32_t newpop = __popc(filter);

        //compute position of result
        if (predicate){
            int32_t offset = filterout + __popc(filter & prevwrapmask);
            output[(offset % warpSize) + (offset/warpSize)*blockDim.x + warpid * warpSize] = input[i];
        }

        filterout += newpop;

        if (filterout >= warpSize){
            // if (laneid == 0) bcount[warpid] = atomicAdd(elems, warpSize);
            // int32_t elems_old              = bcount[warpid];
#if __CUDA_ARCH__ < 300
            if (laneid == 0) bcount[warpid] = atomicAdd(elems, warpSize);
            int32_t elems_old = bcount[warpid];
#else
            int32_t tmp;
            if (laneid == 0) tmp = atomicAdd(elems, warpSize);
            int32_t elems_old = __shfl(tmp, 0);
#endif
            b_dev[elems_old + laneid]      = output[i];
            output[i]                      = output[i + blockDim.x];
            filterout                     -= warpSize;
        }
    }
    __syncthreads();

#if __CUDA_ARCH__ < 300
    if (laneid == 0) bcount[warpid] = atomicAdd(elems, filterout);
    int32_t elems_old = bcount[warpid];
#else
    int32_t tmp;
    if (laneid == 0) tmp = atomicAdd(elems, filterout);
    int32_t elems_old = __shfl(tmp, 0);
#endif
    if (laneid < filterout) b_dev[elems_old + laneid] = output[i];

    __syncthreads();
    if (i == 0) b_dev[*elems] = -1;
}

int32_t a[N];
int32_t b[N];

void stable_select_cpu(int32_t *a, int32_t *b, int N){
    int i = 0;
    for (int j = 0 ; j < N ; ++j) if (a[j] < 50) b[i++] = a[j];
    b[i] = -1;
}

int main(){
    srand(time(0));

    for (auto &x: a) x = rand() % 100;
    
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
    int32_t *a_pinned, *a_dev;
    int32_t *b_pinned, *b_dev;
    cudaEvent_t start, stop, start1, stop1, start2, stop2;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    dim3 dimBlock( 1024, 1 );
    dim3 dimGrid( 1, 1 );

    cudaMallocHost((void**)&a_pinned, N*sizeof(int32_t));
    cudaMallocHost((void**)&b_pinned, N*sizeof(int32_t));

    memcpy(a_pinned, a, N*sizeof(int32_t));

    gpu(cudaMalloc( (void**)&a_dev, N*sizeof(int32_t)));
    gpu(cudaMalloc( (void**)&b_dev, N*sizeof(int32_t)));

    cudaEventRecord(start1);
    unstable_select<<<dimGrid, dimBlock, (3 * dimBlock.x + BRDCSTMEM(dimBlock) + 1)*sizeof(int32_t)>>>(a_pinned, b_pinned, N);
#ifndef NDEBUG
    gpu( cudaPeekAtLastError() );
    gpu( cudaDeviceSynchronize() );
#endif
    cudaEventRecord(stop1);

    cudaEventRecord(start);
    gpu(cudaMemcpy( a_dev, a_pinned, N*sizeof(int32_t), cudaMemcpyDefault));
    cout << (3 * dimBlock.x + (dimBlock.x / WARPSIZE) + 1)*sizeof(int32_t) << endl;
    cudaEventRecord(start2);
    unstable_select<<<dimGrid, dimBlock, (3 * dimBlock.x + BRDCSTMEM(dimBlock) + 1)*sizeof(int32_t)>>>(a_dev, b_dev, N);
#ifndef NDEBUG
    gpu( cudaPeekAtLastError() );
    gpu( cudaDeviceSynchronize() );
#endif
    cudaEventRecord(stop2);
    gpu(cudaMemcpy(a_pinned, b_dev, N*sizeof(int32_t), cudaMemcpyDefault));
    cudaEventRecord(stop);
    gpu(cudaFree(a_dev));
    gpu(cudaFree(b_dev));

    cudaEventSynchronize(stop);

    int results1 = N;
    for (int i = 0 ; i < N ; ++i) {
        if (b_pinned[i] == -1) {
            results1 = i;
            break;
        } else {
            assert(b_pinned[i] < 50);
        }
    }
    // int results = N;
    // for (int i = 0 ; i < N ; ++i) {
    //     if (a_pinned[i] == -1) {
    //         results = i;
    //         break;
    //     }
    // }
    int results2 = N;
    for (int i = 0 ; i < N ; ++i) {
        if (b[i] == -1) {
            results2 = i;
            break;
        } else {
            assert(b[i] < 50);
        }
    }
    // cout << results << " " << results1 << " " << results2 << " " << a_pinned[4] << endl;

    // assert(results1 == results2);
    if (results1 != results2){
        cout << "Wrong results!!!!!!" << endl;
    }

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
