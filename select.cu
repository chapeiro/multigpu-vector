#include <cstdio>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cassert>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 1*1024*4;

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
__device__ int32_t cnt = 0;

__global__ void unstable_select(int32_t *a_dev, int32_t *b_dev, int N){
    // int32_t *input  = (int32_t *) (s               );
    int32_t width = blockDim.x * blockDim.y;
    int32_t gridwidth = gridDim.x * gridDim.y;
    int32_t bigwidth = width * gridwidth;
    volatile int32_t *output = (int32_t *) (s + 4*width);
#if __CUDA_ARCH__ < 300 || defined (NUSE_SHFL)
    volatile int32_t *bcount = (int32_t *) (s + 9*width);
#endif
    int32_t *elems  = &cnt;//(int32_t *) (s + 9*width+BRDCSTMEM(blockDim));

    int32_t i       = threadIdx.x + threadIdx.y * blockDim.x;
    int32_t blocki  = blockIdx.x  +  blockIdx.y *  gridDim.x;
    int32_t laneid  = i % warpSize;

#if __CUDA_ARCH__ < 300 || defined (NUSE_SHFL)
    int32_t warpid  = i / warpSize;
#endif
    int32_t prevwrapmask = (1 << laneid) - 1;

    int32_t filterout = 0;

    // if (i == 0) *elems = 0;
    __syncthreads();
    
    volatile int32_t *wrapoutput = output + 5 * warpSize * warpid;

    //read from global memory
    for (int j = 0 ; j < N/4 ; j += bigwidth){
        bool predicate[4] = {false, false, false, false};
        vec4 tmp = reinterpret_cast<vec4*>(a_dev)[i+j+blocki*width];

        #pragma unroll
        for (int k = 0 ; k < 4 ; ++k){
            if (4*(i+j+blocki*width)+k < N){
                // input[blockDim.x*k + i] = tmp.i[k];
                assert(tmp.i[k] > 0);
                assert(tmp.i[k] <= 100);
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
                reinterpret_cast<vec4*>(b_dev)[elems_old/4 + laneid] = tmp_out;
                // b_dev[elems_old + laneid]      = output[i];
                wrapoutput[laneid]             = wrapoutput[laneid + 4*warpSize];
                filterout                     -= 4*warpSize;
            }
        }
    }
    __syncthreads(); //this is needed to guarantee that all previous writes to b_dev are aligned (and also vectorizable, aligned to vec4)

#if __CUDA_ARCH__ < 300
    if (laneid == 0) bcount[warpid] = atomicAdd(elems, filterout);
    int32_t elems_old = bcount[warpid];
#else
    int32_t tmp;
    if (laneid == 0) tmp = atomicAdd(elems, filterout);
    int32_t elems_old = __shfl(tmp, 0);
#endif
    int32_t k = 0;
    while (laneid + k * warpSize < filterout) {
        b_dev[elems_old + laneid + k * warpSize] = wrapoutput[k*warpSize + laneid];
        ++k;
    }

    // __syncthreads(); //this is needed to guarantee that all previous writes to b_dev are aligned
    // if (i == 0) b_dev[*elems] = -1;

    if (laneid == 0) printf("%d\n", *elems);
    __syncthreads();
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
    dim3 dimBlock(32, 1 );
    dim3 dimGrid( 2, 1 );

    cudaMallocHost((void**)&a_pinned, N*sizeof(int32_t));
    cudaMallocHost((void**)&b_pinned, N*sizeof(int32_t));

    memcpy(a_pinned, a, N*sizeof(int32_t));

    int32_t zero = 0;
#ifndef NTESTUVA
    gpu(cudaMemcpyToSymbol(cnt, &zero, sizeof(int32_t)));
    cudaEventRecord(start1);
    unstable_select<<<dimGrid, dimBlock, (9 * dimBlock.x * dimBlock.y + BRDCSTMEM(dimBlock) + 1)*sizeof(int32_t)>>>(a_pinned, b_pinned, N);
#ifndef NDEBUG
    gpu( cudaPeekAtLastError() );
    gpu( cudaDeviceSynchronize() );
#endif
    cudaEventRecord(stop1);
#endif

    cudaDeviceSynchronize();
    int32_t t1;
    gpu(cudaMemcpyFromSymbol(&t1, cnt, sizeof(int32_t)));

    cudaDeviceSynchronize();
#ifndef NTESTMEMCPY
    int32_t *a_dev, *b_dev;

    gpu(cudaMalloc( (void**)&a_dev, N*sizeof(int32_t)));
    gpu(cudaMalloc( (void**)&b_dev, N*sizeof(int32_t)));

    cudaEventRecord(start);
    gpu(cudaMemcpyToSymbol(cnt, &zero, sizeof(int32_t)));
    gpu(cudaMemcpy( a_dev, a_pinned, N*sizeof(int32_t), cudaMemcpyDefault));
    cout << (9 * dimBlock.x * dimBlock.y + BRDCSTMEM(dimBlock) + 1)*sizeof(int32_t) << endl;
    cudaEventRecord(start2);
    unstable_select<<<dimGrid, dimBlock, (9 * dimBlock.x * dimBlock.y + BRDCSTMEM(dimBlock) + 1)*sizeof(int32_t)>>>(a_dev, b_dev, N);
#ifndef NDEBUG
    gpu( cudaPeekAtLastError() );
    gpu( cudaDeviceSynchronize() );
#endif
    cudaEventRecord(stop2);
    gpu(cudaMemcpy(a_pinned, b_dev, N*sizeof(int32_t), cudaMemcpyDefault));
    cudaEventRecord(stop);
    gpu(cudaFree(a_dev));
    gpu(cudaFree(b_dev));
#endif

    cudaEventSynchronize(stop);

    cudaDeviceSynchronize();
    int32_t t;
    gpu(cudaMemcpyFromSymbol(&t, cnt, sizeof(int32_t)));
    cudaDeviceSynchronize();
    cout << zero << endl;
#ifndef NDEBUG
    int results2 = N;
    for (int i = 0 ; i < N ; ++i) {
        if (b[i] == -1) {
            results2 = i;
            break;
        } else {
            assert(b[i] <= 50);
            assert(b[i] > 0);
        }
    }
#ifndef NTESTUVA
    int results1 = t1;
    for (int i = 0 ; i < results1 ; ++i) {
        if (b_pinned[i] == -1) {
            results1 = i;
            break;
        } else {
            if (b_pinned[i] <= 0 || b_pinned[i] > 50){
                cout << b_pinned[i] << " " << i << endl;
            }
            // assert(b_pinned[i] <= 50);
            // assert(b_pinned[i] > 0);
        }
    }
#else
    int results1 = results2;
#endif
#ifndef NTESTMEMCPY
    int results = t;
    for (int i = 0 ; i < results ; ++i) {
        if (a_pinned[i] == -1) {
            results = i;
            break;
        } else {
            if (a_pinned[i] <= 0 || a_pinned[i] > 50){
                cout << a_pinned[i] << " " << i << endl;
            }
            // assert(a_pinned[i] <= 50);
            // assert(a_pinned[i] > 0);
        }
    }
#else
    int results = results2;
#endif
    cout << results << " " << results1 << " " << results2 << " " << a_pinned[4] << endl;
    cout << results << " " << t << " " << results2 << " " << a_pinned[4] << endl;

    // assert(results1 == results2);
    if (results1 != results2){
        cout << "Wrong results!!!!!!" << endl;
    } else {
        sort(b_pinned, b_pinned + results1);
        sort(b       , b        + results1);
        for (int i = 0 ; i < results1 ; ++i){
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
