#include <iostream>
#include <chrono>
#include <algorithm>
#include "common.cuh"
#include "select.cuh"

using namespace std;

#ifndef NVAL
#define NVAL (1024*64*1024*4)
#endif

constexpr int N = NVAL;

uint32_t unstable_select_gpu2(int32_t *src, int32_t *dst, uint32_t N, int32_t pred, const dim3 &dimGrid, const dim3 &dimBlock, cudaEvent_t &start, cudaEvent_t &stop){
    int32_t *buffer;
    int32_t grid_size = dimGrid.x * dimGrid.y * dimGrid.z;

    gpu(cudaSetDevice(0));
    gpu(cudaMalloc((void**)&buffer, (grid_size * 4 * WARPSIZE + 2)* sizeof(int32_t)));

    uint32_t* counters = (uint32_t *) (buffer + grid_size * 4 * WARPSIZE);
    
    // initialize global counters
    gpu(cudaMemset(buffer + grid_size * 4 * WARPSIZE - grid_size, 0, (grid_size + 2) * sizeof(int32_t)));

    int32_t *buffer2;
    int32_t *dst2;

    gpu(cudaSetDevice(1));
    cudaMalloc((void**)&dst2, N/2*sizeof(int32_t));

    gpu(cudaMalloc((void**)&buffer2, (grid_size * 4 * WARPSIZE + 2)* sizeof(int32_t)));

    uint32_t* counters2 = (uint32_t *) (buffer2 + grid_size * 4 * WARPSIZE);

    // initialize global counters
    gpu(cudaMemset(buffer2 + grid_size * 4 * WARPSIZE - grid_size, 0, (grid_size + 2) * sizeof(int32_t)));

    gpu(cudaSetDevice(0));
    cudaEventRecord(start);

    size_t shared_mem = (9 * dimBlock.x * dimBlock.y + BRDCSTMEM(dimBlock) + ((dimBlock.x * dimBlock.y) / WARPSIZE))*sizeof(int32_t);

    // run kernel
    unstable_select<<<dimGrid, dimBlock, shared_mem>>>(src, dst, N/2, pred, buffer, counters, counters+1);

    gpu(cudaSetDevice(1));

    // run kernel
    cudaStreamWaitEvent(NULL, start, 0); //only for correctly counting the time for both kernels
    unstable_select<<<dimGrid, dimBlock, shared_mem>>>(src+N/2, dst2, N/2, pred, buffer2, counters2, counters2+1);
    
    gpu(cudaSetDevice(0));
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

    gpu(cudaSetDevice(1));
#ifndef NDEBUG
    gpu(cudaPeekAtLastError()  );
    gpu(cudaDeviceSynchronize());
#endif

    gpu(cudaMemcpy(h_counters, counters2, 2 * sizeof(uint32_t), cudaMemcpyDefault));
    uint32_t h_output_size2 = h_counters[0];
    uint32_t h_buffer_end2  = h_counters[1];
    uint32_t h_buffer_start2= (h_counters[1]/(4*WARPSIZE))*(4*WARPSIZE);
    uint32_t h_buffer_size2 = h_buffer_end2 - h_buffer_start2;
    assert(h_buffer_start2 % (4*WARPSIZE) == 0);
    assert(h_buffer_end2 >= h_buffer_start2);
    assert(h_buffer_size2 < 4*WARPSIZE);

    // combine results
    if (h_buffer_size2 > 0) cudaMemcpy(dst+h_output_size+h_buffer_size+h_output_size2, buffer2+h_buffer_start2, h_buffer_size2 * sizeof(int32_t), cudaMemcpyDefault);
    

    gpu(cudaSetDevice(1));
    gpu(cudaMemcpy(dst+h_output_size+h_buffer_size, dst2, h_output_size2*sizeof(int32_t), cudaMemcpyDefault));
    
    gpu(cudaSetDevice(0));
    cudaEventRecord(stop);
    
    gpu(cudaSetDevice(0));
    gpu(cudaFree(dst2));
    return h_output_size+h_buffer_size+h_output_size2+h_buffer_size2;
}


int32_t *a;
int32_t *b;

int main(){
    gpu(cudaSetDevice(1));
    gpu(cudaFree(0)); //initialize devices on demand
    gpu(cudaSetDevice(0));
    gpu(cudaFree(0)); //initialize devices on demand
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
    int results1;
    {
    auto ts = chrono::high_resolution_clock::now();
     results1 = unstable_select_gpu2(a_pinned, b_pinned, N, 50, dimGrid, dimBlock, start1, stop1);
    auto te   = chrono::high_resolution_clock::now();
    auto diff = te - ts;
    auto millis = chrono::duration<double, milli>(diff).count();
    cout << millis << " ms" << endl;
    }
#else
    int results1 = 0;
#endif

#ifndef NTESTMEMCPY
    int32_t *a_dev, *b_dev;

    gpu(cudaMalloc( (void**)&a_dev, N*sizeof(int32_t)));
    gpu(cudaMalloc( (void**)&b_dev, N*sizeof(int32_t)));

    cudaEventRecord(start);
    gpu(cudaMemcpy( a_dev, a_pinned, N*sizeof(int32_t), cudaMemcpyDefault));
    
    int results2 = unstable_select_gpu(a_dev, b_dev, N, 50, dimGrid, dimBlock, start2, stop2);

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
