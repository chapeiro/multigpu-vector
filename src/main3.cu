#include <iostream>
#include <chrono>
#include <algorithm>
#include "common.cuh"
#include "select.cuh"
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
// #include <functional>

using namespace std;

#ifndef NVAL
#define NVAL (10000*1024*4)
#endif

constexpr int N = NVAL;

vector<pair<int32_t *, uint32_t>> data_pool;

mutex data_pool_mutex;

condition_variable cv;
unsigned int remaining_sources = 0;

__host__ void consume(unstable_select_gpu<> *filter, int32_t *dst, uint32_t *res){
    set_device_on_scope(filter->device);
    do {
        unique_lock<mutex> lock(data_pool_mutex);

        cv.wait(lock, []{return !data_pool.empty() || (data_pool.empty() && remaining_sources == 0);});

        if (data_pool.empty()){
            assert(remaining_sources == 0);
            lock.unlock();
            break;
        }

        pair<int32_t *, uint32_t> p = data_pool.back();
        data_pool.pop_back();
        lock.unlock();
        filter->next(dst, p.first, p.second);
        cudaStreamSynchronize(filter->stream);
        filter->next(dst, p.first, p.second);
    } while (false);
    // *res = filter->next(dst);
}


void generate_data(int32_t *src, uint32_t N, uint32_t buff_size){
    for (uint32_t i = 0 ; i < N ; i += buff_size){
        unique_lock<mutex> lock(data_pool_mutex);
        data_pool.emplace_back(src+i, min(N - i, buff_size));
        cv.notify_one();
        lock.unlock();
        this_thread::sleep_for(chrono::milliseconds(250));
    }
    --remaining_sources;
    cv.notify_all();
}

__host__ uint32_t unstable_select_gpu_caller2(int32_t *src, int32_t *dst, uint32_t N, int32_t pred, const dim3 &dimGrid, const dim3 &dimBlock, cudaEvent_t &start, cudaEvent_t &stop){
    int32_t *dst2;
    cudaMalloc((void**)&dst2, N*sizeof(int32_t));
    data_pool.clear();
    remaining_sources = 1;

    cudaStream_t sta, stb;
    cudaStreamCreateWithFlags(&sta, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stb, cudaStreamNonBlocking);

    uint32_t res1 = 0;
    uint32_t res2 = 0;

    thread g(generate_data, src, N, N/2);//WARPSIZE*64);
        this_thread::sleep_for(chrono::seconds(1));

    unstable_select_gpu<> filter1(dimGrid, dimBlock, sta);
    unstable_select_gpu<> filter2(dimGrid, dimBlock, stb);

    cudaEventRecord(start, 0);

    thread t1(consume, &filter1, dst , &res1);
    thread t2(consume, &filter2, dst2, &res2);
    t1.join();
    t2.join();
    g.join();
    
    gpu(cudaMemcpy(dst+res1, dst2, res2*sizeof(int32_t), cudaMemcpyDefault));
    cudaEventRecord(stop, 0);
    gpu(cudaFree(dst2));
    
    return res1 + res2;
}

int32_t *a;
int32_t *b;

int main(){
    buffpool = new buffer_pool<int32_t>(1024, 0);
    gpu(cudaSetDeviceFlags(cudaDeviceScheduleYield));
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

#ifndef NTESTUVA
    int results1 = unstable_select_gpu_caller2(a_pinned, b_pinned, N, 50, dimGrid, dimBlock, start1, stop1);
#else
    int results1 = 0;
#endif

#ifndef NTESTMEMCPY
    int32_t *a_dev, *b_dev;

    gpu(cudaMalloc( (void**)&a_dev, N*sizeof(int32_t)));
    gpu(cudaMalloc( (void**)&b_dev, N*sizeof(int32_t)));

    cudaEventRecord(start);
    gpu(cudaMemcpy( a_dev, a_pinned, N*sizeof(int32_t), cudaMemcpyDefault));
    
    int results2 = unstable_select_gpu_caller2(a_dev, b_dev, N, 50, dimGrid, dimBlock, start2, stop2);

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
