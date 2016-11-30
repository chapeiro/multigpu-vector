#include <iostream>
#include <chrono>
#include <algorithm>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include "common.cuh"
#include "select.cuh"
#include <iomanip>
#include <cuda_profiler_api.h>
// #include <functional>

using namespace std;

#ifndef NVAL
#define NVAL (64*1024*1024*4)
#endif

constexpr int N = NVAL;

vector<pair<int32_t *, uint32_t>> data_pool;

mutex data_pool_mutex;

condition_variable cv;
unsigned int remaining_sources = 0;

vector<buffer_pool<int32_t>::buffer_t *> data_pool2;

mutex data_pool_mutex2;

condition_variable cv2;
unsigned int remaining_sources2 = 0;


mutex data_pool_mutex_cons_ready;

condition_variable cv_cons_ready;
unsigned int consumers_not_loaded = 2;//4;

__host__ void consume(unstable_select_gpu<> *filter, buffer_pool<int32_t> *outpool){
    set_device_on_scope(filter->device);
    outpool->register_producer(filter);
    {
        unique_lock<mutex> lock(data_pool_mutex_cons_ready);
        --consumers_not_loaded;
        lock.unlock();
        cv_cons_ready.notify_all();
    }
    auto start = std::chrono::system_clock::now();
    do {
        unique_lock<mutex> lock(data_pool_mutex);

        cout << data_pool.size() << endl;
        cv.wait(lock, []{return !data_pool.empty() || (data_pool.empty() && remaining_sources == 0);});

        if (data_pool.empty()){
            assert(remaining_sources == 0);
            lock.unlock();
            break;
        }

        pair<int32_t *, uint32_t> p = data_pool.back();
        data_pool.pop_back();
        lock.unlock();
        cout << "starting filter instance..." << endl;
        filter->next(p.first, p.second);
        cudaStreamSynchronize(filter->stream);
        cout << "ended filter instance..." << endl;
        // outpool->acquire_buffer_blocked();
        // filter->next(dst, p.first, p.second);
    } while (true);
    filter->next();
    cudaStreamSynchronize(filter->stream);
    outpool->unregister_producer(filter);
    auto end = std::chrono::system_clock::now();
    auto dur = end - start;
    cout << "duration: " << dur.count() << endl;
}


__host__ void generator2(buffer_pool<int32_t> *src, int device = 0){
    {
        unique_lock<mutex> lock(data_pool_mutex_cons_ready);

        cv_cons_ready.wait(lock, []{return consumers_not_loaded == 0;});

        lock.unlock();
    }
    // --remaining_sources2;return;
    set_device_on_scope d(device);
    buffer_pool<int32_t>::buffer_t ** buff_d;
    buffer_pool<int32_t>::buffer_t ** buff_ret;
    gpu(cudaMalloc(&buff_d, sizeof(buffer_pool<int32_t>::buffer_t *)));
    cudaMallocHost(&buff_ret, sizeof(buffer_pool<int32_t>::buffer_t *));
    cudaStream_t strm;
    cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking);

    do {
        cout << "poll " << endl;
        buffer_pool<int32_t>::buffer_t * buff = src->h_acquire_buffer_blocked(buff_d, buff_ret, strm);
        cout << "]]]]]]" << buff << endl;
        if (buff == (buffer_pool<int32_t>::buffer_t *) 1) {
            this_thread::sleep_for(chrono::milliseconds(1));
            continue;
        }
        if (!src->is_valid(buff)) break;

        unique_lock<mutex> lock(data_pool_mutex2);
        data_pool2.emplace_back(buff);
        cv2.notify_all();
        lock.unlock();
    } while(true);
    --remaining_sources2;
    cv2.notify_all();

    cudaStreamDestroy(strm);
    cudaFree(buff_d);
    cudaFreeHost(buff_ret);
}


__host__ void consume2(int32_t *dst, uint32_t *res, int device = 0){
    set_device_on_scope d(device);
    cudaStream_t strm2;
    cudaStreamCreateWithFlags(&strm2, cudaStreamNonBlocking);
    buffpool->register_producer(NULL);
    buffer_pool<int32_t>::buffer_t::inspector_t insp(strm2);
    do {
        unique_lock<mutex> lock(data_pool_mutex2);

        cv2.wait(lock, []{return !data_pool2.empty() || (data_pool2.empty() && remaining_sources2 == 0);});

        if (data_pool2.empty()){
            assert(remaining_sources2 == 0);
            lock.unlock();
            break;
        }

        buffer_pool<int32_t>::buffer_t *p = data_pool2.back(); //FIXME: release buffer back to device
        data_pool2.pop_back();

        insp.load(p, true);

        uint32_t cnt = insp.count();

        const int32_t  *data = (const int32_t *) insp.data();
        uint32_t start = *res;
        *res += cnt;
        lock.unlock();

        cout << data << endl;
        gpu(cudaMemcpyAsync(dst+start, data, sizeof(int32_t)*cnt, cudaMemcpyDefault, strm2));
        cout << "---------------------------------------------------------------consumed: " << dec << (uint64_t) *res << endl;

        buffpool->h_release_buffer(p, strm2);
    } while (true);
    gpu(cudaStreamSynchronize(strm2));
    buffpool->unregister_producer(NULL);
    // *res = filter->next(dst);
    cudaStreamDestroy(strm2);
}

void generate_data(int32_t *src, uint32_t N, uint32_t buff_size){
    for (uint32_t i = 0 ; i < N ; i += buff_size){
        unique_lock<mutex> lock(data_pool_mutex);
        data_pool.emplace_back(src+i, min(N - i, buff_size));
        lock.unlock();
        cv.notify_all();
        // this_thread::sleep_for(chrono::milliseconds(250));
    }
    --remaining_sources;
    cv.notify_all();
}

__host__ uint32_t unstable_select_gpu_caller2(int32_t *src, int32_t *dst, uint32_t N, int32_t pred, const dim3 &dimGrid, const dim3 &dimBlock, cudaEvent_t &start, cudaEvent_t &stop){
    data_pool.clear();

    remaining_sources = 1;
    remaining_sources2 = 1;

    cudaStream_t sta, stb;//, stc, std;
    cudaStreamCreateWithFlags(&sta, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stb, cudaStreamNonBlocking);
    // cudaStreamCreateWithFlags(&stc, cudaStreamNonBlocking);
    // cudaStreamCreateWithFlags(&std, cudaStreamNonBlocking);

    uint32_t res1 = 0;

    thread g(generate_data, src, N, DEFAULT_BUFF_CAP);//WARPSIZE*64);

    buffer_pool<int32_t> *outpool = cuda_new<buffer_pool<int32_t>>(0, 1024, 0, 0);

    unstable_select_gpu<> filter1(dimGrid, dimBlock, outpool, sta);
    unstable_select_gpu<> filter2(dimGrid, dimBlock, outpool, stb);
    // unstable_select_gpu<> filter3(dimGrid, dimBlock, outpool, stc);
    // unstable_select_gpu<> filter4(dimGrid, dimBlock, outpool, stc);

    gpu(cudaDeviceSynchronize());

    cudaProfilerStart();
    cudaEventRecord(start, 0);

    thread t1(consume, &filter1, outpool);
    thread t2(consume, &filter2, outpool);
    // thread t2b(consume, &filter3, outpool);
    // thread t2c(consume, &filter4, outpool);
    thread t3(generator2, outpool, 0);
    thread t4(consume2, dst, &res1, 0);
    t1.join();
    t2.join();
    // t2b.join();
    // t2c.join();
    t3.join();
    t4.join();
    g.join();

    cudaEventRecord(stop, 0);
    cudaProfilerStop();

    cuda_delete(outpool);
    
    return res1;
}

int32_t *a;
int32_t *b;

int main(){
    setbuf(stdout, NULL);
    // gpu(cudaSetDeviceFlags(cudaDeviceScheduleYield));
    srand(time(0));

    buffpool = cuda_new<buffer_pool<int32_t>>(0, 1024, 1024, 0);
    gpu(cudaMemAdvise(&buffpool, sizeof(buffer_pool<int32_t> *), cudaMemAdviseSetReadMostly, 0));
    gpu(cudaDeviceSynchronize());
    cout << "asdasf " << endl;
    
    a = (int32_t*) malloc(N*sizeof(int32_t));
    b = (int32_t*) malloc(N*sizeof(int32_t));

    for (int i = 0 ; i < N ; ++i) a[i] = rand() % 100 + 1;
    
    // char *ad;
    // int *bd;
    // const int csize = N*sizeof(char);
    // const int isize = N*sizeof(int);

    double millis = 0;
#ifndef NCPU
    {
        auto start = chrono::high_resolution_clock::now();
        stable_select_cpu(a, b, N);
        auto end   = chrono::high_resolution_clock::now();
        auto diff = end - start;

        millis = chrono::duration<double, milli>(diff).count();
        cout << millis << " ms " << endl;
    }
#ifndef NCPU
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
    cout << results << endl;
#endif
#endif
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
    // int results = N;
    // for (int i = 0 ; i < N ; ++i) {
    //     if (b[i] == -1) {
    //         results = i;
    //         break;
    //     } else {
    //         assert(b[i] <= 50);
    //         assert(b[i] > 0);
    //     }
    // }
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
    return 0;
    // assert(results1 == results2);
    if (results != results1){
        cout << "Wrong results!!!!!!" << endl;
    } else {
#ifndef __CUDA_ARCH__
        sort(b_pinned, b_pinned + results1);
        sort(b       , b        + results);
        int errors = 0;
        int i1 = 0;
        int i2 = 0;
        while (i1 < results && i2 < results1){
            if (b[i1] != b_pinned[i2]){
                ++errors;
                if (b[i1] < b_pinned[i2]){
                    cout << "Lost  " << b_pinned[i2] << " (vs " << b[i1] << ") @" << i2 << " !!!!!!" << endl;
                    ++i1;
                } else {
                    cout << "Extra " << b_pinned[i2] << " (vs " << b[i1] << ") @" << i2 << " !!!!!!" << endl;
                    ++i2;
                }
            } else {
                ++i1;
                ++i2;
            }
        }
        int missing = (results - i1) + (results - i2);
        errors += missing;
        if (missing) cout << "Missing : " << missing << endl;
        if (errors ) {
            cout << "Total errors: " << errors << endl;
            exit(-1);
        }
#endif
    }
#endif
    gpu(cudaFreeHost(a_pinned));
    gpu(cudaFreeHost(b_pinned));
    
    cudaDeviceSynchronize();
    cudaDeviceReset();

    return EXIT_SUCCESS;
}
