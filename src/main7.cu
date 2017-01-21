#include <iostream>
#include <chrono>
#include <algorithm>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include "common.cuh"
// #include "select.cuh"
#include <iomanip>
#include "buffer_manager.cuh"
#include <cuda_profiler_api.h>
#include <chrono>

#include "operators/exchange.cuh"
#include "operators/materializer.cuh"
#include "operators/generators.cuh"
#include "operators/aggregation.cuh"

#include "operators/gpu_to_cpu.cuh"
// #include <functional>

#include "operators/select3.cuh"
#include "operators/hashjoin.cuh"

// #include <nvToolsExt.h>

using namespace std;

#ifndef NVAL
#define NVAL (64*1024*1024*4)
#endif

constexpr int N = NVAL;

int32_t *a;
int32_t *b;
int32_t *c;

// __global__ void launch_close_pipeline2(d_operator_t * parent){
//     parent->close();
//     // variant::apply_visitor(close{}, *parent);
// }

size_t stable_select_cpu(int32_t *a, int32_t *b, int N){
    size_t i = 0;
    for (size_t j = 0 ; j < N ; ++j) if (a[j] <= 5000) b[i++] = a[j];
    return i;
}

int32_t hist[5001];

size_t sum_select_cpu(int32_t *a, int32_t *b, int N){
    int32_t s = 0;
    for (size_t j = 0 ; j < N ; ++j) if (a[j] <= 5000) s += a[j];
    b[0] = s;
    return 1;
}

size_t sum_selfjoin_select_cpu(int32_t *a, int32_t *b, int N){
    int32_t s = 0;
    for (size_t j = 0 ; j < N ; ++j) if (a[j] <= 5000) ++hist[a[j]];
    for (int i = 1 ; i <= 5000 ; ++i) s += hist[i] * hist[i];
    b[0] = s;
    return 1;
}

constexpr int log_htsize = 20;
int32_t ht[1 << log_htsize];

uint32_t h_hashMurmur(uint32_t x){
    x ^= x >> 16;
    x *= 0x85ebca6b;
    x ^= x >> 13;
    x *= 0xc2b2ae35;
    x ^= x >> 16;
    return x & ((1 << log_htsize) - 1);
}

size_t sum_hashjoin_select_cpu(int32_t *a, int32_t *b, int32_t *probe, int32_t *next, int N, int M){
    int32_t s = 0;
    for (size_t j = 0 ; j < N ; ++j){
        if (a[j] <= 5000) {
            uint32_t bucket = h_hashMurmur(a[j]);
            next[j]    = ht[bucket];
            ht[bucket] = j;
        }
    }
    for (size_t j = 0 ; j < M ; ++j){
        uint32_t bucket  = h_hashMurmur(probe[j]);
        int32_t  current = ht[bucket];
        while (current >= 0){
            if (probe[j] == a[current]) ++s;
            current = next[current];
        }
    }
    b[0] = s;
    return 1;
}

int main(){
    setbuf(stdout, NULL);
    // gpu(cudaSetDeviceFlags(cudaDeviceScheduleYield));
    srand(time(0));
    buffer_manager<int32_t>::init();

    for (int i = 0 ; i < (1 << log_htsize) ; ++i) ht[i] = -1;
    
    // a = (int32_t*) malloc(N*sizeof(int32_t));

    gpu(cudaMallocHost(&a, N*sizeof(int32_t)));

    b = (int32_t*) malloc(N*sizeof(int32_t));
    c = (int32_t*) malloc(N*sizeof(int32_t));
    int32_t *d = (int32_t*) malloc(N*sizeof(int32_t));
    
    // nvtxMarkA("Start");

    for (int i = 0 ; i < N ; ++i) a[i] = rand() % 10000 + 1;

    // nvtxMarkA("End");
    
    // size_t M;
    // {
    //     auto start = chrono::system_clock::now();
    //     // M = sum_selfjoin_select_cpu(a, c, N);
    //     M = 1;//sum_hashjoin_select_cpu(a, c, a, d, N, N);
    //     auto end   = chrono::system_clock::now();
    //     cout << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
    // }

    int32_t *dst;
    gpu(cudaMallocHost(&dst, sizeof(int32_t)*N));

    h_operator_t * omat     = h_operator_t::create<materializer>(dst);


    launch_conf conf0{dim3(1), dim3(1), 0, -1};

    vector<p_operator_t> parents2        = {omat };
    vector<launch_conf > parent_conf2    = {conf0};

    h_operator_t * oprod5   = h_operator_t::create<exchange>(parents2, parent_conf2);

    launch_conf conf1{dim3(16), dim3(1024), 40960, 1};

    d_operator_t * oprod4   = d_operator_t::create<gpu_to_cpu<WARPSIZE, 64, buffer_t *>>(conf1, oprod5, conf1.device);

    d_operator_t * oprod3b  = d_operator_t::create<aggregation<WARPSIZE, 0>>(conf1, oprod4, 8000, conf1.grid_size(), conf1.device);

    d_operator_t * oprod3a  = d_operator_t::create<hashjoin<WARPSIZE>>(conf1, oprod3b, log_htsize, N, conf1);

    d_operator_t * builder;

    hashjoin<WARPSIZE> * h = oprod3a->get<hashjoin<WARPSIZE> *>();

    gpu(cudaMemcpy(&builder, &(h->builder), sizeof(d_operator_t *), cudaMemcpyDefault));

    d_operator_t * oprod3   = d_operator_t::create<unstable_select<>>(conf1, builder, conf1.grid_size(), conf1.device);

    vector<p_operator_t> parents         = {oprod3};
    vector<launch_conf > parent_conf     = {conf1 };

    h_operator_t * oprod    = h_operator_t::create<exchange>(parents, parent_conf);

    generator *gen  = new generator(oprod, a, N);

    vector<p_operator_t> parent_hj      = {oprod3a};

    h_operator_t * oprodhj  = h_operator_t::create<exchange>(parent_hj, parent_conf);

    generator *gen2 = new generator(oprodhj, a, N);

    gpu(cudaProfilerStart());

    // nvtxMarkA("Start");
    auto start = chrono::system_clock::now();
    {
        auto start = chrono::system_clock::now();

        gen->open();
        gen->close();

        auto end   = chrono::system_clock::now();
        cout << "Tm: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
    }
    {
        auto start = chrono::system_clock::now();

        gen2->open();
        gen2->close();

        auto end   = chrono::system_clock::now();
        cout << "Tm: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
    }
    auto end   = chrono::system_clock::now();
    // nvtxMarkA("End");

    gpu(cudaProfilerStop());

    cout << "Total: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;

    const materializer * mat = omat->get<materializer *>();
    vector<int32_t> results(mat->dst, mat->dst + mat->size);

    cout << results.size() << endl;

    size_t M;
    {
        auto start = chrono::system_clock::now();
        // M = stable_select_cpu(a, c, N);
        // M = sum_select_cpu(a, c, N);
        // M = sum_selfjoin_select_cpu(a, c, N);
        M = sum_hashjoin_select_cpu(a, c, a, d, N, N);
        auto end   = chrono::system_clock::now();
        cout << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
    }
    
    if (results.size() != M){
        cout << "Wrong output size!" << results.size() << " vs " << M << endl;
        // assert(false);
        return -1;
        // return 0;
    }
    cout << results[0] << ((results[0] == c[0]) ? " = " : " ! ") << c[0] << endl;
    // return 0;
#ifndef __CUDA_ARCH__
    sort(c, c + M);
    sort(results.begin(), results.end());

    bool failed = false;
    for (int i = 0 ; i < M ; ++i){
        if (c[i] != results[i]){
            cout << "Wrong result " << results[i] << " vs " << c[i] << " at (sorted) " << i << endl;
            failed = true;
        }
    }
    // assert(!failed);
    if (failed) return -1;
    cout << "End of Test" << endl;
#endif

    return 0;
}