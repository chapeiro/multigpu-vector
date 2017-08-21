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
#include "operators/split.cuh"
#include "operators/union_all.cuh"
#include "operators/union_all_cpu.cuh"
#include "operators/mem_move.cuh"

#include "argument_parsing.cuh"
// #include <nvToolsExt.h>

#include "operators/map.cuh"
#include "operators/aggr.cuh"

using namespace std;

params_t par;

int32_t *a;
int32_t *s;
int32_t *b;
int32_t *c;

__global__ __launch_bounds__(65536, 4) void launch_consume_pipeline(const int32_t * __restrict__ a, const int32_t * __restrict__ b, uint32_t N, map<aggr<int32_t, int32_t>, product, int32_t, int32_t, int32_t> * m){
    assert(N % (vector_size * get_total_num_of_warps()) == 0);

    for (int j = 0 ; j < N ; j += vector_size * get_total_num_of_warps()){
        m->consume_warp(a + j + vector_size * get_global_warpid(), b + j + vector_size * get_global_warpid(), j / vector_size, 0);
    }
}

template<typename T>
__global__ __launch_bounds__(65536, 4) void launch_close_pipeline(T * op){
    op->at_close();
}

int main(int argc, char *argv[]){
    int tmp = parse_args(argc, argv, par);
    if (tmp) return tmp;
    cout << par << endl;
    setbuf(stdout, NULL);
    // gpu(cudaSetDeviceFlags(cudaDeviceScheduleYield));

    // for (auto &t: thrds) t.join();
    srand(time(0));

    int32_t *a;
    int32_t *b;

    int32_t *d_a;
    int32_t *d_b;

    gpu(cudaMallocHost(&a, par.N*sizeof(int32_t)));
    gpu(cudaMallocHost(&b, par.N*sizeof(int32_t)));
    gpu(cudaMalloc(&d_a, par.N*sizeof(int32_t)));
    gpu(cudaMalloc(&d_b, par.N*sizeof(int32_t)));

    std::random_device r;

    std::mt19937 e1(r());
    {
        auto start = chrono::system_clock::now();

        for (size_t i = 0 ; i < par.N ; ++i) a[i] = e1() % 10000 + 1;//min(par.M, par.N) + 1;
        for (size_t i = 0 ; i < par.N ; ++i) b[i] = e1() % 10000 + 1;//min(par.M, par.N) + 1;

        auto end   = chrono::system_clock::now();
        cout << "Trand: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
    }

    {
        auto start = chrono::system_clock::now();
        int32_t s2 = 0;
        for (size_t i = 0 ; i < par.N ; ++i) s2 += a[i] * b[i];
        cout << "Aggr : " << s2 << endl;
        auto end   = chrono::system_clock::now();
        cout << "T: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
    }
    
    gpu(cudaMemcpy(d_a, a, par.N*sizeof(int32_t), cudaMemcpyDefault));
    gpu(cudaMemcpy(d_b, b, par.N*sizeof(int32_t), cudaMemcpyDefault));

    launch_conf conf{dim3(32), dim3(1024), 0, 0};

    aggr<int32_t, int32_t> * s = cuda_new<aggr<int32_t, int32_t>>(conf.device, (d_operator_t *) NULL, conf);

    map<aggr<int32_t, int32_t>, product, int32_t, int32_t, int32_t> * prod = cuda_new<map<aggr<int32_t, int32_t>, product, int32_t, int32_t, int32_t>>(conf.device, s, product{}, conf);

    {
        auto start = chrono::system_clock::now();
        launch_consume_pipeline<<<conf.gridDim, conf.blockDim, conf.shared_mem, 0>>>(d_a, d_b, par.N, prod);
        gpu(cudaDeviceSynchronize());
        auto end   = chrono::system_clock::now();
        cout << "T: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
    }

    launch_close_pipeline<<<conf.gridDim, conf.blockDim, conf.shared_mem, 0>>>(s);
    gpu(cudaDeviceSynchronize());

    return 0;
}