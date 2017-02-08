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

#include "exchange.cuh"
#include "materializer.cuh"
#include "generators.cuh"
#include "aggregation.cuh"

#include "gpu_to_cpu.cuh"
// #include <functional>

#include "select3.cuh"

// #include <nvToolsExt.h>

using namespace std;

#ifndef NVAL
#define NVAL (64*1024*1024*4)
#endif

constexpr int N = NVAL;

int32_t *a;
int32_t *b;
int32_t *c;

__global__ void launch_close_pipeline2(d_operator_t * parent){
    parent->close();
    // variant::apply_visitor(close{}, *parent);
}

size_t stable_select_cpu(int32_t *a, int32_t *b, int N){
    size_t i = 0;
    for (size_t j = 0 ; j < N ; ++j) if (a[j] <= 50) b[i++] = a[j];
    return i;
}

size_t sum_select_cpu(int32_t *a, int32_t *b, int N){
    int32_t s = 0;
    for (size_t j = 0 ; j < N ; ++j) if (a[j] <= 50) s += a[j];
    b[0] = s;
    return 1;
}

int main(){
    setbuf(stdout, NULL);
    // gpu(cudaSetDeviceFlags(cudaDeviceScheduleYield));
    srand(time(0));
    buffer_manager<int32_t>::init();
    
    // a = (int32_t*) malloc(N*sizeof(int32_t));

    gpu(cudaMallocHost(&a, N*sizeof(int32_t)));

    b = (int32_t*) malloc(N*sizeof(int32_t));
    c = (int32_t*) malloc(N*sizeof(int32_t));
    
    // nvtxMarkA("Start");

    for (int i = 0 ; i < N ; ++i) a[i] = rand() % 100 + 1;

    // nvtxMarkA("End");
    
    size_t M;
    {
        auto start = chrono::system_clock::now();
        M = sum_select_cpu(a, c, N);
        auto end   = chrono::system_clock::now();
        cout << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
    }

    int32_t *dst;
    gpu(cudaMallocHost(&dst, sizeof(int32_t)*N));

    h_operator_t * omat = h_operator_t::create<materializer>(dst);
    
    // vector<int>         prod_loc2        = {1};
    // vector<int>         prodout_loc2     = {1};
    // vector<int>         prodout_size2    = {1024};
    // vector<int>         prod2out2        = {0};
    // vector<Operator *>  parents2         = {omat};
    // vector<dim3>        parent_dimGrid2  = {dim3(1)};
    // vector<dim3>        parent_dimBlock2 = {dim3(1)};

    // exchange *exc2 = new exchange(prod_loc2, 
    //                                 prodout_loc2, 
    //                                 prodout_size2,
    //                                 prod2out2,
    //                                 parents2,
    //                                 parent_dimGrid2,
    //                                 parent_dimBlock2,
    //                                 {0});

    // Operator * oprod2 = cuda_new<Operator>(1, exc2->prods[0]);

    // dim3 gridsel(8);

    // unstable_select<> *s = cuda_new<unstable_select<>>(1, oprod2, gridsel.x*gridsel.y*gridsel.z, 1);
    // Operator * oprod3 = cuda_new<Operator>(1, s);

    vector<int>         prod_loc2        = {-1};
    vector<int>         prodout_loc2     = {-1};
    vector<int>         prodout_size2    = {1024};
    vector<int>         prod2out2        = {0};
    vector<p_operator_t> parents2        = {omat};
    vector<dim3>        parent_dimGrid2  = {dim3(1)};
    vector<dim3>        parent_dimBlock2 = {dim3(1)};

    exchange *exc2 = new exchange(prod_loc2, 
                                    prodout_loc2, 
                                    prodout_size2,
                                    prod2out2,
                                    parents2,
                                    parent_dimGrid2,
                                    parent_dimBlock2,
                                    {0});

    dim3 gridsel(16);

    h_operator_t * oprod5 = exc2->prods[0];

    d_operator_t * oprod4 = d_operator_t::create<gpu_to_cpu<WARPSIZE, 64, buffer_t *>>(1, oprod5, 1);

    d_operator_t * oprod3b = d_operator_t::create<aggregation<WARPSIZE, 0>>(1, oprod4, 8000, gridsel.x*gridsel.y*gridsel.z, 1);

    d_operator_t * oprod3 = d_operator_t::create<unstable_select<>>(1, oprod3b, gridsel.x*gridsel.y*gridsel.z, 1);


    vector<int>         prod_loc        = {-1};
    vector<int>         prodout_loc     = {-1};
    vector<int>         prodout_size    = {1024};
    vector<int>         prod2out        = {0};
    vector<p_operator_t> parents        = {oprod3};
    vector<dim3>        parent_dimGrid  = {gridsel};
    vector<dim3>        parent_dimBlock = {dim3(1024)};
    vector<int>         shared_mem      = {40960};

    exchange *exc = new exchange(prod_loc, 
                                    prodout_loc, 
                                    prodout_size,
                                    prod2out,
                                    parents,
                                    parent_dimGrid,
                                    parent_dimBlock,
                                    shared_mem); //FIXME: shared memory...

    h_operator_t * oprod = exc->prods[0];

    generator *gen = new generator(oprod, a, N);

    gpu(cudaProfilerStart());

    // nvtxMarkA("Start");
    auto start = chrono::system_clock::now();
    // gen->consume(NULL);
    // exc2->join();
    gen->close();
    {
        auto start = chrono::system_clock::now();

        for (auto &t: exc->prods) t->close();
        exc->join();

        set_device_on_scope d(1);
        // launch_close_pipeline2<<<parent_dimGrid[0], parent_dimBlock[0], 40960, 0>>>(oprod3);
        // gpu(cudaDeviceSynchronize());
        cout << "kkkkkkkkkkkkk " << endl;
        launch_close_pipeline2<<<parent_dimGrid[0], parent_dimBlock[0], shared_mem[0], 0>>>(oprod3b);
        cout << "kkkkkkkkkkkkk " << endl;
        launch_close_pipeline2<<<parent_dimGrid[0], parent_dimBlock[0], shared_mem[0], 0>>>(oprod4);
        gpu(cudaDeviceSynchronize());

        cout << "kkkkkkkkkkkkk " << endl;
        for (auto &t: exc2->prods) t->close();

        cout << "kkkkkkkkkkkkk " << endl;
        exc2->join();

        cout << "kkkkkkkkkkkkk " << endl;
        omat->close();

        auto end   = chrono::system_clock::now();
        cout << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
        // assert(false);
    }
    auto end   = chrono::system_clock::now();
    // nvtxMarkA("End");

    gpu(cudaProfilerStop());

    cout << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
    // gen->join();
    delete gen;
    delete exc;
    // delete exc2;

    const materializer * mat = omat->get<materializer *>();
    vector<int32_t> results(mat->dst, mat->dst + mat->size);

    cout << results.size() << endl;
    
    if (results.size() != M){
        cout << "Wrong output size!" << endl;
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
            cout << "Wrong result " << results[i] << "vs" << c[i] << " at (sorted) " << i << endl;
            failed = true;
        }
    }
    // assert(!failed);
    if (failed) return -1;
    cout << "End of Test" << endl;
#endif

    return 0;
}