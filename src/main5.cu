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
// #include <functional>

#include "select2.cuh"

// #include <nvToolsExt.h>

using namespace std;

#ifndef NVAL
#define NVAL (64*1024*1024*4)
#endif

constexpr int N = NVAL;

int32_t *a;
int32_t *b;

int main(){
    setbuf(stdout, NULL);
    // gpu(cudaSetDeviceFlags(cudaDeviceScheduleYield));
    srand(time(0));
    buffer_manager<int32_t>::init();
    
    a = (int32_t*) malloc(N*sizeof(int32_t));
    b = (int32_t*) malloc(N*sizeof(int32_t));
    
    // nvtxMarkA("Start");

    for (int i = 0 ; i < N ; ++i) a[i] = rand() % 100 + 1;

    // nvtxMarkA("End");

    int32_t *dst;
    gpu(cudaMallocHost(&dst, sizeof(int32_t)*N));

    materializer *mat = new materializer(NULL, dst);//, cout);
    Operator * omat = new Operator(mat);
    
    vector<int>         prod_loc2        = {1};
    vector<int>         prodout_loc2     = {1};
    vector<int>         prodout_size2    = {1024};
    vector<int>         prod2out2        = {0};
    vector<Operator *>  parents2         = {omat};
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

    Operator * oprod2 = cuda_new<Operator>(1, exc2->prods[0]);

    dim3 gridsel(1);

    unstable_select<> *s = cuda_new<unstable_select<>>(1, oprod2, gridsel.x*gridsel.y*gridsel.z, 1);
    Operator * oprod3 = cuda_new<Operator>(1, s);

    vector<int>         prod_loc        = {-1};
    vector<int>         prodout_loc     = {-1};
    vector<int>         prodout_size    = {1024};
    vector<int>         prod2out        = {0};
    vector<Operator *>  parents         = {oprod3};
    vector<dim3>        parent_dimGrid  = {gridsel};
    vector<dim3>        parent_dimBlock = {dim3(1024)};

    exchange *exc = new exchange(prod_loc, 
                                    prodout_loc, 
                                    prodout_size,
                                    prod2out,
                                    parents,
                                    parent_dimGrid,
                                    parent_dimBlock,
                                    {40960}); //FIXME: shared memory...

    Operator * oprod = new Operator(exc->prods[0]);

    generator *gen = new generator(oprod, a, N);

    gpu(cudaProfilerStart());

    auto start = chrono::system_clock::now();
    gen->consume(NULL);
    // exc2->join();
    gen->join();
    {
        auto start = chrono::system_clock::now();
        exc->join();
        exc2->join();
        auto end   = chrono::system_clock::now();
        cout << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
    }
    auto end   = chrono::system_clock::now();

    gpu(cudaProfilerStop());

    cout << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
    // gen->join();
    delete gen;
    delete exc;
    // delete exc2;

    vector<int32_t> results(mat->dst, mat->dst + mat->size);

    cout << results.size() << endl;
    if (results.size() != N){
        cout << "Wrong output size!" << endl;
        // assert(false);
        // return -1;
        return 0;
    }
    return 0;
#ifndef __CUDA_ARCH__
    sort(a, a + N);
    sort(results.begin(), results.end());

    bool failed = false;
    for (int i = 0 ; i < N ; ++i){
        if (a[i] != results[i]){
            cout << "Wrong result " << results[i] << "vs" << a[i] << " at (sorted) " << i << endl;
            failed = true;
        }
    }
    assert(!failed);
    cout << "End of Test" << endl;
#endif

    return 0;
}