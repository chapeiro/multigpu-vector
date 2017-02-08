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

vector<d_operator_t *> hash_exchange(vector<p_operator_t> parents, vector<launch_conf> parents_conf, vector<launch_conf> child_conf){
    assert(parents.size() == parents_conf.size());
    vector<p_operator_t> out;
    vector<h_operator_t *> exchs;
    for (int i = 0 ; i < parents.size() ; ++i){
        int ch_same_dev = 0;
        for (int j = 0 ; j < child_conf.size() ; ++j){
            if (child_conf[j].device == parents_conf[i].device) ++ch_same_dev;
        }
        if (ch_same_dev && parents_conf[i].device >= 0){
            out.emplace_back(d_operator_t::create<     union_all<>         >(parents_conf[i], parents[i].d, ch_same_dev+1, parents_conf[i], parents_conf[i].device));
        } else if (ch_same_dev && parents_conf[i].device < 0){
            out.emplace_back(h_operator_t::create<     union_all_cpu       >(parents[i].h, ch_same_dev+1));
        } else {
            out.emplace_back(parents[i]);
        }

        vector<p_operator_t> xpar        = {out[i]         };
        vector<launch_conf > xconf       = {parents_conf[i]};

        h_operator_t * x = h_operator_t::create<     exchange            >(xpar, xconf);

        int tmp = child_conf.size() - (parents_conf[i].device >= 0 ? ch_same_dev : 0);
        if (tmp > 1){
            exchs.emplace_back(h_operator_t::create<     union_all_cpu       >(x   , tmp));
        } else {
            exchs.emplace_back(x);
        }
    }

    vector<d_operator_t *> in;
    for (int i = 0 ; i < child_conf.size() ; ++i){
        vector<d_operator_t *> transp;
        for (int j = 0 ; j < parents_conf.size() ; ++j){
            if (parents_conf[j].device == child_conf[i].device){
                assert(parents_conf[j].device >= 0);
                transp.emplace_back(out[j].d);
            } else {
                transp.emplace_back(d_operator_t::create<     gpu_to_cpu<>        >(child_conf[i], exchs[j], child_conf[i].device));
            }
        }
        in.emplace_back(d_operator_t::create<     split<>             >(child_conf[i], transp, child_conf[i], child_conf[i].device));
    }
    return in;
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

{
    set_device_on_scope d(0);
    gpu(cudaThreadSetLimit(cudaLimitStackSize, 8*1024));
}{    set_device_on_scope d(1);
    gpu(cudaThreadSetLimit(cudaLimitStackSize, 8*1024));
}
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

    launch_conf    mat_conf{dim3(1), dim3(1), 0, -1};
    h_operator_t * mat                                                          = h_operator_t::create<     materializer        >(dst);
    
    vector<p_operator_t> xparent        = {mat     };
    vector<launch_conf > xconf          = {mat_conf};

    h_operator_t * xroot                                                        = h_operator_t::create<     exchange            >(xparent, xconf);
    h_operator_t * xroot_pre                                                    = h_operator_t::create<     union_all_cpu       >(xroot  , 2);

    launch_conf conf0hj{dim3(32), dim3(1024), 0, 0};
    launch_conf conf1hj{dim3(32), dim3(1024), 0, 1};

    d_operator_t * xg2c0                                                        = d_operator_t::create<     gpu_to_cpu<>        >(conf0hj, xroot_pre, conf0hj.device);
    d_operator_t * xg2c1                                                        = d_operator_t::create<     gpu_to_cpu<>        >(conf1hj, xroot_pre, conf1hj.device);

    d_operator_t * hj0                                                          = d_operator_t::create<     hashjoin<>          >(conf0hj, xg2c0, log_htsize, N, conf0hj);
    d_operator_t * hj1                                                          = d_operator_t::create<     hashjoin<>          >(conf1hj, xg2c1, log_htsize, N, conf1hj);

    d_operator_t * uhj0                                                         = d_operator_t::create<     union_all<>         >(conf0hj, hj0, 2, conf0hj, conf0hj.device);
    d_operator_t * uhj1                                                         = d_operator_t::create<     union_all<>         >(conf1hj, hj1, 2, conf1hj, conf1hj.device);


    launch_conf conf0{dim3(16), dim3(1024), 40960, 0};
    launch_conf conf1{dim3(16), dim3(1024), 40960, 1};

    vector<d_operator_t *> hexin = hash_exchange(vector<p_operator_t>{hj0, hj1}, {conf0hj, conf1hj}, {conf0, conf1});

    // d_operator_t * uhj0                                                         = d_operator_t::create<     union_all<>         >(conf0, hj0, 2, conf0, conf0.device);
    // d_operator_t * uhj1                                                         = d_operator_t::create<     union_all<>         >(conf1, hj1, 2, conf1, conf1.device);


    // vector<p_operator_t> hashx0_p        = {uhj1    };
    // vector<launch_conf > hashx0_c        = {conf1   };
    // vector<p_operator_t> hashx1_p        = {uhj0    };
    // vector<launch_conf > hashx1_c        = {conf0   };

    // h_operator_t * hashx0                                                       = h_operator_t::create<     exchange            >(hashx0_p, hashx0_c);
    // h_operator_t * hashx1                                                       = h_operator_t::create<     exchange            >(hashx1_p, hashx1_c);

    // d_operator_t * xg2chx0                                                      = d_operator_t::create<     gpu_to_cpu<>        >(conf0, hashx0, conf0.device);
    // d_operator_t * xg2chx1                                                      = d_operator_t::create<     gpu_to_cpu<>        >(conf1, hashx1, conf1.device);

    // vector<d_operator_t *> shj0_p        = {uhj0, xg2chx0};
    // vector<d_operator_t *> shj1_p        = {xg2chx1, uhj1};

    d_operator_t * shj0 = hexin[0];//                                                        = d_operator_t::create<     split<>             >(conf0, shj0_p, conf0, conf0.device);
    d_operator_t * shj1 = hexin[1];//                                                        = d_operator_t::create<     split<>             >(conf1, shj1_p, conf1, conf1.device);

    vector<p_operator_t> xgparent        = {shj0 , shj1 };
    vector<launch_conf > xgconf          = {conf0, conf1};

    h_operator_t * xgenhj                                                       = h_operator_t::create<     exchange            >(xgparent, xgconf);

    generator *genhj                                                            = new generator(xgenhj, a, N);

    // =====================================================================================================================================================

    d_operator_t * hjbuilder0;
    gpu(cudaMemcpy(&hjbuilder0, &(hj0->get<hashjoin<> *>()->builder), sizeof(d_operator_t *), cudaMemcpyDefault));

    d_operator_t * hjbuilder1;
    gpu(cudaMemcpy(&hjbuilder1, &(hj1->get<hashjoin<> *>()->builder), sizeof(d_operator_t *), cudaMemcpyDefault));

    // d_operator_t * uhjb0                                                        = d_operator_t::create<     union_all<>         >(conf0, hjbuilder0, 2, conf0, conf0.device);
    // d_operator_t * uhjb1                                                        = d_operator_t::create<     union_all<>         >(conf1, hjbuilder1, 2, conf1, conf1.device);


    // vector<p_operator_t> hashx0b_p        = {uhjb1    };
    // vector<launch_conf > hashx0b_c        = {conf1   };
    // vector<p_operator_t> hashx1b_p        = {uhjb0    };
    // vector<launch_conf > hashx1b_c        = {conf0   };

    // h_operator_t * hashx0b                                                      = h_operator_t::create<     exchange            >(hashx0b_p, hashx0b_c);
    // h_operator_t * hashx1b                                                      = h_operator_t::create<     exchange            >(hashx1b_p, hashx1b_c);

    // d_operator_t * xg2chbx0                                                     = d_operator_t::create<     gpu_to_cpu<>        >(conf0, hashx0b, conf0.device);
    // d_operator_t * xg2chbx1                                                     = d_operator_t::create<     gpu_to_cpu<>        >(conf1, hashx1b, conf1.device);

    // vector<d_operator_t *> shj0b_p        = {uhjb0, xg2chbx0};
    // vector<d_operator_t *> shj1b_p        = {xg2chbx1, uhjb1};

    // d_operator_t * shjb0                                                        = d_operator_t::create<     split<>             >(conf0, shj0b_p, conf0, conf0.device);
    // d_operator_t * shjb1                                                        = d_operator_t::create<     split<>             >(conf1, shj1b_p, conf1, conf1.device);

    vector<d_operator_t *> hexbin                                               = hash_exchange(vector<p_operator_t>{hjbuilder0, hjbuilder1}, {conf0, conf1}, {conf0, conf1});


    d_operator_t * sel0                                                         = d_operator_t::create<     unstable_select<WARPSIZE, int32_t>   >(conf0, hexbin[0], conf0.get_blocks_per_grid(), conf0.device);
    d_operator_t * sel1                                                         = d_operator_t::create<     unstable_select<WARPSIZE, int32_t>   >(conf1, hexbin[1], conf1.get_blocks_per_grid(), conf1.device);

    vector<p_operator_t> xgenhjb_p        = {sel0 , sel1 };
    vector<launch_conf > xgenhjb_c        = {conf0, conf1};

    h_operator_t * xgenhjb                                                      = h_operator_t::create<     exchange            >(xgenhjb_p, xgenhjb_c);

    generator *genhjb                                                           = new generator(xgenhjb, a, N);

    gpu(cudaProfilerStart());

    // nvtxMarkA("Start");
    auto start = chrono::system_clock::now();
    {
        auto start = chrono::system_clock::now();

        genhjb->open();
        genhjb->close();

        auto end   = chrono::system_clock::now();
        cout << "Tm: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
    }
    {
        auto start = chrono::system_clock::now();

        genhj->open();
        genhj->close();

        auto end   = chrono::system_clock::now();
        cout << "Tm: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
    }
    auto end   = chrono::system_clock::now();
    // nvtxMarkA("End");

    gpu(cudaProfilerStop());

    cout << "Total: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;

    const materializer * omat = mat->get<materializer *>();
    vector<int32_t> results(omat->dst, omat->dst + omat->size);

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
    
    // if (results.size() != M){
    //     cout << "Wrong output size!" << results.size() << " vs " << M << endl;
    //     // assert(false);
    //     return -1;
    //     // return 0;
    // }
    cout << results[0]+results[1] << ((results[0]+results[1] == c[0]) ? " = " : " ! ") << c[0] << endl;
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