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

#include <getopt.h>

// #include <nvToolsExt.h>

using namespace std;

#ifndef NVAL
#define NVAL (64*1024*1024*4)
#endif

int N = NVAL;
int M = NVAL;

int32_t *a;
int32_t *s;
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
        // if (a[j] <= 5000) {
            uint32_t bucket = h_hashMurmur(a[j]);
            next[j]    = ht[bucket];
            ht[bucket] = j;
        // }
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

pair<vector<generator *>, materializer *> create_select_pipeline(int32_t *dst, int32_t *src, int N, vector<launch_conf> confs){
    launch_conf    mat_conf{dim3(1), dim3(1), 0, -1};
    h_operator_t * mat                                                          = h_operator_t::create<     materializer        >(dst);
    
    vector<p_operator_t> xparent        = {mat     };
    vector<launch_conf > xconf          = {mat_conf};

    h_operator_t * xroot                                                        = h_operator_t::create<     exchange            >(xparent, xconf);
    h_operator_t * xroot_pre                                                    = (confs.size() > 1) ? h_operator_t::create<     union_all_cpu       >(xroot  , confs.size()) : xroot;
    
    vector<p_operator_t> pipelines;
    for (const auto &conf: confs){
        d_operator_t * g2c                                                      = d_operator_t::create<     gpu_to_cpu<>                         >(conf, xroot_pre, conf.device);
        d_operator_t * sel                                                      = d_operator_t::create<     unstable_select<WARPSIZE, int32_t>   >(conf, g2c, conf.get_blocks_per_grid(), conf.device);

        pipelines.emplace_back(sel);
    }

    h_operator_t * xgen                                                         = h_operator_t::create<     exchange            >(pipelines, confs);

    generator    * gen                                                          = new generator(xgen, src, N);

    return make_pair(vector<generator *>{gen}, mat->get<materializer *>());
}

pair<vector<generator *>, materializer *> create_sum_select_pipeline(int32_t *dst, int32_t *src, int N, vector<launch_conf> confs){
    launch_conf    mat_conf{dim3(1), dim3(1), 0, -1};
    h_operator_t * mat                                                          = h_operator_t::create<     materializer        >(dst);
    
    vector<p_operator_t> xparent        = {mat     };
    vector<launch_conf > xconf          = {mat_conf};

    h_operator_t * xroot                                                        = h_operator_t::create<     exchange            >(xparent, xconf);
    h_operator_t * xroot_pre                                                    = (confs.size() > 1) ? h_operator_t::create<     union_all_cpu       >(xroot  , confs.size()) : xroot;
    
    vector<p_operator_t> pipelines;
    for (const auto &conf: confs){
        d_operator_t * g2c                                                      = d_operator_t::create<     gpu_to_cpu<>                         >(conf, xroot_pre, conf.device);
        d_operator_t * aggr                                                     = d_operator_t::create<     aggregation<>                        >(conf, g2c , 8000, conf.get_blocks_per_grid(), conf.device);
        d_operator_t * sel                                                      = d_operator_t::create<     unstable_select<WARPSIZE, int32_t>   >(conf, aggr, conf.get_blocks_per_grid(), conf.device);

        pipelines.emplace_back(sel);
    }

    h_operator_t * xgen                                                         = h_operator_t::create<     exchange            >(pipelines, confs);

    generator    * gen                                                          = new generator(xgen, src, N);

    return make_pair(vector<generator *>{gen}, mat->get<materializer *>());
}

pair<vector<generator *>, materializer *> create_select_pipeline_exp(int32_t *dst, int32_t *src, int N, vector<launch_conf> confs){
    launch_conf    mat_conf{dim3(1), dim3(1), 0, -1};
    h_operator_t * mat                                                          = h_operator_t::create<     materializer        >(dst);
    
    vector<p_operator_t> xparent        = {mat     };
    vector<launch_conf > xconf          = {mat_conf};

    h_operator_t * xroot                                                        = h_operator_t::create<     exchange            >(xparent, xconf);
    h_operator_t * xroot_pre                                                    = (confs.size() > 1) ? h_operator_t::create<     union_all_cpu       >(xroot  , confs.size()) : xroot;
    
    vector<p_operator_t> pipelines;
    for (const auto &conf: confs){
        d_operator_t * g2c                                                      = d_operator_t::create<     gpu_to_cpu<>                         >(conf, xroot_pre, conf.device);

        d_operator_t * g2cu                                                     = d_operator_t::create<     union_all<>                          >(conf, g2c, 2, conf, conf.device);

        vector<d_operator_t *> shj0b_p        = {g2cu, g2cu};

        d_operator_t * spl                                                      = d_operator_t::create<     split<>                              >(conf, shj0b_p, conf, conf.device);
        
        d_operator_t * sel                                                      = d_operator_t::create<     unstable_select<WARPSIZE, int32_t>   >(conf, spl, conf.get_blocks_per_grid(), conf.device);

        pipelines.emplace_back(sel);
    }

    h_operator_t * xgen                                                         = h_operator_t::create<     exchange            >(pipelines, confs);

    generator    * gen                                                          = new generator(xgen, src, N);

    return make_pair(vector<generator *>{gen}, mat->get<materializer *>());
}

pair<vector<generator *>, materializer *> create_count_join_select_pipeline(int32_t *dst, int32_t *R, int RN, int32_t *L, int LN, vector<launch_conf> confs){

    launch_conf    mat_conf{dim3(1), dim3(1), 0, -1};
    h_operator_t * mat                                                          = h_operator_t::create<     materializer        >(dst);
    
    vector<p_operator_t> xparent        = {mat     };
    vector<launch_conf > xconf          = {mat_conf};

    h_operator_t * xroot                                                        = h_operator_t::create<     exchange            >(xparent, xconf);
    h_operator_t * xroot_pre                                                    = h_operator_t::create<     union_all_cpu       >(xroot  , 2);

    launch_conf conf0hj{dim3(128), dim3(1024), 0, 0};
    launch_conf conf1hj{dim3(128), dim3(1024), 0, 1};

    d_operator_t * xg2c0                                                        = d_operator_t::create<     gpu_to_cpu<>        >(conf0hj, xroot_pre, conf0hj.device);
    d_operator_t * xg2c1                                                        = d_operator_t::create<     gpu_to_cpu<>        >(conf1hj, xroot_pre, conf1hj.device);

    d_operator_t * hj0                                                          = d_operator_t::create<     hashjoin<>          >(conf0hj, xg2c0, log_htsize, RN, conf0hj);
    d_operator_t * hj1                                                          = d_operator_t::create<     hashjoin<>          >(conf1hj, xg2c1, log_htsize, RN, conf1hj);

    // d_operator_t * u0 = d_operator_t::create<     union_all<>         >(conf0hj, hj0, 2, conf0hj, conf0hj.device);
    // d_operator_t * s0 = d_operator_t::create<     split<>             >(conf0hj, vector<d_operator_t *>{u0, u0}, vector<launch_conf>{conf0hj, conf0hj}, conf0hj, conf0hj.device);

    // d_operator_t * u1 = d_operator_t::create<     union_all<>         >(conf1hj, hj1, 2, conf1hj, conf1hj.device);
    // d_operator_t * s1 = d_operator_t::create<     split<>             >(conf1hj, vector<d_operator_t *>{u1, u1}, vector<launch_conf>{conf1hj, conf1hj}, conf1hj, conf1hj.device);



    launch_conf conf0{confs[0]};
    launch_conf conf1{confs[1]};

    vector<d_operator_t *> hexin = hash_exchange(vector<p_operator_t>{hj0, hj1}, {conf0hj, conf1hj}, {conf0, conf1});

    vector<p_operator_t> xgparent        = {hexin[0], hexin[1]};
    vector<launch_conf > xgconf          = {conf0   , conf1   };

    h_operator_t * xgenhj                                                       = h_operator_t::create<     exchange            >(xgparent, xgconf);

    generator *genhj                                                            = new generator(xgenhj, L, LN);

    // =====================================================================================================================================================

    d_operator_t * hjbuilder0;
    gpu(cudaMemcpy(&hjbuilder0, &(hj0->get<hashjoin<> *>()->builder), sizeof(d_operator_t *), cudaMemcpyDefault));

    d_operator_t * hjbuilder1;
    gpu(cudaMemcpy(&hjbuilder1, &(hj1->get<hashjoin<> *>()->builder), sizeof(d_operator_t *), cudaMemcpyDefault));

    vector<d_operator_t *> hexbin                                               = hash_exchange(vector<p_operator_t>{hjbuilder0, hjbuilder1}, {conf0, conf1}, {conf0, conf1});

    // d_operator_t * sel0                                                         = d_operator_t::create<     unstable_select<WARPSIZE, int32_t>   >(conf0, hexbin[0], conf0.get_blocks_per_grid(), conf0.device);
    // d_operator_t * sel1                                                         = d_operator_t::create<     unstable_select<WARPSIZE, int32_t>   >(conf1, hexbin[1], conf1.get_blocks_per_grid(), conf1.device);

    d_operator_t * sel0                                                         = hexbin[0];
    d_operator_t * sel1                                                         = hexbin[1];

    vector<p_operator_t> xgenhjb_p        = {sel0 , sel1 };
    vector<launch_conf > xgenhjb_c        = {conf0, conf1};

    h_operator_t * xgenhjb                                                      = h_operator_t::create<     exchange            >(xgenhjb_p, xgenhjb_c);

    generator *genhjb                                                           = new generator(xgenhjb, R, RN);

    return make_pair(vector<generator *>{genhjb, genhj}, mat->get<materializer *>());
}

pair<vector<generator *>, materializer *> create_count_join_1G_pipeline(int32_t *dst, int32_t *R, int RN, int32_t *L, int LN, vector<launch_conf> confs){

    launch_conf    mat_conf{dim3(1), dim3(1), 0, -1};
    h_operator_t * mat                                                          = h_operator_t::create<     materializer        >(dst);
    
    vector<p_operator_t> xparent        = {mat     };
    vector<launch_conf > xconf          = {mat_conf};

    h_operator_t * xroot                                                        = h_operator_t::create<     exchange            >(xparent, xconf);
    
    launch_conf conf0hj{dim3(128), dim3(1024), 0, 0};
    launch_conf conf1hj{dim3(128), dim3(1024), 0, 1};

    d_operator_t * xg2c0                                                        = d_operator_t::create<     gpu_to_cpu<>        >(conf0hj, xroot, conf0hj.device);

    d_operator_t * hj0                                                          = d_operator_t::create<     hashjoin<>          >(conf0hj, xg2c0, log_htsize, RN, conf0hj);

    launch_conf conf0{confs[0]};

    vector<p_operator_t> xgparent        = {hj0};
    vector<launch_conf > xgconf          = {conf0hj};

    h_operator_t * xgenhj                                                       = h_operator_t::create<     exchange            >(xgparent, xgconf);

    generator *genhj                                                            = new generator(xgenhj, L, LN);

    // =====================================================================================================================================================

    d_operator_t * hjbuilder0;
    gpu(cudaMemcpy(&hjbuilder0, &(hj0->get<hashjoin<> *>()->builder), sizeof(d_operator_t *), cudaMemcpyDefault));

    // vector<d_operator_t *> hexbin                                               = hash_exchange(vector<p_operator_t>{hjbuilder0, hjbuilder1}, {conf0, conf1}, {conf0, conf1});

    vector<p_operator_t> xgenhjb_p        = {hjbuilder0};
    vector<launch_conf > xgenhjb_c        = {conf0};

    h_operator_t * xgenhjb                                                      = h_operator_t::create<     exchange            >(xgenhjb_p, xgenhjb_c);

    generator *genhjb                                                           = new generator(xgenhjb, R, RN);

    return make_pair(vector<generator *>{genhjb, genhj}, mat->get<materializer *>());
}

int parse_args(int argc, char *argv[]){
    int c;
    while ((c = getopt (argc, argv, "N:M:")) != -1){
        switch (c){
            case 'N':
                {
                    char * tmp   = optarg;
                    char * start = optarg;
                    int val = 1;
                    while (*tmp){
                        char t = *tmp;
                        if (t >= '0' && t <= '9'){
                            ++tmp;
                            continue;
                        } else {
                            *tmp = 0;
                            val *= atoi(start);
                            *tmp = t;
                            start = tmp + 1;

                            ++tmp;
                            if (t == '*') continue;

                            if      (t == 'K') val *= 1024;
                            else if (t == 'M') val *= 1024*1024;
                            else if (t == 'G') val *= 1024*1024*1024;
                            else {
                                cout << "Invalid entry for option -N: " << optarg << endl;
                                return -2;
                            }
                            if (*tmp) {
                                cout << "Invalid entry for option -N: " << optarg << endl;
                                return -3;
                            }
                            break;
                        }
                    }
                    N = val;
                    break;
                }
            case 'M':
                {
                    char * tmp   = optarg;
                    char * start = optarg;
                    int val = 1;
                    while (*tmp){
                        char t = *tmp;
                        if (t >= '0' && t <= '9'){
                            ++tmp;
                            continue;
                        } else {
                            *tmp = 0;
                            val *= atoi(start);
                            *tmp = t;
                            start = tmp + 1;

                            ++tmp;
                            if (t == '*') continue;

                            if      (t == 'K') val *= 1024;
                            else if (t == 'M') val *= 1024*1024;
                            else if (t == 'G') val *= 1024*1024*1024;
                            else {
                                cout << "Invalid entry for option -M: " << optarg << endl;
                                return -2;
                            }
                            if (*tmp) {
                                cout << "Invalid entry for option -M: " << optarg << endl;
                                return -3;
                            }
                            break;
                        }
                    }
                    M = val;
                    break;
                }
            case '?':
                if (optopt == 'N')
                    fprintf (stderr, "Option -%c requires an argument.\n", optopt);
                else if (isprint(optopt))
                    fprintf (stderr, "Unknown option `-%c'.\n", optopt);
                else
                    fprintf (stderr, "Unknown option character `\\x%x'.\n", optopt);
                return -4;
            default:
                assert(false);
                break;
        }
    }
    return 0;
}


int main(int argc, char *argv[]){
    int tmp = parse_args(argc, argv);
    if (tmp) return tmp;
    cout << N << " " << M << endl;
    setbuf(stdout, NULL);
    // gpu(cudaSetDeviceFlags(cudaDeviceScheduleYield));
    srand(time(0));
    buffer_manager<int32_t>::init();

    for (int i = 0 ; i < (1 << log_htsize) ; ++i) ht[i] = -1;
    
    // a = (int32_t*) malloc(N*sizeof(int32_t));

    gpu(cudaMallocHost(&a, N*sizeof(int32_t)));
    gpu(cudaMallocHost(&s, M*sizeof(int32_t)));

    b = (int32_t*) malloc(N*sizeof(int32_t));
    c = (int32_t*) malloc(N*sizeof(int32_t));
    int32_t *d = (int32_t*) malloc(N*sizeof(int32_t));
    
    // nvtxMarkA("Start");

    for (int i = 0 ; i < N ; ++i) a[i] = rand() % min(N, M) + 1;
    for (int i = 0 ; i < M ; ++i) s[i] = rand() % min(N, M) + 1;

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

    launch_conf conf0{dim3(128), dim3(1024), 40960, 0};
    launch_conf conf1{dim3(128), dim3(1024), 40960, 1};
    vector<launch_conf> confs{conf0, conf1};
    
    // pair<vector<generator *>, materializer *> p = create_sum_select_pipeline(dst, a, N, confs);
    pair<vector<generator *>, materializer *> p = create_count_join_select_pipeline(dst, a, N, s, M, confs);

    materializer * omat = p.second;

    gpu(cudaProfilerStart());

    // nvtxMarkA("Start");
    auto start = chrono::system_clock::now();
    for (const auto &gen: p.first){
        auto start = chrono::system_clock::now();

        gen->open();
        gen->close();

        auto end   = chrono::system_clock::now();
        cout << "Tm: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
        // cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    }
    auto end   = chrono::system_clock::now();
    // nvtxMarkA("End");

    gpu(cudaProfilerStop());

    cout << "Total: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;

    vector<int32_t> results(omat->dst, omat->dst + omat->size);

    cout << results.size() << endl;

//     size_t M;
//     {
//         auto start = chrono::system_clock::now();
//         // M = stable_select_cpu(a, c, N);
//         // M = sum_select_cpu(a, c, N);
//         // M = sum_selfjoin_select_cpu(a, c, N);
//         M = sum_hashjoin_select_cpu(a, c, a, d, N, N);
//         auto end   = chrono::system_clock::now();
//         cout << "CPU: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
//     }
    
//     cout << results[0] + results[1] << ((results[0] == c[0]) ? " = " : " ! ") << c[0] << endl;
//     if (results.size() != M){
//         cout << "Wrong output size! " << results.size() << " vs " << M << endl;
//         // assert(false);
//         return -1;
//         // return 0;
//     }
// //     // return 0;
// #ifndef __CUDA_ARCH__
//     sort(c, c + M);
//     sort(results.begin(), results.end());

//     bool failed = false;
//     for (int i = 0 ; i < M ; ++i){
//         if (c[i] != results[i]){
//             cout << "Wrong result " << results[i] << " vs " << c[i] << " at (sorted) " << i << endl;
//             failed = true;
//         }
//     }
//     // assert(!failed);
//     if (failed) return -1;
//     cout << "End of Test" << endl;
// #endif

    return 0;
}