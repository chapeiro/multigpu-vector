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

using namespace std;

params_t par;

int32_t *a;
int32_t *s;
int32_t *b;
int32_t *c;

// __global__ void launch_close_pipeline2(d_operator_t * parent){
//     parent->close();
//     // variant::apply_visitor(close{}, *parent);
// }

size_t stable_select_cpu(int32_t *a, int32_t *b, size_t N, int32_t thres = 5000){
    size_t i = 0;
    for (size_t j = 0 ; j < N ; ++j) if (a[j] <= thres) b[i++] = a[j];
    return i;
}

int32_t hist[5001];

size_t sum_select_cpu(int32_t *a, int32_t *b, size_t N, int32_t thres = 5000){
    int32_t s = 0;
    for (size_t j = 0 ; j < N ; ++j) if (a[j] <= thres) s += a[j];
    b[0] = s;
    return 1;
}

size_t sum_selfjoin_select_cpu(int32_t *a, int32_t *b, size_t N){
    int32_t s = 0;
    for (size_t j = 0 ; j < N ; ++j) if (a[j] <= 5000) ++hist[a[j]];
    for (int32_t i = 1 ; i <= 5000 ; ++i) s += hist[i] * hist[i];
    b[0] = s;
    return 1;
}

constexpr int log_htsize = 29;
// int32_t ht[1 << log_htsize];

// uint32_t h_hashMurmur(uint32_t x){
//     x ^= x >> 16;
//     x *= 0x85ebca6b;
//     x ^= x >> 13;
//     x *= 0xc2b2ae35;
//     x ^= x >> 16;
//     return x & ((1 << log_htsize) - 1);
// }

// size_t sum_hashjoin_select_cpu(int32_t *a, int32_t *b, int32_t *probe, int32_t *next, int N, int M){
//     int32_t s = 0;
//     for (size_t j = 0 ; j < N ; ++j){
//         // if (a[j] <= 5000) {
//             uint32_t bucket = h_hashMurmur(a[j]);
//             next[j]    = ht[bucket];
//             ht[bucket] = j;
//         // }
//     }
//     for (size_t j = 0 ; j < M ; ++j){
//         uint32_t bucket  = h_hashMurmur(probe[j]);
//         int32_t  current = ht[bucket];
//         while (current >= 0){
//             if (probe[j] == a[current]) ++s;
//             current = next[current];
//         }
//     }
//     b[0] = s;
//     return 1;
// }

vector<d_operator_t *> hash_exchange(vector<p_operator_t> parents, vector<launch_conf> parents_conf, vector<launch_conf> child_conf){
    assert(parents.size() == parents_conf.size());
    vector<p_operator_t> out;
    vector<h_operator_t *> exchs;
    for (size_t i = 0 ; i < parents.size() ; ++i){
        int ch_same_dev = 0;
        for (size_t j = 0 ; j < child_conf.size() ; ++j){
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
    for (size_t i = 0 ; i < child_conf.size() ; ++i){
        vector<d_operator_t *> transp;
        for (size_t j = 0 ; j < parents_conf.size() ; ++j){
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

pair<vector<generator *>, materializer *> create_select_pipeline(int32_t *dst, int32_t *src, size_t N, vector<launch_conf> confs, int32_t thres = 5000){
    launch_conf    mat_conf{dim3(1), dim3(1), 0, -1};
    h_operator_t * mat                                                          = h_operator_t::create<     materializer        >(dst);
    
    vector<p_operator_t> xparent        = {mat     };
    vector<launch_conf > xconf          = {mat_conf};

    h_operator_t * xroot                                                        = h_operator_t::create<     exchange            >(xparent, xconf);
    h_operator_t * xroot_pre                                                    = (confs.size() > 1) ? h_operator_t::create<     union_all_cpu       >(xroot  , confs.size()) : xroot;
    
    vector<p_operator_t> pipelines;
    for (const auto &conf: confs){
        h_operator_t * mmv                                                      = h_operator_t::create<     mem_move                             >(xroot_pre);
        d_operator_t * g2c                                                      = d_operator_t::create<     gpu_to_cpu<>                         >(conf, mmv, conf.device);
        d_operator_t * sel                                                      = d_operator_t::create<     unstable_select<WARPSIZE, less_eq_than<int32_t>, int32_t>   >(conf, g2c, less_eq_than<int32_t>(thres), conf.get_blocks_per_grid(), conf.device);

        pipelines.emplace_back(sel);
    }

    h_operator_t * xgen                                                         = h_operator_t::create<     exchange            >(pipelines, confs);

    generator    * gen                                                          = new generator(xgen, src, N);

    return make_pair(vector<generator *>{gen}, mat->get<materializer *>());
}

pair<vector<generator *>, materializer *> create_select_all_pipeline(int32_t *dst, int32_t *src, size_t N, vector<launch_conf> confs, int32_t thres = 5000){
    launch_conf    mat_conf{dim3(1), dim3(1), 0, -1};
    h_operator_t * mat                                                          = h_operator_t::create<     materializer        >(dst);
    
    vector<p_operator_t> xparent        = {mat     };
    vector<launch_conf > xconf          = {mat_conf};

    h_operator_t * xroot                                                        = h_operator_t::create<     exchange            >(xparent, xconf);
    h_operator_t * xroot_pre                                                    = (confs.size() > 1) ? h_operator_t::create<     union_all_cpu       >(xroot  , confs.size()) : xroot;
    
    vector<p_operator_t> pipelines;
    for (const auto &conf: confs){
        h_operator_t * mmv                                                      = h_operator_t::create<     mem_move                             >(xroot_pre);
        d_operator_t * g2c                                                      = d_operator_t::create<     gpu_to_cpu<>                         >(conf, mmv, conf.device);
        // d_operator_t * sel                                                      = d_operator_t::create<     unstable_select<WARPSIZE, less_eq_than<int32_t>, int32_t>   >(conf, g2c, less_eq_than<int32_t>(thres), conf.get_blocks_per_grid(), conf.device);

        pipelines.emplace_back(g2c);
    }

    h_operator_t * xgen                                                         = h_operator_t::create<     exchange            >(pipelines, confs);

    generator    * gen                                                          = new generator(xgen, src, N);

    return make_pair(vector<generator *>{gen}, mat->get<materializer *>());
}

pair<vector<generator *>, materializer *> create_sum_select_pipeline(int32_t *dst, int32_t *src, size_t N, vector<launch_conf> confs, int32_t thres = 5000){
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
        d_operator_t * sel                                                      = d_operator_t::create<     unstable_select<WARPSIZE, less_eq_than<int32_t>, int32_t>   >(conf, aggr, less_eq_than<int32_t>(thres), conf.get_blocks_per_grid(), conf.device);

        pipelines.emplace_back(sel);
    }

    h_operator_t * xgen                                                         = h_operator_t::create<     exchange            >(pipelines, confs);

    generator    * gen                                                          = new generator(xgen, src, N);

    return make_pair(vector<generator *>{gen}, mat->get<materializer *>());
}

pair<vector<generator *>, materializer *> create_select_pipeline_exp(int32_t *dst, int32_t *src, size_t N, vector<launch_conf> confs){
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
        
        d_operator_t * sel                                                      = d_operator_t::create<     unstable_select<WARPSIZE, less_eq_than<int32_t>, int32_t>   >(conf, spl, less_eq_than<int32_t>(5000), conf.get_blocks_per_grid(), conf.device);

        pipelines.emplace_back(sel);
    }

    h_operator_t * xgen                                                         = h_operator_t::create<     exchange            >(pipelines, confs);

    generator    * gen                                                          = new generator(xgen, src, N);

    return make_pair(vector<generator *>{gen}, mat->get<materializer *>());
}

pair<vector<generator *>, materializer *> create_count_join_select_pipeline(int32_t *dst, int32_t *R, size_t RN, int32_t *L, size_t LN, vector<launch_conf> confs){

    launch_conf    mat_conf{dim3(1), dim3(1), 0, -1};
    h_operator_t * mat                                                          = h_operator_t::create<     materializer        >(dst);
    
    vector<p_operator_t> xparent        = {mat     };
    vector<launch_conf > xconf          = {mat_conf};

    h_operator_t * xroot                                                        = h_operator_t::create<     exchange            >(xparent, xconf);
    h_operator_t * xroot_pre                                                    = h_operator_t::create<     union_all_cpu       >(mat  , 2);

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

    // d_operator_t * sel0                                                         = d_operator_t::create<     unstable_select<WARPSIZE, less_eq_than<int32_t>, int32_t>   >(conf0, hexbin[0], less_eq_than<int32_t>(5000), conf0.get_blocks_per_grid(), conf0.device);
    // d_operator_t * sel1                                                         = d_operator_t::create<     unstable_select<WARPSIZE, less_eq_than<int32_t>, int32_t>   >(conf1, hexbin[1], less_eq_than<int32_t>(5000), conf1.get_blocks_per_grid(), conf1.device);

    d_operator_t * sel0                                                         = hexbin[0];
    d_operator_t * sel1                                                         = hexbin[1];

    vector<p_operator_t> xgenhjb_p        = {sel0 , sel1 };
    vector<launch_conf > xgenhjb_c        = {conf0, conf1};

    h_operator_t * xgenhjb                                                      = h_operator_t::create<     exchange            >(xgenhjb_p, xgenhjb_c);

    generator *genhjb                                                           = new generator(xgenhjb, R, RN);

    return make_pair(vector<generator *>{genhjb, genhj}, mat->get<materializer *>());
}

pair<vector<generator *>, materializer *> create_count_join_1G_pipeline(int32_t *dst, int32_t *R, size_t RN, int32_t *L, size_t LN, vector<launch_conf> confs){

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

void thread_warm_up(){}

int main(int argc, char *argv[]){
    int tmp = parse_args(argc, argv, par);
    if (tmp) return tmp;
    cout << par << endl;
    setbuf(stdout, NULL);
    // gpu(cudaSetDeviceFlags(cudaDeviceScheduleYield));

    //warm up threads
    vector<thread> thrds;
    for (int i = 0 ; i < 10 ; ++i) thrds.emplace_back(thread_warm_up);

    for (auto &t: thrds) t.join();
    srand(time(0));
// {
//     set_device_on_scope d(0);
//     cudaSetDeviceFlags(cudaDeviceMapHost);
// }{    set_device_on_scope d(1);
//     cudaSetDeviceFlags(cudaDeviceMapHost);
// // }
// int size = 64;
// int buff_buffer_size = 8;
// int buff_keep_threshold = 16;
//     int devices = 2;
//         gpu(cudaGetDeviceCount(&devices));
//         devive_buffs_mutex = new mutex[devices];
//         device_buffs_pool  = new vector<buffer<int32_t, DEFAULT_BUFF_CAP, vec4>  *>[devices];
//         release_streams    = new cudaStream_t[devices];

//         gpu(cudaSetDevice(1));
//         gpu(cudaSetDevice(0));
//         for (int j = 0; j < devices; ++j) {
//             set_device_on_scope d(j);

//             for (int i = 0 ; i < devices ; ++i) {
//                 if (i != j) {
//                     int t;
//                     cudaDeviceCanAccessPeer(&t, j, i);
//                     if (t){
//                         cudaDeviceEnablePeerAccess(i, 0);
//                     } else {
//                         cout << "Warning: P2P disabled for : " << j << "->" << i << endl;
//                     }
//                 }
//             }


//             vector<buffer_t *> buffs;
//             for (size_t i = 0 ; i < size ; ++i) {
//                 int32_t *data;
//                 gpu(cudaMalloc(&data, 4*1024*1024 * sizeof(int32_t)));
//             }//buffs.push_back(cuda_new<buffer_t>(j, j));
//             // T      *mem;
//             // size_t  pitch;
//             // gpu(cudaMallocPitch(&mem, &pitch, buffer_t::capacity()*sizeof(T), size));

//             // vector<buffer_t *> buffs;
//             // for (size_t i = 0 ; i < size ; ++i) buffs.push_back(cuda_new<buffer_t>(j, (T*) (((char *) mem)+i*pitch), j));

//             // pool_t * tmp =  cuda_new<pool_t>(j, size, buffs, j);
//             // gpu(cudaMemcpyToSymbol(pool    , &tmp, sizeof(pool_t *)));
//             // gpu(cudaMemcpyToSymbol(deviceId,   &j, sizeof(int     )));

//             // gpu(cudaStreamCreateWithFlags(&(release_streams[j]), cudaStreamNonBlocking));
//             // gpu(cudaStreamCreate(&(release_streams[j])));
//             // break;
//         }
//         set_device_on_scope d(0);

//         int32_t      *mem;
//         gpu(cudaMallocHost(&mem, buffer_t::capacity()*sizeof(int32_t)*size));

//         vector<buffer_t *> buffs;
//         for (size_t i = 0 ; i < size ; ++i) buffs.push_back(cuda_new<buffer_t>(-1, mem + i * buffer_t::capacity(), -1));
//         h_pool = new h_pool_t(size, buffs);

//         device_buff      = new buffer_t**[devices];
//         device_buff_size = buff_buffer_size;
//         keep_threshold   = buff_keep_threshold;
//         buffer_t **tmp;
//         // gpu(cudaMallocHost(&tmp, device_buff_size*sizeof(buffer_t *)*devices));
//         for (int i = 0 ; i < devices ; ++i) {
//             gpu(cudaMallocHost(&tmp, device_buff_size*sizeof(buffer_t *)));
//             device_buff[i] = tmp;
//             // device_buff[i] = tmp + device_buff_size*sizeof(buffer_t *)*i;
//         }

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int i = 0 ; i < 44 ; i += 2) CPU_SET(i, &cpuset);
    // CPU_SET(2, &cpuset);
    // CPU_SET(4, &cpuset);
    // CPU_SET(6, &cpuset);
    // CPU_SET(8, &cpuset);

    int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
      std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n";
    }
    this_thread::yield();

    buffer_manager<int32_t>::init();


// {
//     // set_device_on_scope d(0);
//     // gpu(cudaSetDevice(0));

//     int32_t * h_data_in[2];
//     int32_t * d_data_in[2];
//     cudaStream_t strm[2];

//     const size_t memsize = sizeof(int32_t)*1024*1024*64;

//     for (int i = 0 ; i < 2 ; ++i){
//         gpu(cudaMallocHost(&h_data_in[i], memsize));//, cudaHostAllocDefault));
//         gpu(cudaMalloc(&d_data_in[i], memsize));

//         gpu(cudaStreamCreateWithFlags(&strm[i], cudaStreamNonBlocking));
//         // gpu(cudaStreamCreate(&strm[i]));
//     }



//     // gpu(cudaSetDevice(0));
//     // gpu(cudaDeviceSynchronize());
//     // gpu(cudaSetDevice(1));
//     // gpu(cudaDeviceSynchronize());
//     // gpu(cudaSetDevice(0));

// //    gpu(cudaDeviceSynchronize());
//     this_thread::sleep_for(chrono::milliseconds(3000));
//     {
//         auto start = chrono::system_clock::now();
//         const int its = 32;
//         for (int i = 0 ; i < its ; ++i){

//             gpu(cudaMemcpyAsync(h_data_in[i & 1], d_data_in[i & 1], memsize, cudaMemcpyDefault, strm[0]));
//             gpu(cudaMemcpyAsync(d_data_in[1 ^ (i & 1)], h_data_in[1 ^ (i & 1)], memsize, cudaMemcpyDefault, strm[1]));
            
//             gpu(cudaStreamSynchronize(strm[0]));
//             gpu(cudaStreamSynchronize(strm[1]));
//         }
//             // gpu(cudaStreamSynchronize(strm[0]));
//             // gpu(cudaStreamSynchronize(strm[1]));
//         auto end   = chrono::system_clock::now();

//         cout << "Total: " << (((memsize*1000.0*its)/(1024*1024*1024))/(chrono::duration_cast<chrono::milliseconds>(end - start).count())) << " GBps" << endl;
//         cout << "Total: " << (chrono::duration_cast<chrono::milliseconds>(end - start).count()) << " ms" << endl;
//     }

//     {
//         auto start = chrono::system_clock::now();
//         const int its = 16;
//         for (int i = 0 ; i < its ; ++i){

//             gpu(cudaMemcpyAsync(h_data_in[i & 1], d_data_in[i & 1], memsize, cudaMemcpyDefault, strm[0]));
//             gpu(cudaMemcpyAsync(d_data_in[1 ^ (i & 1)], h_data_in[1 ^ (i & 1)], memsize, cudaMemcpyDefault, strm[1]));
            
//             // gpu(cudaStreamSynchronize(strm[0]));
//             // gpu(cudaStreamSynchronize(strm[1]));
//         }
//             gpu(cudaStreamSynchronize(strm[0]));
//             gpu(cudaStreamSynchronize(strm[1]));
//         auto end   = chrono::system_clock::now();

//         cout << "Total: " << (((memsize*1000.0*its)/(1024*1024*1024))/(chrono::duration_cast<chrono::milliseconds>(end - start).count())) << " GBps" << endl;
//         cout << "Total: " << (chrono::duration_cast<chrono::milliseconds>(end - start).count()) << " ms" << endl;
//     }
// }
//     return 0;
    // for (int i = 0 ; i < (1 << log_htsize) ; ++i) ht[i] = -1;
    
    // a = (int32_t*) malloc(N*sizeof(int32_t));

    gpu(cudaMallocHost(&a, par.N*sizeof(int32_t)));
    // gpu(cudaMallocHost(&s, par.M*sizeof(int32_t)));
    s = (int32_t *) malloc(par.M*sizeof(int32_t));

    b = (int32_t*) malloc(par.N*sizeof(int32_t));
    c = (int32_t*) malloc(par.N*sizeof(int32_t));
    int32_t *d = (int32_t*) malloc(par.N*sizeof(int32_t));
    
    // nvtxMarkA("Start");
    // return 0;

    std::random_device r;

    // Choose a random mean between 1 and 6
    std::mt19937 e1(r());
    // std::uniform_int_distribution<int32_t> uniform_dist(1, min(par.M, par.N));

    {
        auto start = chrono::system_clock::now();

        for (size_t i = 0 ; i < par.N ; ++i) a[i] = e1() % 10000 + 1;//min(par.M, par.N) + 1;
        for (size_t i = 0 ; i < par.M ; ++i) s[i] = e1() % 10000 + 1;//min(par.M, par.N) + 1;

        auto end   = chrono::system_clock::now();
        cout << "Trand: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
        // cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    }

// {
//     set_device_on_scope d(0);
//     gpu(cudaThreadSetLimit(cudaLimitStackSize, 8*1024));
// }{    set_device_on_scope d(1);
//     gpu(cudaThreadSetLimit(cudaLimitStackSize, 8*1024));
// }
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
    gpu(cudaMallocHost(&dst, sizeof(int32_t)*par.N));

    launch_conf conf0{dim3(128), dim3(1024), 0, 0};
    launch_conf conf1{dim3(128), dim3(1024), 0, 1};
    vector<launch_conf> confs{conf0, conf1};
    
    pair<vector<generator *>, materializer *> p = create_select_all_pipeline(dst, a, par.N, confs);
    // pair<vector<generator *>, materializer *> p = create_select_pipeline(dst, a, par.N, confs, par.thres);
    // pair<vector<generator *>, materializer *> p = create_sum_select_pipeline(dst, a, par.N, confs, par.thres);
    // pair<vector<generator *>, materializer *> p = create_count_join_select_pipeline(dst, a, par.N, s, par.M, confs);

    materializer * omat = p.second;

    for (int i = 0 ; i < 2 ; ++i) {
        set_device_on_scope d(i);
        gpu(cudaProfilerStart());
    }

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

    for (int i = 0 ; i < 2 ; ++i) {
        set_device_on_scope d(i);
        gpu(cudaProfilerStop());
    }

    cout << "Total: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;

    return 0;

    vector<int32_t> results(omat->dst, omat->dst + omat->size);

    cout << results.size() << endl;

    // size_t M1;
    // {
    //     auto start = chrono::system_clock::now();
    //     // M1 = stable_select_cpu(a, c, N, thres);
    //     // M1 = sum_select_cpu(a, c, N, thres);
    //     // M1 = sum_selfjoin_select_cpu(a, c, N);
    //     // M1 = sum_hashjoin_select_cpu(a, c, s, d, N, M);
    //     auto end   = chrono::system_clock::now();
    //     cout << "CPU: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
    // }
    
    // // cout << results.size() << ((results.size() == M1) ? " = " : " ! ") << M1 << endl;
    cout << results[0] + results[1] << ((results[0] == c[0]) ? " = " : " ! ") << c[0] << endl;
//     if (results.size() != M1){
//         cout << "Wrong output size! " << results.size() << " vs " << M1 << endl;
//         // assert(false);
//         return -1;
//         // return 0;
//     }
// //     // return 0;
// #ifndef __CUDA_ARCH__
//     sort(c, c + M1);
//     sort(results.begin(), results.end());

//     bool failed = false;
//     for (int i = 0 ; i < M1 ; ++i){
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