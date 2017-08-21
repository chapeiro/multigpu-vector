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
// #include "operators/materializer.cuh"
#include "operators/generators.cuh"
// #include "operators/aggregation.cuh"

// #include "operators/gpu_to_cpu.cuh"
// #include <functional>

// #include "operators/select3.cuh"
// #include "operators/split.cuh"
// #include "operators/union_all.cuh"
// #include "operators/union_all_cpu.cuh"

#include "argument_parsing.cuh"
// #include <nvToolsExt.h>

#include "operators/map.cuh"
#include "operators/count.cuh"
#include "operators/aggr.cuh"
#include "operators/print.cuh"
#include "operators/map1.cuh"
#include "operators/sync.cuh"
#include "operators/apply_selection.cuh"
#include "operators/cpu_to_gpu.cuh"
#include "operators/mem_move.cuh"
#include "operators/delayed_barrier.cuh"
#include "operators/hashjoin.cuh"

#include "operators/d_operator.cuh"

#include "dfunctional.cuh"

using namespace std;

params_t par;

int32_t *a;
int32_t *s;
int32_t *b;
int32_t *c;

__global__ __launch_bounds__(65536, 4) void launch_consume_pipeline(const int32_t * __restrict__ a, const int32_t * __restrict__ b, uint32_t N, map<product, int32_t, int32_t, int32_t> * m){
    assert(N % (vector_size * get_total_num_of_warps()) == 0);

    for (int j = 0 ; j < N ; j += vector_size * get_total_num_of_warps()){
        m->consume_warp(a + j + vector_size * get_global_warpid(), b + j + vector_size * get_global_warpid(), vector_size, j / vector_size, 0);
    }
}

template<typename T>
__global__ __launch_bounds__(65536, 4) void launch_consume_pipeline(const int32_t * __restrict__ a, uint32_t N, T * m, cid_t cid){
    assert(N % (vector_size * get_total_num_of_warps()) == 0);

    for (int j = 0 ; j < N ; j += vector_size * get_total_num_of_warps()){
        m->consume_warp(a + j + vector_size * get_global_warpid(), vector_size, (vid_t) (j / vector_size), cid);
    }
}

template<typename T>
__global__ __launch_bounds__(65536, 4) void launch_close_pipeline(T * op){
    op->at_close();
}

// int32_t * lo_quantity, int32_t * lo_discount, int32_t * lo_orderdate, int32_t * lo_extendedprice, size_t lo_N, 

vector<vector<generator *>> sbbQ1_1_dates(int32_t * d_year, int32_t * d_datekey, size_t d_N, launch_conf conf){
    auto d_print_cnt            = cuda_new<dprint>(conf.device);

    auto d_cnt_dates_in_1993    = cuda_new<dcount<int32_t, int32_t>>(conf.device, d_operator<int32_t>(conf, d_print_cnt), conf);

    auto d_datekey_1993         = cuda_new<apply_selection<int32_t>>(conf.device, d_operator<int32_t>(conf, d_cnt_dates_in_1993), conf);

    typedef synchro<int32_t, sel_t> sync_t;

    auto d_sync                 = cuda_new<sync_t>(conf.device, d_operator<int32_t, sel_t>(conf, d_datekey_1993), 64, 0, conf);

    typedef map1<equal_tov<int32_t>, int32_t, sel_t> map_s;

    auto sel_d_year_1993        = cuda_new<map_s>(conf.device, d_operator<sel_t>(conf, d_sync), equal_tov<int32_t>{1993}, conf);

    auto c2g_d_year             = new cpu_to_gpu<int32_t>(cuda_new<d_operator<int32_t>>(conf.device, conf, sel_d_year_1993), conf);
    auto c2g_d_datekey          = new cpu_to_gpu<int32_t>(cuda_new<d_operator<int32_t>>(conf.device, conf,          d_sync), conf);

    c2g_d_datekey->serialize_with(c2g_d_year);

    auto memmv_d_year           = new mem_move<int32_t>(   c2g_d_year, conf.device);
    auto memmv_d_datekey        = new mem_move<int32_t>(c2g_d_datekey, conf.device);

    auto barrier_d_year         = new delayed_barrier<int32_t>(   memmv_d_year);
    auto barrier_d_datekey      = new delayed_barrier<int32_t>(memmv_d_datekey);

    barrier_d_datekey->leave_behind(   barrier_d_year, 4);
    barrier_d_year   ->leave_behind(barrier_d_datekey, 0);

    auto gen_d_year             = new generator(h_operator<int32_t>(   barrier_d_year),    d_year, d_N, 1);
    auto gen_d_datekey          = new generator(h_operator<int32_t>(barrier_d_datekey), d_datekey, d_N, 0);

    return {{gen_d_year, gen_d_datekey}};
}


vector<vector<generator *>> sbbQ1_1(int32_t * lo_orderdate, int32_t * lo_quantity, size_t lo_N, int32_t * d_year, int32_t * d_datekey, size_t d_N, launch_conf conf){
    auto d_print_cnt                = cuda_new<dprint>(conf.device);

    auto hj                         = cuda_new<hashjoin<int32_t>>(conf.device, d_operator<int32_t>(conf, d_print_cnt), 28, d_N, conf);

    auto d_datekey_1993             = cuda_new<apply_selection<int32_t>>(conf.device, hj->get_builder(), conf);

    typedef synchro<int32_t, sel_t> sync_t;

    auto d_sync                     = cuda_new<sync_t>(conf.device, d_operator<int32_t, sel_t>(conf, d_datekey_1993), 32, 0, conf);

    typedef map1<equal_tov<int32_t>, int32_t, sel_t> map_s;

    auto sel_d_year_1993            = cuda_new<map_s>(conf.device, d_operator<sel_t>(conf, d_sync), equal_tov<int32_t>{1993}, conf);

    auto c2g_d_year                 = new cpu_to_gpu<int32_t>(cuda_new<d_operator<int32_t>>(conf.device, conf, sel_d_year_1993), conf);
    auto c2g_d_datekey              = new cpu_to_gpu<int32_t>(cuda_new<d_operator<int32_t>>(conf.device, conf,          d_sync), conf);

    c2g_d_datekey->serialize_with(c2g_d_year);

    auto memmv_d_year               = new mem_move<int32_t>(   c2g_d_year, conf.device);
    auto memmv_d_datekey            = new mem_move<int32_t>(c2g_d_datekey, conf.device);

    auto barrier_d_year             = new delayed_barrier<int32_t>(   memmv_d_year);
    auto barrier_d_datekey          = new delayed_barrier<int32_t>(memmv_d_datekey);

    barrier_d_datekey->leave_behind(   barrier_d_year, 4);
    barrier_d_year   ->leave_behind(barrier_d_datekey, 0);

    auto gen_d_year                 = new generator(h_operator<int32_t>(   barrier_d_year),    d_year, d_N, 1);
    auto gen_d_datekey              = new generator(h_operator<int32_t>(barrier_d_datekey), d_datekey, d_N, 0);

    auto lo_orderdate_q_lt_25       = cuda_new<apply_selection<int32_t>>(conf.device, hj, conf);

    auto lo_sync_sel_orderdate      = cuda_new<sync_t>(conf.device, d_operator<int32_t, sel_t>(conf, lo_orderdate_q_lt_25), 32, 2, conf);

    auto c2g_lo_orderdate           = new cpu_to_gpu<int32_t>(cuda_new<d_operator<int32_t>>(conf.device, conf, lo_sync_sel_orderdate), conf);
    auto memmv_lo_orderdate         = new mem_move<int32_t>(c2g_lo_orderdate, conf.device);

    typedef map1<less_than<int32_t>, int32_t, sel_t> map_l_t;

    auto sv_lo_quantity_lt_25       = cuda_new<map_l_t>(conf.device, d_operator<sel_t>(conf, lo_sync_sel_orderdate), less_than<int32_t>{25}, conf);

    auto c2g_lo_quantity            = new cpu_to_gpu<int32_t>(cuda_new<d_operator<int32_t>>(conf.device, conf, sv_lo_quantity_lt_25), conf);
    auto memmv_lo_quantity          = new mem_move<int32_t>(c2g_lo_quantity, conf.device);

    c2g_lo_quantity->serialize_with(c2g_lo_orderdate);
    
    auto barrier_lo_orderdate       = new delayed_barrier<int32_t>(memmv_lo_orderdate);
    auto barrier_lo_quantity        = new delayed_barrier<int32_t>(memmv_lo_quantity);

    barrier_lo_orderdate->leave_behind( barrier_lo_quantity, 4);
    barrier_lo_quantity ->leave_behind(barrier_lo_orderdate, 0);

    auto gen_lo_orderdate           = new generator(h_operator<int32_t>(barrier_lo_orderdate), lo_orderdate, lo_N, 2);
    auto gen_lo_quantity            = new generator(h_operator<int32_t>(barrier_lo_quantity), lo_quantity, lo_N, 3);

    return {{gen_d_year, gen_d_datekey}, {gen_lo_orderdate, gen_lo_quantity}};
}

int main(int argc, char *argv[]){
    int tmp = parse_args(argc, argv, par);
    if (tmp) return tmp;
    cout << par << endl;
    setbuf(stdout, NULL);

    gpu(cudaFree(0));

    // gpu(cudaSetDeviceFlags(cudaDeviceScheduleYield));

    // for (auto &t: thrds) t.join();
    srand(time(0));

    buffer_manager<int32_t>::init();
    
    cout << "------------" << endl;
    int32_t *a;
    int32_t *b;

    gpu(cudaMallocHost(&a, par.N*sizeof(int32_t)));
    gpu(cudaMallocHost(&b, par.N*sizeof(int32_t)));

    int32_t *src_a = a;
    int32_t *src_b = b;

    std::random_device r;

    std::mt19937 e1(r());
    cout << "------------" << endl;
    {
        auto start = chrono::system_clock::now();

        for (size_t i = 0 ; i < par.N ; ++i) a[i] = 1993;//e1() % 7 + 1992;//e1() % 10 + 1;//min(par.M, par.N) + 1;
        for (size_t i = 0 ; i < par.N ; ++i) b[i] = i;//e1();// % 10000 + 1;//i % 2;//min(par.M, par.N) + 1;

        auto end   = chrono::system_clock::now();
        cout << "Trand: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
    }

    {
        auto start = chrono::system_clock::now();
        int32_t s2 = 0;
        for (size_t i = 0 ; i < par.N ; ++i) if (a[i] == 1993) s2 += 1;//b[i];
        cout << "Aggr : " << s2 << endl;
        auto end   = chrono::system_clock::now();
        cout << "T: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
    }
    
    if (par.src_at_device){
        gpu(cudaMalloc(&src_a, par.N*sizeof(int32_t)));
        gpu(cudaMalloc(&src_b, par.N*sizeof(int32_t)));
        gpu(cudaMemcpy(src_a, a, par.N*sizeof(int32_t), cudaMemcpyDefault));
        gpu(cudaMemcpy(src_b, b, par.N*sizeof(int32_t), cudaMemcpyDefault));
    }
    launch_conf conf[2] = {{dim3(32), dim3(1024), 0, 0}, {dim3(32), dim3(1024), 0, 1}};

    // vector<h_operator<int32_t>> t0;
    // vector<h_operator<int32_t>> t1;

    // for (int i = 0 ; i < par.gpus ; ++i){
    //     dprint                     * p = cuda_new<dprint>(conf[i].device);

    //     typedef aggr<int32_t, int32_t> aggr_t;

    //     aggr_t                    * ag = cuda_new<aggr_t>(conf[i].device, d_operator<int32_t>(conf[i], p), conf[i]);

    //     typedef apply_selection<int32_t> apply_sel;

    //     apply_sel                  * s = cuda_new<apply_sel>(conf[i].device, d_operator<int32_t>(conf[i], ag), conf[i]);

    //     typedef synchro<int32_t, sel_t> sync_t;

    //     sync_t                    * sy = cuda_new<sync_t>(conf[i].device, d_operator<int32_t, sel_t>(conf[i], s), 64, 0, conf[i]);

    //     typedef map1<equal_tov<int32_t>, int32_t, sel_t> map_s;

    //     map_s                      * m = cuda_new<map_s>(conf[i].device, d_operator<sel_t>(conf[i], sy), equal_tov<int32_t>{2}, conf[i]);

    //     // d_operator<int32_t> *o = cuda_new<d_operator<int32_t>>(conf[i].device, conf[i], m);
    //     cpu_to_gpu<int32_t>     * c2g0 = new cpu_to_gpu<int32_t>(cuda_new<d_operator<int32_t>>(conf[i].device, conf[i], sy), conf[i]);

    //     cpu_to_gpu<int32_t>     * c2g1 = new cpu_to_gpu<int32_t>(cuda_new<d_operator<int32_t>>(conf[i].device, conf[i],  m), conf[i]);
    //     c2g1->serialize_with(c2g0);

    //     mem_move<int32_t>         * m0 = new mem_move<int32_t>(c2g0, conf[i].device);
    //     mem_move<int32_t>         * m1 = new mem_move<int32_t>(c2g1, conf[i].device);

    //     delayed_barrier<int32_t>  * b0 = new delayed_barrier<int32_t>(m0);
    //     delayed_barrier<int32_t>  * b1 = new delayed_barrier<int32_t>(m1);

    //     b1->leave_behind(b0, 0);
    //     b0->leave_behind(b1, 4);

    //     t0.emplace_back(b0);
    //     t1.emplace_back(b1);
    // }

    // router             * rtable = new router;

    // exchange<int32_t>    * exc0 = new exchange<int32_t>(t0, rtable);
    // exchange<int32_t>    * exc1 = new exchange<int32_t>(t1, rtable);

    // generator * g0              = new generator(h_operator<int32_t>(exc0), src_b, par.N, 0);
    // generator * g1              = new generator(h_operator<int32_t>(exc1), src_a, par.N, 1);


    // buffer_t * buff = buffer_manager<int32_t>::h_get_buffer(conf.device);

    // buff->data = d_a;
    // buff->cnt  = par.N;

    auto pipelines = sbbQ1_1(src_a, src_b, par.N, conf[0]);

    {
        auto start = chrono::system_clock::now();
    
        for (auto &p: pipelines){
            auto start = chrono::system_clock::now();

            vector<thread> gthreads;

            for (auto t: p) gthreads.emplace_back([t]{
                t->open();
                t->close();
            });

            for (auto &t: gthreads) t.join();

            auto end   = chrono::system_clock::now();
            cout << "\nTpip: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
        }

        auto end   = chrono::system_clock::now();
        cout << "Ttotal: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
    }

    buffer_manager<int32_t>::destroy();

    // d_operator<int32_t> w(conf, p);
    // launch_close_pipeline<<<conf.gridDim, conf.blockDim, conf.shared_mem, 0>>>(m);
    // gpu(cudaDeviceSynchronize());

    return 0;
}