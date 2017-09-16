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

#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <string>
#include <fstream>
#include <sstream>

#include "numa_utils.cuh"

#include "vanilla_kernels/vanilla_kernels.cuh"

#include "operators/exchange.cuh"
#include "operators/builders/hash_exchange.cuh"
// #include "operators/materializer.cuh"
#include "operators/generators.cuh"
// #include "operators/aggregation.cuh"

#include "operators/gpu_to_cpu.cuh"
#include "operators/cpu_print.cuh"
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
#include "operators/mem_move_local_to.cuh"
#include "operators/delayed_barrier.cuh"
#include "operators/hashjoin.cuh"
#include "operators/broadcast.cuh"

#include "operators/d_operator.cuh"

#include "dfunctional.cuh"

using namespace std;

params_t par;

int32_t *a;
int32_t *s;
int32_t *b;
int32_t *c;

__global__ __launch_bounds__(65536, 4) void launch_consume_pipeline(const int32_t * __restrict__ a, const int32_t * __restrict__ b, uint32_t N, map2<product, int32_t, int32_t, int32_t> * m){
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

vector<vector<generator *>> ssbQ1_1_dates(int32_t * d_year, int32_t * d_datekey, size_t d_N, launch_conf conf){
    auto d_print_cnt            = cuda_new<dprint>(conf.device);

    auto d_cnt_dates_in_1993    = cuda_new<dcount<int32_t, int32_t>>(conf.device, d_operator<int32_t>(conf, d_print_cnt), conf);

    auto d_datekey_1993         = cuda_new<apply_selection<int32_t>>(conf.device, d_operator<int32_t>(conf, d_cnt_dates_in_1993), conf);

    typedef synchro<int32_t, sel_t> sync_t;

    auto d_sync                 = cuda_new<sync_t>(conf.device, d_operator<int32_t, sel_t>(conf, d_datekey_1993), 64, 0, 0, conf);

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


tuple<generator *, delayed_barrier<int32_t> *, cpu_to_gpu<int32_t> *> gen2gpu(int32_t * src, size_t N, cid_t cid, d_operator<int32_t> * parent, launch_conf conf){
    auto c2g                 = new cpu_to_gpu<int32_t>(parent, conf);
    auto memmv               = new mem_move<int32_t>(c2g, conf.device);
    auto memmv_local         = new mem_move_local_to<int32_t>(memmv, numa_node_of_gpu(conf.device));

    auto barrier             = new delayed_barrier<int32_t>(memmv_local);

    auto gen                 = new generator(h_operator<int32_t>(barrier), src, N, cid);

    return make_tuple(gen, barrier, c2g);
}

tuple<delayed_barrier<int32_t> *, cpu_to_gpu<int32_t> *, mem_move<int32_t> *> mv2gpu(d_operator<int32_t> * parent, launch_conf conf){
    auto c2g                 = new cpu_to_gpu<int32_t>(parent, conf);
    auto memmv               = new mem_move<int32_t>(c2g, conf.device);
    auto memmv_local         = new mem_move_local_to<int32_t>(memmv, numa_node_of_gpu(conf.device));

    auto barrier             = new delayed_barrier<int32_t>(memmv_local);

    return make_tuple(barrier, c2g, memmv);
}


vector<vector<generator *>> ssbQ1_1(int32_t * lo_extendedprice, int32_t * lo_discount, int32_t * lo_orderdate, int32_t * lo_quantity, size_t lo_N, int32_t * d_year, int32_t * d_datekey, size_t d_N, launch_conf conf){
    // auto d_print_cnt                = cuda_new<dprint>(conf.device);

    auto cprint                     = new cpu_print<int32_t>();

    auto memmv_cprint               = new mem_move<int32_t>(cprint, -1);

    auto g2c                        = cuda_new<gpu_to_cpu<64, int32_t>>(conf.device, memmv_cprint, 19, conf.device);

    auto hj                         = cuda_new<hashjoin<int32_t, int32_t>>(conf.device, d_operator<int32_t>(conf, g2c), 24, d_N, conf);

    typedef synchro<int32_t, sel_t>     sync_t ;
    typedef synchro<sel_t, sel_t>       sync2_t;
    typedef synchro<int32_t, int32_t>   sync3_t;

    auto sync_hj                    = cuda_new<sync3_t>(conf.device, d_operator<int32_t, int32_t>(conf, hj), 16, 12, 13, conf);

    auto d_datekey_1993             = cuda_new<apply_selection<int32_t>>(conf.device, hj->get_builder(), conf);


    auto d_sync                     = cuda_new<sync_t>(conf.device, d_operator<int32_t, sel_t>(conf, d_datekey_1993), 16, 7, 0, conf);

    typedef map1<equal_tov<int32_t>, int32_t, sel_t> map_s;

    auto sel_d_year_1993            = cuda_new<map_s>(conf.device, d_operator<sel_t>(conf, d_sync), equal_tov<int32_t>{1993}, conf);

    auto c2g_d_year                 = new cpu_to_gpu<int32_t>(cuda_new<d_operator<int32_t>>(conf.device, conf, sel_d_year_1993), conf);
    auto c2g_d_datekey              = new cpu_to_gpu<int32_t>(cuda_new<d_operator<int32_t>>(conf.device, conf,          d_sync), conf);

    auto memmv_d_year               = new mem_move<int32_t>(   c2g_d_year, conf.device);
    auto memmv_d_datekey            = new mem_move<int32_t>(c2g_d_datekey, conf.device);

    c2g_d_datekey->serialize_with(c2g_d_year);
    memmv_d_datekey->serialize_with(memmv_d_year);

    auto barrier_d_year             = new delayed_barrier<int32_t>(   memmv_d_year);
    auto barrier_d_datekey          = new delayed_barrier<int32_t>(memmv_d_datekey);

    barrier_d_datekey->leave_behind(   barrier_d_year, 1);
    barrier_d_year   ->leave_behind(barrier_d_datekey, 0);

    auto gen_d_year                 = new generator(h_operator<int32_t>(   barrier_d_year),    d_year, d_N, 1);
    auto gen_d_datekey              = new generator(h_operator<int32_t>(barrier_d_datekey), d_datekey, d_N, 7);

    auto lo_orderdate_q_lt_25       = cuda_new<apply_selection<int32_t>>(conf.device, d_operator<int32_t>(conf, sync_hj), conf, 12);

    auto sel_prod                   = cuda_new<apply_selection<int32_t>>(conf.device, d_operator<int32_t>(conf, sync_hj), conf, 17);

    auto lo_sync_sel_disc           = cuda_new<sync_t>(conf.device, d_operator<int32_t, sel_t>(conf, sel_prod), 16, 15, 16, conf);

    auto lo_sync_sel_orderdate      = cuda_new<sync_t>(conf.device, d_operator<int32_t, sel_t>(conf, lo_orderdate_q_lt_25), 16, 2, 11, conf);

    auto c2g_lo_orderdate           = new cpu_to_gpu<int32_t>(cuda_new<d_operator<int32_t>>(conf.device, conf, lo_sync_sel_orderdate), conf);
    auto memmv_lo_orderdate         = new mem_move<int32_t>(c2g_lo_orderdate, conf.device);

    typedef map1<less_than<int32_t>, int32_t, sel_t> map_l_t;
    //lo_sync_sel_orderdate
    auto brdcst_lo_disc_quant_f     = cuda_new<broadcast<sel_t>>(conf.device, vector<d_operator<sel_t>>{d_operator<sel_t>(conf, lo_sync_sel_orderdate), d_operator<sel_t>(conf, lo_sync_sel_disc)}, conf);

    auto lo_cond                    = cuda_new<map<log_and<sel_t>, sel_t, sel_t, sel_t>>(conf.device, d_operator<sel_t>(conf, brdcst_lo_disc_quant_f), log_and<sel_t>{}, conf);

    auto sync_disc_quant            = cuda_new<sync2_t>(conf.device, d_operator<sel_t, sel_t>(conf, lo_cond), 16, 4, 14, conf);

    auto sv_lo_quantity_lt_25       = cuda_new<map_l_t>(conf.device, d_operator<sel_t>(conf, sync_disc_quant), less_than<int32_t>{25}, conf);

    auto c2g_lo_quantity            = new cpu_to_gpu<int32_t>(cuda_new<d_operator<int32_t>>(conf.device, conf, sv_lo_quantity_lt_25), conf);
    auto memmv_lo_quantity          = new mem_move<int32_t>(c2g_lo_quantity, conf.device);

    c2g_lo_quantity->serialize_with(c2g_lo_orderdate);
    memmv_lo_quantity->serialize_with(memmv_lo_orderdate);
    
    auto barrier_lo_orderdate       = new delayed_barrier<int32_t>(memmv_lo_orderdate);
    auto barrier_lo_quantity        = new delayed_barrier<int32_t>(memmv_lo_quantity);

    barrier_lo_orderdate->leave_behind( barrier_lo_quantity, 1);
    barrier_lo_quantity ->leave_behind(barrier_lo_orderdate, 0);


    auto gen_lo_orderdate           = new generator(h_operator<int32_t>(barrier_lo_orderdate), lo_orderdate, lo_N, 2);
    auto gen_lo_quantity            = new generator(h_operator<int32_t>(barrier_lo_quantity), lo_quantity, lo_N, 3);

    auto sv_lo_discount_bt_1_3      = cuda_new<map1<in_range<int32_t>, int32_t, sel_t>>(conf.device, d_operator<sel_t>(conf, sync_disc_quant), in_range<int32_t>{1, 3}, conf);

    auto lo_prod                    = cuda_new<map<product, int32_t, int32_t, int32_t>>(conf.device, d_operator<int32_t>(conf, lo_sync_sel_disc), product{}, conf);

    auto lo_sync_prod               = cuda_new<sync3_t>(conf.device, d_operator<int32_t, int32_t>(conf, lo_prod), 16, 4, 15, conf);

    auto brdcst_lo_discount         = cuda_new<broadcast<int32_t>>(conf.device, vector<d_operator<int32_t>>{d_operator<int32_t>(conf, sv_lo_discount_bt_1_3), d_operator<int32_t>(conf, lo_sync_prod)}, conf);

    auto c2g_lo_discount            = new cpu_to_gpu<int32_t>(cuda_new<d_operator<int32_t>>(conf.device, conf, brdcst_lo_discount), conf);
    auto memmv_lo_discount          = new mem_move<int32_t>(c2g_lo_discount, conf.device);

    auto barrier_lo_discount        = new delayed_barrier<int32_t>(memmv_lo_discount);

    c2g_lo_discount->serialize_with(c2g_lo_orderdate);
    memmv_lo_discount->serialize_with(memmv_lo_orderdate);

    barrier_lo_discount->leave_behind(barrier_lo_quantity, 1);
    barrier_lo_quantity->leave_behind(barrier_lo_discount, 0);

    auto gen_lo_discount            = new generator(h_operator<int32_t>(barrier_lo_discount), lo_discount, lo_N, 4);


    auto c2g_lo_extendedprice            = new cpu_to_gpu<int32_t>(cuda_new<d_operator<int32_t>>(conf.device, conf, lo_sync_prod), conf);
    auto memmv_lo_extendedprice          = new mem_move<int32_t>(c2g_lo_extendedprice, conf.device);

    auto barrier_lo_extendedprice        = new delayed_barrier<int32_t>(memmv_lo_extendedprice);

    c2g_lo_extendedprice->serialize_with(c2g_lo_orderdate);
    memmv_lo_extendedprice->serialize_with(memmv_lo_orderdate);

    barrier_lo_extendedprice->leave_behind(barrier_lo_quantity, 1);
    barrier_lo_quantity->leave_behind(barrier_lo_extendedprice, 0);

    auto gen_lo_extendedprice            = new generator(h_operator<int32_t>(barrier_lo_extendedprice), lo_extendedprice, lo_N, 6);

    return {{gen_d_year, gen_d_datekey}, {gen_lo_orderdate, gen_lo_quantity, gen_lo_discount, gen_lo_extendedprice}};
}


vector<vector<generator *>> ssbQ1_1_2gpus(int32_t * lo_extendedprice, int32_t * lo_discount, int32_t * lo_orderdate, int32_t * lo_quantity, size_t lo_N, int32_t * d_year, int32_t * d_datekey, size_t d_N, vector<launch_conf> confs){
    vector<d_operator<int32_t, int32_t>> hjs;
    vector<d_operator<int32_t>> hj_builders;

    vector<generator *> d_gen;
    vector<generator *> lo_gen;
    
    for (const auto &conf: confs){

        auto cprint                     = new cpu_print<int32_t>();

        auto memmv_cprint               = new mem_move<int32_t>(cprint, -1);

        auto g2c                        = cuda_new<gpu_to_cpu<64, int32_t>>(conf.device, memmv_cprint, 19, conf.device);

        auto hj                         = cuda_new<hashjoin<int32_t, int32_t>>(conf.device, d_operator<int32_t>(conf, g2c), 24, d_N, conf);

        hjs.emplace_back(conf, hj);
        hj_builders.emplace_back(hj->get_builder());
    }

    hash_exchange<int32_t, int32_t> he(hjs         , confs);
    hash_exchange<int32_t> heb(hj_builders, confs);
    
    for (size_t i = 0 ; i < confs.size() ; ++i){
        launch_conf conf = confs[i];
        // auto d_print_cnt                = cuda_new<dprint>(conf.device);
        typedef synchro<int32_t, sel_t>     sync_t ;
        typedef synchro<sel_t, sel_t>       sync2_t;
        typedef synchro<int32_t, int32_t>   sync3_t;

        auto sync_hj                    = cuda_new<sync3_t>(conf.device, he.get_entry(i), 16, 12, 13, conf);

        auto d_datekey_1993             = cuda_new<apply_selection<int32_t>>(conf.device, heb.get_entry(i), conf);


        auto d_sync                     = cuda_new<sync_t>(conf.device, d_operator<int32_t, sel_t>(conf, d_datekey_1993), 16, 7, 0, conf);

        typedef map1<equal_tov<int32_t>, int32_t, sel_t> map_s;

        auto sel_d_year_1993            = cuda_new<map_s>(conf.device, d_operator<sel_t>(conf, d_sync), equal_tov<int32_t>{1993}, conf);

        auto c2g_d_year                 = new cpu_to_gpu<int32_t>(cuda_new<d_operator<int32_t>>(conf.device, conf, sel_d_year_1993), conf);
        auto c2g_d_datekey              = new cpu_to_gpu<int32_t>(cuda_new<d_operator<int32_t>>(conf.device, conf,          d_sync), conf);

        heb.c2g0_list[i]->serialize_with(c2g_d_year);
        c2g_d_datekey->serialize_with(c2g_d_year);

        auto memmv_d_year               = new mem_move<int32_t>(   c2g_d_year, conf.device);
        auto memmv_d_datekey            = new mem_move<int32_t>(c2g_d_datekey, conf.device);

        auto barrier_d_year             = new delayed_barrier<int32_t>(   memmv_d_year);
        auto barrier_d_datekey          = new delayed_barrier<int32_t>(memmv_d_datekey);

        memmv_d_datekey->serialize_with(memmv_d_year);

        barrier_d_datekey->leave_behind(   barrier_d_year, 1);
        barrier_d_year   ->leave_behind(barrier_d_datekey, 0);

        auto gen_d_year                 = new generator(h_operator<int32_t>(   barrier_d_year),    d_year + (d_N / 2) * i, (i ? d_N - (d_N / 2) * i : (d_N / 2)), 1);
        auto gen_d_datekey              = new generator(h_operator<int32_t>(barrier_d_datekey), d_datekey + (d_N / 2) * i, (i ? d_N - (d_N / 2) * i : (d_N / 2)), 7);
        d_gen.push_back(gen_d_year   );
        d_gen.push_back(gen_d_datekey);

        auto lo_orderdate_q_lt_25       = cuda_new<apply_selection<int32_t>>(conf.device, d_operator<int32_t>(conf, sync_hj), conf, 12);

        auto sel_prod                   = cuda_new<apply_selection<int32_t>>(conf.device, d_operator<int32_t>(conf, sync_hj), conf, 17);

        auto lo_sync_sel_disc           = cuda_new<sync_t>(conf.device, d_operator<int32_t, sel_t>(conf, sel_prod), 16, 15, 16, conf);

        auto lo_sync_sel_orderdate      = cuda_new<sync_t>(conf.device, d_operator<int32_t, sel_t>(conf, lo_orderdate_q_lt_25), 16, 2, 11, conf);

        auto c2g_lo_orderdate           = new cpu_to_gpu<int32_t>(cuda_new<d_operator<int32_t>>(conf.device, conf, lo_sync_sel_orderdate), conf);
        auto memmv_lo_orderdate         = new mem_move<int32_t>(c2g_lo_orderdate, conf.device);
        auto memmv_local_lo_orderdate   = new mem_move_local_to<int32_t>(memmv_lo_orderdate, numa_node_of_gpu(conf.device));

        typedef map1<less_than<int32_t>, int32_t, sel_t> map_l_t;
        //lo_sync_sel_orderdate
        auto brdcst_lo_disc_quant_f     = cuda_new<broadcast<sel_t>>(conf.device, vector<d_operator<sel_t>>{d_operator<sel_t>(conf, lo_sync_sel_orderdate), d_operator<sel_t>(conf, lo_sync_sel_disc)}, conf);

        auto lo_cond                    = cuda_new<map<log_and<sel_t>, sel_t, sel_t, sel_t>>(conf.device, d_operator<sel_t>(conf, brdcst_lo_disc_quant_f), log_and<sel_t>{}, conf);

        auto sync_disc_quant            = cuda_new<sync2_t>(conf.device, d_operator<sel_t, sel_t>(conf, lo_cond), 16, 4, 14, conf);

        auto sv_lo_quantity_lt_25       = cuda_new<map_l_t>(conf.device, d_operator<sel_t>(conf, sync_disc_quant), less_than<int32_t>{25}, conf);

        auto c2g_lo_quantity            = new cpu_to_gpu<int32_t>(cuda_new<d_operator<int32_t>>(conf.device, conf, sv_lo_quantity_lt_25), conf);
        auto memmv_lo_quantity          = new mem_move<int32_t>(c2g_lo_quantity, conf.device);
        auto memmv_local_lo_quantity    = new mem_move_local_to<int32_t>(memmv_lo_quantity, numa_node_of_gpu(conf.device));

        // he.c2g0_list[i]->serialize_with(memmv_lo_orderdate);
        // he.c2g1_list[i]->serialize_with(memmv_lo_orderdate);
        memmv_lo_quantity->serialize_with(memmv_lo_orderdate);

        he.c2g0_list[i]->serialize_with(c2g_lo_orderdate);
        he.c2g1_list[i]->serialize_with(c2g_lo_orderdate);
        c2g_lo_quantity->serialize_with(c2g_lo_orderdate);
        
        auto barrier_lo_orderdate       = new delayed_barrier<int32_t>(memmv_local_lo_orderdate);
        auto barrier_lo_quantity        = new delayed_barrier<int32_t>(memmv_local_lo_quantity);

        barrier_lo_orderdate->leave_behind( barrier_lo_quantity, 2);
        barrier_lo_quantity ->leave_behind(barrier_lo_orderdate, 2);


        auto gen_lo_orderdate           = new generator(h_operator<int32_t>(barrier_lo_orderdate), lo_orderdate + (lo_N / 2) * i, (i ? lo_N - (lo_N / 2) * i : (lo_N / 2)), 2);
        auto gen_lo_quantity            = new generator(h_operator<int32_t>(barrier_lo_quantity), lo_quantity + (lo_N / 2) * i, (i ? lo_N - (lo_N / 2) * i : (lo_N / 2)), 3);

        auto sv_lo_discount_bt_1_3      = cuda_new<map1<in_range<int32_t>, int32_t, sel_t>>(conf.device, d_operator<sel_t>(conf, sync_disc_quant), in_range<int32_t>{1, 3}, conf);

        auto lo_prod                    = cuda_new<map<product, int32_t, int32_t, int32_t>>(conf.device, d_operator<int32_t>(conf, lo_sync_sel_disc), product{}, conf);

        auto lo_sync_prod               = cuda_new<sync3_t>(conf.device, d_operator<int32_t, int32_t>(conf, lo_prod), 16, 4, 15, conf);

        auto brdcst_lo_discount         = cuda_new<broadcast<int32_t>>(conf.device, vector<d_operator<int32_t>>{d_operator<int32_t>(conf, sv_lo_discount_bt_1_3), d_operator<int32_t>(conf, lo_sync_prod)}, conf);

        auto c2g_lo_discount            = new cpu_to_gpu<int32_t>(cuda_new<d_operator<int32_t>>(conf.device, conf, brdcst_lo_discount), conf);
        auto memmv_lo_discount          = new mem_move<int32_t>(c2g_lo_discount, conf.device);
        auto memmv_local_lo_discount    = new mem_move_local_to<int32_t>(memmv_lo_discount, numa_node_of_gpu(conf.device));

        auto barrier_lo_discount        = new delayed_barrier<int32_t>(memmv_local_lo_discount);

        c2g_lo_discount->serialize_with(c2g_lo_orderdate);
        memmv_lo_discount->serialize_with(memmv_lo_orderdate);

        barrier_lo_discount->leave_behind(barrier_lo_quantity, 2);
        barrier_lo_quantity->leave_behind(barrier_lo_discount, 2);

        auto gen_lo_discount            = new generator(h_operator<int32_t>(barrier_lo_discount), lo_discount + (lo_N / 2) * i, (i ? lo_N - (lo_N / 2) * i : (lo_N / 2)), 4);


        auto c2g_lo_extendedprice            = new cpu_to_gpu<int32_t>(cuda_new<d_operator<int32_t>>(conf.device, conf, lo_sync_prod), conf);
        auto memmv_lo_extendedprice          = new mem_move<int32_t>(c2g_lo_extendedprice, conf.device);
        auto memmv_local_lo_extendedprice    = new mem_move_local_to<int32_t>(memmv_lo_extendedprice, numa_node_of_gpu(conf.device));

        auto barrier_lo_extendedprice        = new delayed_barrier<int32_t>(memmv_local_lo_extendedprice);

        c2g_lo_extendedprice->serialize_with(c2g_lo_orderdate);
        memmv_lo_extendedprice->serialize_with(memmv_lo_orderdate);

        barrier_lo_extendedprice->leave_behind(barrier_lo_quantity, 2);
        barrier_lo_quantity->leave_behind(barrier_lo_extendedprice, 2);

        auto gen_lo_extendedprice            = new generator(h_operator<int32_t>(barrier_lo_extendedprice), lo_extendedprice + (lo_N / 2) * i, (i ? lo_N - (lo_N / 2) * i : (lo_N / 2)), 6);

        lo_gen.push_back(gen_lo_orderdate);
        lo_gen.push_back(gen_lo_quantity);
        lo_gen.push_back(gen_lo_discount);
        lo_gen.push_back(gen_lo_extendedprice);
    }
    return {d_gen, lo_gen};
}


template<typename T>
T * open_file(string s, size_t size){
    size_t fsize = size * sizeof(T);
    int fd = open(s.c_str(), O_RDONLY);
    assert(fd != -1);

    char *buf = (char*) mmap(NULL, fsize, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
    assert(buf != MAP_FAILED);

    return (T *) buf;
}


template<typename T>
T * open_file_and_load_to_pinned(string s, size_t size){
    size_t fsize = size * sizeof(T);

    int fd = open(s.c_str(), O_RDONLY);
    assert(fd != -1);

    T *buf = (T *) mmap(NULL, fsize, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
    assert(((char *) buf) != MAP_FAILED);

    T *mem;
    gpu(cudaMallocHost(&mem, fsize));
    // for (size_t i = 0 ; i < size ; ++i) mem[i] = buf[i];
    gpu(cudaMemcpy(mem, buf, fsize, cudaMemcpyDefault));

    munmap(buf, fsize);
    return mem;
}

// template<typename T>
// T * open_file_and_load_to_pinned(string s, size_t size){
//     size_t fsize = size * sizeof(T);

//     ifstream file(s.c_str(), ios::binary);
//     // file.unsetf(std::ios::skipws);

//     // int fd = open(s.c_str(), O_RDONLY);
//     // assert(fd != -1);

//     // T *buf = (T *) mmap(NULL, fsize, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
//     // assert(((char *) buf) != MAP_FAILED);

//     T *mem;
//     gpu(cudaMallocHost(&mem, fsize));
//     // for (size_t i = 0 ; i < size ; ++i) mem[i] = buf[i];
//     // gpu(cudaMemcpy(mem, buf, fsize, cudaMemcpyDefault));

//     file.read((char *) mem, fsize);
//     // copy(istream_iterator<uint8_t>(file), istream_iterator<uint8_t>(), mem);

//     // munmap(buf, fsize);
//     return mem;
// }

template<typename T>
T * open_file_and_load_for_write(string s, size_t size){
    size_t fsize = size * sizeof(T);

    int fd = open(s.c_str(), O_RDWR | O_CREAT);
    assert(fd != -1);
    
    if (lseek(fd, fsize-1, SEEK_SET) == -1) assert(false);
    write(fd, "", 1);

    char *buf = (char*) mmap(NULL, fsize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

    return (T *) buf;
}


size_t extract_ints(string from, int32_t ** to, int columns, int *column, size_t N){
    int32_t * s = to[0];
    ifstream ifile(from.c_str());

    size_t l = 0;
    string line; // we read the full line here
    while (getline(ifile, line)){
        istringstream iss{line}; // construct a string stream from line

        // read the tokens from current line separated by comma
        vector<string> tokens; // here we store the tokens
        string token; // current token
        
        int i = 0;
        int j = 0;
        while (getline(iss, token, '|')) if (j < columns && column[j] == i++) {
            *(to[j++]++) = stoi(token);
        }

        if (++l % (N/1000) == 0) cout << (1000*l)/N << endl;
    }
    return to[0] - s;
}

void convert_lineorder(size_t N = 59986214){
    int cls = 4;
    string s[4] = {
                    "/cloud_store/periklis/data100/lo_orderdate.bin",
                    "/cloud_store/periklis/data100/lo_quantity.bin",
                    "/cloud_store/periklis/data100/lo_extendedprice.bin",
                    "/cloud_store/periklis/data100/lo_discount.bin"
    };
    int clm[] = {5, 8, 9, 11};
    int32_t *x[cls];
    int32_t *y[cls];
    for (int i = 0 ; i < cls ; ++i) y[i] = x[i] = open_file_and_load_for_write<int32_t>(s[i], N);
    size_t lo_N = extract_ints("/cloud_store/periklis/data100/lineorder.tbl", x, cls, clm, N);
    assert(N == lo_N);
    for (int i = 0 ; i < cls ; ++i) {
        msync(y[i], lo_N * sizeof(int32_t), MS_SYNC);
        munmap(y[i], lo_N * sizeof(int32_t));
    }
}

void create_rand_file(string name, size_t N){
    int32_t *x = open_file_and_load_for_write<int32_t>(name, N);
    std::random_device r;
    std::mt19937 e1(r());
    for (size_t i = 0 ; i < N ; ++i) x[i] = e1();

    msync( x, N * sizeof(int32_t), MS_SYNC);
    munmap(x, N * sizeof(int32_t));
}


void convert_date(size_t N = 2556){
    int cls = 2;
    string s[2] = {
                    "../data100/d_datekey.bin",
                    "../data100/d_year.bin"
    };
    int clm[2] = {0, 4};
    int32_t *x[cls];
    int32_t *y[cls];
    for (int i = 0 ; i < cls ; ++i) y[i] = x[i] = open_file_and_load_for_write<int32_t>(s[i], N);
    size_t lo_N = extract_ints("../ssb-dbgen/date.tbl", x, cls, clm, N);
cout << N << " " << lo_N << endl;
    assert(N == lo_N);
    for (int i = 0 ; i < cls ; ++i) {
        msync(y[i], lo_N * sizeof(int32_t), MS_SYNC);
        munmap(y[i], lo_N * sizeof(int32_t));
    }
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

void cpu_ssbQ1_1(int32_t * __restrict__ lo_extendedprice, int32_t * __restrict__ lo_discount, int32_t * __restrict__ lo_orderdate, int32_t * __restrict__ lo_quantity, size_t lo_N, int32_t * __restrict__ d_year, int32_t * __restrict__ d_datekey, size_t d_N){
    memset(ht, -1, sizeof(int32_t) * (1 << log_htsize));

    int32_t next[d_N];

        auto start = chrono::system_clock::now();
    int32_t s = 0;
    int32_t c = 0;
    for (size_t j = 0 ; j < d_N ; ++j){
        if (d_year[j] == 1993) {
            uint32_t bucket = h_hashMurmur(d_datekey[j]);
            next[j]    = ht[bucket];
            ht[bucket] = j;
            ++c;
        }
    }
    cout << "===" << c << endl;
    c = 0;
    int32_t g = 0;
    for (size_t j = 0 ; j < lo_N ; ++j){
        if (lo_discount[j] >= 1 && lo_discount[j] <= 3 && lo_quantity[j] < 25){
            uint32_t bucket  = h_hashMurmur(lo_orderdate[j]);
            int32_t  current = ht[bucket];
            while (current >= 0){
                if (lo_orderdate[j] == d_datekey[current]) {
                    s += lo_discount[j] * lo_extendedprice[j];
                    ++g;
                }
                current = next[current];
            }
            ++c;
        }
    }
        auto end   = chrono::system_clock::now();
        cout << "Ttotal: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
    // cout << "===" << c << endl;
    cout << "===" << s << " " << c << " " << g << endl;
}



void cpu_select_sum(int32_t * __restrict__ a, int32_t * __restrict__ b, size_t N, int32_t thres){
    int32_t res = 0;
        auto start = chrono::system_clock::now();
    for (size_t i = 0 ; i < N ; ++i) if (a[i] < thres) res += b[i];
        auto end   = chrono::system_clock::now();
        cout << "Ttotal: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
    cout << "===" << res << endl;
}

void cpu_select_sum_multicore(int32_t * __restrict__ a, int32_t * __restrict__ b, size_t N, int32_t thres, int cores){
    size_t bucket_size = (N + cores - 1) / cores;
    atomic<int32_t> gres(0);
    auto start = chrono::system_clock::now();
    thread * ts[cores];
    for (int j = 0 ; j < cores ; ++j){
        ts[j] = new thread([thres, &gres](int32_t * __restrict__ a, int32_t * __restrict__ b, size_t N){
            int32_t res = 0;
            for (size_t i = 0 ; i < N ; ++i) if (a[i] < thres) res += b[i];
            gres.fetch_add(res);
        }, a + bucket_size * j, b + bucket_size * j, min((size_t) N, (size_t) (bucket_size * (j + 1))) - bucket_size * j);
    }

    for (int j = 0 ; j < cores ; ++j) ts[j]->join();
        auto end   = chrono::system_clock::now();
        cout << "Ttotal: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
    cout << "===" << gres << endl;
}

vector<vector<generator *>> select_sum(int32_t * a, int32_t * b, size_t N, int32_t thres, launch_conf conf){
    auto cprint     = new cpu_print<int32_t>();

    auto memmv_cprint = new mem_move<int32_t>(cprint, -1);

    auto g2c        = cuda_new<gpu_to_cpu<64, int32_t>>(conf.device, memmv_cprint, 4, conf.device);

    auto suma       = cuda_new<aggr<int32_t, int32_t>>(conf.device, d_operator<int32_t>(conf, g2c ), conf);

    auto sel_b      = cuda_new<apply_selection<int32_t>>(conf.device, d_operator<int32_t>(conf, suma), conf);

    auto sync       = cuda_new<synchro<int32_t, sel_t>>(conf.device, d_operator<int32_t, sel_t>(conf, sel_b), 16, 2, 3, conf);

    typedef map1<less_than<int32_t>, int32_t, sel_t> map_s;

    auto sel        = cuda_new<map_s>(conf.device, d_operator<sel_t>(conf, sync), less_than<int32_t>{thres}, conf);

    auto gen2gpu_a  = gen2gpu(a, N, 1, cuda_new<d_operator<int32_t>>(conf.device, conf, sel ), conf);
    auto gen2gpu_b  = gen2gpu(b, N, 2, cuda_new<d_operator<int32_t>>(conf.device, conf, sync), conf);

    get<2>(gen2gpu_a)->serialize_with(get<2>(gen2gpu_b));

    get<1>(gen2gpu_a)->leave_behind(get<1>(gen2gpu_b), 4);
    get<1>(gen2gpu_b)->leave_behind(get<1>(gen2gpu_a), 4);

    return {{get<0>(gen2gpu_a), get<0>(gen2gpu_b)}};
}

vector<vector<generator *>> select_sum_multigpu2(int32_t * a, int32_t * b, size_t N, int32_t thres, vector<launch_conf> confs){
    auto cprint     = new cpu_print<int32_t>();

    vector<h_operator<int32_t>> gpus_a;
    vector<h_operator<int32_t>> gpus_b;
    for (size_t i = 0 ; i < confs.size() ; ++i){
        launch_conf conf = confs[i];

        auto memmv_cprint = new mem_move<int32_t>(cprint, -1);

        auto g2c        = cuda_new<gpu_to_cpu<64, int32_t>>(conf.device, memmv_cprint, 4, conf.device);

        auto suma       = cuda_new<aggr<int32_t, int32_t>>(conf.device, d_operator<int32_t>(conf, g2c ), conf);

        auto sel_b      = cuda_new<apply_selection<int32_t>>(conf.device, d_operator<int32_t>(conf, suma), conf);

        auto sync       = cuda_new<synchro<int32_t, sel_t>>(conf.device, d_operator<int32_t, sel_t>(conf, sel_b), 16, 2, 3, conf);

        typedef map1<less_than<int32_t>, int32_t, sel_t> map_s;

        auto sel        = cuda_new<map_s>(conf.device, d_operator<sel_t>(conf, sync), less_than<int32_t>{thres}, conf);

        auto mv_a  = mv2gpu(cuda_new<d_operator<int32_t>>(conf.device, conf, sel ), conf);
        auto mv_b  = mv2gpu(cuda_new<d_operator<int32_t>>(conf.device, conf, sync), conf);

        get<1>(mv_a)->serialize_with(get<1>(mv_b));
        get<2>(mv_a)->serialize_with(get<2>(mv_b));

        get<0>(mv_a)->leave_behind(get<0>(mv_b), 4);
        get<0>(mv_b)->leave_behind(get<0>(mv_a), 4);

        gpus_a.emplace_back(get<0>(mv_a));
        gpus_b.emplace_back(get<0>(mv_b));
    }
    router * r = new router;

    auto exch_a = new exchange<int32_t>(gpus_a, r);
    auto exch_b = new exchange<int32_t>(gpus_b, r);

    auto gen_a  = new generator(h_operator<int32_t>(exch_a), a, N, 1);
    auto gen_b  = new generator(h_operator<int32_t>(exch_b), b, N, 2);

    return {{gen_a, gen_b}};
    // return {{get<0>(gen2gpu_a), get<0>(gen2gpu_b)}};
}

vector<vector<generator *>> select_sum_multigpu(int32_t * a, int32_t * b, size_t N, int32_t thres, vector<launch_conf> confs){
    auto cprint     = new cpu_print<int32_t>();

    vector<generator *> gens;
    for (size_t i = 0 ; i < confs.size() ; ++i){
        launch_conf conf = confs[i];

        auto memmv_cprint = new mem_move<int32_t>(cprint, -1);

        auto g2c        = cuda_new<gpu_to_cpu<64, int32_t>>(conf.device, memmv_cprint, 4, conf.device);

        auto suma       = cuda_new<aggr<int32_t, int32_t>>(conf.device, d_operator<int32_t>(conf, g2c ), conf);

        auto sel_b      = cuda_new<apply_selection<int32_t>>(conf.device, d_operator<int32_t>(conf, suma), conf);

        auto sync       = cuda_new<synchro<int32_t, sel_t>>(conf.device, d_operator<int32_t, sel_t>(conf, sel_b), 16, 2, 3, conf);

        typedef map1<less_than<int32_t>, int32_t, sel_t> map_s;

        auto sel        = cuda_new<map_s>(conf.device, d_operator<sel_t>(conf, sync), less_than<int32_t>{thres}, conf);

        auto gen2gpu_a  = gen2gpu(a + (N / 2) * i, (i ? N - (N / 2) * i : (N / 2)), 1, cuda_new<d_operator<int32_t>>(conf.device, conf, sel ), conf);
        auto gen2gpu_b  = gen2gpu(b + (N / 2) * i, (i ? N - (N / 2) * i : (N / 2)), 2, cuda_new<d_operator<int32_t>>(conf.device, conf, sync), conf);

        get<2>(gen2gpu_a)->serialize_with(get<2>(gen2gpu_b));

        get<1>(gen2gpu_a)->leave_behind(get<1>(gen2gpu_b), 4);
        get<1>(gen2gpu_b)->leave_behind(get<1>(gen2gpu_a), 4);

        gens.push_back(get<0>(gen2gpu_a));
        gens.push_back(get<0>(gen2gpu_b));
    }

    return {gens};
    // return {{get<0>(gen2gpu_a), get<0>(gen2gpu_b)}};
}


int main(int argc, char *argv[]){
    // convert_lineorder(600037902);//252332317);
    // convert_date(2185);
    // return 0;
    int tmp = parse_args(argc, argv, par);
    if (tmp) return tmp;
    cout << par << endl;

    // create_rand_file("/cloud_store/periklis/urandom/entries_" + to_string(par.N/(1024*1024*1024)) + "G_0.bin", par.N);
    // create_rand_file("/cloud_store/periklis/urandom/entries_" + to_string(par.N/(1024*1024*1024)) + "G_1.bin", par.N);
    // return 0;
    setbuf(stdout, NULL);

    gpu(cudaSetDeviceFlags(cudaDeviceScheduleYield));

    gpu(cudaFree(0));

    // gpu(cudaDeviceSetLimit(cudaLimitStackSize, 40960));


    // for (auto &t: thrds) t.join();
    srand(time(0));

    buffer_manager<int32_t>::init();

    int32_t *src_lo_quantity     ;
    int32_t *src_lo_orderdate    ;
    int32_t *src_lo_discount     ;
    int32_t *src_lo_extendedprice;
    int32_t *src_d_year          ;
    int32_t *src_d_datekey       ;
    int32_t *lo_quantity         ;
    int32_t *lo_orderdate        ;
    int32_t *lo_discount         ;
    int32_t *lo_extendedprice    ;
    int32_t *d_year              ;
    int32_t *d_datekey           ;

    int32_t *a;
    int32_t *b;

    int32_t *src_a;
    int32_t *src_b;

    size_t lo_N = 239944856/4; //2400151608/4;//
    size_t d_N  = 10224/4;

    if (par.query == SSBQ1_1){
        {
            auto start = chrono::system_clock::now();
            
            // src_lo_quantity        = open_file_and_load_to_pinned<int32_t>("/cloud_store/periklis/data100/lo_quantity.bin" , lo_N);
            // src_lo_orderdate       = open_file_and_load_to_pinned<int32_t>("/cloud_store/periklis/data100/lo_orderdate.bin", lo_N);
            // src_lo_discount        = open_file_and_load_to_pinned<int32_t>("/cloud_store/periklis/data100/lo_discount.bin", lo_N);
            // src_lo_extendedprice   = open_file_and_load_to_pinned<int32_t>("/cloud_store/periklis/data100/lo_extendedprice.bin", lo_N);
            src_lo_quantity        = open_file_and_load_to_pinned<int32_t>("../data10/lo_quantity.bin" , lo_N);
            src_lo_orderdate       = open_file_and_load_to_pinned<int32_t>("../data10/lo_orderdate.bin", lo_N);
            src_lo_discount        = open_file_and_load_to_pinned<int32_t>("../data10/lo_discount.bin", lo_N);
            src_lo_extendedprice   = open_file_and_load_to_pinned<int32_t>("../data10/lo_extendedprice.bin", lo_N);
            src_d_year             = open_file_and_load_to_pinned<int32_t>("../data10/d_year.bin", d_N);
            src_d_datekey          = open_file_and_load_to_pinned<int32_t>("../data10/d_datekey.bin", d_N);

            auto end   = chrono::system_clock::now();
            cout << "TloadSSB: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
        }
        // cout << src_d_year[0] << endl;
        // cout << src_d_year[1] << endl;
        // cout << src_d_year[2] << endl;
        // cout << src_d_year[3] << endl;
        // cout << src_d_year[4] << endl;

        // return 0;

        lo_quantity            = src_lo_quantity ;
        lo_orderdate           = src_lo_orderdate;
        lo_discount            = src_lo_discount ;
        lo_extendedprice       = src_lo_extendedprice ;
        d_year                 = src_d_year      ;
        d_datekey              = src_d_datekey   ;

        if (par.src_at_device){
            gpu(cudaMalloc(&lo_quantity         , lo_N*sizeof(int32_t)));
            gpu(cudaMalloc(&lo_orderdate        , lo_N*sizeof(int32_t)));
            gpu(cudaMalloc(&lo_discount         , lo_N*sizeof(int32_t)));
            gpu(cudaMalloc(&lo_extendedprice    , lo_N*sizeof(int32_t)));
            gpu(cudaMalloc(&d_year              , d_N *sizeof(int32_t)));
            gpu(cudaMalloc(&d_datekey           , d_N *sizeof(int32_t)));

            gpu(cudaMemcpy(lo_quantity      , src_lo_quantity       , lo_N*sizeof(int32_t), cudaMemcpyDefault));
            gpu(cudaMemcpy(lo_orderdate     , src_lo_orderdate      , lo_N*sizeof(int32_t), cudaMemcpyDefault));
            gpu(cudaMemcpy(lo_discount      , src_lo_discount       , lo_N*sizeof(int32_t), cudaMemcpyDefault));
            gpu(cudaMemcpy(lo_extendedprice , src_lo_extendedprice  , lo_N*sizeof(int32_t), cudaMemcpyDefault));
            gpu(cudaMemcpy(d_year           , src_d_year            , d_N *sizeof(int32_t), cudaMemcpyDefault));
            gpu(cudaMemcpy(d_datekey        , src_d_datekey         , d_N *sizeof(int32_t), cudaMemcpyDefault));
        }
    } else if (par.query == SELSUM){
        // {
        //     auto start = chrono::system_clock::now();
        //     thread t1([&src_a](size_t N){
        //         src_a = open_file_and_load_to_pinned<int32_t>("/cloud_store/periklis/urandom/entries_16G_0.bin" , N);
        //     }, par.N);
        //     src_b = open_file_and_load_to_pinned<int32_t>("/cloud_store/periklis/urandom/entries_16G_1.bin" , par.N);
        //     t1.join();
        //     auto end   = chrono::system_clock::now();
        //     cout << "Tload: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
        // }
    gpu(cudaMallocHost(&src_a, par.N*sizeof(int32_t)));
    gpu(cudaMallocHost(&src_b, par.N*sizeof(int32_t)));
   std::random_device r;

    std::mt19937 e1(r());
    {
        auto start = chrono::system_clock::now();

        for (size_t i = 0 ; i < par.N ; ++i) src_a[i] = e1() % 10000 + 1;//e1() % 10 + 1;//min(par.M, par.N) + 1;
        for (size_t i = 0 ; i < par.N ; ++i) src_b[i] = e1() % 10000 + 1;//i % 2;//min(par.M, par.N) + 1;

        auto end   = chrono::system_clock::now();
        cout << "Trand: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
    }

        a = src_a;
        b = src_b;

        if (par.src_at_device){
            gpu(cudaMalloc(&a         , par.N*sizeof(int32_t)));
            gpu(cudaMalloc(&b         , par.N*sizeof(int32_t)));

            gpu(cudaMemcpy(a, src_a, par.N*sizeof(int32_t), cudaMemcpyDefault));
            gpu(cudaMemcpy(b, src_b, par.N*sizeof(int32_t), cudaMemcpyDefault));
        }
    }
    vector<launch_conf> conf{launch_conf{dim3(32), dim3(1024), 0, 0}, launch_conf{dim3(32), dim3(1024), 0, 1}};


    if (par.query == SSBQ1_1){
        cpu_ssbQ1_1(src_lo_extendedprice, src_lo_discount, src_lo_orderdate, src_lo_quantity, lo_N, src_d_year, src_d_datekey, d_N);
    } else if (par.query == SELSUM){
        cpu_select_sum(src_a, src_b, par.N, par.thres);
        cpu_select_sum_multicore(src_a, src_b, par.N, par.thres, 48);

        {
            set_device_on_scope d(conf[0].device);
            int32_t * result;
            gpu(cudaMalloc(&result, sizeof(int32_t)));
            int32_t h_result = 0;
            gpu(cudaMemcpy(result, &h_result, sizeof(int32_t), cudaMemcpyDefault));
            {
                auto start = chrono::system_clock::now();
                sum_select_less_than<<<conf[0].gridDim, conf[0].blockDim, conf[0].shared_mem, 0>>>(a, b, par.N, par.thres, result);
                gpu(cudaDeviceSynchronize());
                auto end   = chrono::system_clock::now();
                cout << "\nTvanilla: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
            }
            gpu(cudaMemcpy(&h_result, result, sizeof(int32_t), cudaMemcpyDefault));
            gpu(cudaFree(result));
            cout << "=======================================" << h_result << endl;
        }
        {
            set_device_on_scope d(conf[0].device);
            int32_t * result;
            gpu(cudaMalloc(&result, sizeof(int32_t)));
            int32_t h_result = 0;
            gpu(cudaMemcpy(result, &h_result, sizeof(int32_t), cudaMemcpyDefault));
            {
                auto start = chrono::system_clock::now();
                sum_select_less_than2<<<conf[0].gridDim, conf[0].blockDim, conf[0].shared_mem, 0>>>(a, b, par.N, par.thres, result);
                gpu(cudaDeviceSynchronize());
                auto end   = chrono::system_clock::now();
                cout << "\nTvanilla: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
            }
            gpu(cudaMemcpy(&h_result, result, sizeof(int32_t), cudaMemcpyDefault));
            gpu(cudaFree(result));
            cout << "=======================================" << h_result << endl;
        }
    }

    vector<vector<generator *>> pipelines;

    // for (par.gpus = 1 ; par.gpus <= 2 ; ++par.gpus){
        if (par.query == SSBQ1_1){
            if (par.gpus == 1){
                pipelines = ssbQ1_1(lo_extendedprice, lo_discount, lo_orderdate, lo_quantity, lo_N, d_year, d_datekey, d_N, conf[0]);
            } else {
                pipelines = ssbQ1_1_2gpus(lo_extendedprice, lo_discount, lo_orderdate, lo_quantity, lo_N, d_year, d_datekey, d_N, conf);
            }
        } else if (par.query == SELSUM){
            if (par.gpus == 1){
                pipelines = select_sum(a, b, par.N, par.thres, conf[0]);
            } else {
                pipelines = select_sum_multigpu2(a, b, par.N, par.thres, conf);
            }
        }

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
    // }

    buffer_manager<int32_t>::destroy();

    // d_operator<int32_t> w(conf, p);
    // launch_close_pipeline<<<conf.gridDim, conf.blockDim, conf.shared_mem, 0>>>(m);
    // gpu(cudaDeviceSynchronize());

    return 0;
}