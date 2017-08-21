#ifndef HASH_EXCHANGE_CUH_
#define HASH_EXCHANGE_CUH_

#include "../d_operator.cuh"
#include <vector>


#include "../cpu_to_gpu.cuh"
#include "../gpu_to_cpu.cuh"
#include "../mem_move.cuh"
#include "../union_all.cuh"
#include "../colsplit.cuh"
#include "../h_colsplit.cuh"
#include "../split.cuh"
#include "../apply_selection.cuh"
#include "../delayed_barrier.cuh"
#include "../sync.cuh"

using namespace std;


template<typename... T>
class hash_exchange{
    vector<d_operator<T...>  > entry_points;
public:
    vector<cpu_to_gpu<typename tuple_element<0, tuple<T...>>::type> *> c2g0_list;
    vector<cpu_to_gpu<typename tuple_element<1, tuple<T...>>::type> *> c2g1_list;
public:
    hash_exchange(vector<d_operator<T...>> parents, vector<launch_conf> confs){
        static_assert(sizeof...(T) == 2, "Unimplemented for sizeof...(T) != 2");
        assert(parents.size() == 2);

        vector<d_operator<T...>> unions;
        for (size_t i = 0 ; i < parents.size() ; ++i){
            unions.emplace_back(confs[i], cuda_new<union_all<T...>>(confs[i].device, parents[i], parents.size(), confs[i]));
        }

        vector<h_operator<T...>> dst_in;
        for (size_t i = 0 ; i < parents.size() ; ++i){
            auto s          = cuda_new<synchro<T...>>(confs[i].device, unions[i], 16, 70 + i, 90 + i, confs[i]);

            auto c2g0        = new cpu_to_gpu<typename tuple_element<0, tuple<T...>>::type>(cuda_new<d_operator<typename tuple_element<0, tuple<T...>>::type>>(confs[i].device, confs[i], s), confs[i]);
            auto c2g1        = new cpu_to_gpu<typename tuple_element<1, tuple<T...>>::type>(cuda_new<d_operator<typename tuple_element<1, tuple<T...>>::type>>(confs[i].device, confs[i], s), confs[i]);

            c2g0_list.emplace_back(c2g0);
            c2g1_list.emplace_back(c2g1);

            auto mmv0       = new mem_move<typename tuple_element<0, tuple<T...>>::type>(c2g0, confs[i].device);
            auto mmv1       = new mem_move<typename tuple_element<1, tuple<T...>>::type>(c2g1, confs[i].device);


            auto barrier_mmv0  = new delayed_barrier<typename tuple_element<0, tuple<T...>>::type>(mmv0);
            auto barrier_mmv1  = new delayed_barrier<typename tuple_element<1, tuple<T...>>::type>(mmv1);

            barrier_mmv1->leave_behind(barrier_mmv0, 2);
            barrier_mmv0->leave_behind(barrier_mmv1, 2);

            auto cmv        = new h_colsplit<T...>(barrier_mmv0, barrier_mmv1, 70 + i, 80 + i);
            //TODO: need a union if more than 2 nodes
            dst_in.emplace_back(cmv);
        }
        
        for (size_t i = 0 ; i < parents.size() ; ++i){
            vector<d_operator<T..., sel_t>> colsplits;
            for (size_t j = 0 ; j < parents.size() ; ++j){
                d_operator<T...> targ;
                if (i != j){
                    auto g2c        = cuda_new<gpu_to_cpu<64, T...>>(confs[i].device, dst_in[j], 30 + i, confs[i].device);
                    targ            = d_operator<T...>(confs[i], g2c);
                } else {
                    targ            = unions[j];
                }
                auto s          = cuda_new<synchro<T...>>(confs[i].device, targ, 1, 40 + j, 50 + i, confs[i]);

                auto appl_sel0  = cuda_new<apply_selection<typename tuple_element<0, tuple<T...>>::type>>(confs[i].device, d_operator<typename tuple_element<0, tuple<T...>>::type>(confs[i], s), confs[i], 40 + j);
                auto appl_sel1  = cuda_new<apply_selection<typename tuple_element<1, tuple<T...>>::type>>(confs[i].device, d_operator<typename tuple_element<1, tuple<T...>>::type>(confs[i], s), confs[i], 60 + j);

                colsplits.emplace_back(confs[i], cuda_new<colsplit<T..., sel_t>>(confs[i].device, d_operator<typename tuple_element<0, tuple<T...>>::type, sel_t>(confs[i], appl_sel0), d_operator<typename tuple_element<1, tuple<T...>>::type, sel_t>(confs[i], appl_sel1), 40 + j, 60 + j, confs[i]));
            }
            auto spl            = cuda_new<split<T...>>(confs[i].device, colsplits, confs[i], confs[i].device);
            
            entry_points.emplace_back(confs[i], spl);
        }
    }

    d_operator<T...> get_entry(size_t i){
        assert(i < entry_points.size());
        return entry_points[i];
    }
};

template<typename T>
class hash_exchange<T>{
    vector<d_operator<T>  > entry_points;
    vector<d_operator<T>> unions;
public:
    vector<cpu_to_gpu<T> *> c2g0_list;
public:
    hash_exchange(vector<d_operator<T>> parents, vector<launch_conf> confs){
        assert(parents.size() == 2);

        for (size_t i = 0 ; i < parents.size() ; ++i){
            unions.emplace_back(confs[i], cuda_new<union_all<T>>(confs[i].device, parents[i], parents.size(), confs[i]));
        }

        vector<h_operator<T>> dst_in;
        for (size_t i = 0 ; i < parents.size() ; ++i){
            auto c2g0        = new cpu_to_gpu<T>(cuda_new<d_operator<T>>(confs[i].device, unions[i]), confs[i]);

            c2g0_list.emplace_back(c2g0);

            auto mmv0       = new mem_move<T>(c2g0, confs[i].device);

            //TODO: need a union if more than 2 nodes
            dst_in.emplace_back(mmv0);
        }
        
        for (size_t i = 0 ; i < parents.size() ; ++i){
            vector<d_operator<T, sel_t>> colsplits;
            for (size_t j = 0 ; j < parents.size() ; ++j){
                d_operator<T> targ;
                if (i != j){
                    auto g2c        = cuda_new<gpu_to_cpu<64, T>>(confs[i].device, dst_in[j], 30 + i, confs[i].device);
                    targ            = d_operator<T>(confs[i], g2c);
                } else {
                    targ            = unions[j];
                }
                auto appl_sel0  = cuda_new<apply_selection<T>>(confs[i].device, targ, confs[i], 40 + j);

                colsplits.emplace_back(confs[i], appl_sel0);
            }
            auto spl            = cuda_new<split<T>>(confs[i].device, colsplits, confs[i], confs[i].device);
            
            entry_points.emplace_back(confs[i], spl);
        }
    }

    d_operator<T> get_entry(size_t i){
        assert(i < entry_points.size());
        return entry_points[i];
    }
};

#endif /* HASH_EXCHANGE_CUH_ */