#ifndef EXCHANGE_CUH_
#define EXCHANGE_CUH_

#include <vector>
#include <queue>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include "h_operator.cuh"

struct router{
private:
    mutex                       guard;
    unordered_map<vid_t, int>   route_table;

public:
    template<typename... T, typename F>
    int get_mapping(const T *... src, cnt_t N, vid_t vid, cid_t cid, F pol){
        lock_guard<mutex> g(guard);
        auto s = route_table.find(vid);
        if (s != route_table.end()) return s->second;
        int policy = pol(src..., N, vid, cid);
        route_table.emplace(make_pair(vid, policy));
        return policy;
    }
};

template<typename... T>
class exchange{
private:
    vector<thread>                                                  firers ;
    vector<h_operator<T...>>                                        parents;

    atomic<int>                                                     remaining_producers;

    queue<tuple<tuple<const T *...>, cnt_t, vid_t, cid_t>>         *ready_pool;
    mutex                                                          *ready_pool_mutex;
    condition_variable                                             *ready_pool_cv;

    router                                                         *r;

public:
    exchange(const vector<h_operator<T...>> &parents, router *r);

private:
    __host__ void fire(h_operator<T...> op, int i);
    __host__ bool get_ready(tuple<tuple<const T *...>, cnt_t, vid_t, cid_t> &p, int i);
    __host__ void producer_ended();

public:
    __host__ void open();

    __host__ void consume(const T * ... src, cnt_t N, vid_t vid, cid_t cid);

    __host__ void close();

    ~exchange();
};

#endif /* EXCHANGE_CUH_ */