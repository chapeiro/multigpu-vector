#ifndef DELAYED_BARRIER_CUH_
#define DELAYED_BARRIER_CUH_

#include "h_operator.cuh"
#include <mutex>
#include <condition_variable>
#include <vector>

using namespace std;

class delayed_barrier_state{
private:
    int64_t             consumed ;
    mutex               m        ;
    condition_variable  cv       ;
public:
    __host__ delayed_barrier_state(): consumed(0){};

    __host__ void block_until(int64_t at_least_consumed);

    __host__ inline int64_t get_consumed() const{
        return consumed;
    }

    __host__ void mark_consumed();
};


template<typename... T>
class delayed_barrier{
private:
    h_operator<T...>                                parent;
    delayed_barrier_state                           state ;

    vector<pair<delayed_barrier_state *, int64_t>>  follow;
public:
    __host__ delayed_barrier(h_operator<T...> parent);

    __host__ void open();

    __host__ void consume(const T * ... src, cnt_t N, vid_t vid, cid_t cid);

    template<typename... Tw>
    __host__ void leave_behind(delayed_barrier<Tw...> *t, int64_t leave_back);

    __host__ void close();

    __host__ ~delayed_barrier();
};

#endif /* DELAYED_BARRIER_CUH_ */