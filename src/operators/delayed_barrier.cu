#include "delayed_barrier.cuh"

using namespace std;

template<typename... T>
__host__ delayed_barrier<T...>::delayed_barrier(h_operator<T...> parent): parent(parent){
}

template<typename... T>
__host__ void delayed_barrier<T...>::open(){
    parent.open();
}

template<typename... T>
__host__ void delayed_barrier<T...>::close(){
    parent.close();
}

template<typename... T>
__host__ void delayed_barrier<T...>::consume(const T * ... src, cnt_t N, vid_t vid, cid_t cid){
    for (auto t: follow) t.first->block_until(state.get_consumed() - t.second + 1);
    parent.consume(src..., N, vid, cid);
    
    state.mark_consumed();
}

template<typename... T>
template<typename... Tw>
__host__ void delayed_barrier<T...>::leave_behind(delayed_barrier<Tw...> *t, int64_t leave_back){
    follow.emplace_back(&(t->state), leave_back);
}

__host__ void delayed_barrier_state::block_until(int64_t at_least_consumed){
    unique_lock<mutex> lk(m);
    cv.wait(lk, [this, at_least_consumed]{return consumed >= at_least_consumed;});
    lk.unlock();
}


__host__ void delayed_barrier_state::mark_consumed(){
    ++consumed;
    cv.notify_all();
}

template class delayed_barrier<int32_t>;
template void  delayed_barrier<int32_t>::leave_behind<int32_t>(delayed_barrier<int32_t> *, int64_t);