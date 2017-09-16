#include "exchange.cuh"

using namespace std;

template<typename... T>
__host__ exchange<T...>::exchange(const vector<h_operator<T...>> &parents, router *r): parents(parents), remaining_producers(1), r(r){
    ready_pool       = new queue<tuple<tuple<const T *...>, cnt_t, vid_t, cid_t>>[parents.size()];
    ready_pool_mutex = new mutex                                                 [parents.size()];
    ready_pool_cv    = new condition_variable                                    [parents.size()];
}

template<typename... T>
__host__ void exchange<T...>::open(){
    for (int i = 0 ; i < parents.size() ; ++i) firers.emplace_back(&exchange::fire, this, parents[i], i);
}

template<typename... T>
__host__ void exchange<T...>::fire(h_operator<T...> op, int i){
    op.open();
    
    auto start = chrono::system_clock::now();
    do {
        tuple<tuple<const T *...>, cnt_t, vid_t, cid_t> p;
        if (!get_ready(p, i)) break;

        op.consume(get<0>(p), get<1>(p), get<2>(p), get<3>(p));
        this_thread::yield();
    } while (true);
    
    auto end   = chrono::system_clock::now();
    cout << "T:" << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;

    op.close();
}

template<typename... T>
__host__ void exchange<T...>::consume(const T * ... src, cnt_t N, vid_t vid, cid_t cid){
    int target = r->get_mapping<T...>(src..., N, vid, cid, [this](const T * ... src, cnt_t N, vid_t vid, cid_t cid){return rand() % parents.size();});
    assert(target >= 0 && target < parents.size());

    std::unique_lock<mutex> lock(ready_pool_mutex[target]);
    ready_pool[target].emplace(make_tuple(src...), N, vid, cid);
    ready_pool_cv[target].notify_all();
    lock.unlock();
}

template<typename... T>
__host__ bool exchange<T...>::get_ready(tuple<tuple<const T *...>, cnt_t, vid_t, cid_t> &p, int i){
    std::unique_lock<mutex> lock(ready_pool_mutex[i]);

    ready_pool_cv[i].wait(lock, [this, i](){return !ready_pool[i].empty() || (ready_pool[i].empty() && remaining_producers <= 0);});

    if (ready_pool[i].empty()){
        assert(remaining_producers == 0);
        lock.unlock();
        return false;
    }

    p = ready_pool[i].front();
    ready_pool[i].pop();

    lock.unlock();
    return true;
}

template<typename... T>
__host__ void exchange<T...>::producer_ended(){
    --remaining_producers;
    assert(remaining_producers >= 0);
    for (int i = 0 ; i < parents.size() ; ++i) ready_pool_cv[i].notify_all();
}

template<typename... T>
__host__ exchange<T...>::~exchange(){
    for (int i = 0 ; i < parents.size() ; ++i) assert(ready_pool[i].empty());
}

template<typename... T>
__host__ void exchange<T...>::close(){
    producer_ended();
    for (auto &t: firers ) t.join();
}

template class exchange<int32_t>;
