#ifndef DATA_GENERATORS_CUH_
#define DATA_GENERATORS_CUH_

#include "h_operator.cuh"
#include <tuple>
#include <chrono>

using namespace std;

class generator{
private:
    h_operator<int32_t>                 parent;
    cid_t                               cid;
    int32_t                            *src;
    size_t                              N;
public:
    __host__ generator(h_operator<int32_t> parent, int32_t *src, size_t N, cid_t cid);

    __host__ void open();

    __host__ void consume(cnt_t N, vid_t vid, cid_t cid);

    __host__ void close();

    __host__ ~generator();
};

template<int index, typename... Ts>
struct increment_tuple {
    void operator() (tuple<Ts...>& t, size_t inc) {
        get<index>(t) += inc;
        increment_tuple<index - 1, Ts...>{}(t, inc);
    }
};

template<typename... Ts>
struct increment_tuple<0, Ts...> {
    void operator() (tuple<Ts...>& t, size_t inc) {
        get<0>(t) += inc;
    }
};

template<typename... Ts>
void increment(tuple<Ts...>& t, size_t inc) {
    const auto size = tuple_size<tuple<Ts...>>::value;
    increment_tuple<size - 1, Ts...>{}(t, inc);
}

template<typename... T>
class multigenerator{
private:
    h_operator<T...>                 parent;
    cid_t                               cid;
    tuple<T * ...>                      src;
    size_t                              N;
public:
    __host__ multigenerator(h_operator<T...> parent, T * ... src, size_t N, cid_t cid):
            parent(parent), src(src...), N(N), cid(cid){
    }

    __host__ multigenerator(T * ... src, size_t N, cid_t cid):
            src(src...), N(N), cid(cid){
    }

    __host__ void open(){
        parent.open();
        {
            vid_t vid = 0;
            // buffer_pool<int32_t>::buffer_t::inspector_t insp(strm);
            auto start = chrono::system_clock::now();
            while (N > 0){
            // {
                // buffer_pool<int32_t>::buffer_t * buff = buffer_manager<int32_t>::get_buffer();

                // buffer_pool<int32_t>::buffer_t * buff = buffer_manager<int32_t>::h_get_buffer(1);

                size_t m = min(N, (size_t) h_vector_size);

                // insp.load(buff, true);
                // insp.overwrite(src, m, false);

                // insp.save(buff, true);
                // memcpy(buff->data, src, m * sizeof(int32_t));
                // buff->data = src;
                // buff->cnt  = m;
                
                assert(m > 0);
                parent.consume(src, m, vid, cid);

                vid += m;
                N   -= m;
                increment(src, m);
            }
            auto end   = chrono::system_clock::now();
            cout << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
        }
    }

    __host__ void consume(cnt_t N, vid_t vid, cid_t cid){
        assert(false);
    }

    __host__ void close(){
        // {
        //     auto start = chrono::system_clock::now();
        //     // gpu(cudaStreamSynchronize(strm));
        //     auto end   = chrono::system_clock::now();
        //     cout << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms." << endl;
        // }
        parent.close();
    }


    __host__ ~multigenerator(){}
};

#endif /* DATA_GENERATORS_CUH_ */