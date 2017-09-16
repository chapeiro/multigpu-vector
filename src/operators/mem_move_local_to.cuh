#ifndef MEM_MOVE_LOCAL_TO_CUH_
#define MEM_MOVE_LOCAL_TO_CUH_

#include "h_operator.cuh"
#include <thread>

using namespace std;

template<typename T>
class mem_move_local_to{
private:
    h_operator<T>                                   parent;
    cudaStream_t                                    strm;
    int                                             target_device;
    thread                                         *t;
public:
    __host__ mem_move_local_to(h_operator<T> parent, int target_device);

    __host__ void open();

    __host__ void consume(const T * __restrict__ src, cnt_t N, vid_t vid, cid_t cid);

    template<typename Tw>
    __host__ void serialize_with(mem_move_local_to<Tw> *t);

    __host__ void close();

    template<typename U>
    friend class mem_move_local_to;
};

template<typename... T>
class mem_multimove_local_to{
private:
    h_operator<T...>                                parent;
    cudaStream_t                                    strm;
    int                                             target_device;
    thread                                         *t;
public:
    __host__ mem_multimove_local_to(h_operator<T...> parent, int target_device);

    __host__ void open();

    __host__ void consume(const T * __restrict__ ... src, cnt_t N, vid_t vid, cid_t cid);

    template<typename... Tw>
    __host__ void serialize_with(mem_multimove_local_to<Tw...> *t);

    __host__ void close();

    template<typename... U>
    friend class mem_multimove_local_to;
};

#endif /* MEM_MOVE_LOCAL_TO_CUH_ */