#ifndef HASHJOIN_CUH_
#define HASHJOIN_CUH_

#include "d_operator.cuh"

using namespace std;

template<typename Teq, typename Tpayload>
class hashjoin;

template<typename T>
struct __align__(16) node{
    int32_t         next;
    T               data;
    vid_t           oid ;

public:
    __device__ node(){}

    __device__ node(const node<T> &n): next(n.next), data(n.data), oid(n.oid){}
    __device__ node(const volatile node<T> &n): next(n.next), data(n.data), oid(n.oid){}
};

template<typename T>
class hashjoin_builder{
private:
    vid_t                           index;
    volatile int32_t              * first;
    volatile node<T>              * next;
    int32_t                         log_table_size;
#ifndef NDEBUG
    int32_t                         max_size;
#endif

public:
    __host__ hashjoin_builder(int32_t log_table_size, volatile int32_t * first, volatile node<T> * next
#ifndef NDEBUG
                                , int32_t max_size
#endif
                                );
    

    __host__   void before_open();
    __device__ void at_open();

    __device__ void consume_open();
    __device__ void consume_warp(const T * __restrict__ src, cnt_t N, vid_t vid, cid_t cid) __restrict__;
    __device__ void consume_close();

    __device__ void at_close();
    __host__   void after_close();

    __host__ ~hashjoin_builder(){}
};

template<typename Teq, typename Tpayload = void>
class hashjoin{
private:
    int32_t                         out[vector_size];
    d_operator<int32_t>             parent;
    int32_t                         log_table_size;
    volatile int32_t              * first;
    volatile node<Teq>            * next;
public:
    d_operator<Teq>                 builder;

    __host__ hashjoin(d_operator<int32_t> parent, int32_t log_table_size, int32_t max_size, const launch_conf &conf);

    __host__ d_operator<Teq> get_builder();

    __host__   void before_open();
    __device__ void at_open();

    __device__ void consume_open();
    template<typename Tt = Tpayload, typename = typename std::enable_if< is_same<Tt, void>::value, void>::type>
    __device__ void consume_warp(const Teq * __restrict__ src, cnt_t N, vid_t vid, cid_t cid) __restrict__;

    template<typename Tt = Tpayload, typename = typename std::enable_if<!is_same<Tt, void>::value, void>::type>
    __device__ void consume_warp(const Teq * __restrict__ src, const Tpayload * __restrict__ payload, cnt_t N, vid_t vid, cid_t cid) __restrict__;
    __device__ void consume_close();

    __device__ void at_close();
    __host__   void after_close();

    __host__ ~hashjoin();
};

#endif /* HASHJOIN_CUH_ */