#ifndef OPERATOR_CUH_
#define OPERATOR_CUH_

#include "../external/variant/variant/variant.h"
#include "buffer_pool.cuh"

using namespace std;

class generator;
class materializer;
class exchange;
class producer;
template<size_t warp_size, typename T>
class unstable_select;
template<size_t warp_size, size_t size, typename T>
class gpu_to_cpu_dev;

typedef buffer_pool<int32_t> buffer_pool_t;

typedef buffer_pool_t::buffer_t buffer_t;

// typedef variant::variant<generator *, materializer *, producer *> Operator;


// struct close{
//     template<typename T>
//     __host__ __device__ void operator()(T op) const;
// };


class d_operator_t{
private:
    typedef variant::variant<unstable_select<32, int32_t> *, gpu_to_cpu_dev<32, 64, buffer_t *> *> op_t;
    op_t op;

public:
    template<typename Op>
    __host__ d_operator_t(Op * op): op(op){}

    __device__ void open();
    __device__ void consume_open();
    __device__ void consume_warp(const int32_t *x, unsigned int N);
    __device__ void consume_close();
    __device__ void close();

    __host__ ~d_operator_t();
};

class h_operator_t{
private:
    typedef variant::variant<generator *, materializer *, producer *> op_t;
    op_t op;

public:
    template<typename Op>
    __host__ h_operator_t(Op * op): op(op){}

    __host__ void open();
    __host__ void consume(buffer_t * buff);
    __host__ void close();

    __host__ ~h_operator_t();
};

union p_operator_t{
    d_operator_t *d;
    h_operator_t *h;

    p_operator_t(){}
    p_operator_t(d_operator_t *d): d(d){}
    p_operator_t(h_operator_t *h): h(h){}
};

#endif /* OPERATOR_CUH_ */