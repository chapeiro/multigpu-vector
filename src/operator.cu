#include "operator.cuh"
#include "buffer_pool.cuh"

#include "aggregation.cuh"
#include "generators.cuh"
#include "materializer.cuh"
#include "exchange.cuh"
#include "select3.cuh"
#include "gpu_to_cpu.cuh"

// template<typename T>
// __host__ __device__ void push::operator()(T op) const{
//    op->consume(buff);
// }

// template<typename T>
// __host__ __device__ void close::operator()(T op) const{
//    op->join();
// }


// template __host__ __device__ void push::operator()<generator    *>(generator    * op) const;
// template __host__ __device__ void push::operator()<materializer *>(materializer * op) const;
// template __host__ __device__ void push::operator()<consumer     *>(consumer     * op) const;
// template __host__ __device__ void push::operator()<producer     *>(producer     * op) const;

// template __host__ __device__ void close::operator()<generator    *>(generator    * op) const;
// template __host__ __device__ void close::operator()<materializer *>(materializer * op) const;
// template __host__ __device__ void close::operator()<consumer     *>(consumer     * op) const;
// template __host__ __device__ void close::operator()<producer     *>(producer     * op) const;

class open_op{
public:
    template<typename T>
    __host__ __device__ void operator()(T op) const{
        op->open();
    }
};

class close_op{
public:
    template<typename T>
    __host__ __device__ void operator()(T op) const{
        op->close();
    }
};

class push_op{
private:
    buffer_t *buff;
public:
    __host__ push_op(buffer_t *buff): buff(buff){}

    template<typename T>
    __host__ void operator()(T op) const{
        op->consume(buff);
    }
};

class push_warp_op{
private:
    const int32_t *x;
    unsigned int   N;
public:
    __device__ push_warp_op(const int32_t *x, unsigned int N): x(x), N(N){}

    template<typename T>
    __device__ void operator()(T op) const{
        op->consume_warp(x, N);
    }
};

class push_open_op{
public:
    template<typename T>
    __device__ void operator()(T op) const{
        op->consume_open();
    }
};

class push_close_op{
public:
    template<typename T>
    __device__ void operator()(T op) const{
        op->consume_close();
    }
};

class delete_op{
public:
    template<typename T>
    __host__ void operator()(T op) const{
        cuda_delete(op);
    }
};

__device__ void d_operator_t::open(){
    variant::apply_visitor(open_op{}, op);
}

__device__ void d_operator_t::consume_open(){
    variant::apply_visitor(push_open_op{}, op);
}

__device__ void d_operator_t::consume_warp(const int32_t *x, unsigned int N){
    variant::apply_visitor(push_warp_op{x, N}, op);
}

__device__ void d_operator_t::consume_close(){
    variant::apply_visitor(push_close_op{}, op);
}

__device__ void d_operator_t::close(){
    variant::apply_visitor(close_op{}, op);
}

__host__ d_operator_t::~d_operator_t(){
    variant::apply_visitor(delete_op{}, op);
}

__host__ void h_operator_t::open(){
    variant::apply_visitor(open_op{}, op);
}

__host__ void h_operator_t::consume(buffer_t * buff){
    variant::apply_visitor(push_op{buff}, op);
}

__host__ void h_operator_t::close(){
    variant::apply_visitor(close_op{}, op);
}

__host__ h_operator_t::~h_operator_t(){
    variant::apply_visitor(delete_op{}, op);
}
