#include "operator.cuh"
#include "buffer_pool.cuh"

#include "generators.cuh"
#include "materializer.cuh"
#include "exchange.cuh"
#include "select2.cuh"
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
        op->join();
    }
};

class push_op{
private:
    buffer_t *buff;
public:
    __host__ __device__ push_op(buffer_t *buff): buff(buff){}

    template<typename T>
    __host__ __device__ void operator()(T op) const{
        op->consume(buff);
    }
};

class delete_op{
public:
    template<typename T>
    __host__ void operator()(T op) const{
        cuda_delete(op);
    }
};

// template<typename Op, typename... Args>
// __host__ Operator::Operator(int device, Args... args){
//     op = cuda_new<Op>(device, args);
// }
// template<typename Op>
// __host__ Operator::Operator(Op * op): op(op){
// }

// __device__ __host__ void Operator::open(){
//     variant::apply_visitor(open_op{}, op);
// }

__device__ __host__ void Operator::consume(buffer_t * buff){
    variant::apply_visitor(push_op{buff}, op);
}

__device__ __host__ void Operator::close(){
    variant::apply_visitor(close_op{}, op);
}

__host__ Operator::~Operator(){
    variant::apply_visitor(delete_op{}, op);
}