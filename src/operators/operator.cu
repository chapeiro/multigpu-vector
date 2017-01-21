#include "operator.cuh"

#include "aggregation.cuh"
#include "generators.cuh"
#include "materializer.cuh"
#include "exchange.cuh"
#include "select3.cuh"
#include "gpu_to_cpu.cuh"
#include "hashjoin.cuh"

class open_op{
public:
    template<typename T>
    __host__ void operator()(T op) const{
        op->open();
    }
};

class close_op{
public:
    template<typename T>
    __host__ void operator()(T op) const{
        op->close();
    }
};

class before_open_op{
public:
    template<typename T>
    __host__ void operator()(T op) const{
        op->before_open();
    }
};

class after_close_op{
public:
    template<typename T>
    __host__ void operator()(T op) const{
        op->after_close();
    }
};

class at_open_op{
public:
    template<typename T>
    __device__ void operator()(T op) const{
        op->at_open();
    }
};

class at_close_op{
public:
    template<typename T>
    __device__ void operator()(T op) const{
        op->at_close();
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



__global__ void launch_open_pipeline3(d_operator_t *op){
    variant::apply_visitor(at_open_op{}, op->op);
}


__global__ void launch_close_pipeline3(d_operator_t *op){
    variant::apply_visitor(at_close_op{}, op->op);
}

// __device__ void d_operator_t::open(){
//     variant::apply_visitor(open_op{}, op);
// }
__host__ void d_operator_t::open(){
    d_operator_t tmp;
    gpu(cudaMemcpy(&tmp, this, sizeof(d_operator_t), cudaMemcpyDefault));

    variant::apply_visitor(before_open_op{}, tmp.op);

    set_device_on_scope d(tmp.conf.device);
    launch_open_pipeline3<<<tmp.conf.gridDim, tmp.conf.blockDim, tmp.conf.shared_mem, 0>>>(this);
    gpu(cudaDeviceSynchronize());
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

__host__ void d_operator_t::close(){
    d_operator_t tmp;
    gpu(cudaMemcpy(&tmp, this, sizeof(d_operator_t), cudaMemcpyDefault));

    set_device_on_scope d(tmp.conf.device);
    launch_close_pipeline3<<<tmp.conf.gridDim, tmp.conf.blockDim, tmp.conf.shared_mem, 0>>>(this);
    gpu(cudaDeviceSynchronize());

    variant::apply_visitor(after_close_op{}, tmp.op);
}

__host__ d_operator_t::~d_operator_t(){
    // variant::apply_visitor(delete_op{}, op);
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
    // variant::apply_visitor(delete_op{}, op);
}
