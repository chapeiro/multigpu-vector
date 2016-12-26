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

typedef buffer_pool<int32_t> buffer_pool_t;

typedef buffer_pool_t::buffer_t buffer_t;

// typedef variant::variant<generator *, materializer *, producer *> Operator;


// struct close{
//     template<typename T>
//     __host__ __device__ void operator()(T op) const;
// };

class Operator{
private:
    typedef variant::variant<generator *, materializer *, producer *, unstable_select<32, int32_t> *> op_t;
    op_t op;

public:
    template<typename Op>
    __host__ Operator(Op * op): op(op){
    }

    // __device__ __host__ void open();

    __device__ __host__ void consume(buffer_t * buff);
    
    __device__ __host__ void close();

    __host__ ~Operator();
};
// template<typename T>
// class Operator{
// protected:
//     vector<Operator *> parents;

// public:
//     Operator(const vector<Operator *> &parents): parents(parents){}

//     // virtual __host__ int getSharedMemoryRequirements(){
//     //     int s = 0;
//     //     for (const auto &p: parents) if (p) s += 0;//FIXME: may be invoked on object of another device eg CPU2GPU p->getSharedMemoryRequirements();
//     //     return s;
//     // }

//     __host__ __device__ void consume(buffer_pool<int32_t>::buffer_t * data){
//         T::consume(data);
//     }
// };

#endif /* OPERATOR_CUH_ */