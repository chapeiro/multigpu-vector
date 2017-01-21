#ifndef EXCHANGE_CUH_
#define EXCHANGE_CUH_

#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include "operator.cuh"
#include "buffer_pool.cuh"

// typedef void (Operator::*con)(buffer_pool<int32_t>::buffer_t *);

typedef buffer_pool<int32_t> buffer_pool_t;


class producer{
public:
    typedef buffer_pool_t::buffer_t buffer_t;
private:
    buffer_pool_t * outpool;
    exchange      * exc;


public:
    __host__ producer(buffer_pool_t * outpool): outpool(outpool), exc(NULL){
        if (outpool) outpool->register_producer(this);
    }
    __host__ producer(exchange      * exc)    : outpool(NULL   ), exc(exc ){}

    __host__ ~producer();

    __host__ __device__ void open(){};
    __host__ __device__ void close();

    __host__ __device__ void consume(buffer_t * data);
};

class consumer{
public:
    typedef buffer_pool_t::buffer_t buffer_t;
private:
    int             device;
    int             shared_mem;
    cudaStream_t    strm;
    cudaStream_t    strm2;
    dim3            dimGrid;
    dim3            dimBlock;
    p_operator_t    parent;
    vector<thread>  execs;
    
public:
    __host__ consumer(p_operator_t parent, dim3 dimGrid, dim3 dimBlock, int shared_mem);

    __host__ void open();

    __host__ void consume(buffer_t * data);

    __host__ void close();

    __host__ ~consumer();
};




class exchange{
public:
    typedef buffer_pool<int32_t> buffer_pool_t;

public:
    vector<thread>                      firers;
    
    atomic<int>                         remaining_producers;

    vector<buffer_pool_t::buffer_t *>   ready_pool;
    mutex                               ready_pool_mutex;
    condition_variable                  ready_pool_cv;

    __host__ void poll(buffer_pool_t * target);
    __host__ void fire(consumer *cons);
public:
    exchange(const vector<p_operator_t> &parents,
                const vector<launch_conf> &parent_conf);

public: //FIXME: friends...
    __host__ void set_ready(buffer_pool_t::buffer_t * buff);
    __host__ buffer_pool_t::buffer_t * get_ready();
    __host__ void producer_ended();

public:
    __host__ void open();

    __host__ void consume(buffer_t * data);

    __host__ void close();

    ~exchange();
};

#endif /* EXCHANGE_CUH_ */