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

    __host__ __device__ void join();

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
    Operator       *parent;
    buffer_t      **buff;
    
public:
    __host__ consumer(Operator *parent, dim3 dimGrid, dim3 dimBlock, int shared_mem);

    __host__ __device__ void consume(buffer_t * data);

    __host__ __device__ void join();

    __host__ ~consumer();
};


class exchange{
public:
    typedef buffer_pool<int32_t> buffer_pool_t;

public:
    vector<buffer_pool_t *>             prod_output_holders;
    vector<producer      *>             prods;
    vector<thread>                      pollers;
    vector<thread>                      firers;

    atomic<int>                         remaining_producers;

    vector<buffer_pool_t::buffer_t *>   ready_pool;
    mutex                               ready_pool_mutex;
    condition_variable                  ready_pool_cv;

    __host__ void poll(buffer_pool_t * target);
    __host__ void fire(consumer *cons);
public:
    exchange(const vector<int> &prod_loc, const vector<int> &prodout_loc, 
                    const vector<int> &prodout_size, const vector<int> &prod2out,
                    const vector<Operator *> &parents,
                    const vector<dim3> &parent_dimGrid,
                    const vector<dim3> &parent_dimBlock,
                    const vector<int> &shared_mem);

public: //FIXME: friends...
    __host__ void set_ready(buffer_pool_t::buffer_t * buff);
    __host__ buffer_pool_t::buffer_t * get_ready();
    __host__ void producer_ended();

public:
    void join();
    ~exchange();
};

#endif /* EXCHANGE_CUH_ */