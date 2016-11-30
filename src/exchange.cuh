#ifndef EXCHANGE_CUH_
#define EXCHANGE_CUH_

#include <cstdint>
#include <iostream>
#include <cassert>
#include <vector>
#include "common.cuh"

using namespace std;

struct producer_conf{
    int             device;
    cudaStream_t    stream;
    size_t          n_buffs;
}

struct consumer_conf{
    int device;
    cudaStream_t stream;
}

template<size_t warp_size = WARPSIZE, typename T = int32_t>
class producer{
    static_assert(sizeof(T)*8 == 32, "Operator is only implemented for data types of 32-bits size");

    T*          buff_storage;
    uint32_t   *buff_elems  ; //NOTE: will we have a locality problem ? does cude prefetch ? I do not think so...

    T*         *cons_returns;
    uint32_t   *cons_tags   ;

public:
    producer(size_t buffs, size_t buffer_size, size_t consumers, cudaStream_t stream, int dev){
        set_device_on_scope d(dev);

        size_t loc_mem = buffs * (buffer_size + 1);     //memory for buffers and counts of buffers

        size_t rem_mem = consumers * warp_size;   //consumer data queue
        size_t tag_mem = consumers;               //consumer data queue

        gpu(cudaMalloc((void**)&buff_storage, loc_mem * sizeof(T )));

        buff_elems = buff_storage + buffs * buffer_size;

        gpu(cudaMalloc((void**)&cons_returns , rem_mem * sizeof(T*)));
        gpu(cudaMalloc((void**)&cons_tags    , consumers * sizeof(uint32_t)));

        gpu(cudaMemset(cons_returns, 0, rem_mem   * sizeof(T*)      , stream));
        gpu(cudaMemset(cons_tags   , 0, consumers * sizeof(uint32_t), stream));

    }


}



template<size_t warp_size = WARPSIZE, typename T = int32_t>
class exchange{
    static_assert(sizeof(T)*8 == 32, "Operator is only implemented for data types of 32-bits size");
private:
    T                 *buffer       ;
    uint32_t          *counters     ;

    const size_t       grid_size    ;
    const size_t       shared_mem   ;
    const dim3         dimGrid      ;
    const dim3         dimBlock     ;
    const cudaStream_t stream       ;
    const int          device       ;
public:
    exchange(const vector<producer_conf> &producers, const vector<consumer_conf> &consumers, size_t buffer_size){
            for (const auto &prod: producers){
                set_device_on_scope d(prod.device);

                size_t loc_mem = buffer_size * prod.buffs; //memory for buffers
                loc_mem       += buffs[i];                 //counts for buffers

                size_t rem_mem = consumers * warp_size;    //consumer data queue
                rem_mem       += consumers;                //consumer data tag

                gpu(cudaMalloc((void**)&buffer, ((grid_size + 1) * 4 * warp_size + 3) * sizeof(T)));
                gpu(cudaMalloc((void**)&buffer, ((grid_size + 1) * 4 * warp_size + 3) * sizeof(T)));

            }
            set_device_on_scope d(device);

            counters = (uint32_t *) (buffer + (grid_size + 1) * 4 * warp_size);

            gpu(cudaMemsetAsync(buffer + (grid_size + 1) * (4 * warp_size - 1), 0, (grid_size + 4) * sizeof(T), stream));
        }

    uint32_t next(T *dst, T *src = NULL, uint32_t N = 0);

    ~exchange(){
        set_device_on_scope d(device);
        gpu(cudaFree(buffer));
    }
};

#endif /* EXCHANGE_CUH_ */