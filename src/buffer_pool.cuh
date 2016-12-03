#ifndef BUFFER_POOL_CUH_
#define BUFFER_POOL_CUH_

#include "common.cuh"
#include "buffer.cuh"
// #include "optimistic_safe_device_stack.cuh"
#include "lockfree_stack.cuh"
#include <vector>

using namespace std;

// template<typename T, size_t buffer_capacity = DEFAULT_BUFF_CAP, typename buffer_t = buffer<T, buffer_capacity>, typename pool_t = optimistic_safe_device_stack<buffer_t *, (buffer_t *) NULL>>
// class buffer_pool;

template<typename T, size_t buffer_capacity = DEFAULT_BUFF_CAP, typename buffer_t = buffer<T, buffer_capacity>, typename pool_t = lockfree_stack<buffer_t *, (buffer_t *) NULL>>
class buffer_pool;

// template<typename T, size_t buffer_capacity = DEFAULT_BUFF_CAP>
// __global__ void acquire_buffer_for_host_blocked(buffer_pool<T, buffer_capacity> *self, buffer<T, buffer_capacity> ** ret){
//     printf("ttppp--pttt\n");
//     // assert(blockDim.x * blockDim.y * blockDim.z == 1);
//     // assert( gridDim.x *  gridDim.y *  gridDim.z == 1);
//     printf("ttppp--pttt\n");
//     __threadfence();
//     *ret = self->acquire_buffer_blocked();
// }

// __global__ void acquire_buffer_for_host_blocked(buffer_pool<vec4, 1024> *self, buffer<vec4, 1024> ** ret, size_t maxretries);
__global__ void register_producer_for_host(buffer_pool<int32_t, DEFAULT_BUFF_CAP> *self, void* prod);
__global__ void unregister_producer_for_host(buffer_pool<int32_t, DEFAULT_BUFF_CAP> *self, void* prod);
__global__ void acquire_buffer_blocked_unsafe_for_host(buffer_pool<int32_t, DEFAULT_BUFF_CAP> *self, buffer<int32_t, DEFAULT_BUFF_CAP> ** ret);
__global__ void release_buffer_host2(buffer_pool<int32_t, DEFAULT_BUFF_CAP> *self, buffer<int32_t, DEFAULT_BUFF_CAP> *buff);


template<typename T, size_t buffer_capacity, typename buff_t, typename pool_t>
class buffer_pool{
public:
    typedef buff_t buffer_t;

private:
public:
    pool_t             *pool;
    volatile uint32_t   producers;
    volatile uint32_t   available_buffers;

    size_t sz; //FIXME: remove

public:
    __host__ buffer_pool(size_t size, size_t fill_size, int device = 0) : producers(0), available_buffers(fill_size), sz(size){
        vector<buffer_t *> buffs;
        for (size_t i = 0 ; i < fill_size ; ++i) buffs.push_back(cuda_new<buffer_t>(device, device));
        pool = cuda_new<pool_t>(device, size, buffs, device);
    }

    __host__ ~buffer_pool(){
        // assert(pool->full());
        // while (!pool->empty()) cuda_delete(pool->pop_blocked());
        cuda_delete(pool);
    }

    // __device__ inline bool acquire_buffer_blocked_to(buffer_t** volatile ret, buffer_t** replaced){
    //     assert(__ballot(1) == 1);
    //     buffer_t * outbuff = *ret;
    //     if (replaced) *replaced = NULL;
    //     while (!outbuff->may_write()){
    //         buffer_t * tmp;
    //         if (buffer_manager::try_get_buffer(&tmp)){ //NOTE: we can not get the current buffer as answer, as it is not released yet
    //             // assert(tmp != outbuff); //NOTE: we can... it may have been released
    //             buffer_t * oldb = atomicCAS(ret, outbuff, tmp);
    //             if (oldb != outbuff) buffer_manager::release_buffer(tmp);
    //             else                {
    //                 atomicSub((uint32_t *) &available_buffers, 1);
    //                 if (replaced) *replaced = outbuff;
    //                 return true;
    //             }
    //         } else if (!producers && !available_buffers) return false;
    //         outbuff = *ret;
    //     }
    //     return true;
    // }

    __device__ inline buffer_t * acquire_buffer_blocked_unsafe(){
        assert(__ballot(1) == 1);
        int cnt = 0;
        while(cnt++ < 100) {
            buffer_t * tmp;
            if (pool->try_pop(&tmp)){ //NOTE: we can not get the current buffer as answer, as it is not released yet
                atomicSub((uint32_t *) &available_buffers, 1);
                printf("--------------------------->%d\n", cnt);
                return tmp;
            } else if (!producers && !available_buffers) return pool_t::get_invalid();
        }
        printf("--------------------------------_>timeout%d %d %d\n", cnt, producers, available_buffers);
        return (buffer_t *) 1;
    }


    __host__ inline buffer_t* h_acquire_buffer_blocked(buffer_t ** buff, buffer_t ** buff_ret, cudaStream_t strm){
        
        acquire_buffer_blocked_unsafe_for_host<<<1, 1, 0, strm>>>(this, buff_ret);
        
        // cudaMemcpyAsync(buff_ret, buff, sizeof(buffer_t *), cudaMemcpyDefault, strm);
        cudaStreamSynchronize(strm);
        return *buff_ret;



        // return run_on_device(this, &buffer_pool<T, buffer_capacity>::acquire_buffer_blocked_unsafe);
    }

    __host__ __device__ static bool is_valid(buffer_t * x){
        return pool_t::is_valid(x);
    }

    // __device__ inline bool try_acquire_buffer(buffer_t ** ret){
    //     if (pool->try_pop(ret)){
    //         printf("acq %llx from %llx\n", *ret, this);
    //         (*ret)->clean();
    //         return true;
    //     }
    //     return false;
    // }

    __device__ inline void release_buffer(buffer_t * buff){
        printf("rel %llx to %llx\n", buff, this);
        assert(producers);
        atomicAdd((uint32_t *) &available_buffers, 1);
        pool->push(buff);
    }

    __host__ inline void h_release_buffer(buffer_t * buff, cudaStream_t strm){
        release_buffer_host2<<<1, 1, 0, strm>>>(this, buff);
    }

    __host__ __device__  inline void register_producer(void * producer){
#ifdef __CUDA_ARCH__
        printf("------------------------------------------------------------\n");
        atomicAdd((uint32_t *) &producers, 1);
#else
        cudaPointerAttributes attrs;
        gpu(cudaPointerGetAttributes(&attrs, this));
        set_device_on_scope d(attrs.device);
        cudaStream_t strm;
        cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking);
        register_producer_for_host<<<1, 1, 0, strm>>>(this, producer);

        cudaStreamSynchronize(strm);
        cudaStreamDestroy(strm);
#endif
    }

    __host__ __device__ inline void unregister_producer(void * producer){
#ifdef __CUDA_ARCH__
        assert(producers);
        printf("------------------------------------------------------------++\n");
        atomicSub((uint32_t *) &producers, 1);
#else
        cudaPointerAttributes attrs;
        gpu(cudaPointerGetAttributes(&attrs, this));
        set_device_on_scope d(attrs.device);
        cudaStream_t strm;
        cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking);
        unregister_producer_for_host<<<1, 1, 0, strm>>>(this, producer);

        cudaStreamSynchronize(strm);
        cudaStreamDestroy(strm);
#endif
    }

    __device__ inline size_t size() const{
        return pool->size();
    }
};

#endif /* BUFFER_POOL_CUH_ */