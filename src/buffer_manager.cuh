#ifndef BUFFER_MANAGER_CUH_
#define BUFFER_MANAGER_CUH_

#include "buffer.cuh"
#include "lockfree_stack.cuh"
#include <type_traits>

extern __device__ __constant__ lockfree_stack<buffer<int32_t, DEFAULT_BUFF_CAP, vec4>  *, (buffer<int32_t, DEFAULT_BUFF_CAP, vec4>  *) NULL> * pool;
extern __device__ __constant__ int deviceId;

template<typename T>
class buffer_manager;

__global__ void release_buffer_host(buffer<int32_t, DEFAULT_BUFF_CAP> *buff);
__global__ void get_buffer_host    (buffer<int32_t, DEFAULT_BUFF_CAP> **buff);

template<typename T = int32_t>
class buffer_manager{
    static_assert(std::is_same<T, int32_t>::value, "Not implemented yet");
public:
    typedef buffer<int32_t, DEFAULT_BUFF_CAP, vec4> buffer_t;
    typedef lockfree_stack<buffer_t *, (buffer<int32_t, DEFAULT_BUFF_CAP, vec4>  *) NULL>        pool_t;

    static __host__ void init(int size = 1024){
        int devices;
        gpu(cudaGetDeviceCount(&devices));
        for (int j = 0; j < devices; ++j) {
            set_device_on_scope d(j);

            vector<buffer_t *> buffs;
            for (size_t i = 0 ; i < size ; ++i) buffs.push_back(cuda_new<buffer_t>(j, j));

            pool_t * tmp =  cuda_new<pool_t>(j, size, buffs, j);
            gpu(cudaMemcpyToSymbol(pool    , &tmp, sizeof(pool_t *)));
            gpu(cudaMemcpyToSymbol(deviceId,   &j, sizeof(int     )));
        }
    }

    static __device__ buffer_t * get_buffer(){
        return pool->pop();
    }

    static __device__ bool try_get_buffer(buffer_t **ret){
        if (pool->try_pop(ret)){
            (*ret)->clean();
            return true;
        }
        return false;
    }

    static __host__ inline void h_get_buffer(buffer_t **buff, cudaStream_t strm, int dev){
        set_device_on_scope d(dev);
        get_buffer_host<<<1, 1, 0, strm>>>(buff);
        cudaStreamSynchronize(strm);
    }

    static __host__ __device__ void release_buffer(buffer_t * buff, cudaStream_t strm = 0){
#ifdef __CUDA_ARCH__
        assert(strm == 0); //FIXME: something better ?
        if (buff->device == deviceId) pool->push(buff);
        else                          assert(false); //FIXME: IMPORTANT free buffer of another device!
#else
        cudaPointerAttributes attrs;
        cudaPointerGetAttributes(&attrs, buff);
        set_device_on_scope d(attrs.device);
        release_buffer_host<<<1, 1, 0, strm>>>(buff);
#endif
    }

    static __device__ inline bool acquire_buffer_blocked_to(buffer_t** volatile ret, buffer_t** replaced){
        assert(__ballot(1) == 1);
        buffer_t * outbuff = *ret;
        if (replaced) *replaced = NULL;
        while (!outbuff->may_write()){
            buffer_t * tmp;
            if (try_get_buffer(&tmp)){ //NOTE: we can not get the current buffer as answer, as it is not released yet
                // assert(tmp != outbuff); //NOTE: we can... it may have been released
                buffer_t * oldb = atomicCAS(ret, outbuff, tmp); //FIXME: ABA problem
                if (oldb != outbuff) release_buffer(tmp);
                else                {
                    // atomicSub((uint32_t *) &available_buffers, 1);
                    if (replaced) *replaced = outbuff;
                    return true;
                }
            } //else if (!producers && !available_buffers) return false;
            outbuff = *ret;
        }
        return true;
    }

    static __host__ void destroy(); //FIXME: cleanup...
};

#endif /* BUFFER_MANAGER_CUH_ */