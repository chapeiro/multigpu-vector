#ifndef BUFFER_MANAGER_CUH_
#define BUFFER_MANAGER_CUH_

#include "buffer.cuh"
#include "lockfree_stack.cuh"
#include <type_traits>
#include "threadsafe_stack.cuh"

extern __device__ __constant__ lockfree_stack<buffer<int32_t, DEFAULT_BUFF_CAP, vec4>  *, (buffer<int32_t, DEFAULT_BUFF_CAP, vec4>  *) NULL> * pool;
extern __device__ __constant__ int deviceId;

extern threadsafe_stack<buffer<int32_t, DEFAULT_BUFF_CAP, vec4>  *, (buffer<int32_t, DEFAULT_BUFF_CAP, vec4>  *) NULL> * h_pool;

template<typename T>
class buffer_manager;

__global__ void release_buffer_host(buffer<int32_t, DEFAULT_BUFF_CAP> *buff);
__global__ void get_buffer_host    (buffer<int32_t, DEFAULT_BUFF_CAP> **buff, int buffs = 1);

template<typename T = int32_t>
class buffer_manager{
    static_assert(std::is_same<T, int32_t>::value, "Not implemented yet");
public:
    typedef buffer<int32_t, DEFAULT_BUFF_CAP, vec4> buffer_t;
    typedef lockfree_stack<buffer_t *, (buffer<int32_t, DEFAULT_BUFF_CAP, vec4>  *) NULL>        pool_t;
    typedef threadsafe_stack<buffer_t *, (buffer<int32_t, DEFAULT_BUFF_CAP, vec4>  *) NULL>      h_pool_t;

    static __host__ void init(int size = 1024){
        int devices;
        gpu(cudaGetDeviceCount(&devices));
        for (int j = 0; j < devices; ++j) {
            set_device_on_scope d(j);

            for (int i = 0 ; i < devices ; ++i) {
                if (i != j) {
                    int t;
                    cudaDeviceCanAccessPeer(&t, j, i);
                    if (t){
                        cudaDeviceEnablePeerAccess(i, 0);
                    } else {
                        cout << "Warning: P2P disabled for : " << j << "->" << i << endl;
                    }
                }
            }


            vector<buffer_t *> buffs;
            for (size_t i = 0 ; i < size ; ++i) buffs.push_back(cuda_new<buffer_t>(j, j));

            pool_t * tmp =  cuda_new<pool_t>(j, size, buffs, j);
            gpu(cudaMemcpyToSymbol(pool    , &tmp, sizeof(pool_t *)));
            gpu(cudaMemcpyToSymbol(deviceId,   &j, sizeof(int     )));
        }
        vector<buffer_t *> buffs;
        for (size_t i = 0 ; i < size ; ++i) buffs.push_back(cuda_new<buffer_t>(-1, -1));
        h_pool = new h_pool_t(size, buffs);
    }

    static __host__ __device__ buffer_t * get_buffer(){
#ifdef __CUDA_ARCH__
        return pool->pop();
#else
        return h_pool->pop();
#endif
    }

    static __device__ bool try_get_buffer(buffer_t **ret){
#ifdef __CUDA_ARCH__
        if (pool->try_pop(ret)){
#else
        if (h_pool->try_pop(ret)){
#endif
            (*ret)->clean();
            return true;
        }
        return false;
    }

    static __host__ inline void h_get_buffer(buffer_t **buff, cudaStream_t strm, int dev, int buffs = 1){
        set_device_on_scope d(dev);
        get_buffer_host<<<1, 1, 0, strm>>>(buff, buffs);
        cudaStreamSynchronize(strm);
    }

    static __host__ __device__ void release_buffer(buffer_t * buff, cudaStream_t strm){
#ifdef __CUDA_ARCH__
        assert(strm == 0); //FIXME: something better ?
        if (buff->device == deviceId) pool->push(buff);
        else                          assert(false); //FIXME: IMPORTANT free buffer of another device (or host)!
#else
        int dev = get_device(buff);
        if (dev >= 0){
            set_device_on_scope d(dev);
            release_buffer_host<<<1, 1, 0, strm>>>(buff);
            gpu(cudaPeekAtLastError()  );
            gpu(cudaDeviceSynchronize());
        } else {
            h_pool->push(buff);
        }
#endif
    }

    static __host__ __device__ void release_buffer(buffer_t * buff){
#ifdef __CUDA_ARCH__
        release_buffer(buff, 0);
#else
        cudaStream_t strm = 0;
        int dev = get_device(buff);
        if (dev >= 0){
            set_device_on_scope d(dev);

            gpu(cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking));
        }
        release_buffer(buff, strm);
        if (dev >= 0) gpu(cudaStreamDestroy(strm));
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