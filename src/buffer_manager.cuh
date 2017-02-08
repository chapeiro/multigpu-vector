#ifndef BUFFER_MANAGER_CUH_
#define BUFFER_MANAGER_CUH_

#include "buffer.cuh"
#include "threadsafe_device_stack.cuh"
#include <type_traits>
#include "threadsafe_stack.cuh"
#include <mutex>
#include <vector>
#include <numaif.h>
#include <unordered_map>

using namespace std;

extern __device__ __constant__ threadsafe_device_stack<buffer<int32_t, DEFAULT_BUFF_CAP, vec4>  *, (buffer<int32_t, DEFAULT_BUFF_CAP, vec4>  *) NULL> * pool;
extern __device__ __constant__ int deviceId;

extern mutex                                              *devive_buffs_mutex;
extern vector<buffer<int32_t, DEFAULT_BUFF_CAP, vec4>  *> *device_buffs_pool;
extern buffer<int32_t, DEFAULT_BUFF_CAP, vec4>          ***device_buff;
extern int                                                 device_buff_size;
extern int                                                 keep_threshold;

extern cudaStream_t                                       *release_streams;

extern cpu_set_t                                          *gpu_affinity;
extern cpu_set_t                                          *cpu_numa_affinity;

extern threadsafe_stack<buffer<int32_t, DEFAULT_BUFF_CAP, vec4>  *, (buffer<int32_t, DEFAULT_BUFF_CAP, vec4>  *) NULL> ** h_pool;
extern threadsafe_stack<buffer<int32_t, DEFAULT_BUFF_CAP, vec4>  *, (buffer<int32_t, DEFAULT_BUFF_CAP, vec4>  *) NULL> ** h_pool_numa;

extern unordered_map<buffer<int32_t, DEFAULT_BUFF_CAP, vec4> *, int32_t *> buffer_cache;

template<typename T>
class buffer_manager;

__global__ void release_buffer_host(buffer<int32_t, DEFAULT_BUFF_CAP> **buff, int buffs = 1);
__global__ void get_buffer_host    (buffer<int32_t, DEFAULT_BUFF_CAP> **buff, int buffs = 1);

template<typename T = int32_t>
class buffer_manager{
    static_assert(std::is_same<T, int32_t>::value, "Not implemented yet");
public:
    typedef buffer<int32_t, DEFAULT_BUFF_CAP, vec4> buffer_t;
    typedef threadsafe_device_stack<buffer_t *, (buffer_t *) NULL>        pool_t;
    typedef threadsafe_stack       <buffer_t *, (buffer_t *) NULL>      h_pool_t;

    static __host__ void init(int size = 64, int buff_buffer_size = 8, int buff_keep_threshold = 16);

    static __host__ __device__ buffer_t * get_buffer(){
#ifdef __CUDA_ARCH__
        return pool->pop();
#else
        cout << "+" << h_pool[sched_getcpu()] << " " << sched_getcpu() << endl;

        auto buff = h_pool[sched_getcpu()]->pop();
            int status[1];
            int ret_code;
            status[0]=-1;
            ret_code=move_pages(0 /*self memory */, 1, (void **) &buff->data, NULL, status, 0);
            printf("get_buffer: Memory at %p is at %d node (retcode %d) cpu: %d\n", buff->data, status[0], ret_code, sched_getcpu());
        return buff;
#endif
    }

    static __device__ bool try_get_buffer2(buffer_t **ret){
#ifdef __CUDA_ARCH__
        if (pool->try_pop(ret)){
#else
        if (h_pool[sched_getcpu()]->try_pop(ret)){
#endif
            (*ret)->clean();
            return true;
        }
        return false;
    }

    static __host__ inline buffer_t * h_get_buffer(int dev){
        if (dev >= 0){
            unique_lock<std::mutex> lock(devive_buffs_mutex[dev]);
            if (device_buffs_pool[dev].empty()){
                set_device_on_scope d(dev);
                get_buffer_host<<<1, 1, 0, release_streams[dev]>>>(device_buff[dev], device_buff_size);
                gpu(cudaStreamSynchronize(release_streams[dev]));
                device_buffs_pool[dev].insert(device_buffs_pool[dev].end(), device_buff[dev], device_buff[dev]+device_buff_size);
                // gpu(cudaFreeHost(buff));
            }
            buffer_t * ret = device_buffs_pool[dev].back();
            device_buffs_pool[dev].pop_back();
            return ret;
        } else {
            return get_buffer();
        }
    }

    static __host__ __device__ void release_buffer(buffer_t * buff){//, cudaStream_t strm){
#ifdef __CUDA_ARCH__
        // assert(strm == 0); //FIXME: something better ?
        if (buff->device == deviceId) {
            buff->clean();
            __threadfence();
            pool->push(buff);
        } else                          assert(false); //FIXME: IMPORTANT free buffer of another device (or host)!
#else
        int dev = get_device(buff);
        if (dev >= 0){
            set_device_on_scope d(dev);
            unique_lock<std::mutex> lock(devive_buffs_mutex[dev]);
            device_buffs_pool[dev].push_back(buff);
            size_t size = device_buffs_pool[dev].size();
            if (size > keep_threshold){
                for (int i = 0 ; i < device_buff_size ; ++i) device_buff[dev][i] = device_buffs_pool[dev][size-i-1];
                device_buffs_pool[dev].erase(device_buffs_pool[dev].end()-device_buff_size, device_buffs_pool[dev].end());
                release_buffer_host<<<1, 1, 0, release_streams[dev]>>>(device_buff[dev], device_buff_size);
                gpu(cudaStreamSynchronize(release_streams[dev]));
                // gpu(cudaPeekAtLastError()  );
                // gpu(cudaDeviceSynchronize());
            }
        } else {
            int status[1];
            int ret_code;
            status[0]=-1;
            ret_code=move_pages(0 /*self memory */, 1, (void **) &buff->data, NULL, status, 0);
            // printf("-=Memory at %p is at %d node (retcode %d) cpu: %d\n", buff->data, status[0], ret_code, sched_getcpu());

            h_pool_numa[status[0]]->push(buff);
        }
#endif
    }

//     static __host__ __device__ void release_buffer(buffer_t * buff){
// #ifdef __CUDA_ARCH__
//         release_buffer(buff, 0);
// #else
//         cudaStream_t strm = 0;
//         int dev = get_device(buff);
//         if (dev >= 0){
//             set_device_on_scope d(dev);

//             gpu(cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking));
//         }
//         release_buffer(buff, strm);
//         if (dev >= 0) gpu(cudaStreamDestroy(strm));
// #endif
//     }

    static __device__ inline bool acquire_buffer_blocked_to(buffer_t** volatile ret, buffer_t** replaced){
        assert(__ballot(1) == 1);
        buffer_t * outbuff = *ret;
        if (replaced) *replaced = NULL;
        while (!outbuff->may_write()){
            buffer_t * tmp;
            if (pool->try_pop(&tmp)){ //NOTE: we can not get the current buffer as answer, as it is not released yet
                // assert(tmp->cnt == 0);
                tmp->clean();
                __threadfence();
                // if (tmp == outbuff) {
                //     if (*ret != outbuff) release_buffer(tmp);
                //     // else assert(false);
                //     // assert(*ret != outbuff); //FIXME: Should hold! but does not!!!
                //     // release_buffer(tmp);
                // } else {
                    // assert(tmp != outbuff); //NOTE: we can... it may have been released
                    buffer_t * oldb = atomicCAS(ret, outbuff, tmp); //FIXME: ABA problem
                    if (oldb != outbuff) release_buffer(tmp);
                    else                {
                        // atomicSub((uint32_t *) &available_buffers, 1);
                        if (replaced) *replaced = outbuff;
                        return true;
                    }
                // }
            } //else if (!producers && !available_buffers) return false;
            outbuff = *ret;
        }
        return true;
    }

    static __host__ void overwrite(buffer_t * buff, const T * data, uint32_t N, cudaStream_t strm, bool blocking = true);

    static __host__ void destroy(); //FIXME: cleanup...
};

#endif /* BUFFER_MANAGER_CUH_ */