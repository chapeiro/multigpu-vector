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


extern cpu_set_t                                          *gpu_affinity;
extern cpu_set_t                                          *cpu_numa_affinity;

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


    static mutex                                              *devive_buffs_mutex;
    static vector<buffer_t *>                                 *device_buffs_pool;
    static buffer_t                                         ***device_buff;
    static int                                                 device_buff_size;
    static int                                                 keep_threshold;

    static cudaStream_t                                       *release_streams;

    static threadsafe_stack<buffer_t *, (buffer_t *) NULL>   **h_pool;
    static threadsafe_stack<buffer_t *, (buffer_t *) NULL>   **h_pool_numa;

    static unordered_map<buffer_t *, T *>                      buffer_cache;




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

    static __host__ inline buffer_t * h_get_buffer(int dev);

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
            assert(buffer_cache.find(buff) != buffer_cache.end());
            buff->data = buffer_cache.find(buff)->second;
            assert(buff->device < 0);
            assert(get_device(buff->data) < 0);
            int status[1];
            int ret_code;
            status[0]=-1;
            ret_code=move_pages(0 /*self memory */, 1, (void **) &buff->data, NULL, status, 0);
            // printf("-=Memory at %p is at %d node (retcode %d) cpu: %d\n", buff->data, status[0], ret_code, sched_getcpu());
            assert(ret_code == 0);

            printf("===============================================================> %d %p %d %d\n", buff->device, buff->data, get_device(buff->data), status[0]);
            h_pool_numa[status[0]]->push(buff);
            printf("%d %p %d\n", buff->device, buff->data, status[0]);
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


template<typename T>
threadsafe_stack<buffer<int32_t, DEFAULT_BUFF_CAP, vec4>  *, (buffer<int32_t, DEFAULT_BUFF_CAP, vec4>  *) NULL> ** buffer_manager<T>::h_pool;

template<typename T>
threadsafe_stack<buffer<int32_t, DEFAULT_BUFF_CAP, vec4>  *, (buffer<int32_t, DEFAULT_BUFF_CAP, vec4>  *) NULL> ** buffer_manager<T>::h_pool_numa;

template<typename T>
unordered_map<buffer<int32_t, DEFAULT_BUFF_CAP, vec4> *, T *> buffer_manager<T>::buffer_cache;

template<typename T>
mutex                                              *buffer_manager<T>::devive_buffs_mutex;

template<typename T>
vector<buffer<int32_t, DEFAULT_BUFF_CAP, vec4>  *> *buffer_manager<T>::device_buffs_pool;

template<typename T>
buffer<int32_t, DEFAULT_BUFF_CAP, vec4>          ***buffer_manager<T>::device_buff;

template<typename T>
int                                                 buffer_manager<T>::device_buff_size;

template<typename T>
int                                                 buffer_manager<T>::keep_threshold;

template<typename T>
cudaStream_t                                       *buffer_manager<T>::release_streams;

#endif /* BUFFER_MANAGER_CUH_ */