#include "buffer_manager.cuh"

#include <thread>
#include <utmpx.h>
#include <unistd.h>

#include "numa_utils.cuh"

#include <cinttypes>

// __device__ __constant__ threadsafe_device_stack<int32_t *, (int32_t *) NULL> * pool;
// __device__ __constant__ int deviceId;
// __device__ __constant__ void * buff_start;
// __device__ __constant__ void * buff_end  ;

#include <cstdio>
#include <cinttypes>
#include "common/gpu/gpu-common.hpp"
#include "multigpu/buffer_manager.cuh"

extern "C"{

__device__ void dprinti64(int64_t x){
    printf("%" PRId64 "\n", x);
}

__device__ int32_t * get_buffers(){
    uint32_t b = __ballot(1);
    uint32_t m = 1 << get_laneid();
    int32_t * ret;
    do {
        uint32_t leader = b & -b;

        if (leader == m) ret = buffer_manager<int32_t>::get_buffer();

        b ^= leader;
    } while (b);
    return ret;
}

__device__ void release_buffers(int32_t * buff){
    uint32_t b = __ballot(buff != NULL);
    uint32_t m = 1 << get_laneid();
    do {
        uint32_t leader = b & -b;

        if (leader == m) buffer_manager<int32_t>::release_buffer(buff);

        b ^= leader;
    } while (b);
}

}

void initializeModule(CUmodule & cudaModule){
    CUdeviceptr ptr  ;
    size_t      bytes;
    void *      mem  ;

    gpu(cuModuleGetGlobal(&ptr, &bytes, cudaModule, "pool"));
    gpu(cudaMemcpyFromSymbol(&mem, pool      , sizeof(void   *)));
    gpu(cuMemcpyHtoD        (ptr , &mem      , sizeof(void   *)));

    gpu(cuModuleGetGlobal(&ptr, &bytes, cudaModule, "buff_start"));
    gpu(cudaMemcpyFromSymbol(&mem, buff_start, sizeof(void   *)));
    gpu(cuMemcpyHtoD        (ptr , &mem      , sizeof(void   *)));

    gpu(cuModuleGetGlobal(&ptr, &bytes, cudaModule, "buff_end"));
    gpu(cudaMemcpyFromSymbol(&mem, buff_end, sizeof(void   *)));
    gpu(cuMemcpyHtoD        (ptr , &mem      , sizeof(void   *)));

    gpu(cuModuleGetGlobal(&ptr, &bytes, cudaModule, "deviceId"));
    gpu(cudaMemcpyFromSymbol(&mem, deviceId  , sizeof(int)));
    gpu(cuMemcpyHtoD        (ptr , &mem      , sizeof(int)));
}


__global__ void release_buffer_host(void **buff, int buffs){
    assert(blockDim.x * blockDim.y * blockDim.z == 1);
    assert( gridDim.x *  gridDim.y *  gridDim.z == 1);
    for (int i = 0 ; i < buffs ; ++i) buffer_manager<int32_t>::release_buffer((int32_t *) buff[i]);
}

__global__ void get_buffer_host(void **buff, int buffs){
    assert(blockDim.x * blockDim.y * blockDim.z == 1);
    assert( gridDim.x *  gridDim.y *  gridDim.z == 1);
    for (int i = 0 ; i < buffs ; ++i) buff[i] = buffer_manager<int32_t>::get_buffer();
}

int                                                 cpu_cnt;
cpu_set_t                                          *gpu_affinity;
cpu_set_t                                          *cpu_numa_affinity;
int                                                *gpu_numa_node;

#if defined(__clang__) && defined(__CUDA__)
template<typename T>
__device__ T * buffer_manager<T>::get_buffer(){
    return pool->pop();
}

template<typename T>
__host__   T * buffer_manager<T>::get_buffer(){
    return get_buffer_numa(sched_getcpu());
}
#else
template<typename T>
__host__ __device__ T * buffer_manager<T>::get_buffer(){
#ifdef __CUDA_ARCH__
    return pool->pop();
#else
    return get_buffer_numa(sched_getcpu());
#endif
}
#endif

template<typename T>
__host__ void buffer_manager<T>::init(int size, int h_size, int buff_buffer_size, int buff_keep_threshold){
    int devices = get_num_of_gpus();
    buffer_manager<T>::h_size = h_size;
    
    long cores = sysconf(_SC_NPROCESSORS_ONLN);
    assert(cores > 0);
    cpu_cnt = cores;

    for (int i = 0 ; i < devices ; ++i) {
        gpu_run(cudaSetDevice(i));
        gpu_run(cudaFree(0));
    }
    
    gpu_run(cudaSetDevice(0));


    // P2P check & enable
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
    }

    //FIXME: Generalize
    int cpu_numa_nodes = numa_num_task_nodes();

    // std::cout << "CPU numa nodes : " << cpu_numa_nodes << std::endl;
    // std::cout << "CPU cores      : " << cores << std::endl;
    // std::cout << "GPU devices    : " << devices << std::endl;

    terminating        = false;
    device_buffs_mutex = new mutex              [devices];
    device_buffs_cv    = new condition_variable [devices];
    device_buffs_thrds = new thread *           [devices];
    device_buffs_pool  = new vector<T *>        [devices];
    release_streams    = new cudaStream_t       [devices];

    h_pool             = new h_pool_t *         [cores  ];
    h_pool_numa        = new h_pool_t *         [cpu_numa_nodes];

    h_buff_start       = new void              *[devices];
    h_buff_end         = new void              *[devices];

    h_h_buff_start     = new void              *[cpu_numa_nodes];

    device_buff        = new T **[devices];
    device_buff_size   = buff_buffer_size;
    keep_threshold     = buff_keep_threshold;

    buffer_cache.clear();

    // gpu(cudaMallocHost(&tmp, device_buff_size*sizeof(buffer_t *)*devices));
    // for (int i = 0 ; i < devices ; ++i) {
        // device_buff[i] = tmp + device_buff_size*sizeof(buffer_t *)*i;
    // }

    nvmlInit();
    unsigned int device_count;
    nvmlDeviceGetCount(&device_count);
    assert(device_count == devices && "NMVL disagrees with cuda about the number of GPUs");


    gpu_affinity       = new cpu_set_t[devices];
    cpu_numa_affinity  = new cpu_set_t[cpu_numa_nodes];
    gpu_numa_node      = new int      [devices];

    for (int j = 0 ; j < devices        ; ++j) CPU_ZERO(&gpu_affinity[j]);
    for (int j = 0 ; j < cpu_numa_nodes ; ++j) CPU_ZERO(&cpu_numa_affinity[j]);

    // diascld36
    // //FIXME: Generalize
    // for (int i = 0 ; i < 48 ; i += 2) CPU_SET(i, &gpu_affinity[0]);
    // for (int i = 1 ; i < 48 ; i += 2) CPU_SET(i, &gpu_affinity[1]);

    // //FIXME: Generalize
    // cpu_numa_affinity  = new cpu_set_t[cpu_numa_nodes];
    // for (int i = 0 ; i < 48 ; i += 2) CPU_SET(i, &cpu_numa_affinity[0]);
    // for (int i = 1 ; i < 48 ; i += 2) CPU_SET(i, &cpu_numa_affinity[1]);

    for (int j = 0 ; j < devices ; ++j){
        int sets = ((cores + 63) / 64);
        uint64_t cpuSet[sets];
        for (int i = 0 ; i < sets ; ++i) cpuSet[i] = 0;
        nvmlDevice_t device;

        nvmlDeviceGetHandleByIndex(j, &device);
        nvmlDeviceGetCpuAffinity(device, sets, cpuSet);
        for (int i = 0 ; i < sets ; ++i){
            for (int k = 0 ; k < 64 ; ++k){
                if ((cpuSet[i] >> k) & 1){
                    CPU_SET(64 * i + k, &gpu_affinity[j]);
                }
            }
        }
    }

    for (int j = 0 ; j < cores ; ++j){
        CPU_SET(j, &cpu_numa_affinity[numa_node_of_cpu(j)]);
    }

    //numa_node_of_cpu must be set prior to this
    for (int j = 0 ; j < devices        ; ++j) gpu_numa_node[j] = calc_numa_node_of_gpu(j);

    // for (int i = 0 ; i < cores ; ++i){
    //     std::cout << "CPU " << i << " local to GPU ";
    //     for (int j = 0 ; j < devices ; ++j){
    //         if (CPU_ISSET(i, &gpu_affinity[j])) std::cout << j;
    //     }
    //     std::cout << std::endl;
    // }

    // for (int i = 0 ; i < cores ; ++i){
    //     std::cout << "CPU " << i << " local to NUMA ";
    //     for (int j = 0 ; j < cpu_numa_nodes ; ++j){
    //         if (CPU_ISSET(i, &cpu_numa_affinity[j])) std::cout << j;
    //     }
    //     std::cout << std::endl;
    // }


    // // diascld37
    // //FIXME: Generalize
    // for (int i = 0  ; i < 14 ; ++i) CPU_SET(i, &gpu_affinity[0]);
    // for (int i = 0  ; i < 14 ; ++i) CPU_SET(i, &gpu_affinity[1]);
    // for (int i = 14 ; i < 28 ; ++i) CPU_SET(i, &gpu_affinity[2]);
    // for (int i = 14 ; i < 28 ; ++i) CPU_SET(i, &gpu_affinity[3]);

    //FIXME: Generalize
    // cpu_numa_affinity  = new cpu_set_t[cpu_numa_nodes];
    // for (int i = 0  ; i < 14 ; ++i) CPU_SET(i, &cpu_numa_affinity[0]);
    // for (int i = 14 ; i < 28 ; ++i) CPU_SET(i, &cpu_numa_affinity[1]);

    mutex buff_cache;

    vector<thread> buffer_pool_constrs;
    for (int j = 0; j < devices; ++j) {
        buffer_pool_constrs.emplace_back([j, size, &buff_cache]{
                set_device_on_scope d(j);

                set_affinity(&gpu_affinity[j]);

                T      *mem;
                size_t  pitch;
                gpu(cudaMallocPitch(&mem, &pitch, h_vector_size*sizeof(T), size));
                
                vector<T *> buffs;
                
                buffs.reserve(size);
                for (size_t i = 0 ; i < size ; ++i) {
                    T        * m = (T *) (((char *) mem) + i*pitch);
                    // buffer_t * b = cuda_new<buffer_t>(j, m, j);
                    buffs.push_back(m);

                    // cout << "Device " << j << " : data = " << m << endl;
                    assert(get_device(m) == j);
                }
                {
                    lock_guard<mutex> guard(buff_cache);
                    buffer_cache.insert(buffs.begin(), buffs.end());
                }
                
                pool_t * tmp =  cuda_new<pool_t>(j, size, buffs, j);
                gpu(cudaMemcpyToSymbol(pool      , &tmp, sizeof(pool_t *)));
                gpu(cudaMemcpyToSymbol(deviceId  ,   &j, sizeof(int     )));
                gpu(cudaMemcpyToSymbol(buff_start, &mem, sizeof(void   *)));
                void * e = (void *) (((char *) mem) + size*pitch);
                gpu(cudaMemcpyToSymbol(buff_end  ,   &e, sizeof(void   *)));

                h_buff_start[j] = mem;
                h_buff_end  [j] = e  ;
                
                int greatest;
                gpu(cudaDeviceGetStreamPriorityRange(NULL, &greatest));
                gpu(cudaStreamCreateWithPriority(&(release_streams[j]), cudaStreamNonBlocking, greatest));

                T **bf;
                gpu(cudaMallocHost(&bf, std::max(device_buff_size, keep_threshold)*sizeof(T *)));
                device_buff[j] = bf;

                device_buffs_thrds[j] = new thread(dev_buff_manager, j);
            });
    }


    for (int i = 0 ; i < cpu_numa_nodes ; ++i){
        buffer_pool_constrs.emplace_back([i, h_size, cores, &buff_cache]{
            set_affinity(&cpu_numa_affinity[i]);

            T      *mem = (T *) numa_alloc_onnode(h_vector_size*sizeof(T)*h_size, i);
            assert(mem);

            gpu(cudaHostRegister(mem, h_vector_size*sizeof(T)*h_size, 0));
            // T * mem;
            // gpu(cudaMallocHost(&mem, h_vector_size*sizeof(T)*h_size));

            int status[1];
            int ret_code;
            status[0]=-1;
            ret_code=move_pages(0 /*self memory */, 1, (void **) &mem, NULL, status, 0);
            printf("Memory at %p is at %d node (retcode %d, cpu %d)\n", mem, status[0], ret_code, sched_getcpu());

            numa_assert(get_numa_addressed(mem) == i);

            h_h_buff_start[i] = mem;

            vector<T *> buffs;
            buffs.reserve(h_size);
            for (size_t j = 0 ; j < h_size ; ++j) {
                T        * m = mem + j * h_vector_size;
                // buffer_t * b = cuda_new<buffer_t>(-1, m, -1);
                buffs.push_back(m);

                // cout << "NUMA " << get_numa_addressed(m) << " : data = " << m << endl;
                numa_assert(get_numa_addressed(m) == i);
            }

            {
                lock_guard<mutex> guard(buff_cache);
                buffer_cache.insert(buffs.begin(), buffs.end());
            }

            h_pool_t *p         = new h_pool_t(h_size, buffs);
            
            h_pool_numa[i]      = p;


            for (int j = 0 ; j < cores ; ++j){
                if (CPU_ISSET(j, &cpu_numa_affinity[i])) h_pool[j] = p;
            }
        });
    }

    // h_pool_t **numa_h_pools = new h_pool_t *[cpu_numa_nodes];

    // for (int i = 0 ; i < cores ; ++i) numa_node_inited[i] = NULL;

    // for (int i = 0 ; i < cores ; ++i){
    //     int numa_node = numa_node_of_cpu(i);

    //     if (!numa_node_inited[numa_node]){
    //         cpu_set_t cpuset;
    //         CPU_ZERO(&cpuset);
    //         CPU_SET(i, &cpuset);


    //         T      *mem;
    //         gpu(cudaMallocHost(&mem, buffer_t::capacity()*sizeof(T)*size));

    //         vector<buffer_t *> buffs;
    //         for (size_t i = 0 ; i < size ; ++i) buffs.push_back(cuda_new<buffer_t>(-1, mem + i * buffer_t::capacity(), -1));
    //         numa_node_inited[numa_node] = new h_pool_t(size, buffs);
    //     }
    //     h_pool[i] = numa_node_inited[numa_node];
    // }

    // T      *mem;
    // gpu(cudaMallocHost(&mem, buffer_t::capacity()*sizeof(T)*size));

    // vector<buffer_t *> buffs;
    // for (size_t i = 0 ; i < size ; ++i) buffs.push_back(cuda_new<buffer_t>(-1, mem + i * buffer_t::capacity(), -1));
    // h_pool = new h_pool_t(size, buffs);

    for (auto &t: buffer_pool_constrs) t.join();
}

template<typename T>
__host__ void buffer_manager<T>::destroy(){
    int devices;
    gpu(cudaGetDeviceCount(&devices));

    long cores = sysconf(_SC_NPROCESSORS_ONLN);
    assert(cores > 0);

    int cpu_numa_nodes = numa_num_task_nodes();

    terminating = true;

    // device_buffs_mutex = new mutex              [devices];
    // device_buffs_pool  = new vector<buffer_t *> [devices];
    // release_streams    = new cudaStream_t       [devices];

    // h_pool             = new h_pool_t *         [cores  ];
    // h_pool_numa        = new h_pool_t *         [cpu_numa_nodes];

    // device_buff        = new buffer_t**[devices];
    // device_buff_size   = buff_buffer_size;
    // keep_threshold     = buff_keep_threshold;

    // gpu_affinity       = new cpu_set_t[devices];

    // mutex buff_cache;

    vector<thread> buffer_pool_constrs;
    for (int j = 0; j < devices; ++j) {
        buffer_pool_constrs.emplace_back([j]{
                set_device_on_scope d(j);

                device_buffs_cv[j].notify_all();
                device_buffs_thrds[j]->join();

                std::unique_lock<std::mutex> lock(device_buffs_mutex[j]);

                size_t size = device_buffs_pool[j].size();
                assert(size <= keep_threshold);
                for (size_t i = 0 ; i < size ; ++i) device_buff[j][i] = device_buffs_pool[j][i];

                release_buffer_host<<<1, 1, 0, release_streams[j]>>>((void **) device_buff[j], size);
                gpu(cudaStreamSynchronize(release_streams[j]));

                pool_t *tmp;
                gpu(cudaMemcpyFromSymbol(&tmp, pool, sizeof(pool_t *)));
                cuda_delete(tmp);

                T * mem;
                gpu(cudaMemcpyFromSymbol(&mem, buff_start, sizeof(void   *)));
                gpu(cudaFree(mem));

                gpu(cudaStreamDestroy(release_streams[j]));

                gpu(cudaFreeHost(device_buff[j]));
            });
    }
    
    size_t h_size = buffer_manager<T>::h_size;

    for (int i = 0 ; i < cpu_numa_nodes ; ++i){
        buffer_pool_constrs.emplace_back([i, h_size]{
            set_affinity(&cpu_numa_affinity[i]);

            gpu(cudaHostUnregister(h_h_buff_start[i]));
            numa_free(h_h_buff_start[i], h_vector_size * sizeof(T) * h_size);
            
            delete h_pool_numa[i];
        });
    }

    for (auto &t: buffer_pool_constrs) t.join();

    terminating        = false;
    delete[] device_buffs_mutex;
    delete[] device_buffs_cv   ;
    delete[] device_buffs_thrds;
    delete[] device_buffs_pool ;
    delete[] release_streams   ;

    delete[] h_pool            ;
    delete[] h_pool_numa       ;

    delete[] h_buff_start      ;
    delete[] h_buff_end        ;

    delete[] h_h_buff_start    ;

    delete[] device_buff       ;

    buffer_cache.clear();
}

extern "C"{
    void * get_buffer(size_t bytes){
        assert(bytes <= sizeof(int32_t) * h_vector_size); //FIMXE: buffer manager should be able to allocate blocks of arbitary size
        return (void *) buffer_manager<int32_t>::h_get_buffer(-1);
    }

    void   release_buffer(void * buff){
        buffer_manager<int32_t>::release_buffer((int32_t *) buff);
    }
}

template<typename T>
void buffer_manager<T>::dev_buff_manager(int dev){
    set_device_on_scope d(dev);

    std::unique_lock<mutex> lk(device_buffs_mutex[dev]);
    while (true){
        device_buffs_cv[dev].wait(lk, [dev]{return device_buffs_pool[dev].empty() || terminating;});

        if (terminating) break;

        get_buffer_host<<<1, 1, 0, release_streams[dev]>>>((void **) device_buff[dev], device_buff_size);
        gpu(cudaStreamSynchronize(release_streams[dev]));

        device_buffs_pool[dev].insert(device_buffs_pool[dev].end(), device_buff[dev], device_buff[dev]+device_buff_size);

        device_buffs_cv[dev].notify_all();
        // lk.unlock();
    }
}


template<typename T>
__host__ inline T * buffer_manager<T>::h_get_buffer(int dev){
    if (dev >= 0){
        std::unique_lock<std::mutex> lock(device_buffs_mutex[dev]);

        device_buffs_cv[dev].wait(lock, [dev]{return !device_buffs_pool[dev].empty();});

        T * ret = device_buffs_pool[dev].back();
        device_buffs_pool[dev].pop_back();
        device_buffs_cv[dev].notify_all();
        return ret;
    } else {
        return get_buffer();
    }
}


template<typename T>
__host__ void buffer_manager<T>::overwrite(T * buff, const T * data, uint32_t N, cudaStream_t strm, bool blocking){
    gpu(cudaMemcpyAsync(buff, data, N*sizeof(T), cudaMemcpyDefault, strm));
    if (blocking) gpu(cudaStreamSynchronize(strm));
}

template<typename T>
__host__ void buffer_manager<T>::overwrite_bytes(void * buff, const void * data, size_t bytes, cudaStream_t strm, bool blocking){
    gpu(cudaMemcpyAsync(buff, data, bytes, cudaMemcpyDefault, strm));
    if (blocking) gpu(cudaStreamSynchronize(strm));
}

template class buffer_manager<int32_t>;


__global__ void GpuHashRearrange_acq_buffs(void   ** buffs){
    buffs[blockIdx.x] = get_buffers();
}