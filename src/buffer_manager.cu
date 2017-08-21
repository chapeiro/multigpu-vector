#include "buffer_manager.cuh"

#include <thread>
#include <utmpx.h>
#include <unistd.h>

#include "numa_utils.cuh"

__device__ __constant__ threadsafe_device_stack<int32_t *, (int32_t *) NULL> * pool;
__device__ __constant__ int deviceId;
__device__ __constant__ void * buff_start;
__device__ __constant__ void * buff_end  ;


template<typename T>
__global__ void release_buffer_host(T **buff, int buffs){
    assert(blockDim.x * blockDim.y * blockDim.z == 1);
    assert( gridDim.x *  gridDim.y *  gridDim.z == 1);
    for (int i = 0 ; i < buffs ; ++i) buffer_manager<T>::release_buffer(buff[i]);
}

template<typename T>
__global__ void get_buffer_host(T **buff, int buffs){
    assert(blockDim.x * blockDim.y * blockDim.z == 1);
    assert( gridDim.x *  gridDim.y *  gridDim.z == 1);
    for (int i = 0 ; i < buffs ; ++i) buff[i] = buffer_manager<T>::get_buffer();
}

int                                                 cpu_cnt;
cpu_set_t                                          *gpu_affinity;
cpu_set_t                                          *cpu_numa_affinity;

template<typename T>
__host__ void buffer_manager<T>::init(int size, int buff_buffer_size, int buff_keep_threshold){
    int devices;
    gpu(cudaGetDeviceCount(&devices));

    long cores = sysconf(_SC_NPROCESSORS_ONLN);
    assert(cores > 0);
    cpu_cnt = cores;


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
    int cpu_numa_nodes = 2;

    terminating        = false;
    device_buffs_mutex = new mutex              [devices];
    device_buffs_cv    = new condition_variable [devices];
    device_buffs_thrds = new thread *           [devices];
    device_buffs_pool  = new vector<T *>        [devices];
    release_streams    = new cudaStream_t       [devices];

    h_pool             = new h_pool_t *         [cores  ];
    h_pool_numa        = new h_pool_t *         [cpu_numa_nodes];

    device_buff        = new T **[devices];
    device_buff_size   = buff_buffer_size;
    keep_threshold     = buff_keep_threshold;

    // gpu(cudaMallocHost(&tmp, device_buff_size*sizeof(buffer_t *)*devices));
    // for (int i = 0 ; i < devices ; ++i) {
        // device_buff[i] = tmp + device_buff_size*sizeof(buffer_t *)*i;
    // }

    gpu_affinity       = new cpu_set_t[devices];

    for (int j = 0 ; j < devices ; ++j) CPU_ZERO(&gpu_affinity[j]);

    // diascld36
    // //FIXME: Generalize
    // for (int i = 0 ; i < 48 ; i += 2) CPU_SET(i, &gpu_affinity[0]);
    // for (int i = 1 ; i < 48 ; i += 2) CPU_SET(i, &gpu_affinity[1]);

    // //FIXME: Generalize
    // cpu_numa_affinity  = new cpu_set_t[cpu_numa_nodes];
    // for (int i = 0 ; i < 48 ; i += 2) CPU_SET(i, &cpu_numa_affinity[0]);
    // for (int i = 1 ; i < 48 ; i += 2) CPU_SET(i, &cpu_numa_affinity[1]);

    // diascld37
    //FIXME: Generalize
    for (int i = 0  ; i < 14 ; ++i) CPU_SET(i, &gpu_affinity[0]);
    for (int i = 0  ; i < 14 ; ++i) CPU_SET(i, &gpu_affinity[1]);
    for (int i = 14 ; i < 28 ; ++i) CPU_SET(i, &gpu_affinity[2]);
    for (int i = 14 ; i < 28 ; ++i) CPU_SET(i, &gpu_affinity[3]);

    //FIXME: Generalize
    cpu_numa_affinity  = new cpu_set_t[cpu_numa_nodes];
    for (int i = 0  ; i < 14 ; ++i) CPU_SET(i, &cpu_numa_affinity[0]);
    for (int i = 14 ; i < 28 ; ++i) CPU_SET(i, &cpu_numa_affinity[1]);

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

                gpu(cudaStreamCreateWithFlags(&(release_streams[j]), cudaStreamNonBlocking));

                T **bf;
                gpu(cudaMallocHost(&bf, device_buff_size*sizeof(T *)));
                device_buff[j] = bf;

                device_buffs_thrds[j] = new thread(dev_buff_manager, j);
            });
    }


    for (int i = 0 ; i < cpu_numa_nodes ; ++i){
        buffer_pool_constrs.emplace_back([i, size, cores, &buff_cache]{
            set_affinity(&cpu_numa_affinity[i]);

            T      *mem = (T *) numa_alloc_onnode(h_vector_size*sizeof(T)*size, i);
            assert(mem);
            gpu(cudaHostRegister(mem, h_vector_size*sizeof(T)*size, 0));

            // T * mem;
            // gpu(cudaMallocHost(&mem, h_vector_size*sizeof(T)*size));

            int status[1];
            int ret_code;
            status[0]=-1;
            ret_code=move_pages(0 /*self memory */, 1, (void **) &mem, NULL, status, 0);
            printf("Memory at %p is at %d node (retcode %d, cpu %d)\n", mem, status[0], ret_code, sched_getcpu());

            numa_assert(get_numa_addressed(mem) == i);

            vector<T *> buffs;
            buffs.reserve(size);
            for (size_t j = 0 ; j < size ; ++j) {
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

            h_pool_t *p         = new h_pool_t(size, buffs);
            
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

    //FIXME: Generalize
    int cpu_numa_nodes = 2;

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

                unique_lock<std::mutex> lock(device_buffs_mutex[j]);

                size_t size = device_buffs_pool[j].size();
                assert(size <= keep_threshold);
                for (int i = 0 ; i < size ; ++i) device_buff[j][i] = device_buffs_pool[j][i];

                release_buffer_host<<<1, 1, 0, release_streams[j]>>>(device_buff[j], size);
                gpu(cudaStreamSynchronize(release_streams[j]));

                pool_t *tmp;
                gpu(cudaMemcpyFromSymbol(&tmp, pool, sizeof(pool_t *)));
                cuda_delete(tmp);


                gpu(cudaStreamDestroy(release_streams[j]));

                gpu(cudaFreeHost(device_buff[j]));
            });
    }

    for (int i = 0 ; i < cpu_numa_nodes ; ++i){
//         buffer_pool_constrs.emplace_back([i, size, cores, &buff_cache]{
// #ifndef NDEBUG
//             int rc =
// #endif
//             pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpu_numa_affinity[i]);
//             assert(rc == 0);
//             this_thread::yield();

//             T      *mem;
//             gpu(cudaMallocHost(&mem, buffer_t::capacity()*sizeof(T)*size));

//             gpu(cudaHostFree(mem));
            // {
            //     lock_guard<mutex> guard(buff_cache);
            //     for (size_t i = 0 ; i < size ; ++i) {
            //         T        * m = mem + i * buffer_t::capacity();
            //         buffer_t * b = cuda_new<buffer_t>(-1, m, -1);
            //         buffs.push_back(b);
            //         buffer_cache.emplace(b, m);
            //     }
            // }

        delete h_pool_numa[i];
        // });
    }




    for (auto &t: buffer_pool_constrs) t.join();
}


template<typename T>
void buffer_manager<T>::dev_buff_manager(int dev){
    set_device_on_scope d(dev);

    unique_lock<mutex> lk(device_buffs_mutex[dev]);
    while (true){
        device_buffs_cv[dev].wait(lk, [dev]{return device_buffs_pool[dev].empty() || terminating;});

        if (terminating) break;

        get_buffer_host<<<1, 1, 0, release_streams[dev]>>>(device_buff[dev], device_buff_size);
        gpu(cudaStreamSynchronize(release_streams[dev]));

        device_buffs_pool[dev].insert(device_buffs_pool[dev].end(), device_buff[dev], device_buff[dev]+device_buff_size);

        device_buffs_cv[dev].notify_all();
        // lk.unlock();
    }
}


template<typename T>
__host__ inline T * buffer_manager<T>::h_get_buffer(int dev){
    if (dev >= 0){
        unique_lock<std::mutex> lock(device_buffs_mutex[dev]);

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

template class buffer_manager<int32_t>;
template __global__ void release_buffer_host<int32_t>(int32_t **buff, int buffs);
template __global__ void get_buffer_host    <int32_t>(int32_t **buff, int buffs);