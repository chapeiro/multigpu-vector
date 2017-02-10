#include "buffer_manager.cuh"

#include <thread>
#include <utmpx.h>
#include <unistd.h>
#include <numaif.h>

__device__ __constant__ threadsafe_device_stack<buffer<int32_t, DEFAULT_BUFF_CAP, vec4>  *, (buffer<int32_t, DEFAULT_BUFF_CAP, vec4>  *) NULL> * pool;
__device__ __constant__ int deviceId;

// template<typename T>
// threadsafe_stack<buffer<int32_t, DEFAULT_BUFF_CAP, vec4>  *, (buffer<int32_t, DEFAULT_BUFF_CAP, vec4>  *) NULL> ** buffer_manager<T>::h_pool;

// template<typename T>
// threadsafe_stack<buffer<int32_t, DEFAULT_BUFF_CAP, vec4>  *, (buffer<int32_t, DEFAULT_BUFF_CAP, vec4>  *) NULL> ** buffer_manager<T>::h_pool_numa;

// template<typename T>
// unordered_map<buffer<int32_t, DEFAULT_BUFF_CAP, vec4> *, int32_t *> buffer_manager<T>::buffer_cache;

__global__ void release_buffer_host(buffer<int32_t, DEFAULT_BUFF_CAP> **buff, int buffs){
    assert(blockDim.x * blockDim.y * blockDim.z == 1);
    assert( gridDim.x *  gridDim.y *  gridDim.z == 1);
    for (int i = 0 ; i < buffs ; ++i) buffer_manager<int32_t>::release_buffer(buff[i]);
}

__global__ void get_buffer_host(buffer<int32_t, DEFAULT_BUFF_CAP> **buff, int buffs){
    assert(blockDim.x * blockDim.y * blockDim.z == 1);
    assert( gridDim.x *  gridDim.y *  gridDim.z == 1);
    for (int i = 0 ; i < buffs ; ++i) buff[i] = buffer_manager<int32_t>::get_buffer();
}

// template<typename T>
// mutex                                              *buffer_manager<T>::devive_buffs_mutex;

// template<typename T>
// vector<buffer<int32_t, DEFAULT_BUFF_CAP, vec4>  *> *buffer_manager<T>::device_buffs_pool;

// template<typename T>
// buffer<int32_t, DEFAULT_BUFF_CAP, vec4>          ***buffer_manager<T>::device_buff;

// template<typename T>
// int                                                 buffer_manager<T>::device_buff_size;

// template<typename T>
// int                                                 buffer_manager<T>::keep_threshold;

// template<typename T>
// cudaStream_t                                       *buffer_manager<T>::release_streams;
cpu_set_t                                          *gpu_affinity;
cpu_set_t                                          *cpu_numa_affinity;

template<typename T>
__host__ void buffer_manager<T>::init(int size, int buff_buffer_size, int buff_keep_threshold){
    int devices;
    gpu(cudaGetDeviceCount(&devices));

    long cores = sysconf(_SC_NPROCESSORS_ONLN);
    assert(cores > 0);


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

    devive_buffs_mutex = new mutex              [devices];
    device_buffs_pool  = new vector<buffer_t *> [devices];
    release_streams    = new cudaStream_t       [devices];

    h_pool             = new h_pool_t *         [cores  ];
    h_pool_numa        = new h_pool_t *         [cpu_numa_nodes];

    device_buff        = new buffer_t**[devices];
    device_buff_size   = buff_buffer_size;
    keep_threshold     = buff_keep_threshold;

    // gpu(cudaMallocHost(&tmp, device_buff_size*sizeof(buffer_t *)*devices));
    // for (int i = 0 ; i < devices ; ++i) {
        // device_buff[i] = tmp + device_buff_size*sizeof(buffer_t *)*i;
    // }

    gpu_affinity       = new cpu_set_t[devices];

    for (int j = 0 ; j < devices ; ++j) CPU_ZERO(&gpu_affinity[j]);

    //FIXME: Generalize
    for (int i = 0 ; i < 48 ; i += 2) CPU_SET(i, &gpu_affinity[0]);
    for (int i = 1 ; i < 48 ; i += 2) CPU_SET(i, &gpu_affinity[1]);

    //FIXME: Generalize
    cpu_numa_affinity  = new cpu_set_t[cpu_numa_nodes];
    for (int i = 0 ; i < 48 ; i += 2) CPU_SET(i, &cpu_numa_affinity[0]);
    for (int i = 1 ; i < 48 ; i += 2) CPU_SET(i, &cpu_numa_affinity[1]);

    mutex buff_cache;

    vector<thread> buffer_pool_constrs;
    for (int j = 0; j < devices; ++j) {
        buffer_pool_constrs.emplace_back([j, size, &buff_cache]{
                set_device_on_scope d(j);

#ifndef NDEBUG
                int rc = 
#endif
                pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &gpu_affinity[j]);
                assert(rc == 0);

                this_thread::yield();

                T      *mem;
                size_t  pitch;
                gpu(cudaMallocPitch(&mem, &pitch, buffer_t::capacity()*sizeof(T), size));

                vector<buffer_t *> buffs;
                {
                    lock_guard<mutex> guard(buff_cache);
                    for (size_t i = 0 ; i < size ; ++i) {
                        T        * m = (T*) (((char *) mem)+i*pitch);
                        buffer_t * b = cuda_new<buffer_t>(j, m, j);
                        buffs.push_back(b);
                        buffer_cache.emplace(b, m);

                        // cout << "Device " << j << " : buffer " << b << " data :" << m << endl;
                        assert(get_device(b) == j);
                    }
                }
                
                pool_t * tmp =  cuda_new<pool_t>(j, size, buffs, j);
                gpu(cudaMemcpyToSymbol(pool    , &tmp, sizeof(pool_t *)));
                gpu(cudaMemcpyToSymbol(deviceId,   &j, sizeof(int     )));

                gpu(cudaStreamCreateWithFlags(&(release_streams[j]), cudaStreamNonBlocking));

                buffer_t **bf;
                gpu(cudaMallocHost(&bf, device_buff_size*sizeof(buffer_t *)));
                device_buff[j] = bf;
            });
    }


    for (int i = 0 ; i < cpu_numa_nodes ; ++i){
        buffer_pool_constrs.emplace_back([i, size, cores, &buff_cache]{
#ifndef NDEBUG
            int rc =
#endif
            pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpu_numa_affinity[i]);
            assert(rc == 0);
            this_thread::yield();

            T      *mem;
            gpu(cudaMallocHost(&mem, buffer_t::capacity()*sizeof(T)*size));

            int status[1];
            int ret_code;
            status[0]=-1;
            ret_code=move_pages(0 /*self memory */, 1, (void **) &mem, NULL, status, 0);
            printf("Memory at %p is at %d node (retcode %d)\n", mem, status[0], ret_code);

            vector<buffer_t *> buffs;

            {
                lock_guard<mutex> guard(buff_cache);
                for (size_t j = 0 ; j < size ; ++j) {
                    T        * m = mem + j * buffer_t::capacity();
                    buffer_t * b = cuda_new<buffer_t>(-1, m, -1);
                    buffs.push_back(b);
                    buffer_cache.emplace(b, m);

                    // cout << "NUMA " << get_numa_addressed(m) << " : buffer = " << b << " data = " << m << endl;
                    assert(get_numa_addressed(m) == i);
                }
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

    // devive_buffs_mutex = new mutex              [devices];
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

                unique_lock<std::mutex> lock(devive_buffs_mutex[j]);

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
__host__ inline buffer_manager<T>::buffer_t * buffer_manager<T>::h_get_buffer(int dev){
    if (dev >= 0){
        unique_lock<std::mutex> lock(devive_buffs_mutex[dev]);
        if (device_buffs_pool[dev].empty()){
            set_device_on_scope d(dev);
            cout << "astt " << dev << endl;
            get_buffer_host<<<1, 1, 0, release_streams[dev]>>>(device_buff[dev], device_buff_size);
            gpu(cudaStreamSynchronize(release_streams[dev]));
            cout << "astt-" << dev << endl;
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




template<typename T>
__host__ void buffer_manager<T>::overwrite(buffer_t * buff, const T * data, uint32_t N, cudaStream_t strm, bool blocking){
    gpu(cudaMemcpyAsync(buffer_cache.find(buff)->second, data, N*sizeof(T), cudaMemcpyDefault, strm));
    if (blocking) gpu(cudaStreamSynchronize(strm));
}

template class buffer_manager<int32_t>;