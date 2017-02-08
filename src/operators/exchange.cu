#include "exchange.cuh"
#include <iostream>
#include "../common.cuh"
#include "../buffer_manager.cuh"
#include "operator.cuh"

using namespace std;

exchange::exchange(const vector<p_operator_t> &parents,
                    const vector<launch_conf> &parent_conf){
    assert(parents.size()  == parent_conf.size());

    remaining_producers = 1;

    for (size_t i = 0 ; i < parents.size() ; ++i){
        firers.emplace_back(&exchange::fire, this, new consumer(parents[i], parent_conf[i].gridDim, parent_conf[i].blockDim, parent_conf[i].shared_mem));
    }
}

// __host__ void exchange::poll(buffer_pool_t *src){
//     int device = get_device(src);

//     set_device_on_scope d(device);
    
//     buffer_pool_t::buffer_t ** buff_ret;
//     cudaMallocHost(&buff_ret, sizeof(buffer_pool_t::buffer_t *));

//     cudaStream_t strm;
//     cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking);

//         auto start = chrono::system_clock::now();
//     do {
//         // cout << "poll " << this << endl;
//         buffer_pool_t::buffer_t * buff = src->h_acquire_buffer_blocked(buff_ret, strm);
//         // cout << "]]]]]]" << buff << endl;

//         if (buff == (buffer_pool_t::buffer_t *) 1) {
//             // this_thread::sleep_for(chrono::microseconds(100));
//             continue;
//         }

//         if (!src->is_valid(buff)) break;

//         set_ready(buff);
//     } while(true);

//     producer_ended();


//         auto end   = chrono::system_clock::now();
//         cout << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;

//     gpu(cudaStreamDestroy(strm));
//     gpu(cudaFreeHost(buff_ret));
// }

__host__ void exchange::fire(consumer *cons){
    cons->open();
        auto start = chrono::system_clock::now();
    do {
        buffer_pool<int32_t>::buffer_t *p = get_ready();

        if (!p) break;

        cons->consume(p);
        this_thread::yield();
    } while (true);
        auto end   = chrono::system_clock::now();
        cout << "T:" << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;

    cons->close();
}

__host__ void exchange::open(){}

__host__ void exchange::set_ready(buffer_pool_t::buffer_t * buff){
    consume(buff);
}

__host__ void exchange::consume(buffer_t * buff){
    unique_lock<mutex> lock(ready_pool_mutex);
    ready_pool.emplace_back(buff);
    ready_pool_cv.notify_all();
    lock.unlock();
}

__host__ exchange::buffer_pool_t::buffer_t * exchange::get_ready(){
    unique_lock<mutex> lock(ready_pool_mutex);

    ready_pool_cv.wait(lock, [this]{return !ready_pool.empty() || (ready_pool.empty() && remaining_producers <= 0);});

    if (ready_pool.empty()){
        assert(remaining_producers == 0);
        lock.unlock();
        return NULL;
    }

    buffer_pool_t::buffer_t *p = ready_pool.back(); //FIXME: release buffer back to device
    ready_pool.pop_back();

    lock.unlock();
    return p;
}

__host__ void exchange::producer_ended(){
    --remaining_producers;
    assert(remaining_producers >= 0);
    ready_pool_cv.notify_all();
}

__host__ __device__ void producer::consume(buffer_pool_t::buffer_t * buff){
#ifdef __CUDA_ARCH__
    assert(outpool);
    uint32_t laneid;
    asm("mov.u32 %0, %%laneid;" : "=r"(laneid));
    if (laneid == 0) outpool->release_buffer(buff);
#else
    assert(exc);
    exc->set_ready(buff);
#endif
}


__global__ __launch_bounds__(65536, 4) void launch_consume_pipeline(d_operator_t * op, consumer::buffer_t * buff, consumer::buffer_t * prev_buff = NULL){
    // op->consume(buff);
    // variant::apply_visitor(push(buff), *op);

    const int32_t width     = blockDim.x    * blockDim.y;
    const int32_t gridwidth = gridDim.x     * gridDim.y ;
    const int32_t bigwidth  = 4 * (width    * gridwidth);

    // const int32_t i       = threadIdx.x + threadIdx.y * blockDim.x;
    const int32_t blocki  = blockIdx.x + blockIdx.y * gridDim.x;

    const uint32_t N   = min(buff->cnt, buff->capacity());//insp->count();

    // if (blocki == 0 && threadIdx.x == 0 &&  threadIdx.y == 0 && threadIdx.z == 0) buff->cnt = N;
    if (blocki == 0 && threadIdx.x == 0 &&  threadIdx.y == 0 && threadIdx.z == 0 && prev_buff) buffer_manager<int32_t>::release_buffer(prev_buff);

    if (4 * blocki * width < N){ //FIXME: get back!
        const int32_t warpoff = 4*(get_warpid() * warpSize + blocki * width);

        const int32_t *src = buff->data;

        op->consume_open();

        __syncthreads();

        for (int j = warpoff ; j < N ; j += bigwidth){
            // vec4 tmp = reinterpret_cast<const vec4*>(src)[i+j+blocki*width];
            op->consume_warp(src + j, min(N - j, 4*warpSize));
        }

        __syncthreads();

        op->consume_close();
    }
}

__global__ __launch_bounds__(65536, 4) void launch_consume_pipeline(d_operator_t * op, consumer::buffer_t * buff, uint32_t N, consumer::buffer_t * prev_buff = NULL){
    // op->consume(buff);
    // variant::apply_visitor(push(buff), *op);

    const int32_t width     = blockDim.x    * blockDim.y;
    const int32_t gridwidth = gridDim.x     * gridDim.y ;
    const int32_t bigwidth  = 4 * (width    * gridwidth);

    // const int32_t i       = threadIdx.x + threadIdx.y * blockDim.x;
    const int32_t blocki  = blockIdx.x + blockIdx.y * gridDim.x;

    // const uint32_t N   = min(buff->cnt, buff->capacity());//insp->count();

    if (blocki == 0 && threadIdx.x == 0 &&  threadIdx.y == 0 && threadIdx.z == 0) buff->cnt = N;
    if (blocki == 0 && threadIdx.x == 0 &&  threadIdx.y == 0 && threadIdx.z == 0 && prev_buff) buffer_manager<int32_t>::release_buffer(prev_buff);

    if (4 * blocki * width < N){ //FIXME: get back!
        const int32_t warpoff = 4*(get_warpid() * warpSize + blocki * width);

        const int32_t *src = buff->data;

        op->consume_open();

        __syncthreads();

        for (int j = warpoff ; j < N ; j += bigwidth){
            // vec4 tmp = reinterpret_cast<const vec4*>(src)[i+j+blocki*width];
            op->consume_warp(src + j, min(N - j, 4*warpSize));
        }

        __syncthreads();

        op->consume_close();
    }
}

__host__ void consumer::consume(buffer_pool_t::buffer_t * buff_l){
    if (device >= 0){
        // execs.emplace_back([buff_l, this](){
        //parent is running on device
        set_device_on_scope d(device);
        //move buffer
        // assert(buff_l->cnt > 0);
        if (get_device(buff_l) != device){//rand() & 1){
            // cudaEvent_t event;

            // gpu(cudaEventCreate(&event));

            buffer_t * buff = buffer_manager<int32_t>::h_get_buffer(device);

            buffer_pool_t::buffer_t::inspector_t from(strm2);
            buffer_pool_t::buffer_t::inspector_t to  (strm2);

                                                                                // from.load(buff_l, false);
                                                                                // to.load(buff, true);

            const int32_t * d;
            uint32_t        cnt;
            if (get_device(buff_l) < 0){
                from.load(buff_l, true);
                d   = from.data();
                cnt = from.count();
            } else {
                d   = buff_l->data;
                cnt = buff_l->count();
            }
                                        // buffer_t * buff2 = buffer_manager<int32_t>::get_buffer();
                                        // buffer_t * buff3 = buffer_manager<int32_t>::h_get_buffer(device);

                                        // buffer_pool_t::buffer_t::inspector_t from2(strm2);
                                        // buffer_pool_t::buffer_t::inspector_t to2  (strm2);

                                        // from2.load(buff3, false);
                                        // to2.load(buff2, true);
                        
            buffer_manager<int32_t>::overwrite(buff, d, cnt, strm2, true);
            // to.overwrite(buff_l)

                                                                                // to.save(buff, true);

            // cout << to.buff->cnt << " ";
            // to.load(buff, true);

            // cout << to.buff->cnt << " " << buff_l->cnt << endl;
            // assert(to.count() == buff_l->cnt);

            // buff_l = buff;

            // gpu(cudaEventRecord(event, strm2));

            // gpu(cudaStreamWaitEvent(strm, event, 0));

            // gpu(cudaStreamSynchronize(strm2));

        // const size_t memsize = sizeof(int32_t)*1024*1024*4;
            
        //     gpu(cudaMemcpyAsync(d_data_in[1 ^ (i & 1)], h_data_in[1 ^ (i & 1)], memsize, cudaMemcpyDefault, strmt[1]));
        //     gpu(cudaMemcpyAsync(h_data_in[i & 1], d_data_in[i & 1], memsize, cudaMemcpyDefault, strmt[0]));
            

            // gpu(cudaMemcpyAsync((int32_t *) from2.data(), to2.data(), buffer_pool_t::buffer_t::inspector_t::capacity()*sizeof(int32_t), cudaMemcpyDefault, strm2));
            // // gpu(cudaStreamSynchronize(strm2));

            // gpu(cudaMemcpyAsync((int32_t *) from.data(), to.data(), buffer_pool_t::buffer_t::inspector_t::capacity()*sizeof(int32_t), cudaMemcpyDefault, strm));
            // // gpu(cudaMemcpyAsync((int32_t *) to.data(), from.data(), buffer_pool_t::buffer_t::inspector_t::capacity()*sizeof(int32_t), cudaMemcpyDefault, strm2));
            // cout << to.data() << " " << from.data() << " " << from2.data() << " " << to2.data() << endl;
            
            
            launch_consume_pipeline<<<dimGrid, dimBlock, shared_mem, strm>>>(parent.d, buff, from.count(), prev_buff);

            if (get_device(buff_l) != device) {
                buffer_manager<int32_t>::release_buffer(buff_l);
                prev_buff = NULL;
            } else {
                prev_buff = buff;
            }
        } else {
            launch_consume_pipeline<<<dimGrid, dimBlock, shared_mem, strm>>>(parent.d, buff_l, prev_buff);
            prev_buff = buff_l;
        }

        //launch on buffer
#ifndef NQUEUE_WORK
        gpu(cudaStreamSynchronize(strm));
#endif
        // });
    } else {
        //parent is running on host
        parent.h->consume(buff_l); //FIXME : not implemented
        // variant::apply_visitor(push(buff), *parent);
    }
}

__host__ consumer::consumer(p_operator_t parent, dim3 dimGrid, dim3 dimBlock, int shared_mem): parent(parent), dimGrid(dimGrid), dimBlock(dimBlock), shared_mem(shared_mem), prev_buff(NULL){
    assert(parent.h);

    device = get_device(parent.h);

    if (device >= 0){
        set_device_on_scope d(device);
        cout << "asdasd" << endl;

        cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking);
        cudaStreamCreateWithFlags(&strm2, cudaStreamNonBlocking);

        const size_t memsize = sizeof(int32_t)*1024*1024*4;

        for (int i = 0 ; i < 2 ; ++i){
            gpu(cudaMallocHost(&h_data_in[i], memsize));//, cudaHostAllocDefault));
            gpu(cudaMalloc(&d_data_in[i], memsize));

            gpu(cudaStreamCreateWithFlags(&strmt[i], cudaStreamNonBlocking));
        }

        i = 0;

        // shared_mem = 0;//getSharedMemoryRequirements();
    }
}

// __host__ consumer::consumer(h_operator_t *parent, dim3 dimGrid, dim3 dimBlock, int shared_mem): dimGrid(dimGrid), dimBlock(dimBlock), shared_mem(shared_mem){
//     assert(parent);

//     this->parent.h = parent;

//     device = get_device(parent);

//     if (device >= 0){
//         set_device_on_scope d(device);

//         cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking);
//         cudaStreamCreateWithFlags(&strm2, cudaStreamNonBlocking);

//         // shared_mem = 0;//getSharedMemoryRequirements();
//     }
// }

// __global__ void launch_open_pipeline(d_operator_t * parent){
//     parent->open();
//     // variant::apply_visitor(close{}, *parent);
// }

// __global__ void launch_close_pipeline(d_operator_t * parent){
//     parent->close();
//     // variant::apply_visitor(close{}, *parent);
// }

__host__ void consumer::open(){
    if (device >= 0){
#ifndef NDEBUG
        int rc =
#endif
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &gpu_affinity[device]);
        assert(rc == 0);
        this_thread::yield();

        parent.d->open();
    } else {
        parent.h->open();
    }
}

__host__ void consumer::close(){
    for (auto &t: execs) t.join();
    if (device >= 0){
        gpu(cudaStreamSynchronize(strm2));
        gpu(cudaStreamSynchronize(strm));
        parent.d->close();
    } else {
        parent.h->close();
    }
}

__host__ consumer::~consumer(){
    if (device >= 0){
        gpu(cudaStreamDestroy(strm));
    }
}

__host__ exchange::~exchange(){
    assert(ready_pool.empty());
}

__host__ void exchange::close(){
    producer_ended();
    for (auto &t: firers ) t.join();
}

__host__ __device__ void producer::close(){
    if (outpool) {
#ifdef __CUDA_ARCH__
        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 &&
                blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0){
            printf("%d %d %d\n", threadIdx.x, threadIdx.y, threadIdx.z);
            printf("%d %d %d\n",  blockIdx.x,  blockIdx.y,  blockIdx.z);
            outpool->unregister_producer(this);
        }
#else
        assert(false);
#endif
    }
    if (exc    ) exc->producer_ended();
}

__host__ producer::~producer(){}

// template<>
// __host__ __device__ void push::operator()<producer *>(producer * op) const{
//    op->consume(NULL);
// }

// template<>
// __host__ __device__ void push::operator()<consumer *>(consumer * op) const{
//    op->consume(NULL);
// }