#include "exchange.cuh"
#include <iostream>
#include "common.cuh"
#include "buffer_manager.cuh"

using namespace std;

exchange::exchange(const vector<int> &prod_loc, const vector<int> &prodout_loc, 
                    const vector<int> &prodout_size, const vector<int> &prod2out,
                    const vector<Operator *> &parents,
                    const vector<dim3> &parent_dimGrid,
                    const vector<dim3> &parent_dimBlock,
                    const vector<int>  &shared_mem){
    assert(prod_loc.size() == prod2out.size());
    assert(parents.size()  == parent_dimGrid.size());
    assert(parents.size()  == parent_dimBlock.size());

    remaining_producers = prod_loc.size();
    for (size_t i = 0 ; i < prodout_loc.size() ; ++i){
        if (prodout_loc[i] >= 0){
            prod_output_holders.push_back(cuda_new<exchange::buffer_pool_t>(prodout_loc[i], prodout_size[i], 0, prodout_loc[i]));
        } else {
            prod_output_holders.push_back(NULL);
        }
    }

    for (size_t i = 0 ; i < prod_loc.size() ; ++i){
        producer * prod;
        if (prod_loc[i] >= 0){
            assert(prod2out[i] >= 0);
            assert(prod2out[i] < prodout_loc.size());
            prod = cuda_new<producer>(prod_loc[i], prod_output_holders[prod2out[i]]);
        } else {
            prod = new producer(this);
        }
        prods.push_back(prod);
    }

    for (size_t i = 0 ; i < prodout_loc.size() ; ++i){
        if (prod_loc[i] >= 0){
            pollers.emplace_back(&exchange::poll, this, prod_output_holders[i]);
        }
    }

    for (size_t i = 0 ; i < parents.size() ; ++i){
        firers.emplace_back(&exchange::fire, this, new consumer(parents[i], parent_dimGrid[i], parent_dimBlock[i], shared_mem[i]));
    }
}

__host__ void exchange::poll(buffer_pool_t *src){
    int device = get_device(src);

    set_device_on_scope d(device);
    
    buffer_pool_t::buffer_t ** buff_ret;
    cudaMallocHost(&buff_ret, sizeof(buffer_pool_t::buffer_t *));

    cudaStream_t strm;
    cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking);

        auto start = chrono::system_clock::now();
    do {
        // cout << "poll " << this << endl;
        buffer_pool_t::buffer_t * buff = src->h_acquire_buffer_blocked(buff_ret, strm);
        // cout << "]]]]]]" << buff << endl;

        if (buff == (buffer_pool_t::buffer_t *) 1) {
            // this_thread::sleep_for(chrono::microseconds(100));
            continue;
        }

        if (!src->is_valid(buff)) break;

        set_ready(buff);
    } while(true);

    producer_ended();


        auto end   = chrono::system_clock::now();
        cout << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;

    gpu(cudaStreamDestroy(strm));
    gpu(cudaFreeHost(buff_ret));
}

__host__ void exchange::fire(consumer *cons){
        auto start = chrono::system_clock::now();
    do {
        buffer_pool<int32_t>::buffer_t *p = get_ready();

        if (!p) break;

        cons->consume(p);
    } while (true);
        auto end   = chrono::system_clock::now();
        cout << "T:" << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;


    cons->join();
}

__host__ void exchange::set_ready(buffer_pool_t::buffer_t * buff){
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
    assert(remaining_producers == 0);
    ready_pool_cv.notify_all();
}

__device__ size_t t = 0;

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

// __device__ con pfunc1 = (con) producer::consume;

__global__ __launch_bounds__(65536, 4) void launch_consume_pipeline(Operator * op, consumer::buffer_t * buff){
    op->consume(buff);
    // variant::apply_visitor(push(buff), *op);
}

__host__ __device__ void consumer::consume(buffer_pool_t::buffer_t * buff_l){
#ifdef __CUDA_ARCH__
    assert(false);
#else
    if (device >= 0){
        //parent is running on device
        set_device_on_scope d(device);

        cudaEvent_t event;

        gpu(cudaEventCreate(&event));
        

        //move buffer
        buffer_manager<int32_t>::h_get_buffer(this->buff, strm2, device);

        buffer_pool_t::buffer_t::inspector_t from(strm2);
        buffer_pool_t::buffer_t::inspector_t to  (strm2);

        from.load(buff_l, true);
        to.load(*(this->buff), true);
        
        to.overwrite(from.data(), from.count());

        to.save(*(this->buff), true);
        gpu(cudaEventRecord(event, strm2));

        gpu(cudaStreamWaitEvent(strm, event, 0));

        //launch on buffer
        launch_consume_pipeline<<<dimGrid, dimBlock, shared_mem, strm>>>(parent, *(this->buff));
#ifndef NQUEUE_WORK
        gpu(cudaStreamSynchronize(strm));
#endif
    } else {
        //parent is running on host
        parent->consume(buff_l);
        // variant::apply_visitor(push(buff), *parent);
    }
#endif
}

__host__ consumer::consumer(Operator *parent, dim3 dimGrid, dim3 dimBlock, int shared_mem): dimGrid(dimGrid), dimBlock(dimBlock), parent(parent), shared_mem(shared_mem){
    assert(parent);

    device = get_device(parent);

    if (device >= 0){
        set_device_on_scope d(device);

        cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking);
        cudaStreamCreateWithFlags(&strm2, cudaStreamNonBlocking);

        gpu(cudaMallocHost(&buff, sizeof(buffer_pool<int32_t>::buffer_t *)));

        // shared_mem = 0;//getSharedMemoryRequirements();
    }
}
__global__ void launch_close_pipeline(Operator * parent){
    parent->close();
    // variant::apply_visitor(close{}, *parent);
}

__host__ __device__ void consumer::join(){
    if (device >= 0){
        set_device_on_scope d(device);
        launch_close_pipeline<<<dimGrid, dimBlock, shared_mem, strm>>>(parent);
        gpu(cudaStreamSynchronize(strm));
    } else {
        parent->close();
        // variant::apply_visitor(close{}, *parent);
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

__host__ void exchange::join(){
    for (auto &t: firers ) t.join();
    for (auto &t: pollers) t.join();
}

__host__ __device__ void producer::join(){
    if (outpool) {
#ifdef __CUDA_ARCH__
        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0){
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