#include "generators.cuh"
#include "buffer_manager.cuh"

#include <iostream>
#include <chrono>

// #include <thread>
// #include <chrono>

using namespace std;

__host__ generator::generator(h_operator_t * parent, int32_t *src, uint32_t N):
        parent(parent), src(src), N(N){
    // parent->open();

    // cudaStream_t strm;
    gpu(cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking));
    
    // buffer_pool<int32_t>::buffer_t ** buff_ret;
    gpu(cudaMallocHost(&buff_ret, sizeof(buffer_pool<int32_t>::buffer_t *)));

    // buffer_pool<int32_t>::buffer_t::inspector_t insp(strm);
    // while (N > 0){
    //     buffer_pool<int32_t>::buffer_t * buff = buffer_manager<int32_t>::get_buffer();

    //     int m = min(N, buff_size);

    //     insp.load(buff, true);
    //     insp.overwrite(src, m);

    //     insp.save(buff, true);

    //     parent->consume(buff);
    //     // variant::apply_visitor(push(buff), *parent);
    //     // push()(*parent); //->consume(buff);

    //     N   -= m;
    //     src += m;
    // }
    // gpu(cudaStreamSynchronize(strm));
    // gpu(cudaStreamDestroy(strm));
    // gpu(cudaFreeHost(buff_ret));

    // parent->close();
}

__host__ void generator::open(){
    parent->open();
}

__host__ void generator::consume(buffer_pool<int32_t>::buffer_t * data){
    assert(false);
}

__host__ void generator::close(){
    {
        buffer_pool<int32_t>::buffer_t::inspector_t insp(strm);
        auto start = chrono::system_clock::now();
        while (N > 0){
            // buffer_pool<int32_t>::buffer_t * buff = buffer_manager<int32_t>::get_buffer();
            buffer_pool<int32_t>::buffer_t * buff = buffer_manager<int32_t>::h_get_buffer(1);

            int m = min(N, buffer_pool<int32_t>::buffer_t::capacity());

            insp.load(buff, true);
            insp.overwrite(src, m, false);

            insp.save(buff, true);

            parent->consume(buff);

            N   -= m;
            src += m;
        }
        auto end   = chrono::system_clock::now();
        cout << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
    }
    {
        auto start = chrono::system_clock::now();
        gpu(cudaStreamSynchronize(strm));
        auto end   = chrono::system_clock::now();
        cout << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms." << endl;
    }
    parent->close();
}

__host__ generator::~generator(){
    gpu(cudaStreamDestroy(strm));
    gpu(cudaFreeHost(buff_ret));
}



// template<>
// __host__ __device__ void push::operator()<generator *>(generator * op) const{
//    op->consume(NULL);
// }
