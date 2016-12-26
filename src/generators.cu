#include "generators.cuh"
#include "buffer_manager.cuh"

#include <iostream>
#include <chrono>

using namespace std;

__host__ generator::generator(Operator * parent, int32_t *src, uint32_t N):
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

__host__ __device__ void generator::consume(buffer_pool<int32_t>::buffer_t * data){
#ifdef __CUDA_ARCH__
    assert(false);
#else
    buffer_pool<int32_t>::buffer_t::inspector_t insp(strm);
    auto start = chrono::system_clock::now();
    while (N > 0){
        buffer_pool<int32_t>::buffer_t * buff = buffer_manager<int32_t>::get_buffer();

        int m = min(N, buffer_pool<int32_t>::buffer_t::capacity());

        insp.load(buff, true);
        insp.overwrite(src, m);

        insp.save(buff, true);

        parent->consume(buff);

        N   -= m;
        src += m;
    }
    auto end   = chrono::system_clock::now();
    cout << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
#endif
}

__host__ __device__ void generator::join(){
#ifdef __CUDA_ARCH__
    assert(false);
#else
    auto start = chrono::system_clock::now();
    gpu(cudaStreamSynchronize(strm));
    parent->close();
    auto end   = chrono::system_clock::now();
    cout << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
#endif
}

__host__ generator::~generator(){
    gpu(cudaStreamDestroy(strm));
    gpu(cudaFreeHost(buff_ret));
}



// template<>
// __host__ __device__ void push::operator()<generator *>(generator * op) const{
//    op->consume(NULL);
// }
