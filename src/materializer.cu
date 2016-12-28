#include "materializer.cuh"
#include "buffer_manager.cuh"

__host__ materializer::materializer(Operator * parent, int32_t * dst): dst(dst), size(0), ms(0){//, out(&out){
    gpu(cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking));
    insp = new buffer_pool<int32_t>::buffer_t::inspector_t(strm);
    gpu(cudaMallocHost(&out_buff, sizeof(int32_t)*buffer_pool<int32_t>::buffer_t::capacity()));
}

__host__ __device__ void materializer::consume(buffer_pool<int32_t>::buffer_t * data){
#ifdef __CUDA_ARCH__
    assert(false);
#else
    auto start = chrono::system_clock::now();

    insp->load(data, true);
    
    uint32_t       cnt = insp->count();

    insp->read(dst+size, cnt);
    size += cnt;

    // dst.insert(dst.data()+dst.size()-cnt, out_buff, out_buff+cnt);
    // dst.resize(dst.size() + cnt);

    // insp->read(dst.data() + dst.size() - cnt, cnt);

    buffer_manager<int32_t>::release_buffer(data);

    auto end   = chrono::system_clock::now();

    ms += chrono::duration_cast<chrono::microseconds>(end - start);
#endif
}

__host__ __device__ void materializer::join(){
#ifdef __CUDA_ARCH__
    assert(false);
#else
    cout << "=" << chrono::duration_cast<chrono::milliseconds>(ms).count() << endl;
#endif
}


// __host__ __device__ void materializer::consume(buffer_pool<int32_t>::buffer_t * data){
// #ifdef __CUDA_ARCH__
//     assert(false);
// #else
//     auto start = chrono::system_clock::now();

//     insp->load(data, true);
    
//     uint32_t       cnt = insp->count();

//     insp->read(out_buff, cnt);
//     for (uint32_t i = 0 ; i < cnt ; ++i) *out << out_buff[i] << '\n';

//     auto end   = chrono::system_clock::now();

//     ms += chrono::duration_cast<chrono::microseconds>(end - start);
// #endif
// }

// template<>
// __host__ __device__ void push::operator()<materializer *>(materializer * op) const{
//    op->consume(NULL);
// }