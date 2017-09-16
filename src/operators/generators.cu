#include "generators.cuh"
#include "../buffer_manager.cuh"

#include <iostream>
#include <chrono>

// #include <thread>
// #include <chrono>

using namespace std;

__host__ generator::generator(h_operator<int32_t> parent, int32_t *src, size_t N, cid_t cid):
        parent(parent), src(src), N(N), cid(cid){
    // parent->open();

    // cudaStream_t strm;
    // gpu(cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking));
    
    // buffer_pool<int32_t>::buffer_t ** buff_ret;
    // gpu(cudaMallocHost(&buff_ret, sizeof(buffer_pool<int32_t>::buffer_t *)));

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

__host__ void generator::open(){}

__host__ void generator::consume(cnt_t N, vid_t vid, cid_t cid){
    assert(false);
}

__host__ void generator::close(){
    parent.open();
    {
        vid_t vid = 0;
        // buffer_pool<int32_t>::buffer_t::inspector_t insp(strm);
        auto start = chrono::system_clock::now();
        while (N > 0){
            // buffer_pool<int32_t>::buffer_t * buff = buffer_manager<int32_t>::get_buffer();

            // buffer_pool<int32_t>::buffer_t * buff = buffer_manager<int32_t>::h_get_buffer(1);

            size_t m = min(N, (size_t) h_vector_size);

            // insp.load(buff, true);
            // insp.overwrite(src, m, false);

            // insp.save(buff, true);
            // memcpy(buff->data, src, m * sizeof(int32_t));
            // buff->data = src;
            // buff->cnt  = m;
            
            assert(m > 0);
            parent.consume(src, m, vid, cid);

            vid += m;
            N   -= m;
            src += m;
        }
        auto end   = chrono::system_clock::now();
        cout << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
    }
    // {
    //     auto start = chrono::system_clock::now();
    //     // gpu(cudaStreamSynchronize(strm));
    //     auto end   = chrono::system_clock::now();
    //     cout << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms." << endl;
    // }
    parent.close();
}

__host__ generator::~generator(){
    // gpu(cudaStreamDestroy(strm));
    // gpu(cudaFreeHost(buff_ret));
}



// template<>
// __host__ __device__ void push::operator()<generator *>(generator * op) const{
//    op->consume(NULL);
// }



// template<typename... T>
// __host__ multigenerator<T...>::multigenerator(h_operator<T...> parent, T * ... src, size_t N, cid_t cid):
//         parent(parent), src(src...), N(N), cid(cid){
// }

// template<typename... T>
// __host__ multigenerator<T...>::multigenerator(T * ... src, size_t N, cid_t cid):
//         src(src...), N(N), cid(cid){
// }

// template<typename... T>
// __host__ void multigenerator<T...>::open(){}

// template<typename... T>
// __host__ void multigenerator<T...>::consume(cnt_t N, vid_t vid, cid_t cid){
//     assert(false);
// }

// template<int index, typename... Ts>
//  struct increment_tuple {
//      void operator() (tuple<Ts...>& t, size_t inc) {
//          get<index>(t) += inc;
//          increment_tuple<index - 1, Ts...>{}(t, inc);
//      }
//  };

//  template<typename... Ts>
//  struct increment_tuple<0, Ts...> {
//      void operator() (tuple<Ts...>& t, size_t inc) {
//          get<0>(t) += inc;
//      }
//  };

//  template<typename... Ts>
//  void increment(tuple<Ts...>& t, size_t inc) {
//      const auto size = tuple_size<tuple<Ts...>>::value;
//      increment_tuple<size - 1, Ts...>{}(t, inc);
//  }

// template<typename... T>
// __host__ void multigenerator<T...>::close(){
//     parent.open();
//     {
//         vid_t vid = 0;
//         // buffer_pool<int32_t>::buffer_t::inspector_t insp(strm);
//         auto start = chrono::system_clock::now();
//         while (N > 0){
//             // buffer_pool<int32_t>::buffer_t * buff = buffer_manager<int32_t>::get_buffer();

//             // buffer_pool<int32_t>::buffer_t * buff = buffer_manager<int32_t>::h_get_buffer(1);

//             size_t m = min(N, (size_t) h_vector_size);

//             // insp.load(buff, true);
//             // insp.overwrite(src, m, false);

//             // insp.save(buff, true);
//             // memcpy(buff->data, src, m * sizeof(int32_t));
//             // buff->data = src;
//             // buff->cnt  = m;
            
//             assert(m > 0);
//             parent.consume(src, m, vid, cid);

//             vid += m;
//             N   -= m;
//             increment(src, m);
//         }
//         auto end   = chrono::system_clock::now();
//         cout << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
//     }
//     // {
//     //     auto start = chrono::system_clock::now();
//     //     // gpu(cudaStreamSynchronize(strm));
//     //     auto end   = chrono::system_clock::now();
//     //     cout << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms." << endl;
//     // }
//     parent.close();
// }

// template<typename... T>
// __host__ multigenerator<T...>::~multigenerator(){
//     // gpu(cudaStreamDestroy(strm));
//     // gpu(cudaFreeHost(buff_ret));
// }
