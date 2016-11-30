#include <iostream>
#include <chrono>
#include <algorithm>
#include <vector>
#include <thread>
#include <mutex>
// #include <functional>

// namespace std {
//     template<>
//     struct _Reference_wrapper_base<std::thread> {};
// }

#include <future>

using namespace std;

#ifndef NVAL
#define NVAL (10000*1024*4)
#endif

constexpr int N = NVAL;

vector<pair<int32_t *, uint32_t>> data_pool;
mutex data_pool_mutex;

uint32_t consume(int filter, int32_t *dst){
    return 0;
}

int main(){
    // int32_t *dst2;
    // cudaMalloc((void**)&dst2, N*sizeof(int32_t));


    // uint32_t buff_size = WARPSIZE*64;

    // data_pool.clear();

    // for (uint32_t i = 0 ; i < N ; i += buff_size){
    //     data_pool.emplace_back(src+i, min(N - i, buff_size));
    // }
    // cout << data_pool.size() << " " << buff_size << " " << data_pool.back().second << endl;

    // cudaStream_t sta, stb;
    // cudaStreamCreate(&sta);
    // cudaStreamCreate(&stb);


    // unstable_select_gpu<> filter1(dimGrid, dimBlock, sta);
    // unstable_select_gpu<> filter2(dimGrid, dimBlock, stb);

    // cudaEventRecord(start, 0);
    // auto res1 = async(consume, ref(filter1), dst);
    // auto res2 = async(consume, ref(filter2), dst2);

    // gpu(cudaMemcpy(dst+res1.get(), dst2, res2.get()*sizeof(int32_t), cudaMemcpyDefault));
    // uint32_t res = res1.get() + res2.get(); //block until results
    // cudaEventRecord(stop, 0);
    // gpu(cudaFree(dst2));
    // return res;
    return 0;
}
