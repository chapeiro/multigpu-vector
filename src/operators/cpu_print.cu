#include "cpu_print.cuh"
#include <iostream>

using namespace std;

template<typename T>
__host__ void cpu_print<T>::open(){}

template<typename T>
__host__ void cpu_print<T>::close(){}

template<typename T>
__host__ void cpu_print<T>::consume(const T * src, cnt_t N, vid_t vid, cid_t cid){
    // assert(src[0] == 1470726341);
    for (size_t i = 0 ; i < N ; ++i) cout << src[i] << " ";
    cout << endl;
}

template class cpu_print<int32_t>;


