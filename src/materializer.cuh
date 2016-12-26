#ifndef MATERIALIZER_CUH_
#define MATERIALIZER_CUH_

#include "buffer_pool.cuh"
#include "operator.cuh"
#include <vector>
#include <chrono>
#include <deque>

using namespace std;

class materializer{
private:
public://FIXME: remove
    int32_t*                                        dst;
    cudaStream_t                                    strm;
    buffer_pool<int32_t>::buffer_t::inspector_t    *insp;
    int32_t                                        *out_buff;
    size_t                                          size;
    // ostream                                        *out;

    
    chrono::microseconds ms;
public:
    __host__ materializer(Operator * parent, int32_t *dst);
    // __host__ materializer(Operator * parent, ostream &out);

    __host__ __device__ void open(){}

    __host__ __device__ void consume(buffer_pool<int32_t>::buffer_t * data);

    __host__ __device__ void join(){cout << "=" << chrono::duration_cast<chrono::milliseconds>(ms).count() << endl;}
};

#endif /* MATERIALIZER_CUH_ */