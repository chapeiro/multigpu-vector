#ifndef DATA_GENERATORS_CUH_
#define DATA_GENERATORS_CUH_

#include "../buffer_pool.cuh"
#include "operator.cuh"
#include <vector>

using namespace std;

class generator{
private:
    h_operator_t                       *parent;
    cudaStream_t                        strm;
    buffer_pool<int32_t>::buffer_t    **buff_ret;
    int32_t                            *src;
    size_t                              N;
public:
    __host__ generator(h_operator_t * parent, int32_t *src, size_t N);

    __host__ void open();

    __host__ void consume(buffer_pool<int32_t>::buffer_t * data);

    __host__ void close();

    __host__ ~generator();
};

#endif /* DATA_GENERATORS_CUH_ */