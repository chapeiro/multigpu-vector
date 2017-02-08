#ifndef MEM_MOVE_CUH_
#define MEM_MOVE_CUH_

#include "../buffer_pool.cuh"
#include "operator.cuh"

using namespace std;

class mem_move{
private:
    h_operator_t                                  * parent;
    cudaStream_t                                    strm;
public:
    __host__ mem_move(h_operator_t * parent);

    __host__ void open();

    __host__ void consume(buffer_pool<int32_t>::buffer_t * data);

    __host__ void close();
};

#endif /* MEM_MOVE_CUH_ */