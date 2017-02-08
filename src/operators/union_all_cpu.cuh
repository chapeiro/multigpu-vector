#ifndef UNION_ALL_CPU_CUH_
#define UNION_ALL_CPU_CUH_

#include "../buffer_pool.cuh"
#include "operator.cuh"
#include <mutex>

using namespace std;

class union_all_cpu{
private:
public://FIXME: remove
    h_operator_t                                  * parent;
    mutex                                           m;
    const int                                       children;
    int                                             children_closed;
    int                                             children_opened;
public:
    __host__ union_all_cpu(h_operator_t * parent, int children);

    __host__ void open();

    __host__ void consume(buffer_pool<int32_t>::buffer_t * data);

    __host__ void close();
};

#endif /* UNION_ALL_CPU_CUH_ */