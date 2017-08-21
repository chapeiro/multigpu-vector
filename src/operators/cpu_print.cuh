#ifndef CPU_PRINT_CUH_
#define CPU_PRINT_CUH_

#include "../common.cuh"

template<typename T>
class cpu_print{
public:
    __host__ void open();

    __host__ void consume(const T * src, cnt_t N, vid_t vid, cid_t cid);

    __host__ void close();
};

#endif /* CPU_PRINT_CUH_ */