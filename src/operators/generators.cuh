#ifndef DATA_GENERATORS_CUH_
#define DATA_GENERATORS_CUH_

#include "h_operator.cuh"

using namespace std;

class generator{
private:
    h_operator<int32_t>                 parent;
    cid_t                               cid;
    int32_t                            *src;
    size_t                              N;
public:
    __host__ generator(h_operator<int32_t> parent, int32_t *src, size_t N, cid_t cid);

    __host__ void open();

    __host__ void consume(cnt_t N, vid_t vid, cid_t cid);

    __host__ void close();

    __host__ ~generator();
};

#endif /* DATA_GENERATORS_CUH_ */