#ifndef CPU_TO_GPU_CUH_
#define CPU_TO_GPU_CUH_

#include "d_operator.cuh"

template<typename... T>
class cpu_to_gpu{
private:
    launch_conf        conf     ;
    d_operator<T...>  *parent   ;
    cudaStream_t       strm_op  ;
    cudaStream_t       strm_mem ;
    int32_t           *old_buffs; //FIXME: generalize
public:
    __host__ cpu_to_gpu(d_operator<T...> *parent, launch_conf conf);

    __host__ void open();

    __host__ void consume(const T * ... src, cnt_t N, vid_t vid, cid_t cid);

    __host__ void close();

    template<typename... Tw>
    __host__ void serialize_with(cpu_to_gpu<Tw...> *t);

    __host__ ~cpu_to_gpu();
};

#endif /* CPU_TO_GPU_CUH_ */