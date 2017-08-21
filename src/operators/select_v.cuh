#ifndef SELECT_V_CUH_
#define SELECT_V_CUH_

#include "operator.cuh"
#include <vector>
// #include "output_composer.cuh"

using namespace std;

template<typename T>
struct less_eq_than_v{
private:
    const T bound;

public:
    constexpr less_eq_than_v(T bound): bound(bound){}

    constexpr less_eq_than_v(const less_eq_than_v &o): bound(o.bound){}

    __device__ constexpr __forceinline__ bool operator()(const T &x) const{
        return x <= bound;
    }
};

template<size_t warp_size = WARPSIZE, typename F = less_eq_than_v<int32_t>, typename T = int32_t>
class select_v{
private:
    // static_assert(sizeof(T)*8 == 32, "Operator is only implemented for data types of 32-bits size");
    d_operator_t                   *parent;
    sel_t                          *sel_vec_buffer;
    const F                         filt;
public:
    __host__ select_v(d_operator_t * parent, F f, const launch_conf &conf);

    __host__   void before_open();
    __device__ void at_open();

    __device__ void consume_open();
    __device__ void consume_warp(const T * __restrict__ src, vid_t vid, cid_t cid);
    __device__ void consume_close();

    __device__ void at_close();
    __host__   void after_close();

    __host__ ~select_v(){}
};

#endif /* SELECT_V_CUH_ */