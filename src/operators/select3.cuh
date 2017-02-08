#ifndef SELECT3_CUH_
#define SELECT3_CUH_

#include "operator.cuh"
#include <vector>
// #include "output_composer.cuh"

using namespace std;

template<typename T>
struct less_eq_than{
private:
    const T bound;

public:
    constexpr less_eq_than(T bound): bound(bound){}

    constexpr less_eq_than(const less_eq_than &o): bound(o.bound){}

    __device__ constexpr __forceinline__ bool operator()(const T &x) const{
        return x <= bound;
    }
};

template<size_t warp_size = WARPSIZE, typename F = less_eq_than<int32_t>, typename... T>
class unstable_select{
private:
    // static_assert(sizeof(T)*8 == 32, "Operator is only implemented for data types of 32-bits size");
    d_operator_t                  * parent;
    // output_composer<warp_size, T>   output;
    volatile uint32_t               buffer_size;
    uint32_t                        finished;
    int32_t                        *buffer;
    const F                         filt;
public:
    __host__ unstable_select(d_operator_t * parent, F f, int grid_size = 0, int dev = 0);

    __host__   void before_open();
    __device__ void at_open();

    __device__ void consume_open();
    __device__ void consume_warp(const T *... src, unsigned int N);
    __device__ void consume_close();

    __device__ void at_close();
    __host__   void after_close();

    __host__ ~unstable_select(){}
};

#endif /* SELECT3_CUH_ */