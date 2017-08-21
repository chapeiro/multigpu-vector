#ifndef DFUNCTIONAL_CUH_
#define DFUNCTIONAL_CUH_

template<typename T>
struct equal_tov{
    T eq;
public:
    constexpr equal_tov(T x): eq(x){}

    constexpr equal_tov(const equal_tov &o): eq(o.eq){}

    __device__ constexpr __forceinline__ bool operator()(const T &x) const{
        return x == eq;
    }
};

struct product{
public:
    __device__ constexpr __forceinline__ int32_t operator()(const int32_t &x, const int32_t &y) const{
        return x * y;
    }
};

template<typename T>
struct log_and{
public:
    __device__ __forceinline__ int32_t operator()(const T &x, const T &y) const{
        return x && y;
    }
};

template<typename T>
struct in_range{
private:
    T a;
    T b;
public:
    constexpr in_range(T a, T b): a(a), b(b){}

    constexpr in_range(const in_range &o): a(o.a), b(o.b){}

    __device__ constexpr __forceinline__ bool operator()(const T &x) const{
        return (a <= x) && (x <= b);
    }
};

template<typename T>
struct less_than{
    T bound;
public:
    constexpr less_than(T x): bound(x){}

    constexpr less_than(const less_than &o): bound(o.bound){}

    __device__ constexpr __forceinline__ bool operator()(const T &x) const{
        return x < bound;
    }
};
#endif /* DFUNCTIONAL_CUH_ */