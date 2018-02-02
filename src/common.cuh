#ifndef COMMON_CUH_
#define COMMON_CUH_

#include "common/gpu/gpu-common.hpp"
#include <iostream>
#include <cassert>
#include <type_traits>
#include <utmpx.h>
#include <unistd.h>
#include <numaif.h>
#include <tuple>

// #ifndef DEFAULT_BUFF_CAP
// #define DEFAULT_BUFF_CAP (16*1024*1024)
// #endif

// #ifndef WARPSIZE
// #define WARPSIZE (32)
// #endif

// typedef size_t   vid_t;
// typedef uint32_t cid_t;
// typedef uint32_t sel_t;
// typedef uint32_t cnt_t;

// constexpr cnt_t    vector_size   =    32*4*WARPSIZE;
// constexpr cnt_t    h_vector_size = DEFAULT_BUFF_CAP;
// constexpr uint32_t warp_size     =         WARPSIZE;

#ifndef NCUDA
#define gpu(ans) { gpuAssert((ans), __FILE__, __LINE__); }

// __host__ __device__ inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
//     if (code != cudaSuccess) {
// #ifndef __CUDA_ARCH__
//         fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
//         if (abort) exit(code);
// #else
//         printf("GPUassert: %s %s %d\n", "error", file, line);
// #endif
//     }
// }

// #if __CUDA_ARCH__ < 300 || defined (NUSE_SHFL)
#define BRDCSTMEM(blockDim) ((blockDim.x * blockDim.y)/ WARPSIZE)
// #else
// #define BRDCSTMEM(blockDim) (0)
// #endif

// template<typename T,
//          typename std::enable_if<sizeof(T) == sizeof(unsigned long long int),
//             int>::type = 0>
// __device__ T atomicExch(T *address, T val){
//     return (T) atomicExch((unsigned long long int*) address, (unsigned long long int) val);
// }

// template<typename T,
//          typename std::enable_if<sizeof(T) == sizeof(unsigned int) && !std::is_signed<T>::value,
//             int>::type = 0>
// __device__ T atomicExch(T *address, T val){
//     return (T) atomicExch((unsigned int*) address, (unsigned int) val);
// }

// template<typename T,
//          typename std::enable_if<sizeof(T) == sizeof(int) && std::is_signed<T>::value,
//             int>::type = 0>
// __device__ T atomicExch(T *address, T val){
//     return (T) atomicExch((int*) address, (int) val);
// }

template<typename T,
         typename std::enable_if<sizeof(T) == sizeof(unsigned long long int),
            int>::type = 0>
__device__ T atomicCAS(T *address, T comp, T val){
    return (T) atomicCAS((unsigned long long int*) address, (unsigned long long int) comp, (unsigned long long int) val);
}

template<typename T,
         typename std::enable_if<sizeof(T) == sizeof(unsigned int) && !std::is_signed<T>::value,
            int>::type = 0>
__device__ T atomicCAS(T *address, T comp, T val){
    return (T) atomicCAS((unsigned int*) address, (unsigned int) comp, (unsigned int) val);
}

template<typename T,
         typename std::enable_if<sizeof(T) == sizeof(int) && std::is_signed<T>::value,
            int>::type = 0>
__device__ T atomicCAS(T *address, T comp, T val){
    return (T) atomicCAS((int*) address, (int) comp, (int) val);
}

// template<typename T,
//          typename std::enable_if<sizeof(T) == sizeof(unsigned long long int),
//             int>::type = 0>
// __device__ T atomicAdd(T *address, T val){
//     return (T) atomicAdd((unsigned long long int*) address, (unsigned long long int) val);
// }

// template<typename T,
//          typename std::enable_if<sizeof(T) == sizeof(unsigned int) && !std::is_signed<T>::value,
//             int>::type = 0>
// __device__ T atomicAdd(T *address, T val){
//     return (T) atomicAdd((unsigned int*) address, (unsigned int) val);
// }

// template<typename T,
//          typename std::enable_if<sizeof(T) == sizeof(int) && std::is_signed<T>::value,
//             int>::type = 0>
// __device__ T atomicAdd(T *address, T val){
//     return (T) atomicAdd((int*) address, (int) val);
// }

template<typename T,
         typename std::enable_if<sizeof(T) == sizeof(unsigned long long int),
            int>::type = 0>
__device__ T atomicSub(T *address, T val){
    return (T) atomicSub((unsigned long long int*) address, (unsigned long long int) val);
}

template<typename T,
         typename std::enable_if<sizeof(T) == sizeof(unsigned int) && !std::is_signed<T>::value,
            int>::type = 0>
__device__ T atomicSub(T *address, T val){
    return (T) atomicSub((unsigned int*) address, (unsigned int) val);
}

template<typename T,
         typename std::enable_if<sizeof(T) == sizeof(int) && std::is_signed<T>::value,
            int>::type = 0>
__device__ T atomicSub(T *address, T val){
    return (T) atomicSub((int*) address, (int) val);
}

template<typename T, typename... Args>
__host__ T * cuda_new(int dev, Args... args){
    if (dev >= 0){
        set_device_on_scope d(dev);
        T *tmp = new T(args...);
        T *res;
        gpu(cudaMalloc((void**) &res, sizeof(T)));
        gpu(cudaMemcpy(res, tmp, sizeof(T), cudaMemcpyDefault));
        gpu(cudaDeviceSynchronize());
        free(tmp);  //NOTE: bad practice ? we want to allocate tmp by new to
                    //      trigger initialization but we want to free the 
                    //      corresponding memory after moving to device 
                    //      without triggering the destructor
        return res;
    } else {
        T *tmp = new T(args...);
        T *res;
        gpu(cudaMallocHost((void**) &res, sizeof(T)));
        gpu(cudaMemcpy(res, tmp, sizeof(T), cudaMemcpyDefault));
        gpu(cudaDeviceSynchronize());
        free(tmp);  //NOTE: bad practice ? we want to allocate tmp by new to
                    //      trigger initialization but we want to free the 
                    //      corresponding memory after moving to device 
                    //      without triggering the destructor
        return res;
        // return new T(args...);
    }
}


template<typename T, typename... Args>
__host__ void cuda_delete(T *obj, Args... args){
    int device = get_device(obj);
    if (device >= 0){
        T *tmp = (T *) malloc(sizeof(T));
        gpu(cudaDeviceSynchronize());
        gpu(cudaMemcpy(tmp, obj, sizeof(T), cudaMemcpyDefault));
        gpu(cudaFree(obj));
        delete tmp;
    } else {
        T *tmp = (T *) malloc(sizeof(T));
        gpu(cudaDeviceSynchronize());
        gpu(cudaMemcpy(tmp, obj, sizeof(T), cudaMemcpyDefault));
        gpu(cudaFreeHost(obj));
        delete tmp;
        // delete obj;
    }
}

struct launch_conf{
    dim3    gridDim     ;
    dim3    blockDim    ;
    int     shared_mem  ;
    int     device      ;

    __host__ __device__ __forceinline__ launch_conf(): device(-5){}

    __host__ __device__ __forceinline__ launch_conf(const dim3 &gridDim, const dim3 &blockDim, int shared_mem, int device): gridDim(gridDim), blockDim(blockDim), shared_mem(shared_mem), device(device){}
    __host__ __device__ __forceinline__ launch_conf(const launch_conf &o): gridDim(o.gridDim), blockDim(o.blockDim), shared_mem(o.shared_mem), device(o.device){}

    int get_blocks_per_grid() const{
        return gridDim.x  * gridDim.y  * gridDim.z ;
    }

    int get_threads_per_block() const{
        return blockDim.x * blockDim.y * blockDim.z;
    }

    int get_warps_per_block() const{
        return (get_threads_per_block() + WARPSIZE - 1) / WARPSIZE;
    }

    int total_num_of_warps() const{
        return get_blocks_per_grid() * get_warps_per_block();
    }
};

// __device__ __forceinline__ int get_laneid(){
//     uint32_t laneid;
//     asm("mov.u32 %0, %%laneid;" : "=r"(laneid));
//     return laneid;
// }

__device__ __forceinline__ int get_threadid(){
    return threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;
}

__device__ __forceinline__ int get_warpid(){
    return ::get_threadid() / warpSize;
}

__device__ __forceinline__ int get_blockid(){
    return blockIdx.x + gridDim.x * blockIdx.y + gridDim.x * gridDim.y * blockIdx.z;
}

__device__ __forceinline__ int get_threads_per_block(){
    return blockDim.x * blockDim.y * blockDim.z;
}

__device__ __forceinline__ int get_blocks_per_grid(){
    return gridDim.x * gridDim.y * gridDim.z;
}

__device__ __forceinline__ int get_warps_per_block(){
    return (::get_threads_per_block() + WARPSIZE - 1)/WARPSIZE;
}

__device__ __forceinline__ int get_global_warpid(){
    return ::get_blockid() * ::get_warps_per_block() + ::get_warpid();
}

__device__ __forceinline__ int get_global_thread_id(){
    return ::get_blockid() * ::get_threads_per_block() + ::get_threadid();
}

__device__ __forceinline__ int get_total_num_of_warps(){
    return ::get_warps_per_block() * ::get_blocks_per_grid();
}

__device__ __forceinline__ int get_total_num_of_threads(){
    return ::get_threads_per_block() * ::get_blocks_per_grid();
}

__device__ __host__ int __forceinline__ find_nth_set_bit(uint32_t c1, unsigned int n){
    int t, i = n, r = 0;
    uint32_t c2  = c1 - ((c1 >> 1) & 0x55555555);
    uint32_t c4  = ((c2 >> 2) & 0x33333333) + (c2 & 0x33333333);
    uint32_t c8  = ((c4 >> 4) + c4) & 0x0f0f0f0f;
    uint32_t c16 = ((c8 >> 8) + c8);
    uint32_t c32 = ((c16 >> 16) + c16) & 0x3f;
    t = (c16    ) & 0x1f; if (i >= t) { r += 16; i -= t; }
    t = (c8 >> r) & 0x0f; if (i >= t) { r +=  8; i -= t; }
    t = (c4 >> r) & 0x07; if (i >= t) { r +=  4; i -= t; }
    t = (c2 >> r) & 0x03; if (i >= t) { r +=  2; i -= t; }
    t = (c1 >> r) & 0x01; if (i >= t) { r +=  1;         }
    if (n >= c32) r = -1;
    return r; 
}

union vec4{
    int4 vec;
    int  i[4];
};

template<typename T>
__device__ __host__ inline constexpr T round_up(T num, T mult){
    return ((num + mult - 1) / mult) * mult;
}

template<typename T>
__device__ __host__ inline constexpr T round_down(T num, T mult){
    return (num / mult) * mult;
}

#if __CUDA_ARCH__ < 300 || defined (NUSE_SHFL)
template<typename T>
__device__ __forceinline__ T brdcst(T val, uint32_t src){
#ifdef __CUDA_ARCH__
    uint32_t laneid;
    asm("mov.u32 %0, %%laneid;" : "=r"(laneid));

    volatile int32_t *bcount = (int32_t *) (s + 9 * blockDim.x * blockDim.y);
    uint32_t warpid;
    asm("mov.u32 %0, %%warpid;" : "=r"(warpid));

    if (laneid == src) bcount[warpid] = val;
    return bcount[warpid];
#endif
}
#else
   #define brdcst(v, l) (__shfl(v, l))
#endif

template<typename T>
int get_numa_addressed(T * m){
    T * mtest = m;//(T *) (((uintptr_t) m) & ~(((uintptr_t) getpagesize()) - 1));
    int status[1];
    status[0]=-1;
#ifndef NDEBUG
    int ret_code = 
#endif
    move_pages(0 /*self memory */, 1, (void **) &mtest, NULL, status, 0);
    assert(ret_code == 0);
    return status[0];
}

//Can not use this as in the end it calls a host function...
template<typename T, typename Operator, typename... Args>
__global__ void run_on_device_kernel(T obj, Operator op, typename std::result_of<Operator(T, Args...)>::type *ret, Args... args){
    *ret = (obj->*op)(args...);
}

template<typename T, class Operator, class... Args>
__host__ typename std::result_of<Operator(T, Args...)>::type run_on_device(T obj, Operator op, Args... args){
    typedef typename std::result_of<Operator(T, Args...)>::type ret_t;

    ret_t * buff;
    ret_t * buff_ret;
    gpu(cudaMalloc(&buff, sizeof(ret_t)));
    cudaMallocHost(&buff_ret, sizeof(ret_t));
    cudaPointerAttributes attrs;
    gpu(cudaPointerGetAttributes(&attrs, obj));
    set_device_on_scope d(attrs.device);
    cudaStream_t strm;
    cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking);
    run_on_device_kernel<<<1, 1, 0, strm>>>(obj, op, buff, args...);
    
    cudaMemcpyAsync(buff_ret, buff, sizeof(ret_t), cudaMemcpyDefault, strm);
    cudaStreamSynchronize(strm);
    cudaStreamDestroy(strm);
    cudaFree(buff);
    ret_t buff_ret_r = *buff_ret;
    cudaFreeHost(buff_ret);

    return buff_ret_r;
}

// template<typename T, typename Operator, typename... Args>
// __global__ void run_on_device_kernel_void(T obj, Operator op, Args... args){
//     (obj->*op)(args...);
// }

// template<typename T, class Operator, class... Args>
// __host__ void run_on_device_void(T obj, Operator op, Args... args){
//     cudaPointerAttributes attrs;
//     gpu(cudaPointerGetAttributes(&attrs, obj));
//     set_device_on_scope d(attrs.device);
//     cudaStream_t strm;
//     cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking);
//     run_on_device_kernel_void<<<1, 1, 0, strm>>>(obj, op, args...);
    
//     cudaStreamSynchronize(strm);
//     cudaStreamDestroy(strm);
// }

// class Managed{
// public:
//     void *operator new(size_t len){
//         void *ptr;
//         cudaMallocManaged(&ptr, len);
//         return ptr;
//     }

//     void operator delete(void *ptr){
//         cudaFree(ptr);
//     }
// };

// Based on: http://stackoverflow.com/a/10766422/1237824
namespace tuple_expansion{
    using namespace std;
    template <typename F, typename Tuple, bool Done, int Total, int... N>
    struct call_impl{
        __host__ __device__ static void call(F f, Tuple &&t){
            call_impl<F, Tuple, Total == 1 + sizeof...(N), Total, N..., sizeof...(N)>::call(f, std::forward<Tuple>(t));
        }
    };

    template <typename F, typename Tuple, int Total, int... N>
    struct call_impl<F, Tuple, true, Total, N...>{
        __host__ __device__ static void call(F f, Tuple && t){
            f(get<N>(std::forward<Tuple>(t))...);
        }
    };

    template <typename F, typename T, typename Tuple, bool Done, int Total, int... N>
    struct call_impl_obj{
        __host__ __device__ static void call(F f, T * obj, Tuple &&t){
            call_impl_obj<F, T, Tuple, Total == 1 + sizeof...(N), Total, N..., sizeof...(N)>::call(f, obj, std::forward<Tuple>(t));
        }
    };

    template <typename F, typename T, typename Tuple, int Total, int... N>
    struct call_impl_obj<F, T, Tuple, true, Total, N...>{
        __host__ __device__ static void call(F f, T * obj, Tuple && t){
            (obj->*f)(get<N>(std::forward<Tuple>(t))...);
        }
    };
}

template <typename F, typename Tuple>
__host__ __device__ void call(F f, Tuple && t){
    typedef typename std::decay<Tuple>::type ttype;
    tuple_expansion::call_impl<F, Tuple, 0 == std::tuple_size<ttype>::value, std::tuple_size<ttype>::value>::call(f, std::forward<Tuple>(t));
}

template <typename F, typename T, typename Tuple>
__host__ __device__ void call(F f, T * obj, Tuple && t){
    typedef typename std::decay<Tuple>::type ttype;
    tuple_expansion::call_impl_obj<F, T, Tuple, 0 == std::tuple_size<ttype>::value, std::tuple_size<ttype>::value>::call(f, obj, std::forward<Tuple>(t));
}

// template <typename... Ttuple, typename F, typename Tuple>
// __host__ __device__ void call2(F f, Tuple && t){
//     typedef typename std::decay<Tuple>::type ttype;
//     tuple_expansion::call_impl<F, Tuple, 0 == sizeof...(Ttuple), sizeof...(Ttuple)>::call(f, std::forward<Tuple>(t));
// }

// template <typename... Ttuple, typename F, typename T, typename Tuple>
// __host__ __device__ void call2(F f, T * obj, Tuple && t, cnt_t N, vid_t vid, cid_t cid){
//     typedef typename std::decay<Tuple>::type ttype;
//     tuple_expansion::call_impl_obj<F, T, Tuple, 0 == sizeof...(Ttuple), sizeof...(Ttuple)>::call(f, obj, std::forward<Tuple>(t));
// }

// http://stackoverflow.com/a/17426611/1237824
// using aliases for cleaner syntax
template<class T> using Invoke = typename T::type;

template<unsigned...> struct seq{ using type = seq; };

template<class S1, class S2> struct concat;

template<unsigned... I1, unsigned... I2>
struct ::concat<seq<I1...>, seq<I2...>>
  : seq<I1..., (sizeof...(I1)+I2)...>{};

template<class S1, class S2>
using Concat = Invoke<::concat<S1, S2>>;

template<unsigned N> struct gen_seq;
template<unsigned N> using GenSeq = Invoke<gen_seq<N>>;

template<unsigned N>
struct gen_seq : Concat<GenSeq<N/2>, GenSeq<N - N/2>>{};

template<> struct gen_seq<0> : seq<>{};
template<> struct gen_seq<1> : seq<0>{};
#else 
#define gpu(ans) 
#endif

#endif /* COMMON_CUH_ */