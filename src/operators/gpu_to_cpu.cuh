#ifndef GPU_TO_CPU_CUH_
#define GPU_TO_CPU_CUH_

#include "../buffer_manager.cuh"
#include "h_operator.cuh"
#include <thread>
#include <atomic>
#include <thrust/tuple.h>

using namespace std;

template<typename... T>
struct packet{
    thrust::tuple<T * ...>  vectors;
    cnt_t           N                    ;
    vid_t           vid                  ;

    __device__ __host__ volatile packet<T...>& operator=(const packet<T...> &rhs) volatile{
        ((thrust::tuple<T * ...> &) vectors) = rhs.vectors;
        N       = rhs.N;
        vid     = rhs.vid;
        return *this;
    }

    __device__ __host__ packet(){}

    __device__ __host__ packet(volatile packet<T...> &rhs){
        vectors = ((thrust::tuple<T * ...> &) rhs.vectors);
        N       = rhs.N;
        vid     = rhs.vid;
    }
};

template<int rem, int cons, typename P, typename Th, typename... T>
struct packet_handler{
    __device__ static __forceinline__ void write(const Th * __restrict__ srch, const T * __restrict__ ... src, cnt_t base, cnt_t N, P &p){
        Th * dst = get<cons>(p.current_packet.vectors) + base;

        const int laneid = get_laneid();

        for (int i = 0 ; i < vector_size ; i += warp_size) if (i + laneid < N) dst[i + laneid] = srch[i + laneid];

        packet_handler<rem - 1, cons + 1, P, T...>::write(src..., base, N, p);
    }

    __device__ static __forceinline__ void replace(P &p){
        assert(__ballot(1) == 1);
        get<cons>(p.current_packet.vectors) = (Th *) buffer_manager<int32_t>::get_buffer();

        packet_handler<rem - 1, cons + 1, P, T...>::replace(p);
    }
};

template<int cons, typename P, typename Th, typename... T>
struct packet_handler<1, cons, P, Th, T...>{
    static_assert(sizeof...(T) == 0, "Oooopsss...");

    __device__ static __forceinline__ void write(const Th * __restrict__ srch, const T * __restrict__ ... src, cnt_t base, cnt_t N, P &p){
        Th * dst = get<cons>(p.current_packet.vectors) + base;

        const int laneid = get_laneid();

        for (int i = 0 ; i < vector_size ; i += warp_size) if (i + laneid < N) dst[i + laneid] = srch[i + laneid];

        cnt_t filled;
        if (laneid == 0) filled = atomicAdd(&(p.current_packet.N), N);
        filled = brdcst(filled, 0) + N;

        if (filled == h_vector_size){
            //FIXME: push & replace!!!
            if (laneid == 0) {
                p.throw2cpu();
                p.replace_buffers();
            }
        }
    }

    __device__ static __forceinline__ void replace(P &p){
        assert(__ballot(1) == 1);
        get<cons>(p.current_packet.vectors) = (Th *) buffer_manager<int32_t>::get_buffer();

        __threadfence(); //FIXME: is it needed ?
        p.current_packet.vid += p.current_packet.N;
        p.current_packet.N = 0;
        __threadfence(); //FIXME: is it needed ?
        p.packet_used      = 0;
        __threadfence(); //FIXME: is it needed ?
    }
};

template<size_t size = 64, typename... T>
class gpu_to_cpu{
private:
    class gpu_to_cpu_host{
    public:
        h_operator<T...>                        parent;
    private:
        volatile packet<T...>                  *store;
        volatile int                           *flags;
        volatile int                           *eof;
        size_t                                  front;
        int                                     device;

    public:
        void catcher(cid_t ocid, int device);

    public:
        gpu_to_cpu_host(h_operator<T...> parent, volatile packet<T...> *store, volatile int *flags, volatile int *eof);

        ~gpu_to_cpu_host();
    };

public:
    packet<T...>                 current_packet;
    cnt_t                        packet_used;
    __device__ void throw2cpu();
    __device__ void replace_buffers();

private:
    volatile int                 lock;
    volatile int                 end;

    volatile packet<T...>       *store;
    volatile int                *flags;
    volatile int                *eof;
    thread                      *teleporter_catcher;
    gpu_to_cpu_host             *teleporter_catcher_obj;


public:
    __host__ gpu_to_cpu(h_operator<T...> parent, cid_t ocid, int device);

    __host__   void before_open();
    __device__ void at_open();

    __device__ void consume_open();
    __device__ void consume_warp(const T * __restrict__ ... src, cnt_t N, vid_t vid, cid_t cid);
    __device__ void consume_close();

    __device__ void at_close();
    __host__   void after_close();

    __host__ ~gpu_to_cpu();
};


#endif /* GPU_TO_CPU_CUH_ */