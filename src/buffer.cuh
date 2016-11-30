#ifndef BUFFER_CUH_
#define BUFFER_CUH_

#include "common.cuh"
#include <vector>

using namespace std;


template<typename T, uint32_t size, typename T4>
class buffer_inspector;

template<typename T = int32_t, uint32_t size = DEFAULT_BUFF_CAP, typename T4 = vec4>
class buffer {
    static_assert(sizeof(T) == sizeof(int32_t), "Incompatible size");
    static_assert((size % (4*WARPSIZE)) == 0, "Buffer must have a size multiple of 4*warpSize");
public:
    typedef buffer_inspector<T, size, T4> inspector_t;

private:
    uint32_t cnt; //in wrapSize * T
    T      *data;
    
    friend inspector_t;

    __host__ buffer(int t, int b){}

public:
    __host__ buffer(int device): cnt(0){
        set_device_on_scope d(device);
        gpu(cudaMalloc(&data, size * sizeof(T)));
    }

    __host__ ~buffer(){
        cudaFree(data);
    }

    __host__ __device__ uint32_t count() const{
#ifdef __CUDA_ARCH__
        return min(cnt, size);
#else
        uint32_t h_cnt;
        gpu(cudaMemcpy(&h_cnt, &cnt, sizeof(uint32_t), cudaMemcpyDefault));
        return min(h_cnt, size);
#endif
    }

    __host__ __device__ const T *get_data() const{
#ifdef __CUDA_ARCH__
        return data;
#else
        T * h_data;
        gpu(cudaMemcpy(&h_data, &data, sizeof(T *), cudaMemcpyDefault));
        return h_data;
#endif
    }

    __host__ __device__ uint32_t remaining_capacity() const{
        return size - count(); //cap * warpSize;
    }

    __host__ __device__ static constexpr uint32_t capacity(){
        return size;
    }
    __device__ void clean(){
        cnt = 0;
    }

    __device__ bool full() const volatile{
        return cnt >= size;
    }

    __device__ bool empty() const{
        return cnt == 0;
    }

    __device__ __forceinline__ bool try_write(const T4 &x) volatile{
        uint32_t laneid;
        asm("mov.u32 %0, %%laneid;" : "=r"(laneid));
        uint32_t old_cnt;
        if (laneid == 0) old_cnt = atomicAdd((uint32_t *) &cnt, 4*warpSize);
        old_cnt = broadcast(old_cnt, 0);

        if (old_cnt > size - 4*warpSize) return false;
        reinterpret_cast<T4 *>(data + old_cnt)[laneid] = x; //FIXME: someone may have not completed writing out everything when the buffer gets released!
        // data[old_cnt + laneid] = x; //FIXME: someone may have not completed writing out everything when the buffer gets released!
        return true;
    }

    __device__ __forceinline__ bool try_partial_final_write(const T *x, uint32_t N) volatile{
        uint32_t laneid;
        asm("mov.u32 %0, %%laneid;" : "=r"(laneid));
        if (!try_write(reinterpret_cast<const T4 *>(x)[laneid])) return false;
        if (laneid == 0) atomicSub((uint32_t *) &cnt, 4*warpSize - N);
        // if (cnt + N > size) return false;
        // cnt += N;
        return true;
    }

    __device__ __forceinline__ bool may_write() const{
        return (cnt <= size - 4*warpSize);
    }

    // T * begin(){
    //     return data;
    // }

    // const T * begin() const{
    //     return data;
    // }

    // const T * cbegin() const{
    //     return data;
    // }

    // T * end(){
    //     return data + N;
    // }

    // const T * end() const{
    //     return data + N;
    // }

    // const T * cend() const{
    //     return data + N;
    // }
};

template<typename T = int32_t, uint32_t size = DEFAULT_BUFF_CAP, typename T4 = vec4>
class buffer_inspector{
public:
    typedef buffer<T, size, T4> buffer_t;

private:
    buffer_t     buff;
    cudaStream_t strm;

public:
    buffer_inspector(cudaStream_t strm): buff(5, 6), strm(strm){}

    __host__ void load(buffer_t *d_buff, bool blocking = true){
        gpu(cudaMemcpyAsync(&buff, d_buff, sizeof(buffer_t), cudaMemcpyDefault, strm));
        if (blocking) gpu(cudaStreamSynchronize(strm));
    }

    __host__ uint32_t count(){
        return min(buff.cnt, size);
    }

    __host__ const T * data(){
        return buff.data;
    }
};

#endif /* BUFFER_CUH_ */