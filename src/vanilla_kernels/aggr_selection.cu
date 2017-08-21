#include "vanilla_kernels.cuh"

/**
 * SELECT Sum(b)
 * FROM R
 * WHERE a < thres
 *
 * @pre     @p b should be padded with accessible elements to a size multiple of 4 (the content of this positions is irrelevant)
 */
__global__ void sum_select_less_than(int32_t * __restrict__ a, int32_t * __restrict__ b, size_t N, int32_t thres, int32_t * __restrict__ result){
    const int32_t bigwidth = get_total_num_of_threads();
    const int32_t threadid = get_global_thread_id();

    int32_t res = 0;

    for (size_t j = 4*threadid ; j < N ; j += 4*bigwidth){
        bool predicate[4] = {false, false, false, false};

        vec4 tmp  = reinterpret_cast<vec4*>(a)[j/4];

        #pragma unroll
        for (int k = 0 ; k < 4 ; ++k) if (j + k < N) predicate[k] = (tmp.i[k] < thres);

        vec4 tmp2 = reinterpret_cast<vec4*>(b)[j/4];
        #pragma unroll
        for (int k = 0 ; k < 4 ; ++k) if (predicate[k]) res += tmp2.i[k];//b[j + k];
    }

    atomicAdd(result, res);
}

/**
 * SELECT Sum(b)
 * FROM R
 * WHERE a < thres
 */
__global__ void sum_select_less_than2(int32_t * __restrict__ a, int32_t * __restrict__ b, size_t N, int32_t thres, int32_t * __restrict__ result){
    const int32_t bigwidth = get_total_num_of_threads();
    const int32_t threadid = get_global_thread_id();

    int32_t res = 0;

    for (size_t j = threadid ; j < N ; j += bigwidth) if (a[j] < thres) res += b[j];

    atomicAdd(result, res);
}