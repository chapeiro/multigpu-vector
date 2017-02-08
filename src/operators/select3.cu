#include "select3.cuh"

using namespace std;

template<size_t warp_size, typename F, typename... T>
__host__ unstable_select<warp_size, F, T...>::unstable_select(d_operator_t * parent, F f, int grid_size, int dev): 
        parent(parent), buffer_size(0), finished(0), filt(f){
    // output = cuda_new<output_composer<warp_size, T>>(dev, parent, dev);

    assert(dev >= 0);
    set_device_on_scope d(dev);

    gpu(cudaMalloc((void**)&buffer, ((grid_size + 1) * 4 * warp_size + 3) * sizeof(int32_t)));//FIXME: corrent type

    gpu(cudaMemset(buffer, 0, ((grid_size + 1) * 4 * warp_size + 3) * sizeof(int32_t)));//FIXME: corrent type
    // gpu(cudaMemset(buffer + (grid_size + 1) * (4 * warp_size - 1), 0, (grid_size + 4) * sizeof(int32_t)));//FIXME: corrent type
}

template<size_t warp_size, typename F, typename... T>
__device__ void unstable_select<warp_size, F, T...>::consume_warp(const T *... src, unsigned int N){
    extern __shared__ int32_t s[];

    const int32_t warpid            = get_warpid();
    const int32_t laneid            = get_laneid();
    // const int32_t width             = blockDim.x * blockDim.y;

    const int32_t prevwrapmask      = (1 << laneid) - 1;

    // volatile int32_t *fcount      = (int32_t *) s;
    // volatile int32_t *wrapoutbase = (int32_t *) (s + (width + warpSize - 1) / warpSize);
    volatile int32_t *wrapoutbase   = ((int32_t *) s);
    // volatile int32_t *fcount        = ((int32_t *) s) + 5 * warp_size * (((width + warp_size - 1) / warp_size) + 1);
    volatile int32_t *wrapoutput    = wrapoutbase + 5 * warp_size * warpid;

    int32_t filterout = wrapoutput[5*warp_size - 1];

    F filter(filt);

    vec4 x;
    #pragma unroll
    for (int k = 0 ; k < 4 ; ++k){
        if (k*warpSize + laneid < N){
            //compute predicate
            x.i[k] = get<0>(make_tuple(src[k*warpSize + laneid]...));
        }
    }

    #pragma unroll
    for (int k = 0 ; k < 4 ; ++k){
        bool predicate = false;
        if (k*warpSize + laneid < N){
            //compute predicate
            predicate = filter(x.i[k]);
        }

        //aggregate predicate results
        int32_t filter = __ballot(predicate); //filter now contains set bits only for threads

        int32_t newpop = __popc(filter);

        assert(filterout < 4*warpSize);

        //compute position of result
        if (predicate){
            int32_t offset = filterout + __popc(filter & prevwrapmask);
            assert(offset >= 0);
            assert(offset <  5*warpSize);
            wrapoutput[offset] = x.i[k];// x.i[k];//input[blockDim.x*k + i];
        }

        filterout += newpop;

        if (filterout >= 4*warpSize){
            // output.push(wrapoutput);
            parent->consume_warp((const int32_t *) wrapoutput, 4*warpSize);

            wrapoutput[laneid]             = wrapoutput[laneid + 4*warpSize];
            filterout                     -= 4*warpSize;
        }

        // __syncthreads(); //FIXME: this should not be needed, but racecheck produces errors without it
    }

    if (laneid == 0) wrapoutput[5*warp_size - 1] = filterout;
}

template<size_t warp_size, typename F, typename... T>
__device__ void unstable_select<warp_size, F, T...>::consume_close(){
    extern __shared__ int32_t s[];

    const int32_t warpid            = get_warpid();
    const int32_t laneid            = get_laneid();
    // const int32_t width             = blockDim.x * blockDim.y;

    const int32_t gridwidth         = gridDim.x     * gridDim.y ;

    // volatile int32_t *fcount      = (int32_t *) s;
    // volatile int32_t *wrapoutbase = (int32_t *) (s + (width + warpSize - 1) / warpSize);
    volatile int32_t *wrapoutbase = ((int32_t *) s);
    volatile int32_t *wrapoutput  = wrapoutbase + 5 * warp_size * warpid;
    volatile int32_t *fcount      = wrapoutput  + 5 * warp_size - 1;

    int32_t filterout = *fcount;

    for (int32_t m = 1 ; m <= 5 ; ++m){ //fixme: not until 5, but until ceil(log(max warpid)) ? also ceil on target_filter_out condition
        int32_t mask = (1 << m) - 1;

        if (!(warpid & mask)){
            int32_t target_wrapid               = warpid + (1 << (m - 1));
            
            volatile int32_t *target_wrapoutput = wrapoutbase + 5 * warpSize * target_wrapid;

            int32_t target_filter_out           = (target_wrapid < blockDim.x * blockDim.y/warpSize) ? target_wrapoutput[5*warp_size - 1] : 0;
            int32_t target_filter_out_rem       = target_filter_out;

            assert(target_filter_out < 4*warpSize);
            assert(target_filter_out >= 0);

            for (int32_t k = 0; k < (target_filter_out + warpSize - 1)/warpSize ; ++k){
                assert(k < 4);
                if (laneid + k * warpSize < target_filter_out) {
                    assert(filterout + laneid < 5*warpSize);
                    wrapoutput[filterout + laneid] = target_wrapoutput[laneid + k * warpSize];
                }
                int32_t delta = min(target_filter_out_rem, warpSize);
                target_filter_out_rem -= delta;
                filterout += delta;

                if (filterout >= 4*warpSize){
                    // output.push(wrapoutput);
                    parent->consume_warp((const int32_t *) wrapoutput, 4*warpSize);

                    wrapoutput[laneid]             = wrapoutput[laneid + 4*warpSize];
                    filterout                     -= 4*warpSize;
                }
            }

            //no __syncthreads is needed here due to the pattern of accesses on fcount
            if (laneid == 0) *fcount = filterout;
        }
        __syncthreads();
    }

    if (warpid == 0 && filterout){
        int32_t elems_old;
        if (laneid == 0) {
            elems_old = atomicAdd((uint32_t *) &buffer_size, filterout);
            assert((elems_old + filterout) <= (gridwidth + 1) * (4 * warp_size - 1));
        }
        elems_old = brdcst(elems_old, 0);

        volatile int32_t * buffoff = buffer  + elems_old;
        volatile int32_t * aligned = (int32_t *) round_up((uintptr_t) buffoff, warpSize * sizeof(int32_t));
        int32_t preamble  = min((int32_t) (aligned - buffoff), filterout);
        int32_t rem_elems = filterout - preamble;

        if (laneid < preamble){
            buffoff[laneid] = wrapoutput[laneid];
        }

        for (int32_t k = laneid; k < rem_elems ; k += warpSize){
            aligned[k] = wrapoutput[preamble + k];
        }

        int32_t * cnts = buffer + ((4*warpSize-1)*(gridwidth+1));

        int32_t bnum0  = elems_old/(4*warpSize);
        int32_t bnum1  = (elems_old + filterout)/(4*warpSize);

        int32_t nset0  = (bnum0 == bnum1) ? filterout : (bnum1 * 4 * warpSize - elems_old);
        int32_t nset1  = filterout - nset0;

        int32_t totcnt0;
        if (laneid == 0) totcnt0 = atomicAdd(cnts + bnum0, nset0);
        totcnt0 = brdcst(totcnt0, 0) + nset0;

        int32_t totcnt1 = -1;
        if (nset1){
            if (laneid == 0) totcnt1 = atomicAdd(cnts + bnum1, nset1);
            totcnt1 = brdcst(totcnt1, 0) + nset1;
        }

        if (totcnt0 >= 4*warpSize){
            if (totcnt0 > 4*warpSize) printf("%d %d %d %d\n", elems_old, filterout, (gridwidth + 1) * (4 * warp_size - 1), totcnt0);
            assert(totcnt0 <= 4*warpSize);
            // output.push(buffer+bnum0*(4*warpSize));
            parent->consume_warp(buffer+bnum0*(4*warpSize), 4*warpSize);

            if (laneid == 0) cnts[bnum0] = 0; //clean up for next round
        }
        if (totcnt1 >= 4*warpSize){
            if (totcnt1 > 4*warpSize) printf("%d\n", totcnt1);
            assert(totcnt1 <= 4*warpSize);
            // output.push(buffer+bnum1*(4*warpSize));
            parent->consume_warp(buffer+bnum1*(4*warpSize), 4*warpSize);

            if (laneid == 0) cnts[bnum1] = 0; //clean up for next round
        }
    }
    if (warpid == 0) {
        int32_t * cnts = buffer + ((4*warpSize-1)*(gridwidth+1));

        int32_t finished_old;
        if (laneid == 0) finished_old = atomicAdd(&finished, 1);
        finished_old = brdcst(finished_old, 0);

        if (finished_old == gridwidth - 1){ //every other block has finished
            int32_t buffelems   = buffer_size;
            int32_t start       = round_down(buffelems, 4*warpSize);
            int32_t *buffstart  = buffer + start;

            // vec4 tmp_out;
            #pragma unroll
            for (int k = 0 ; k < 4 ; ++k) buffer[k*warpSize + laneid] = buffstart[k*warpSize + laneid];
            // reinterpret_cast<vec4*>(buffer)[laneid] = tmp_out;
            // parent->consume_warp(buffstart, buffelems - start);

            if (laneid == 0) {
                buffer_size                     = buffelems - start;
                finished                        = 0;
                cnts[buffelems/(4*warpSize)]    = 0;
                cnts[0]                         = buffelems - start;
                assert(buffelems - start <= 4*warpSize);
            }
        }
    }

    __syncthreads();

    parent->consume_close();
}

template<size_t warp_size, typename F, typename... T>
__device__ void unstable_select<warp_size, F, T...>::consume_open(){
    parent->consume_open();

    __syncthreads();

    extern __shared__ int32_t s[];

    const int32_t laneid     = get_laneid();
    const int32_t warpid     = get_warpid();
    // const int32_t width      = blockDim.x * blockDim.y;

    volatile int32_t *wrapoutbase   = ((int32_t *) s);
    volatile int32_t *wrapoutput    = wrapoutbase + 5 * warp_size * warpid;
    volatile int32_t *fcount        = wrapoutput  + 5 * warp_size - 1;

    if (laneid == 0) *fcount = 0;
}

template<size_t warp_size, typename F, typename... T>
__device__ void unstable_select<warp_size, F, T...>::at_open(){}

template<size_t warp_size, typename F, typename... T>
__device__ void unstable_select<warp_size, F, T...>::at_close(){
    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0){
        const uint32_t laneid = get_laneid();

        parent->consume_open();

        __syncthreads();
        if (get_warpid() == 0){
            parent->consume_warp(buffer, buffer_size);
        }
        __syncthreads();

        parent->consume_close();
    }
}

template<size_t warp_size, typename F, typename... T>
__host__ void unstable_select<warp_size, F, T...>::before_open(){
    decltype(this->parent) p;
    gpu(cudaMemcpy(&p, &(this->parent), sizeof(decltype(this->parent)), cudaMemcpyDefault));
    p->open();
}

template<size_t warp_size, typename F, typename... T>
__host__ void unstable_select<warp_size, F, T...>::after_close(){
    decltype(this->parent) p;
    gpu(cudaMemcpy(&p, &(this->parent), sizeof(decltype(this->parent)), cudaMemcpyDefault));
    p->close();
}

template class unstable_select<WARPSIZE, less_eq_than<int32_t>, int32_t>;
