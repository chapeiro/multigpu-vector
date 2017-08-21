#ifndef D_OPERATOR_CUH_
#define D_OPERATOR_CUH_

#include "../common.cuh"

using namespace std;


template<typename Op, typename... Tin>
__device__ void __forceinline__ consume_warp(void * p, cnt_t N, vid_t v, cid_t c, const Tin * ... src){
    ((Op *) p)->consume_warp(src..., N, v, c);
}

template<typename Op>
__device__ void __forceinline__ consume_open(void * p){
    ((Op *) p)->consume_open();
}

template<typename Op>
__device__ void __forceinline__ consume_close(void * p){
    ((Op *) p)->consume_close();
}

template<typename Op>
__device__ void __forceinline__ at_open(void * p){
    ((Op *) p)->at_open();
}

template<typename Op>
__device__ void __forceinline__ at_close(void * p){
    ((Op *) p)->at_close();
}

template<typename Op>
__host__ void __forceinline__ before_open(void * p){
    ((Op *) p)->before_open();
}

template<typename Op>
__host__ void __forceinline__ after_close(void * p){
    ((Op *) p)->after_close();
}

template<typename Op, typename... Tin>
__global__ void get_addresses(
                    void (**cons_warp)(void *, cnt_t, vid_t, cid_t, const Tin * ... s),
                    void (**cons_open)(void *),
                    void (**cons_close)(void *),
                    void (**op_open)(void *),
                    void (**op_close)(void *)
                ){
    *cons_warp  = &consume_warp <Op, Tin...>;
    *cons_open  = &consume_open <Op>;
    *cons_close = &consume_close<Op>;
    *op_open    = &at_open      <Op>;
    *op_close   = &at_close     <Op>;
}

template<typename... Tin>
class d_operator;

template<typename... Tin>
__global__ void launch_open_pipeline4(d_operator<Tin...> *op){
    op->op_open(op->op);
}

template<typename... Tin>
__global__ void launch_close_pipeline4(d_operator<Tin...> *op){
    op->op_close(op->op);
}

template<typename... Tin>
class d_operator{
    typedef void (*consume_warp_t )(void *, cnt_t, vid_t, cid_t, const Tin * ... s);
    typedef void (*consume_open_t )(void *);
    typedef void (*consume_close_t)(void *);
    typedef void (*at_close_t     )(void *);
    typedef void (*at_open_t      )(void *);
    typedef void (*after_close_t  )(void *);
    typedef void (*before_open_t  )(void *);
private:
    void           * op;
    consume_warp_t   cons_warp  ;
    consume_open_t   cons_open  ;
    consume_close_t  cons_close ;
    at_open_t        op_open    ;
    at_close_t       op_close   ;
    after_close_t    after_close;
    before_open_t    before_open;

    launch_conf      conf;

    friend void launch_close_pipeline4<Tin...>(d_operator<Tin...> *op);
    friend void launch_open_pipeline4 <Tin...>(d_operator<Tin...> *op);

public:
    __host__ d_operator(): op(NULL){}

    template<typename Op>
    __host__ d_operator(launch_conf lc, Op * op): op(op), conf(lc){
        consume_warp_t * cons_warp ;
        consume_open_t * cons_open ;
        consume_close_t* cons_close;
        at_open_t      * op_open   ;
        at_close_t     * op_close  ;

        set_device_on_scope d(lc.device);
        assert(lc.device == get_device(op));

        gpu(cudaMallocHost(&cons_warp , sizeof(consume_warp_t )));
        gpu(cudaMallocHost(&cons_open , sizeof(consume_open_t )));
        gpu(cudaMallocHost(&cons_close, sizeof(consume_close_t)));
        gpu(cudaMallocHost(&op_open   , sizeof(at_open_t      )));
        gpu(cudaMallocHost(&op_close  , sizeof(at_close_t     )));
        get_addresses<Op, Tin...><<<1, 1, 0, 0>>>(cons_warp, cons_open, cons_close, op_open, op_close);
        gpu(cudaDeviceSynchronize());
        this->cons_warp   = *cons_warp ;
        this->cons_open   = *cons_open ;
        this->cons_close  = *cons_close;
        this->op_open     = *op_open   ;
        this->op_close    = *op_close  ;
        this->before_open = &::before_open<Op>;
        this->after_close = &::after_close<Op>;


        gpu(cudaFreeHost(cons_warp ));
        gpu(cudaFreeHost(cons_open ));
        gpu(cudaFreeHost(cons_close));
        gpu(cudaFreeHost(op_open   ));
        gpu(cudaFreeHost(op_close  ));
    }

    __device__ void __forceinline__ consume_warp(const Tin * ... src, cnt_t N, vid_t v, cid_t c){
        cons_warp(op, N, v, c, src...);
    }

    __device__ void __forceinline__ consume_warp(const tuple<const Tin * ...> &src, cnt_t N, vid_t v, cid_t c){
        call(static_cast<void (d_operator<Tin...>::*)(const Tin * ... src, cnt_t N, vid_t v, cid_t c)>(&d_operator<Tin...>::consume_warp), this, tuple_cat(src, make_tuple(N, v, c)));
    }

    __device__ void __forceinline__ consume_open(){
        cons_open(op);
    }

    __device__ void __forceinline__ consume_close(){
        cons_close(op);
    }

    __host__ void __forceinline__ open(){
        d_operator<Tin...> tmp;
        gpu(cudaMemcpy(&tmp, this, sizeof(d_operator<Tin...>), cudaMemcpyDefault));
        assert(tmp.conf.device == get_device(this));

        set_device_on_scope d(tmp.conf.device);
        tmp.before_open(tmp.op);

        launch_open_pipeline4<<<tmp.conf.gridDim, tmp.conf.blockDim, tmp.conf.shared_mem, 0>>>(this);
        gpu(cudaDeviceSynchronize());
    }

    __host__ void __forceinline__ close(){
        d_operator<Tin...> tmp;
        gpu(cudaMemcpy(&tmp, this, sizeof(d_operator<Tin...>), cudaMemcpyDefault));
        assert(tmp.conf.device == get_device(this));

        set_device_on_scope d(tmp.conf.device);
        launch_close_pipeline4<<<tmp.conf.gridDim, tmp.conf.blockDim, tmp.conf.shared_mem, 0>>>(this);
        gpu(cudaDeviceSynchronize());

        tmp.after_close(tmp.op);
    }

};


#endif /* D_OPERATOR_CUH_ */