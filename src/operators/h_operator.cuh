#ifndef H_OPERATOR_CUH_
#define H_OPERATOR_CUH_

#include "../common.cuh"

using namespace std;

template<typename Op, typename... Tin>
__host__ void __forceinline__ consume(void * p, cnt_t N, vid_t v, cid_t c, const Tin * ... src){
    ((Op *) p)->consume(src..., N, v, c);
}

template<typename Op>
__host__ void __forceinline__ open(void * p){
    ((Op *) p)->open();
}

template<typename Op>
__host__ void __forceinline__ close(void * p){
    ((Op *) p)->close();
}

template<typename... Tin>
class h_operator{
    typedef void (*consume_t)(void *, cnt_t, vid_t, cid_t, const Tin * ... s);
    typedef void (*close_t  )(void *);
    typedef void (*open_t   )(void *);
private:
    void           * op         ;
    consume_t        cons       ;
    open_t           op_open    ;
    close_t          op_close   ;
public:
    __host__ h_operator(): op(NULL){}

    template<typename Op>
    __host__ h_operator(Op * op):   op      (op                     ),
                                    cons    (&::consume<Op, Tin...> ),
                                    op_open (&::open<Op>            ),
                                    op_close(&::close<Op>           ){}

    __host__ void __forceinline__ consume(const Tin * ... src, cnt_t N, vid_t v, cid_t c){
        cons(op, N, v, c, src...);
    }

    __host__ void __forceinline__ consume(const tuple<const Tin * ...> &src, cnt_t N, vid_t v, cid_t c){
        call(static_cast<void (h_operator<Tin...>::*)(const Tin * ... src, cnt_t N, vid_t v, cid_t c)>(&h_operator<Tin...>::consume), this, tuple_cat(src, make_tuple(N, v, c)));
    }

    __host__ void __forceinline__ open(){
        op_open(op);
    }

    __host__ void __forceinline__ close(){
        op_close(op);
    }
};


#endif /* H_OPERATOR_CUH_ */