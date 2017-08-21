#include "colsplit.cuh"

template<typename TL, typename TR, typename... Tload>
__host__ colsplit<TL, TR, Tload...>::colsplit(d_operator<TL, Tload...> parentL, d_operator<TR, Tload...> parentR, cid_t cidL, cid_t cidR, const launch_conf &conf): 
        parentL(parentL), parentR(parentR), cidL(cidL), cidR(cidR){}


template<typename TL, typename TR, typename... Tload>
__host__   void colsplit<TL, TR, Tload...>::before_open(){
    parentL.open();
    parentR.open();
}

template<typename TL, typename TR, typename... Tload>
__device__ void colsplit<TL, TR, Tload...>::at_open(){}

template<typename TL, typename TR, typename... Tload>
__device__ void colsplit<TL, TR, Tload...>::consume_open(){
    parentL.consume_open();
    parentR.consume_open();
}

template<typename TL, typename TR, typename... Tload>
__device__ void colsplit<TL, TR, Tload...>::consume_warp(const TL * __restrict__ srcL, const TR * __restrict__ srcR, const Tload * __restrict__ ... sel, cnt_t N, vid_t vid, cid_t cid){
    parentL.consume_warp(srcL, sel..., N, vid, cidL);
    parentR.consume_warp(srcR, sel..., N, vid, cidR);
}

template<typename TL, typename TR, typename... Tload>
__device__ void colsplit<TL, TR, Tload...>::consume_close(){
    parentL.consume_close();
    parentR.consume_close();
}

template<typename TL, typename TR, typename... Tload>
__device__ void colsplit<TL, TR, Tload...>::at_close(){}

template<typename TL, typename TR, typename... Tload>
__host__   void colsplit<TL, TR, Tload...>::after_close(){
    parentL.close();
    parentR.close();
}

template class colsplit<int32_t, int32_t>;
template class colsplit<int32_t, int32_t, sel_t>;