#include "h_colsplit.cuh"

template<typename TL, typename TR, typename... Tload>
__host__ h_colsplit<TL, TR, Tload...>::h_colsplit(h_operator<TL, Tload...> parentL, h_operator<TR, Tload...> parentR, cid_t cidL, cid_t cidR): 
        parentL(parentL), parentR(parentR), cidL(cidL), cidR(cidR){}


template<typename TL, typename TR, typename... Tload>
__host__ void h_colsplit<TL, TR, Tload...>::open(){
    parentL.open();
    parentR.open();
}

template<typename TL, typename TR, typename... Tload>
__host__ void h_colsplit<TL, TR, Tload...>::consume(const TL * __restrict__ srcL, const TR * __restrict__ srcR, const Tload * __restrict__ ... sel, cnt_t N, vid_t vid, cid_t cid){
    parentL.consume(srcL, sel..., N, vid, cidL);
    parentR.consume(srcR, sel..., N, vid, cidR);
}

template<typename TL, typename TR, typename... Tload>
__host__ void h_colsplit<TL, TR, Tload...>::close(){
    parentL.close();
    parentR.close();
}

template class h_colsplit<int32_t, int32_t>;
template class h_colsplit<int32_t, int32_t, sel_t>;