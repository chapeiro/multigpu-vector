#ifndef H_COLSPLIT_CUH_
#define H_COLSPLIT_CUH_

#include "h_operator.cuh"

using namespace std;

template<typename TL, typename TR, typename... Tload>
class h_colsplit{
private:
    h_operator<TL, Tload...>    parentL;
    h_operator<TR, Tload...>    parentR;
    cid_t                       cidL;
    cid_t                       cidR;
public:
    __host__ h_colsplit(h_operator<TL, Tload...> parentL, h_operator<TR, Tload...> parentR, cid_t cidL, cid_t cidR);

    __host__ void open();

    __host__ void consume(const TL * __restrict__ srcL, const TR * __restrict__ srcR, const Tload * __restrict__ ... sel, cnt_t N, vid_t vid, cid_t cid);

    __host__ void close();
};

#endif /* H_COLSPLIT_CUH_ */