#ifndef ARGUMENT_PARSING_CUH_
#define ARGUMENT_PARSING_CUH_

#include "common.cuh"
#include <iostream>

#ifndef NVAL
#define NVAL (64*1024*1024*4)
#endif

#ifndef MVAL
#define MVAL NVAL
#endif

struct params_t{
    size_t  N;
    size_t  M;
    int32_t thres;

    params_t(): N(NVAL), M(MVAL), thres(5000){}

    friend std::ostream& operator<< (std::ostream& out, const params_t& p);
};

int parse_args(int argc, char *argv[], params_t &params);


#endif /* ARGUMENT_PARSING_CUH_ */