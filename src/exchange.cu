#include <iostream>
#include <chrono>
#include <algorithm>
#include "common.cuh"

using namespace std;

#ifndef NVAL
#define NVAL (10000*1024*4)
#endif

constexpr int N = NVAL;




__global__ void gen_test(cudaEvent_t *store_to, cudaEvent_t *ready, cudaEvent_t *consumed){
    // gpu(cudaEventCreate(store_to));
    gpu(cudaEventRecord(*ready));
    gpu(cudaEventSynchronize(*consumed));
}


__global__ void cons_test(cudaEvent_t *store_to, cudaEvent_t *ready, cudaEvent_t *consumed){
    gpu(cudaEventSynchronize(*ready));
    gpu(cudaEventRecord(*consumed));
}


int32_t *a;
int32_t *b;

int main(){
    cudaEvent_t store_to, ready, consumed;
    cudaStream_t stra, strb;

    gpu(cudaEventCreate(&ready));

    gpu(cudaStreamCreate(&stra));
    gpu(cudaStreamCreate(&strb));


    cons_test<<<1, 32, 0, stra>>>(&store_to, &ready, &consumed);
    gen_test <<<1, 32, 0, strb>>>(&store_to, &ready, &consumed);

    cudaDeviceSynchronize();
    cudaDeviceReset();

    return EXIT_SUCCESS;
}
